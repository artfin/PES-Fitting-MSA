import logging

import torch

from build_model import build_network
from losses import WMSELoss_Ratio, WRMSELoss_Ratio, WMSELoss_Boltzmann, WRMSELoss_Boltzmann, WMSELoss_PS, WRMSELoss_PS
from distributed import reduce_mean, is_main_process
from .base import BaseTrainer, DEVICE, PRINT_PRECISION, USE_WANDB
from .multibatch import MultibatchMixin

if USE_WANDB:
    import wandb


class EnergyTrainer(MultibatchMixin, BaseTrainer):
    """Energy-only trainer. Simplest case - no gradients, no dipoles."""

    def build_model(self):
        cfg_model = self.cfg.get('MODEL', None)
        return build_network(
            cfg_model,
            hidden_dims=self.cfg['MODEL']['HIDDEN_DIMS'],
            input_features=self.train.NPOLY,
            output_features=1
        )

    def build_loss(self):
        known_options = (
            'NAME', 'WEIGHT_TYPE', 'DWT', 'EREF', 'EMAX',
            'FOCAL_GAMMA', 'FOCAL_EMA_DECAY',
            # These are allowed in config but not used by EnergyTrainer
            'USE_GRADIENTS', 'USE_GRADIENTS_AFTER_EPOCH', 'G_LAMBDA',
            'G_LAMBDA_RAMP_EPOCHS', 'TRUST_THRESHOLD', 'TRUST_THRESHOLD_START',
            'TRUST_THRESHOLD_RAMP_EPOCHS', 'TRUST_SOFT_SCALE', 'TRUST_SOFT_CUTOFF',
            'GRADIENT_TRUST_THRESHOLD', 'GRADIENT_TRUST_SOFT_SCALE',
            'USE_HUBER_GRADIENT', 'HUBER_DELTA', 'USE_MGDA',
            'MGDA_ALPHA_MIN', 'MGDA_ALPHA_MAX', 'MGDA_EMA_DECAY', 'LAMBDA_Q'
        )
        for option in self.cfg_loss.keys():
            assert option.upper() in known_options, "[build_loss] unknown option: {}".format(option)

        # base.__init__ and base.train_epoch read these unconditionally, but the
        # gradient-specific keys are absent from pure-energy configs. Mirror the
        # defaults that base.build_loss sets so EnergyTrainer configs don't KeyError.
        self.cfg_loss.setdefault('USE_GRADIENTS', False)
        self.cfg_loss.setdefault('USE_GRADIENTS_AFTER_EPOCH', None)
        self.cfg_loss.setdefault('FOCAL_GAMMA', 0.0)
        self.cfg_loss.setdefault('FOCAL_EMA_DECAY', 0.95)

        loss_name = self.cfg_loss['NAME']
        weight_type = self.cfg_loss['WEIGHT_TYPE']
        dwt = self.cfg_loss.get('DWT', 1.0)
        focal_gamma = self.cfg_loss.get('FOCAL_GAMMA', 0.0)
        focal_ema_decay = self.cfg_loss.get('FOCAL_EMA_DECAY', 0.95)

        if loss_name == 'WMSE' and weight_type == 'Ratio':
            loss_fn = WMSELoss_Ratio(dwt=dwt, focal_gamma=focal_gamma, focal_ema_decay=focal_ema_decay)
        elif loss_name == 'WRMSE' and weight_type == 'Ratio':
            loss_fn = WRMSELoss_Ratio(dwt=dwt, focal_gamma=focal_gamma, focal_ema_decay=focal_ema_decay)
        elif loss_name == 'WMSE' and weight_type == 'Boltzmann':
            Eref = self.cfg_loss.get('EREF', 2000.0)
            loss_fn = WMSELoss_Boltzmann(Eref=Eref)
        elif loss_name == 'WRMSE' and weight_type == 'Boltzmann':
            Eref = self.cfg_loss.get('EREF', 2000.0)
            loss_fn = WRMSELoss_Boltzmann(Eref=Eref)
        elif loss_name == 'WMSE' and weight_type == 'PS':
            Emax = self.cfg_loss.get('EMAX', 2000.0)
            loss_fn = WMSELoss_PS(Emax=Emax)
        elif loss_name == 'WRMSE' and weight_type == 'PS':
            Emax = self.cfg_loss.get('EMAX', 2000.0)
            loss_fn = WRMSELoss_PS(Emax=Emax)
        else:
            raise ValueError("EnergyTrainer: unsupported loss config: NAME={}, WEIGHT_TYPE={}".format(
                loss_name, weight_type))

        logging.info("Build loss function: {}".format(loss_fn))
        return loss_fn

    def prepare_data_for_device(self):
        """Move energy tensors to device."""
        multibatch = bool(self.cfg_batch.get('MULTIBATCH_ENABLED', False))

        if multibatch:
            if torch.cuda.is_available():
                self.train.X = self.train.X.pin_memory()
                self.train.y = self.train.y.pin_memory()
            self.val.X = self.val.X.to(self.device)
            self.val.y = self.val.y.to(self.device)
        else:
            self.train.X = self.train.X.to(self.device)
            self.train.y = self.train.y.to(self.device)
            self.val.X = self.val.X.to(self.device)
            self.val.y = self.val.y.to(self.device)

    def prepare_epoch(self, epoch):
        """No special preparation needed for energy-only training."""
        pass

    def compute_loss(self, separate=False):
        """Compute energy-only loss."""
        y_pred = self.model(self.train.X)
        loss = self.loss_fn(self.train.y, y_pred)

        if self.regularization is not None:
            loss = loss + self.regularization(self.model)

        if separate:
            return loss, torch.tensor(0.0, device=self.device)
        return loss

    def evaluate_and_log(self, epoch, current_lr):
        """Evaluate model and log metrics for energy-only training."""
        with torch.no_grad():
            train_y_pred = self.model(self.train.X)
            loss_train = self.loss_fn(self.train.y, train_y_pred)

            val_y_pred = self.model(self.val.X)
            loss_val = self.loss_fn(self.val.y, val_y_pred)

            if self.world_size > 1:
                loss_train = reduce_mean(loss_train)
                loss_val = reduce_mean(loss_val)

            self.loss_val = loss_val

        if self.writer is not None:
            self.writer.add_scalar("loss/train", loss_train, epoch)
            self.writer.add_scalar("loss/val", loss_val, epoch)
            self.writer.add_scalar("lr", current_lr, epoch)

        if is_main_process() and USE_WANDB:
            wandb.log({"loss_train": loss_train, "loss_val": loss_val, "lr": current_lr})

        self._log("Epoch: {0}; loss train: {2:.{1}f} cm-1; loss val: {3:.{1}f} cm-1; lr: {4:.2e}".format(
            epoch, PRINT_PRECISION, loss_train, loss_val, current_lr))

    def model_eval(self):
        """Final evaluation on train/val/test sets."""
        self.test.X = self.test.X.to(self.device)
        self.test.y = self.test.y.to(self.device)

        self.model.eval()

        with torch.no_grad():
            pred_train = self.model(self.train.X)
            loss_train = self.loss_fn(self.train.y, pred_train)

            pred_val = self.model(self.val.X)
            loss_val = self.loss_fn(self.val.y, pred_val)

            pred_test = self.model(self.test.X)
            loss_test = self.loss_fn(self.test.y, pred_test)

        if self.world_size > 1:
            loss_train = reduce_mean(loss_train)
            loss_val = reduce_mean(loss_val)
            loss_test = reduce_mean(loss_test)

        self._log("Model evaluation after training:")
        self._log("Train      loss: {1:.{0}f} cm-1".format(PRINT_PRECISION, loss_train))
        self._log("Validation loss: {1:.{0}f} cm-1".format(PRINT_PRECISION, loss_val))
        self._log("Test       loss: {1:.{0}f} cm-1".format(PRINT_PRECISION, loss_test))

    def supports_mgda(self):
        """Energy-only trainer does not support MGDA."""
        return False
