import logging

import torch

from build_model import build_network
from losses import WRMSELoss_Ratio_dipole
from distributed import is_main_process, reduce_mean
from .base import BaseTrainer, PRINT_PRECISION, USE_WANDB

if USE_WANDB:
    import wandb


class DipoleGRMTrainer(BaseTrainer):
    """Dipole fitting via gradient response matrix (GRM)."""

    def build_model(self):
        cfg_model = self.cfg.get('MODEL', None)
        return build_network(cfg_model, hidden_dims=self.cfg['MODEL']['HIDDEN_DIMS'][0], input_features=self.train.NPOLY, output_features=3)

    def build_loss(self):
        # base.__init__ reads cfg_loss['USE_GRADIENTS'] right after build_loss();
        # dipole configs don't declare the gradient keys, so mirror the defaults.
        self.cfg_loss.setdefault('LAMBDA_Q', 1.0e3)
        self.cfg_loss.setdefault('USE_GRADIENTS_AFTER_EPOCH', None)
        self.cfg_loss.setdefault('USE_GRADIENTS', False)

        assert self.cfg_loss['NAME'] == 'WRMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'Ratio', \
            "Dipole trainers require LOSS NAME=WRMSE, WEIGHT_TYPE=Ratio"
        dwt = self.cfg_loss.get('dwt', 1.0)
        loss_fn = WRMSELoss_Ratio_dipole(dwt=dwt)
        logging.info("Build loss function: {}".format(loss_fn))
        return loss_fn

    def prepare_data_for_device(self):
        self.train.X = self.train.X.to(self.device)
        self.train.y = self.train.y.to(self.device)
        self.val.X = self.val.X.to(self.device)
        self.val.y = self.val.y.to(self.device)
        self.train.grm = self.train.grm.to(self.device)
        self.val.grm   = self.val.grm.to(self.device)

    def compute_loss(self, separate=False):
        y_pred = self.model(self.train.X)
        dip_pred = torch.einsum('ijk,ik->ij', self.train.grm, y_pred)
        loss = self.loss_fn(self.train.y, dip_pred)

        if self.regularization is not None:
            loss = loss + self.regularization(self.model)
        return loss

    def evaluate_and_log(self, epoch, current_lr):
        with torch.no_grad():
            train_y_pred   = self.model(self.train.X)
            dip_pred_train = torch.einsum('ijk,ik->ij', self.train.grm, train_y_pred)
            loss_train     = self.loss_fn(self.train.y, dip_pred_train)

            val_y_pred   = self.model(self.val.X)
            dip_pred_val = torch.einsum('ijk,ik->ij', self.val.grm, val_y_pred)
            loss_val     = self.loss_fn(self.val.y, dip_pred_val)

            if self.world_size > 1:
                # DIPOLE mode: use loss as proxy for RMSE, no trust region
                self.log_distributed_diagnostics(
                    epoch,
                    loss_local=loss_val.item(),
                    e_rmse_local=loss_val.item(),  # Use loss as proxy
                    use_trust_region=False
                )
                loss_train = reduce_mean(loss_train)
                loss_val   = reduce_mean(loss_val)

            # value to be passed to EarlyStopping/ReduceLR mechanisms
            self.loss_val = loss_val

        # log metrics to WANDB to visualize model performance
        if is_main_process() and USE_WANDB:
            wandb.log({"loss_train": loss_train, "loss_val": loss_val})

        self._log("Epoch: {0}; loss train: {2:.{1}f}; loss val: {3:.{1}f}".format(epoch, PRINT_PRECISION, loss_train, loss_val))

    def model_eval(self):
        self.test.X = self.test.X.to(self.device)
        self.test.y = self.test.y.to(self.device)
        self.test.grm = self.test.grm.to(self.device)

        self.model.eval()
        with torch.no_grad():
            dip_train = torch.einsum('ijk,ik->ij', self.train.grm, self.model(self.train.X))
            loss_train = self.loss_fn(self.train.y, dip_train)
            dip_val = torch.einsum('ijk,ik->ij', self.val.grm, self.model(self.val.X))
            loss_val = self.loss_fn(self.val.y, dip_val)
            dip_test = torch.einsum('ijk,ik->ij', self.test.grm, self.model(self.test.X))
            loss_test = self.loss_fn(self.test.y, dip_test)

        if self.world_size > 1:
            loss_train = reduce_mean(loss_train)
            loss_val   = reduce_mean(loss_val)
            loss_test  = reduce_mean(loss_test)

        self._log("Model evaluation after training:")
        self._log("Train      loss: {1:.{0}f}".format(PRINT_PRECISION, loss_train))
        self._log("Validation loss: {1:.{0}f}".format(PRINT_PRECISION, loss_val))
        self._log("Test       loss: {1:.{0}f}".format(PRINT_PRECISION, loss_test))
