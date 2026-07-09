import logging

import torch

from build_model import build_network
from losses import WRMSELoss_Ratio_dipole
from distributed import is_main_process, reduce_mean
from .base import BaseTrainer, PRINT_PRECISION, USE_WANDB

if USE_WANDB:
    import wandb


class DipoleCTrainer(BaseTrainer):
    """Direct dipole component prediction."""

    def build_model(self):
        cfg_model = self.cfg.get('MODEL', None)
        return build_network(cfg_model, input_features=3 * self.train.NATOMS, output_features=1)

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

    def compute_loss(self, separate=False):
        dip_pred = self.model(self.train.X)
        loss = self.loss_fn(self.train.y, dip_pred)

        if self.regularization is not None:
            loss = loss + self.regularization(self.model)
        return loss

    def evaluate_and_log(self, epoch, current_lr):
        with torch.no_grad():
            train_dip_pred = self.model(self.train.X)
            loss_train = self.loss_fn(self.train.y, train_dip_pred)

            val_dip_pred = self.model(self.val.X)
            loss_val = self.loss_fn(self.val.y, val_dip_pred)

            if self.world_size > 1:
                loss_train = reduce_mean(loss_train)
                loss_val   = reduce_mean(loss_val)

            self.loss_val = loss_val

        self._log("Epoch: {0}; loss train: {2:.{1}f}; loss val: {3:.{1}f}".format(epoch, PRINT_PRECISION, loss_train, loss_val))

    def model_eval(self):
        self.test.X = self.test.X.to(self.device)
        self.test.y = self.test.y.to(self.device)

        self.model.eval()
        with torch.no_grad():
            loss_train = self.loss_fn(self.train.y, self.model(self.train.X))
            loss_val   = self.loss_fn(self.val.y, self.model(self.val.X))
            loss_test  = self.loss_fn(self.test.y, self.model(self.test.X))

        if self.world_size > 1:
            loss_train = reduce_mean(loss_train)
            loss_val   = reduce_mean(loss_val)
            loss_test  = reduce_mean(loss_test)

        self._log("Model evaluation after training:")
        self._log("Train      loss: {1:.{0}f}".format(PRINT_PRECISION, loss_train))
        self._log("Validation loss: {1:.{0}f}".format(PRINT_PRECISION, loss_val))
        self._log("Test       loss: {1:.{0}f}".format(PRINT_PRECISION, loss_test))
