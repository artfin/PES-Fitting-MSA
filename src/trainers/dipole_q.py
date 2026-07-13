import logging

import torch

from config import TORCH_FLOAT
from build_model import QModel
from losses import WRMSELoss_Ratio_dipole
from distributed import is_main_process, reduce_mean
from .base import BaseTrainer, PRINT_PRECISION, USE_WANDB

if USE_WANDB:
    import wandb


class DipoleQTrainer(BaseTrainer):
    """Charge-based dipole fitting with neutrality constraint."""

    def build_model(self):
        cfg_model = self.cfg.get('MODEL', None)
        return QModel(cfg_model, input_features=self.train.NPOLY, output_features=[len(natoms) for natoms in self.train.symmetry.values()])

    def build_loss(self):
        # base.__init__ reads cfg_loss['USE_GRADIENTS'] right after build_loss();
        # dipole configs don't declare the gradient keys, so mirror the defaults.
        self.cfg_loss.setdefault('LAMBDA_Q', 1.0e3)
        self.cfg_loss.setdefault('USE_GRADIENTS_AFTER_EPOCH', None)
        self.cfg_loss.setdefault('USE_GRADIENTS', False)

        assert self.cfg_loss['NAME'] == 'WRMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'Ratio', \
            "Dipole trainers require LOSS NAME=WRMSE, WEIGHT_TYPE=Ratio"
        dwt = self.cfg_loss.get('DWT', 1.0)
        loss_fn = WRMSELoss_Ratio_dipole(dwt=dwt)
        logging.info("Build loss function: {}".format(loss_fn))
        return loss_fn

    def prepare_data_for_device(self):
        self.train.X = self.train.X.to(self.device)
        self.train.y = self.train.y.to(self.device)
        self.val.X = self.val.X.to(self.device)
        self.val.y = self.val.y.to(self.device)
        self.train.xyz_ordered = self.train.xyz_ordered.to(self.device)
        self.val.xyz_ordered = self.val.xyz_ordered.to(self.device)
        self.test.xyz_ordered = self.test.xyz_ordered.to(self.device)

    def compute_loss(self, separate=False):
        q_pred   = self.model(self.train.X)
        X_inf    = torch.zeros_like(self.train.X).cpu()
        X_inf_tr = torch.from_numpy(self.xscaler.transform(X_inf)).to(self.device)
        q_inf    = self.model(X_inf_tr)
        q_corr   = q_pred - q_inf
        dip_pred = torch.einsum('ijk,ij->ik', self.train.xyz_ordered.to(TORCH_FLOAT), q_corr)
        qsum     = torch.sum(q_corr, dim=1)
        qreg     = self.cfg_loss['LAMBDA_Q'] * torch.mean(qsum * qsum)
        loss     = self.loss_fn(self.train.y, dip_pred)
        loss = loss + qreg

        if self.regularization is not None:
            loss = loss + self.regularization(self.model)
        return loss

    def evaluate_and_log(self, epoch, current_lr):
        # To disable the gradient calculation, set the .requires_grad attribute of all parameters to False 
        # or wrap the forward pass into with torch.no_grad().
        with torch.no_grad():
            train_q_pred   = self.model(self.train.X)
            train_X_inf    = torch.zeros_like(self.train.X).cpu()
            train_X_inf_tr = torch.from_numpy(self.xscaler.transform(train_X_inf)).to(self.device)
            train_q_inf    = self.model(train_X_inf_tr)
            train_q_corr   = train_q_pred - train_q_inf
            dip_pred_train = torch.einsum('ijk,ij->ik', self.train.xyz_ordered.to(TORCH_FLOAT), train_q_corr)
            loss_train     = self.loss_fn(self.train.y, dip_pred_train)

            val_q_pred   = self.model(self.val.X)
            val_X_inf    = torch.zeros_like(self.val.X).cpu()
            val_X_inf_tr = torch.from_numpy(self.xscaler.transform(val_X_inf)).to(self.device)
            val_q_inf    = self.model(val_X_inf_tr)
            val_q_corr   = val_q_pred - val_q_inf
            dip_pred_val = torch.einsum('ijk,ij->ik', self.val.xyz_ordered.to(TORCH_FLOAT), val_q_corr)
            loss_val     = self.loss_fn(self.val.y, dip_pred_val)

            train_qsum = torch.sum(train_q_corr, dim=1)
            train_qreg = self.cfg_loss['LAMBDA_Q'] * torch.mean(train_qsum * train_qsum)
            val_qsum   = torch.sum(val_q_corr, dim=1)
            val_qreg   = self.cfg_loss['LAMBDA_Q'] * torch.mean(val_qsum * val_qsum)

            if self.world_size > 1:
                loss_train = reduce_mean(loss_train)
                loss_val   = reduce_mean(loss_val)
                train_qreg = reduce_mean(train_qreg)
                val_qreg   = reduce_mean(val_qreg)

            # value to be passed to EarlyStopping/ReduceLR mechanisms
            self.loss_val = loss_val

        # log metrics to WANDB to visualize model performance
        if is_main_process() and USE_WANDB:
            wandb.log({"loss_train": loss_train, "loss_val": loss_val, "train_qreg": train_qreg, "val_qreg": val_qreg, "lr" : current_lr})

        self._log("Epoch: {0}; loss train: {2:.{1}f}; qreg train: {3:{1}f}; loss val: {4:.{1}f}; qreg val: {5:.{1}f}".format(
            epoch, PRINT_PRECISION, loss_train, train_qreg, loss_val, val_qreg
        ))

    def model_eval(self):
        self.test.X = self.test.X.to(self.device)
        self.test.y = self.test.y.to(self.device)

        self.model.eval()

        # To disable the gradient calculation, set the .requires_grad attribute of all parameters to False 
        # or wrap the forward pass into with torch.no_grad().
        with torch.no_grad():
            train_q_pred   = self.model(self.train.X)
            train_X_inf    = torch.zeros_like(self.train.X).cpu()
            train_X_inf_tr = torch.from_numpy(self.xscaler.transform(train_X_inf)).to(self.device)
            train_q_inf    = self.model(train_X_inf_tr)
            train_q_corr   = train_q_pred - train_q_inf
            dip_pred_train = torch.einsum('ijk,ij->ik', self.train.xyz_ordered.to(TORCH_FLOAT), train_q_corr)
            loss_train     = self.loss_fn(self.train.y, dip_pred_train)

            val_q_pred   = self.model(self.val.X)
            val_X_inf    = torch.zeros_like(self.val.X).cpu()
            val_X_inf_tr = torch.from_numpy(self.xscaler.transform(val_X_inf)).to(self.device)
            val_q_inf    = self.model(val_X_inf_tr)
            val_q_corr   = val_q_pred - val_q_inf
            dip_pred_val = torch.einsum('ijk,ij->ik', self.val.xyz_ordered.to(TORCH_FLOAT), val_q_corr)
            loss_val     = self.loss_fn(self.val.y, dip_pred_val)

            test_q_pred   = self.model(self.test.X)
            test_X_inf    = torch.zeros_like(self.test.X).cpu()
            test_X_inf_tr = torch.from_numpy(self.xscaler.transform(test_X_inf)).to(self.device)
            test_q_inf    = self.model(test_X_inf_tr)
            test_q_corr   = test_q_pred - test_q_inf
            dip_pred_test = torch.einsum('ijk,ij->ik', self.test.xyz_ordered.to(TORCH_FLOAT), test_q_corr)
            loss_test     = self.loss_fn(self.test.y, dip_pred_test)

        if self.world_size > 1:
            loss_train = reduce_mean(loss_train)
            loss_val   = reduce_mean(loss_val)
            loss_test  = reduce_mean(loss_test)

        self._log("Model evluation after training:")
        self._log("Train      loss: {1:{0}f}".format(PRINT_PRECISION, loss_train))
        self._log("Validation loss: {1:{0}f}".format(PRINT_PRECISION, loss_val))
        self._log("Test       loss: {1:{0}f}".format(PRINT_PRECISION, loss_test))
