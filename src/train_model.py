import argparse
import collections
import json
import sys
import logging
import random
import os
import time
import timeit
import yaml

import torch.nn
from torch.utils.tensorboard import SummaryWriter

#from pydrive.auth import GoogleAuth
#from pydrive.drive import GoogleDrive

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from dataset import PolyDataset
from make_dataset import make_dataset, make_dataset_fpaths
from build_model import build_network_yaml

import pathlib
BASEDIR = pathlib.Path(__file__).parent.parent.resolve()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRINT_TRAINING_STEPS = 1
PRINT_PRECISION      = 3

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def preprocess_dataset(train, val, test, cfg):
    if cfg['NORMALIZE'] == 'std':
        xscaler = StandardScaler()
        yscaler = StandardScaler()
    else:
        raise ValueError("unreachable")

    train.X = torch.from_numpy(xscaler.fit_transform(train.X))

    try:
        val.X   = torch.from_numpy(xscaler.transform(val.X))
        test.X  = torch.from_numpy(xscaler.transform(test.X))
    except ValueError:
        logging.error("[preprocess_dataset] caught ValueError")
        val.X  = torch.empty((1, 1))
        test.X = torch.empty((1, 1))

    train.y = torch.from_numpy(yscaler.fit_transform(train.y))

    try:
        val.y   = torch.from_numpy(yscaler.transform(val.y))
        test.y  = torch.from_numpy(yscaler.transform(test.y))
    except ValueError:
        logging.error("[preprocess_dataset] caught ValueError")
        val.y = torch.empty(1)
        test.y = torch.empty(1)

    return xscaler, yscaler

    #symmetry = train.symmetry
    #order    = train.order
    #DATASETS_INTERIM = "datasets/interim"
    #BASENAME         = "poly_{}_{}".format(symmetry.replace(" ", "_"), order)
    #interim_train = os.path.join(DATASETS_INTERIM, BASENAME + "-train-norm.json")
    #with open(interim_train, 'w') as fp:
    #    json.dump(dict(NATOMS=train.NATOMS, NMON=train.NMON, NPOLY=train.NPOLY, symmetry=train.symmetry, order=train.order,
    #                       X=train.X.tolist(), y=train.y.tolist()), fp)

    #interim_test = os.path.join(DATASETS_INTERIM, BASENAME  + "-test-norm.json")
    #with open(interim_test, 'w') as fp:
    #    json.dump(dict(NATOMS=test.NATOMS, NMON=test.NMON, NPOLY=test.NPOLY, symmetry=test.symmetry, order=test.order,
    #                       X=test.X.tolist(), y=test.y.tolist()), fp)

    #print("xscaler.mean: {}".format(xscaler.mean_))
    #print("xscaler.std:  {}".format(xscaler.scale_))

    #plt.figure(figsize=(10, 10))
    #plt.scatter(train.X[:,6], np.zeros_like(train.X[:,6]) + 1)
    #plt.show()

def save_checkpoint(model, xscaler, yscaler, meta_info, chk_path):
    logging.info("Saving the checkpoint.")

    #architecture = [m.out_features for m in next(model.modules()) if isinstance(m, torch.nn.modules.linear.Linear)]
    #architecture = tuple(architecture[:-1])

    #import inspect
    #torch_activations = list(zip(*inspect.getmembers(torch.nn.modules.activation, inspect.isclass)))[0]
    #for module in model.modules():
    #    module_str = repr(module).strip("()")
    #    if module_str in torch_activations:
    #        activation = module_str
    #        break

    checkpoint = {
        "model"        :  model.state_dict(),
        "X_mean"       :  xscaler.mean_,
        "X_std"        :  xscaler.scale_,
        "y_mean"       :  yscaler.mean_,
        "y_std"        :  yscaler.scale_,
        "meta_info"    :  meta_info,
    }
    torch.save(checkpoint, chk_path)

class L1Regularization(torch.nn.Module):
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = lambda_

    def __repr__(self):
        return "L1Regularization(lambda={})".format(self.lambda_)

    def forward(self, model):
        l1_norm = torch.tensor(0.).to(dtype=torch.float64, device=DEVICE)
        for p in model.parameters():
            l1_norm += p.abs().sum()

        return self.lambda_ * l1_norm

class L2Regularization(torch.nn.Module):
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = lambda_

    def __repr__(self):
        return "L2Regularization(lambda={})".format(self.lambda_)

    def forward(self, model):
        l2_norm = torch.tensor(0.).to(DEVICE)
        for p in model.parameters():
            l2_norm += (p**2).sum()
        return self.lambda_ * l2_norm

class WMSELoss_Boltzmann(torch.nn.Module):
    def __init__(self, Eref):
        super().__init__()
        self.Eref = torch.tensor(Eref).to(DEVICE)

        self.y_mean = None
        self.y_std  = None

    def set_scale(self, y_mean, y_std):
        self.y_mean = torch.FloatTensor(y_mean.tolist()).to(DEVICE)
        self.y_std  = torch.FloatTensor(y_std.tolist()).to(DEVICE)

    def __repr__(self):
        return "WMSELoss_Boltzmann(Eref={})".format(self.Eref)

    def forward(self, y, y_pred):
        assert self.y_mean is not None
        assert self.y_std is not None

        # descale energies
        yd      = y      * self.y_std + self.y_mean
        yd_pred = y_pred * self.y_std + self.y_mean

        w = torch.exp(-yd / self.Eref)
        w = w / w.max()

        wmse = (w * (yd - yd_pred)**2).mean()
        return wmse

class WRMSELoss_Boltzmann(torch.nn.Module):
    def __init__(self, Eref):
        super().__init__()
        self.Eref = torch.tensor(Eref).to(DEVICE)

        self.y_mean = None
        self.y_std  = None

    def set_scale(self, y_mean, y_std):
        self.y_mean = torch.FloatTensor(y_mean.tolist()).to(DEVICE)
        self.y_std  = torch.FloatTensor(y_std.tolist()).to(DEVICE)

    def __repr__(self):
        return "WRMSELoss_Boltzmann(Eref={})".format(self.Eref)

    def forward(self, y, y_pred):
        assert self.y_mean is not None
        assert self.y_std is not None

        # descale energies
        yd      = y      * self.y_std + self.y_mean
        yd_pred = y_pred * self.y_std + self.y_mean

        w = torch.exp(-yd / self.Eref)
        w = w / w.max()

        wmse = (w * (yd - yd_pred)**2).mean()
        return torch.sqrt(wmse)

class WMSELoss_Ratio(torch.nn.Module):
    def __init__(self, dwt=1.0):
        super().__init__()
        self.dwt    = torch.tensor(dwt).to(DEVICE)

        self.y_mean = None
        self.y_std  = None

    def set_scale(self, y_mean, y_std):
        self.y_mean = torch.FloatTensor(y_mean.tolist()).to(DEVICE)
        self.y_std  = torch.FloatTensor(y_std.tolist()).to(DEVICE)

    def __repr__(self):
        return "WMSELoss_Ratio(dwt={})".format(self.dwt)

    def forward(self, y, y_pred):
        assert self.y_mean is not None
        assert self.y_std is not None

        # descale energies
        yd      = y      * self.y_std + self.y_mean
        yd_pred = y_pred * self.y_std + self.y_mean

        #ydnp = yd.detach().numpy()
        #yd_prednp = yd_pred.detach().numpy()
        #tt = np.hstack((ydnp, yd_prednp))
        #np.savetxt("tmp.txt", tt)
        #assert False

        ymin = yd.min()

        w  = self.dwt / (self.dwt + yd - ymin)
        wmse = (w * (yd - yd_pred)**2).mean()

        #wnp = w.detach().numpy()
        #ydnp = yd.detach().numpy()
        #tt = np.hstack((ydnp, wnp))
        #np.savetxt("ethanol_w.txt", tt)

        return wmse

class WRMSELoss_Ratio_dipole(torch.nn.Module):
    def __init__(self, dwt=1.0):
        super().__init__()
        self.dwt    = torch.tensor(dwt).to(DEVICE)

        self.y_mean = None
        self.y_std  = None

    def set_scale(self, y_mean, y_std):
        self.y_mean = torch.FloatTensor(y_mean.tolist()).to(DEVICE)
        self.y_std  = torch.FloatTensor(y_std.tolist()).to(DEVICE)

    def __repr__(self):
        return "WRMSELoss_Ratio_dipole(dwt={})".format(self.dwt)

    def forward(self, y, y_pred):
        """
        y:      (E, dipx,      dipy,      dipz     )
        y_pred: (   dipx_pred, dipy_pred, dipz_pred)
        """
        assert self.y_mean is not None
        assert self.y_std is not None

        # descale
        dip_pred = y_pred * self.y_std[1:] + self.y_mean[1:]
        yd       = y      * self.y_std     + self.y_mean
        dip      = yd[:, 1:]
        en       = yd[:, 0]

        en_min = en.min()
        w  = self.dwt / (self.dwt + en - en_min)

        dd   = dip - dip_pred
        wdd  = torch.einsum('ij,i->ij', dd, w)
        #wmse = torch.mean(torch.einsum('ij,ij->i', wdd, dd))

        #for k in range(10):
        #    print("dip: {}; dip_pred: {}".format(dip[k].detach().numpy(), dip_pred[k].detach().numpy()))

        #return torch.sqrt(wmse)
        return 1000.0 * torch.mean(torch.abs(wdd))


class WMSELoss_Ratio_wforces(torch.nn.Module):
    def __init__(self, natoms, dwt=1.0, f_lambda=1.0):
        super().__init__()
        self.natoms = natoms
        self.dwt    = torch.tensor(dwt).to(DEVICE)
        self.f_lambda = torch.tensor(f_lambda).to(DEVICE)

        self.en_mean = None
        self.en_std  = None

    def set_scale(self, en_mean, en_std):
        self.en_mean = torch.from_numpy(en_mean).to(DEVICE)
        self.en_std  = torch.from_numpy(en_std).to(DEVICE)

    def __repr__(self):
        return "WMSELoss_Ratio_wforces(natoms={}, dwt={}, f_lambda={})".format(self.natoms, self.dwt, self.f_lambda)

    def forward(self, en, en_pred, forces, forces_pred):
        wmse_en, wmse_forces = self.forward_separate(en, en_pred, forces, forces_pred)
        return wmse_en + wmse_forces

    def descale_energies(self, en):
        return en * self.en_std + self.en_mean

    def forward_separate(self, en, en_pred, forces, forces_pred):
        assert self.en_mean is not None
        assert self.en_std is not None

        en_pred = en_pred.to(DEVICE)

        # descale energies
        # forces are supposed to be already unnormalized
        _en      = self.descale_energies(en)
        _en_pred = self.descale_energies(en_pred)

        enmin   = _en.min()
        w       = self.dwt / (self.dwt + _en - enmin)
        wmse_en = (w * (_en - _en_pred)**2).mean()

        nconfigs = forces.size()[0]

        forces_pred = forces_pred.reshape(nconfigs, self.natoms, 3)

        df = forces - forces_pred
        wdf = torch.einsum('ijk,il->ijk', df, w)
        wmse_forces = self.f_lambda * torch.einsum('ijk,ijk->i', wdf, df).sum() / (3.0 * self.natoms) / nconfigs

        return wmse_en, wmse_forces

class WRMSELoss_Ratio(torch.nn.Module):
    def __init__(self, dwt=1.0):
        super().__init__()
        self.dwt    = torch.tensor(dwt).to(DEVICE)

        self.y_mean = None
        self.y_std  = None

    def set_scale(self, y_mean, y_std):
        self.y_mean = torch.FloatTensor(y_mean.tolist()).to(DEVICE)
        self.y_std  = torch.FloatTensor(y_std.tolist()).to(DEVICE)

    def __repr__(self):
        return "WRMSELoss_Ratio(dwt={})".format(self.dwt)

    def forward(self, y, y_pred):
        assert self.y_mean is not None
        assert self.y_std is not None

        # descale energies
        yd      = y      * self.y_std + self.y_mean
        yd_pred = y_pred * self.y_std + self.y_mean

        ymin = yd.min()

        w  = self.dwt / (self.dwt + yd - ymin)
        wmse = (w * (yd - yd_pred)**2).mean()

        return torch.sqrt(wmse)

class WMSELoss_PS(torch.nn.Module):
    """
    Weighted mean-squared error with
    weight factors suggested by Partridge and Schwenke
    H. Partridge, D. W. Schwenke, J. Chem. Phys. 106, 4618 (1997)
    """
    def __init__(self, Emax=2000.0):
        super().__init__()
        self.Emax   = torch.FloatTensor([Emax]).to(DEVICE)
        self.y_mean = None
        self.y_std  = None

    def set_scale(self, y_mean, y_std):
        self.y_mean = torch.FloatTensor(y_mean.tolist()).to(DEVICE)
        self.y_std  = torch.FloatTensor(y_std.tolist()).to(DEVICE)

    def __repr__(self):
        return "WMSELoss_PS(Emax={})".format(self.Emax)

    def forward(self, y, y_pred):
        assert self.y_mean is not None
        assert self.y_std is not None

        # descale energies
        yd      = y      * self.y_std + self.y_mean
        yd_pred = y_pred * self.y_std + self.y_mean

        Ehat = torch.max(yd, self.Emax.expand_as(yd))
        w = (torch.tanh(-6e-4 * (Ehat - self.Emax.expand_as(Ehat))) + 1.0) / 2.0 / Ehat
        w /= w.max()
        wmse = (w * (yd - yd_pred)**2).mean()

        return wmse

class WRMSELoss_PS(torch.nn.Module):
    """
    Weight factors of the form suggested by Partridge and Schwenke
    H. Partridge, D. W. Schwenke, J. Chem. Phys. 106, 4618 (1997)
    """
    def __init__(self, Emax=2000.0):
        super().__init__()
        self.Emax   = torch.FloatTensor([Emax]).to(DEVICE)
        self.y_mean = None
        self.y_std  = None

    def set_scale(self, y_mean, y_std):
        self.y_mean = torch.FloatTensor(y_mean.tolist()).to(DEVICE)
        self.y_std  = torch.FloatTensor(y_std.tolist()).to(DEVICE)

    def __repr__(self):
        return "WRMSELoss_PS(Emax={})".format(self.Emax)

    def forward(self, y, y_pred):
        assert self.y_mean is not None
        assert self.y_std is not None

        # descale energies
        yd      = y      * self.y_std + self.y_mean
        yd_pred = y_pred * self.y_std + self.y_mean

        N = 1e-4
        Ehat = torch.max(yd, self.Emax.expand_as(yd))
        w = (torch.tanh(-6e-4 * (Ehat - self.Emax.expand_as(Ehat))) + 1.002002002) / 2.002002002 / N / Ehat
        w /= w.max()
        wmse = (w * (yd - yd_pred)**2).mean()

        return torch.sqrt(wmse)

class EarlyStopping:
    def __init__(self, patience, tol, chk_path):
        """
        patience : how many epochs to wait after the last time the monitored quantity [validation loss] has improved
        tol:       minimum change in the monitored quantity to qualify as an improvement
        path:      path for the checkpoint to be saved to
        """
        self.patience = patience
        self.tol      = tol
        self.chk_path = chk_path

        self.counter    = 0
        self.best_score = None
        self.status     = False

    def reset(self):
        self.counter    = 0
        self.best_score = None
        self.status     = False

    def __call__(self, epoch, score, model, xscaler, yscaler, meta_info):
        if self.best_score is None:
            self.best_score = score
            save_checkpoint(model, xscaler, yscaler, meta_info, self.chk_path)
        elif score < self.best_score and (self.best_score - score) > self.tol:
            self.best_score = score
            self.counter = 0
            save_checkpoint(model, xscaler, yscaler, meta_info, self.chk_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.status = True

        if epoch % PRINT_TRAINING_STEPS == 0:
            logging.info("(Early Stopping) Best validation RMSE: {1:.{0}f}; current validation RMSE: {2:.{0}f}".format(PRINT_PRECISION, self.best_score, score))
            logging.info("(Early Stopping) counter: {}; patience: {}; tolerance: {}".format(self.counter, self.patience, self.tol))


def count_params(model):
    nparams = 0
    for name, param in model.named_parameters():
        params = torch.tensor(param.size())
        nparams += torch.prod(params, 0)

    return nparams

class Training:
    def __init__(self, model_folder, model_name, ckh_path, model, cfg, train, val, test, xscaler, yscaler):
        EVENTDIR = "runs"
        if not os.path.isdir(EVENTDIR):
            os.makedirs(EVENTDIR)

        log_dir = os.path.join("runs", model_name)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.model = model
        self.cfg = cfg
        self.cfg_solver = cfg['TRAINING']

        self.train = train
        self.val   = val
        self.test  = test

        self.xscaler = xscaler
        self.yscaler = yscaler

        self.cfg_loss = cfg['LOSS']
        self.loss_fn  = self.build_loss()
        self.loss_fn.set_scale(self.yscaler.mean_, self.yscaler.scale_)

        self.cfg_regularization = cfg.get('REGULARIZATION', None)
        self.regularization = self.build_regularization()

        self.pretraining = False
        if cfg.get('PRETRAINING'):
            self.pretraining = True

            cfg_pretrain = cfg['PRETRAINING']
            self.pretraining_epochs = cfg_pretrain['EPOCHS']
            self.pretraining_optimizer = self.build_optimizer(cfg_pretrain['OPTIMIZER'])

        self.chk_path = chk_path
        self.es = self.build_early_stopper()
        self.meta_info = {
            "NPOLY":    self.train.NPOLY,
            "NMON":     self.train.NMON,
            "NATOMS":   self.train.NATOMS,
            "symmetry": self.train.symmetry,
            "order":    self.train.order,
        }

    def build_regularization(self):
        if self.cfg_regularization is None:
            return None

        if self.cfg_regularization['NAME'] == 'L1':
            lambda_ = self.cfg_regularization['LAMBDA']
            reg = L1Regularization(lambda_)
        elif self.cfg_regularization['NAME'] == 'L2':
            lambda_ = self.cfg_regularization['LAMBDA']
            reg = L2Regularization(lambda_)
        else:
            raise ValueError("unreachable")


        return reg

    def build_optimizer(self, cfg_optimizer):
        if cfg_optimizer['NAME'] == 'LBFGS':
            lr               = cfg_optimizer.get('LR', 1.0)
            tolerance_grad   = cfg_optimizer.get('TOLERANCE_GRAD', 1e-14)
            tolerance_change = cfg_optimizer.get('TOLERANCE_CHANGE', 1e-14)
            max_iter         = cfg_optimizer.get('MAX_ITER', 100)

            optimizer        = torch.optim.LBFGS(self.model.parameters(), lr=lr, line_search_fn='strong_wolfe', tolerance_grad=tolerance_grad,
                                                 tolerance_change=tolerance_change, max_iter=max_iter)
        elif cfg_optimizer['NAME'] == 'Adam':
            lr           = cfg_optimizer.get('LR', 1e-3)
            weight_decay = cfg_optimizer.get('WEIGHT_DECAY', 0.0)
            optimizer    = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError("unreachable")

        logging.info("Build optimizer: {}".format(optimizer))

        return optimizer

    def build_loss(self):
        known_options = ('NAME', 'WEIGHT_TYPE', 'DWT', 'EREF', 'EMAX', 'USE_FORCES', 'USE_FORCES_AFTER_EPOCH', 'F_LAMBDA')
        for option in self.cfg_loss.keys():
            assert option.upper() in known_options, "[build_loss] unknown option: {}".format(option)

        # have all defaults in the same place and set them to configuration if the value is omitted in the YAML file
        use_forces = self.cfg_loss.get('USE_FORCES', False)
        self.cfg_loss['USE_FORCES'] = use_forces

        use_forces_after_epoch = self.cfg_loss.get('USE_FORCES_AFTER_EPOCH', None)
        self.cfg_loss['USE_FORCES_AFTER_EPOCH'] = use_forces_after_epoch

        if self.cfg_loss['NAME'] == 'WRMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'Ratio' and self.cfg['TYPE'] == 'DIPOLE':
            dwt = self.cfg_loss.get('dwt', 1.0)
            loss_fn = WRMSELoss_Ratio_dipole(dwt=dwt)

        elif self.cfg_loss['NAME'] == 'WRMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'Boltzmann' and not use_forces:
            Eref = self.cfg_loss.get('EREF', 2000.0)
            loss_fn = WRMSELoss_Boltzmann(Eref=Eref)
        elif self.cfg_loss['NAME'] == 'WMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'Boltzmann' and not use_forces:
            Eref = self.cfg_loss.get('EREF', 2000.0)
            loss_fn = WMSELoss_Boltzmann(Eref=Eref)

        elif self.cfg_loss['NAME'] == 'WRMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'Ratio' and not use_forces:
            dwt = self.cfg_loss.get('dwt', 1.0)
            loss_fn = WRMSELoss_Ratio(dwt=dwt)
        elif self.cfg_loss['NAME'] == 'WMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'Ratio' and not use_forces:
            dwt = self.cfg_loss.get('dwt', 1.0)
            loss_fn = WMSELoss_Ratio(dwt=dwt)

        elif self.cfg_loss['NAME'] == 'WRMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'PS' and not use_forces:
            Emax = self.cfg_loss.get('EMAX', 2000.0)
            loss_fn = WRMSELoss_PS(Emax=Emax)
        elif self.cfg_loss['NAME'] == 'WMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'PS' and not use_forces:
            Emax = self.cfg_loss.get('EMAX', 2000.0)
            loss_fn = WMSELoss_PS(Emax=Emax)


        elif self.cfg_loss['NAME'] == 'WMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'Ratio' and use_forces:
            dwt = self.cfg_loss.get('dwt', 1.0)
            f_lambda = self.cfg_loss.get('F_LAMBDA', 1.0)
            loss_fn = WMSELoss_Ratio_wforces(natoms=self.train.NATOMS, dwt=dwt, f_lambda=f_lambda)

        else:
            print(self.cfg_loss)
            raise ValueError("unreachable")

        logging.info("Build loss function: {}".format(loss_fn))

        return loss_fn

    def build_scheduler(self):
        cfg_scheduler = self.cfg_solver['SCHEDULER']

        factor         = cfg_scheduler.get('LR_REDUCE_GAMMA', 0.1)
        threshold      = cfg_scheduler.get('THRESHOLD', 0.1)
        threshold_mode = cfg_scheduler.get('THRESHOLD_MODE', 'abs')
        patience       = cfg_scheduler.get('PATIENCE', 10)
        cooldown       = cfg_scheduler.get('COOLDOWN', 0)
        min_lr         = cfg_scheduler.get('MIN_LR', 1e-5)

        if cfg_scheduler['NAME'] == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=factor, threshold=threshold, threshold_mode=threshold_mode,
                                                                   patience=patience, cooldown=cooldown, min_lr=min_lr)
        else:
            raise ValueError("unreachable")

        logging.info("Build scheduler:")
        logging.info(" NAME:            {}".format(cfg_scheduler['NAME']))
        logging.info(" LR_REDUCE_GAMMA: {}".format(factor))
        logging.info(" THRESHOLD:       {}".format(threshold))
        logging.info(" THRESHOLD_MODE:  {}".format(threshold_mode))
        logging.info(" PATIENCE:        {}".format(patience))
        logging.info(" COOLDOWN:        {}".format(cooldown))
        logging.info(" MIN_LR:          {}\n".format(min_lr))

        return scheduler

    def build_early_stopper(self):
        cfg_early_stopping = self.cfg_solver['EARLY_STOPPING']

        patience  = cfg_early_stopping.get('PATIENCE', 1000)
        tolerance = cfg_early_stopping.get('TOLERANCE', 0.1)

        return EarlyStopping(patience=patience, tol=tolerance, chk_path=self.chk_path)

    def reset_weights(self):
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                logging.info(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()

    def load_basic_checkpoint(self):
        self.reset_weights()
        state = torch.load(self.chk_path, map_location=torch.device(DEVICE))
        self.model.load_state_dict(state)

    def run_pretraining(self):
        logging.info("-------------------------------------------------")
        logging.info("----------- Running pretraining -----------------")
        logging.info("-------------------------------------------------")

        for epoch in range(self.pretraining_epochs):
            loss_train = self.train_epoch(epoch, self.pretraining_optimizer)

            with torch.no_grad():
                self.model.eval()

                pred_val = self.model(self.val.X)
                loss_val = self.loss_fn(self.val.y, pred_val)

            logging.info("Epoch: {}; loss train: {:.3f} cm-1; loss val: {:.3f} cm-1".format(
                epoch, loss_train, loss_val
            ))

        torch.save(model.state_dict(), self.chk_path)

        logging.info("-------------------------------------------------")
        logging.info("----------- Pretraining finished. ---------------")
        logging.info("-------------------------------------------------")

    def continue_from_checkpoint(self, chkpath):
        assert os.path.exists(chkpath)

        self.reset_weights()
        checkpoint = torch.load(chkpath, map_location=torch.device(DEVICE))
        self.model.load_state_dict(checkpoint["model"])

        self.train_model()


    def train_model(self):
        self.model = self.model.to(DEVICE)

        self.train.X = self.train.X.to(DEVICE)
        self.train.y = self.train.y.to(DEVICE)
        self.val.X = self.val.X.to(DEVICE)
        self.val.y = self.val.y.to(DEVICE)

        self.loss_fn = self.loss_fn.to(DEVICE)

        if self.train.dX is not None:
            self.train.dX = self.train.dX.to(DEVICE)
            self.train.dy = self.train.dy.to(DEVICE)

            self.val.dX = self.val.dX.to(DEVICE)
            self.val.dy = self.val.dy.to(DEVICE)

        if self.pretraining:
            self.run_pretraining()
            self.load_basic_checkpoint()

        self.optimizer = self.build_optimizer(self.cfg_solver['OPTIMIZER'])
        self.scheduler = self.build_scheduler()

        start = time.time()

        MAX_EPOCHS = self.cfg_solver['MAX_EPOCHS']

        for epoch in range(MAX_EPOCHS):
            # switch into mixed loss function: E + F
            if self.cfg_loss['USE_FORCES_AFTER_EPOCH'] is not None and epoch == self.cfg_loss['USE_FORCES_AFTER_EPOCH']:
                self.cfg_loss['USE_FORCES'] = True
                self.loss_fn = self.build_loss().to(DEVICE)
                self.loss_fn.set_scale(self.yscaler.mean_, self.yscaler.scale_)

                self.es.reset()

            print("loss function: {}".format(self.loss_fn))

            self.train_epoch(epoch, self.optimizer)

            self.scheduler.step(self.loss_val)
            current_lr = self.optimizer.param_groups[0]['lr']

            if epoch % PRINT_TRAINING_STEPS == 0:
                end = time.time()
                logging.info("Elapsed time: {:.0f}s\n".format(end - start))

            # writing all pending events to disk
            self.writer.flush()

            # pass loss values to EarlyStopping mechanism 
            self.es(epoch, self.loss_val, self.model, self.xscaler, self.yscaler, meta_info=self.meta_info)

            if self.es.status:
                logging.info("Invoking early stop.")
                break

        if self.loss_val < self.es.best_score:
            save_checkpoint(self.model, self.xscaler, self.yscaler, self.meta_info, self.chk_path)

        logging.info("\nReloading best model from the last checkpoint")

        self.reset_weights()
        checkpoint = torch.load(self.chk_path, map_location=torch.device(DEVICE))
        self.model.load_state_dict(checkpoint["model"])

        return self.model

    def compute_forces(self, dataset):
        Xtr = dataset.X

        Xtr.requires_grad = True

        y_pred = self.model(Xtr)
        dEdp   = torch.autograd.grad(outputs=y_pred, inputs=Xtr, grad_outputs=torch.ones_like(y_pred), retain_graph=True, create_graph=True)[0]

        Xtr.requires_grad = False

        # take into account normalization of polynomials
        # now we have derivatives of energy w.r.t. to polynomials
        x_scale = torch.from_numpy(self.xscaler.scale_).to(DEVICE)
        dEdp = torch.div(dEdp, x_scale)

        # force = -dE/dx = -\sigma(E) * dE/d(poly) * d(poly)/dx
        # `torch.einsum` throws a Runtime error without an explicit conversion to Double 
        dEdx = torch.einsum('ij,ijk -> ik', dEdp.double(), dataset.dX.double())

        # take into account normalization of model energy
        y_scale = torch.from_numpy(self.yscaler.scale_).to(DEVICE)
        dEdx = -torch.mul(dEdx, y_scale)

        return y_pred, dEdx

    def train_epoch(self, epoch, optimizer):
        CLOSURE_CALL_COUNT = 0

        def closure():
            nonlocal CLOSURE_CALL_COUNT
            CLOSURE_CALL_COUNT = CLOSURE_CALL_COUNT + 1

            optimizer.zero_grad()

            if self.cfg_loss['USE_FORCES']:
                train_y_pred, train_dy_pred = self.compute_forces(self.train)
                loss = self.loss_fn(self.train.y, train_y_pred, self.train.dy, train_dy_pred)
            elif self.cfg['TYPE'] == 'DIPOLE':
                # y_pred:    [(d, a1), (d, a2), (d, a3)] -- scalar products with anchor vectors 
                # dip_pred:  g @ y_pred                  -- Cartesian components of the predicted dipole
                y_pred = self.model(self.train.X)
                dip_pred = torch.einsum('ijk,ik->ij', self.train.grm, y_pred)

                loss = self.loss_fn(self.train.y, y_pred)
            else:
                y_pred = self.model(self.train.X)
                loss = self.loss_fn(self.train.y, y_pred)

            if self.regularization is not None:
                loss = loss + self.regularization(self.model)

            loss.backward(retain_graph=True)
            return loss

        # Calling model.train() will change the behavior of some layers such as nn.Dropout and nn.BatchNormXd
        self.model.train()

        start_time = timeit.default_timer()
        optimizer.step(closure)
        elapsed = timeit.default_timer() - start_time
        logging.info("Optimizer makes step in {:.2f}s".format(elapsed))
        logging.info("CLOSURE_CALL_COUNT = {}".format(CLOSURE_CALL_COUNT))

        # Calling model.eval() will change the behavior of some layers, 
        # such as nn.Dropout, which will be disabled, and nn.BatchNormXd, which will use the running stats during evaluation.
        self.model.eval()

        if self.cfg_loss['USE_FORCES']:
            train_y_pred, train_dy_pred = self.compute_forces(self.train)
            loss_train_e, loss_train_f = self.loss_fn.forward_separate(self.train.y, train_y_pred, self.train.dy, train_dy_pred)

            val_y_pred, val_dy_pred = self.compute_forces(self.val)
            loss_val_e, loss_val_f = self.loss_fn.forward_separate(self.val.y, val_y_pred, self.val.dy, val_dy_pred)

            train_e_d    = self.loss_fn.descale_energies(self.train.y)
            train_e_pred = self.loss_fn.descale_energies(train_y_pred)
            train_e_mae  = torch.mean(torch.abs(train_e_d - train_e_pred))
            train_e_rmse = torch.sqrt(torch.mean((train_e_d - train_e_pred)*(train_e_d - train_e_pred)))

            val_e_d    = self.loss_fn.descale_energies(self.val.y)
            val_e_pred = self.loss_fn.descale_energies(val_y_pred)
            val_e_mae  = torch.mean(torch.abs(val_e_d - val_e_pred)) 
            val_e_rmse = torch.sqrt(torch.mean((val_e_d - val_e_pred) * (val_e_d - val_e_pred)))

            natoms   = self.train.NATOMS
            train_dy = train.dy.reshape(-1, 3 * natoms)
            val_dy   = val.dy.reshape(-1, 3 * natoms)
            train_f_mae  = torch.mean(torch.sum(torch.abs(train_dy - train_dy_pred), dim=1) / (3 * natoms))
            val_f_mae    = torch.mean(torch.sum(torch.abs(val_dy - val_dy_pred), dim=1) / (3 * natoms))
            train_f_rmse = torch.sqrt(torch.mean(torch.sum((train_dy - train_dy_pred) * (train_dy - train_dy_pred), dim=1) / (3 * natoms)))
            val_f_rmse   = torch.sqrt(torch.mean(torch.sum((val_dy - val_dy_pred) * (val_dy - val_dy_pred), dim=1) / (3 * natoms)))	

            logging.info("Epoch: {}; (energy) loss train: {:.3f}; (force) loss train: {:.3f}\n \
                                           (energy) loss val:   {:.3f}; (force) loss val:   {:.3f}\n \
                                           (energy) MAE train:  {:.3f} cm-1; (force) MAE train:  {:.3f} cm-1/bohr\n \
                                           (energy) MAE val:    {:.3f} cm-1; (force) MAE val:    {:.3f} cm-1/bohr\n \
                                           (energy) RMSE train: {:.3f} cm-1; (force) RMSE train: {:.3f} cm-1/bohr\n \
                                           (energy) RMSE val:   {:.3f} cm-1; (force) RMSE val:   {:.3f} cm-1/bohr".format(
                epoch, loss_train_e, loss_train_f, loss_val_e, loss_val_f, train_e_mae, train_f_mae, val_e_mae, val_f_mae, train_e_rmse, val_e_rmse, train_f_rmse, val_f_rmse
            ))

            # value to be passed to EarlyStopping/ReduceLR mechanisms
            self.loss_val = loss_val_e

            self.writer.add_scalar("loss/train", loss_train_e, epoch)
            self.writer.add_scalar("loss/val", loss_val_e, epoch)

        else:
            # To disable the gradient calculation, set the .requires_grad attribute of all parameters to False 
            # or wrap the forward pass into with torch.no_grad().
            with torch.no_grad():
                train_y_pred = self.model(self.train.X)
                loss_train   = self.loss_fn(self.train.y, train_y_pred)

                val_y_pred = self.model(self.val.X)
                loss_val   = self.loss_fn(self.val.y, val_y_pred)

                # value to be passed to EarlyStopping/ReduceLR mechanisms
                self.loss_val = loss_val

            logging.info("Epoch: {0}; loss train: {2:.{1}f} cm-1; loss val: {3:.{1}f} cm-1".format(epoch, PRINT_PRECISION, loss_train, loss_val))

            self.writer.add_scalar("loss/train", loss_train, epoch)
            self.writer.add_scalar("loss/val", loss_val, epoch)


    def model_eval(self):
        self.test.X = self.test.X.to(DEVICE)
        self.test.y = self.test.y.to(DEVICE)

        if self.test.dX is not None:
            self.test.dX = self.test.dX.to(DEVICE)
            self.test.dy = self.test.dy.to(DEVICE)

        # Calling model.eval() will change the behavior of some layers, 
        # such as nn.Dropout, which will be disabled, and nn.BatchNormXd, which will use the running stats during evaluation.
        self.model.eval()

        if self.cfg_loss['USE_FORCES']:
            train_y_pred, train_dy_pred = self.compute_forces(self.train)
            loss_train_e, loss_train_f = self.loss_fn.forward_separate(self.train.y, train_y_pred, self.train.dy, train_dy_pred)

            val_y_pred, val_dy_pred = self.compute_forces(self.val)
            loss_val_e, loss_val_f = self.loss_fn.forward_separate(self.val.y, val_y_pred, self.val.dy, val_dy_pred)

            test_y_pred, test_dy_pred = self.compute_forces(self.test)
            loss_test_e, loss_test_f = self.loss_fn.forward_separate(self.test.y, test_y_pred, self.test.dy, test_dy_pred)

            logging.info("Model evaluation after training:")
            logging.info("Train      loss: {1:.{0}f} cm-1; force loss: {2:.{0}f} cm-1/bohr".format(PRINT_PRECISION, loss_train_e, loss_train_f))
            logging.info("Validation loss: {1:.{0}f} cm-1; force loss: {2:.{0}f} cm-1/bohr".format(PRINT_PRECISION, loss_val_e, loss_val_f))
            logging.info("Test       loss: {1:.{0}f} cm-1; force loss: {2:.{0}f} cm-1/bohr".format(PRINT_PRECISION, loss_test_e, loss_test_f))
        else:
            # To disable the gradient calculation, set the .requires_grad attribute of all parameters to False 
            # or wrap the forward pass into with torch.no_grad().
            with torch.no_grad():
                pred_train = self.model(self.train.X)
                loss_train = self.loss_fn(self.train.y, pred_train)

                pred_val   = self.model(self.val.X)
                loss_val   = self.loss_fn(self.val.y, pred_val)

                pred_test  = self.model(self.test.X)
                loss_test  = self.loss_fn(self.test.y, pred_test)

            logging.info("Model evaluation after training:")
            logging.info("Train      loss: {1:.{0}f} cm-1".format(PRINT_PRECISION, loss_train))
            logging.info("Validation loss: {1:.{0}f} cm-1".format(PRINT_PRECISION, loss_val))
            logging.info("Test       loss: {1:.{0}f} cm-1".format(PRINT_PRECISION, loss_test))

def setup_google_folder():
    assert os.path.exists('client_secrets.json')
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()

    drive = GoogleDrive(gauth)

    folderName = "PES-Fitting-MSA"

    folders = drive.ListFile(
        {'q': "title='" + folderName + "' and mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()

    for folder in folders:
        if folder['title'] == folderName:
            file = drive.CreateFile({'parents': [{'id': folder['id']}]})
            file.SetContentFile('README.md')
            file.Upload()

def load_cfg(cfg_path):
    with open(cfg_path, mode="r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.info(exc)

    known_groups = ('TYPE', 'DATASET', 'MODEL', 'LOSS', 'TRAINING', 'PRINT_PRECISION')
    for group in cfg.keys():
        assert group in known_groups, "Unknown group: {}".format(group)

    return cfg

def load_dataset(cfg_dataset, typ):
    from enum import Enum, auto
    class KeywordType:
        KEYWORD_OPTIONAL = auto()
        KEYWORD_REQUIRED = auto()

    KEYWORDS = [
        ('NAME', KeywordType.KEYWORD_REQUIRED, None), # `str` 
        # FILE ORGANIZATION 
        #  `list` : paths (relative to BASEDIR) to the .xyz/.npz files 
        ('SOURCE', KeywordType.KEYWORD_REQUIRED, None),
        #  `str`  : path to store pickled train/val/test datasets 
        ('INTERIM_FOLDER', KeywordType.KEYWORD_OPTIONAL, os.path.join(BASEDIR, "datasets", "interim")),
        #  `str`  : path to folder with files to compute invariant polynomials: .f90 to compute polynomials (and their derivatives) + .MONO + .POLY 
        ('EXTERNAL_FOLDER', KeywordType.KEYWORD_OPTIONAL, os.path.join(BASEDIR, "datasets", "external")),
        # DATA SELECTION 
        ('LOAD_FORCES',  KeywordType.KEYWORD_OPTIONAL, False), # `bool` : whether to load forces from dataset
        ('ENERGY_LIMIT', KeywordType.KEYWORD_OPTIONAL, None),  # `bool` : NOT SUPPORTED now -- set an upper bound on energies in the training dataset 
        # DATASET PREPROCESSING
        ('NORMALIZE',        KeywordType.KEYWORD_REQUIRED, None),  # `str`  : how to perform data normalization  
        ('ANCHOR_POSITIONS', KeywordType.KEYWORD_OPTIONAL, None),  # [REQUIRED for TYPE=dipole] `int`s : select atoms whose radius-vectors to use as basis 
        # PIP CONSTRUCTION
        ('ORDER',         KeywordType.KEYWORD_REQUIRED, None),  # `int`  : maximum order of PIPs 
        ('SYMMETRY',      KeywordType.KEYWORD_REQUIRED, None),  # `int`s : permutational symmetry of the molecule | molecular pair
        ('PURIFY',        KeywordType.KEYWORD_OPTIONAL, False), # `bool` : use purified basis of PIPs
        ('ATOM_MAPPING',  KeywordType.KEYWORD_OPTIONAL, False), # `list` : mapping atoms->monomer (which atom belongs to which monomer)
        ('VARIABLES' ,    KeywordType.KEYWORD_REQUIRED, None), # `dict` : mapping interatomic distances->polynomial variables 
    ]

    from operator import itemgetter
    for keyword in cfg_dataset.keys():
        assert keyword in list(map(itemgetter(0), KEYWORDS)), "Unknown keyword: {}".format(keyword)

    for keyword, keyword_type, default_value in KEYWORDS:
        if keyword_type == KeywordType.KEYWORD_REQUIRED:
            assert keyword in cfg_dataset, "Required keyword {} is missing".format(keyword)
        elif keyword_type == KeywordType.KEYWORD_OPTIONAL:
            cfg_dataset.setdefault(keyword, default_value)

    if typ == 'DIPOLE':
        assert 'ANCHOR_POSITIONS' in cfg_dataset
        assert not cfg_dataset['LOAD_FORCES']

    VARIABLES_BLOCK_REQUIRED = ('INTRAMOLECULAR', 'INTERMOLECULAR', 'EXP_LAMBDA')
    for keyword in VARIABLES_BLOCK_REQUIRED:
        assert keyword in cfg_dataset['VARIABLES']

    cfg_dataset['TYPE'] = typ

    if not os.path.isdir(cfg_dataset['INTERIM_FOLDER']):
        os.makedirs(cfg_dataset['INTERIM_FOLDER']) # can create nested directories

    if not os.path.isdir(cfg_dataset['EXTERNAL_FOLDER']):
        os.makedirs(cfg_dataset['EXTERNAL_FOLDER']) # can create nested directories

    logging.info("Dataset options:")
    for keyword, value in cfg_dataset.items():
        logging.info("{:>25}: \t {}".format(keyword, value))

    train_fpath, val_fpath, test_fpath = make_dataset_fpaths(cfg_dataset)
    if not os.path.isfile(train_fpath) or not os.path.isfile(val_fpath) or not os.path.isfile(test_fpath):
        logging.info("Invoking make_dataset to create polynomial dataset")

        # we suppose that paths in YAML configuration are relative to BASEDIR (repo folder)
        source = [os.path.join(BASEDIR, path) for path in cfg_dataset['SOURCE']]

        dataset_fpaths = {"train" : train_fpath, "val": val_fpath, "test" : test_fpath}
        make_dataset(cfg_dataset, dataset_fpaths)
    else:
        logging.info("Dataset found.")

    train = PolyDataset.from_pickle(train_fpath)
    assert train.energy_limit == cfg_dataset['ENERGY_LIMIT']
    assert train.purify       == cfg_dataset['PURIFY']
    logging.info("Loading training dataset: {}; len: {}".format(train_fpath, len(train.y)))

    val   = PolyDataset.from_pickle(val_fpath)
    assert val.energy_limit == cfg_dataset['ENERGY_LIMIT']
    assert val.purify       == cfg_dataset['PURIFY']
    logging.info("Loading validation dataset: {}; len: {}".format(val_fpath, len(val.y)))

    test  = PolyDataset.from_pickle(test_fpath)
    assert test.energy_limit == cfg_dataset['ENERGY_LIMIT']
    assert test.purify       == cfg_dataset['PURIFY']
    logging.info("Loading testing dataset: {}; len: {}".format(test_fpath, len(test.y)))

    return train, val, test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", required=True, type=str, help="path to folder with YAML configuration file")
    parser.add_argument("--model_name",   required=True, type=str, help="the name of the YAML configuration file without extension")
    parser.add_argument("--log_name",     required=False, type=str, default=None, help="name of the logging file without extension")
    parser.add_argument("--chk_name",     required=False, type=str, default=None, help="name of the general checkpoint without extension")
    args = parser.parse_args()

    MODEL_FOLDER = os.path.join(BASEDIR, args.model_folder)
    MODEL_NAME   = args.model_name

    assert os.path.isdir(MODEL_FOLDER), "Path to folder is invalid: {}".format(MODEL_FOLDER)

    cfg_path = os.path.join(MODEL_FOLDER, MODEL_NAME + ".yaml")
    assert os.path.isfile(cfg_path), "YAML configuration file does not exist at {}".format(cfg_path)

    cfg = load_cfg(cfg_path)
    logging.info("loaded configuration file from {}".format(cfg_path))

    if 'PRINT_PRECISION' in cfg:
        PRINT_PRECISION = cfg['PRINT_PRECISION'] 

    if args.log_name is not None:
        log_path = os.path.join(MODEL_FOLDER, args.log_name + ".log")
    else:
        log_path = os.path.join(MODEL_FOLDER, MODEL_NAME + ".log")

    if args.chk_name is not None:
        chk_path = os.path.join(MODEL_FOLDER, args.chk_name + ".pt")
    else:
        chk_path = os.path.join(MODEL_FOLDER, MODEL_NAME + ".pt")

    if os.path.exists(log_path):
        os.remove(log_path)

    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.handlers = []

    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.INFO)

    seed_torch()
    if DEVICE.type == 'cuda':
        logging.info("CUDA Device found: {}".format(torch.cuda.get_device_name(0)))
        logging.info("Memory usage:")
        logging.info("Allocated: {} GB".format(round(torch.cuda.memory_allocated(0)/1024**3, 1)))
    else:
        logging.info("No CUDA Device Found. Using CPU")

    import psutil
    logging.info("[psutil] Memory status: \n {}".format(psutil.virtual_memory()))

    assert 'TYPE' in cfg
    typ = cfg['TYPE']
    assert typ in ('ENERGY', 'DIPOLE')

    cfg_dataset = cfg['DATASET']
    train, val, test = load_dataset(cfg_dataset, typ)
    xscaler, yscaler = preprocess_dataset(train, val, test, cfg_dataset)

    cfg_model = cfg['MODEL']
    if typ == 'ENERGY':
        model = build_network_yaml(cfg_model, input_features=train.NPOLY, output_features=1)
    elif typ == 'DIPOLE':
        model = build_network_yaml(cfg_model, input_features=train.NPOLY, output_features=3)

    nparams = count_params(model)
    logging.info("Number of parameters: {}".format(nparams))

    t = Training(MODEL_FOLDER, MODEL_NAME, chk_path, model, cfg, train, val, test, xscaler, yscaler)

    model = t.train_model()
    t.model_eval()
