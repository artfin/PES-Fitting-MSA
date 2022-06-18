import argparse
import collections
import json
import logging
import random
import os
import sys
import time
import yaml

import torch.nn
from torch.utils.tensorboard import SummaryWriter

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

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

def preprocess_dataset(train, val, test, cfg_preprocess):
    if cfg_preprocess['NORMALIZE'] == 'std':
        xscaler = StandardScaler()
        yscaler = StandardScaler()
    else:
        raise ValueError("unreachable")

    train.X = torch.from_numpy(xscaler.fit_transform(train.X))

    try:
        val.X   = torch.from_numpy(xscaler.transform(val.X))
        test.X  = torch.from_numpy(xscaler.transform(test.X))
    except ValueError:
        val.X  = torch.empty((1, 1))
        test.X = torch.empty((1, 1))

    train.y = torch.from_numpy(yscaler.fit_transform(train.y))

    try:
        val.y   = torch.from_numpy(yscaler.transform(val.y))
        test.y  = torch.from_numpy(yscaler.transform(test.y))
    except ValueError:
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

        ymin = yd.min()

        w  = self.dwt / (self.dwt + yd - ymin)
        wmse = (w * (yd - yd_pred)**2).mean()

        return wmse

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
    def __init__(self, patience=10, tol=0.1, chk_path='checkpoint.pt'):
        """
        patience : how many epochs to wait after the last time the monitored quantity [validation loss] has improved
        tol:       minimum change in the monitored quantity to qualify as an improvement
        path:      path for the checkpoint to be saved to
        """
        self.patience = patience
        self.tol = tol
        self.chk_path = chk_path

        self.counter = 0
        self.best_score = None
        self.status = False

    def __call__(self, epoch, score, model, xscaler, yscaler, meta_info):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, xscaler, yscaler, meta_info)

        elif score < self.best_score and (self.best_score - score) > self.tol:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model, xscaler, yscaler, meta_info)

        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.status = True

        if epoch % PRINT_TRAINING_STEPS == 0:
            logging.info("(Early Stopping) Best validation RMSE: {:.2f}; current validation RMSE: {:.2f}".format(self.best_score, score))
            logging.info("(Early Stopping) ES counter: {}; ES patience: {}".format(self.counter, self.patience))

    def save_checkpoint(self, model, xscaler, yscaler, meta_info):
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
        torch.save(checkpoint, self.chk_path)

def count_params(model):
    nparams = 0
    for name, param in model.named_parameters():
        params = torch.tensor(param.size())
        nparams += torch.prod(params, 0)

    return nparams

class Training:
    def __init__(self, model, cfg, train, val, test, xscaler, yscaler, model_name=None):
        if model_name is None:
            model_name = os.path.split(cfg['OUTPUT_PATH'])[1]

        log_dir = os.path.join("runs", model_name)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.model = model
        self.cfg_solver = cfg['TRAINING']

        self.train = train
        self.val   = val
        self.test  = test

        self.xscaler = xscaler
        self.yscaler = yscaler

        self.cfg_loss = cfg['LOSS']
        self.loss_fn = self.build_loss()

        self.cfg_regularization = cfg.get('REGULARIZATION', None)
        self.regularization = self.build_regularization()

        # passing mean and scale of energies to obtain absolute energies from normalized ones
        self.loss_fn.set_scale(self.yscaler.mean_, self.yscaler.scale_)

        self.pretraining = False
        if cfg.get('PRETRAINING'):
            self.pretraining = True

            cfg_pretrain = cfg['PRETRAINING']
            self.pretraining_epochs = cfg_pretrain['EPOCHS']
            self.pretraining_optimizer = self.build_optimizer(cfg_pretrain['OPTIMIZER'])

        output_path = cfg.get('OUTPUT_PATH', '')
        self.es = self.build_early_stopper(output_path=output_path)
        self.meta_info = {
            "NPOLY":    self.train.NPOLY,
            "NMON":     self.train.NMON,
            "NATOMS":   self.train.NATOMS,
            "symmetry": self.train.symmetry,
            "order":    self.train.order,
            "mask" :    self.train.mask,
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

        logging.info("Build regularization module: {}".format(reg))

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
        if self.cfg_loss['NAME'] == 'WRMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'Boltzmann':
            Eref = self.cfg_loss.get('EREF', 2000.0)
            loss_fn = WRMSELoss_Boltzmann(Eref=Eref)
        elif self.cfg_loss['NAME'] == 'WMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'Boltzmann':
            Eref = self.cfg_loss.get('EREF', 2000.0)
            loss_fn = WMSELoss_Boltzmann(Eref=Eref)
        elif self.cfg_loss['NAME'] == 'WRMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'Ratio':
            dwt = self.cfg_loss.get('dwt', 1.0)
            loss_fn = WRMSELoss_Ratio(dwt=dwt)
        elif self.cfg_loss['NAME'] == 'WMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'Ratio':
            dwt = self.cfg_loss.get('dwt', 1.0)
            loss_fn = WMSELoss_Ratio(dwt=dwt)
        elif self.cfg_loss['NAME'] == 'WRMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'PS':
            Emax = self.cfg_loss.get('EMAX', 2000.0)
            loss_fn = WRMSELoss_PS(Emax=Emax)
        elif self.cfg_loss['NAME'] == 'WMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'PS':
            Emax = self.cfg_loss.get('EMAX', 2000.0)
            loss_fn = WMSELoss_PS(Emax=Emax)
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

    def build_early_stopper(self, output_path):
        cfg_early_stopping = self.cfg_solver['EARLY_STOPPING']

        patience = cfg_early_stopping.get('PATIENCE', 10)
        tolerance = cfg_early_stopping.get('TOLERANCE', 0.1)

        self.chk_path = os.path.join(output_path, 'checkpoint.pt')
        es = EarlyStopping(patience=patience, tol=tolerance, chk_path=self.chk_path)
        return es

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

        if self.pretraining:
            self.run_pretraining()
            self.load_basic_checkpoint()

        self.optimizer = self.build_optimizer(self.cfg_solver['OPTIMIZER'])
        self.scheduler = self.build_scheduler()

        start = time.time()

        MAX_EPOCHS = self.cfg_solver['MAX_EPOCHS']

        for epoch in range(MAX_EPOCHS):
            loss_train = self.train_epoch(epoch, self.optimizer)

            with torch.no_grad():
                self.model.eval()

                pred_val = self.model(self.val.X)
                loss_val = self.loss_fn(self.val.y, pred_val)
                self.writer.add_scalar("loss/val", loss_val, epoch)

            self.scheduler.step(loss_val)
            current_lr = self.optimizer.param_groups[0]['lr']

            if epoch % PRINT_TRAINING_STEPS == 0:
                logging.info("Current learning rate: {:.2e}".format(current_lr))
                logging.info("Epoch: {}; loss train: {:.3f} cm-1; loss val: {:.3f} cm-1".format(
                    epoch, loss_train, loss_val
                ))

                end = time.time()
                logging.info("Elapsed time: {:.0f}s\n".format(end - start))

            # writing all pending events to disk
            self.writer.flush()

            self.es(epoch, loss_val, self.model, self.xscaler, self.yscaler, meta_info=self.meta_info)
            if self.es.status:
                logging.info("Invoking early stop.")
                break

        if loss_val < self.es.best_score:
            self.es.save_checkpoint(self.model, self.xscaler, self.yscaler, meta_info=self.meta_info)

        logging.info("\nReloading best model from the last checkpoint")

        self.reset_weights()
        checkpoint = torch.load(self.chk_path, map_location=torch.device(DEVICE))
        self.model.load_state_dict(checkpoint["model"])

        return self.model

    def train_epoch(self, epoch, optimizer):
        def closure():
            optimizer.zero_grad()
            y_pred = self.model(self.train.X)

            loss = self.loss_fn(self.train.y, y_pred)
            if self.regularization is not None:
                loss = loss + self.regularization(self.model)

            loss.backward()
            return loss

        self.model.train()
        optimizer.step(closure)
        loss_train = closure()

        self.writer.add_scalar("loss/train", loss_train, epoch)
        return loss_train

    def model_eval(self):
        self.test.X = self.test.X.to(DEVICE)
        self.test.y = self.test.y.to(DEVICE)

        with torch.no_grad():
            self.model.eval()

            pred_train        = self.model(self.train.X)
            loss_train        = self.loss_fn(self.train.y, pred_train)

            pred_val        = self.model(self.val.X)
            loss_val        = self.loss_fn(self.val.y, pred_val)

            pred_test        = self.model(self.test.X)
            loss_test        = self.loss_fn(self.test.y, pred_test)

        logging.info("Model evaluation after training:")
        logging.info("Train      loss: {:.2f} cm-1".format(loss_train))
        logging.info("Validation loss: {:.2f} cm-1".format(loss_val))
        logging.info("Test       loss: {:.2f} cm-1".format(loss_test))

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

def load_dataset(cfg_dataset):
    known_options = ('ORDER', 'SYMMETRY', 'TYPE', 'INTRAMOLECULAR_TO_ZERO', 'PURIFY', 'NORMALIZE', 'ENERGY_LIMIT')

    for option in cfg_dataset.keys():
        assert option in known_options, "Unknown option: {}".format(option)

    order        = cfg_dataset['ORDER']
    symmetry     = cfg_dataset['SYMMETRY']
    typ          = cfg_dataset['TYPE'].lower()
    energy_limit = cfg_dataset.get('ENERGY_LIMIT', None)
    intramz      = cfg_dataset.get('INTRAMOLECULAR_TO_ZERO', False)
    purify       = cfg_dataset.get('PURIFY', False)

    assert order in (3, 4, 5)
    assert typ in ('rigid', 'nonrigid', 'nonrigid-clip')

    logging.info("Dataset options:")
    logging.info("order:        {}".format(order))
    logging.info("symmetry:     {}".format(symmetry))
    logging.info("typ:          {}".format(typ))
    logging.info("energy_limit: {}".format(energy_limit))
    logging.info("intramz:      {}".format(intramz))
    logging.info("purify:       {}".format(purify))

    train_fpath, val_fpath, test_fpath = make_dataset_fpaths(order, symmetry, typ, energy_limit, intramz, purify)
    if not os.path.isfile(train_fpath) or not os.path.isfile(val_fpath) or not os.path.isfile(test_fpath):
        logging.info("Invoking make_dataset to create polynomial dataset")
        make_dataset(order=order, symmetry=symmetry, typ=typ, energy_limit=energy_limit, intramz=intramz, purify=purify)
    else:
        logging.info("Dataset found.")

    logging.info("Loading training dataset:   {}".format(train_fpath))
    logging.info("Loading validation dataset: {}".format(val_fpath))
    logging.info("Loading testing dataset:    {}".format(test_fpath))

    train = PolyDataset.from_pickle(train_fpath)
    assert train.energy_limit == energy_limit
    assert train.intramz      == intramz
    assert train.purify       == purify

    val   = PolyDataset.from_pickle(val_fpath)
    assert val.energy_limit == energy_limit
    assert val.intramz      == intramz
    assert val.purify       == purify

    test  = PolyDataset.from_pickle(test_fpath)
    assert test.energy_limit == energy_limit
    assert test.intramz      == intramz
    assert test.purify       == purify

    return train, val, test

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(levelname)s] %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    parser = argparse.ArgumentParser()
    parser.add_argument("model_folder", type=str, help="path to folder with YAML configuration file")
    parser.add_argument("model_name", type=str, help="the name of the YAML configuration file [without extension]")
    args = parser.parse_args()

    MODEL_FOLDER = os.path.join(BASEDIR, args.model_folder)
    MODEL        = args.model_name

    assert os.path.isdir(MODEL_FOLDER), "Path to folder is invalid: {}".format(MODEL_FOLDER)

    cfg_path = os.path.join(MODEL_FOLDER, MODEL + ".yaml")
    assert os.path.isfile(cfg_path), "YAML configuration file does not exist at {}".format(cfg_path)

    seed_torch()
    if DEVICE.type == 'cuda':
        logging.info(torch.cuda.get_device_name(0))
        logging.info("Memory usage:")
        logging.info("Allocated: {} GB".format(round(torch.cuda.memory_allocated(0)/1024**3, 1)))

    with open(cfg_path, mode="r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.info(exc)

    logging.info("loaded configuration file from {}".format(cfg_path))

    log_path = os.path.join(MODEL_FOLDER, MODEL + ".log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)

    cfg_dataset = cfg['DATASET']
    train, val, test = load_dataset(cfg_dataset)
    xscaler, yscaler = preprocess_dataset(train, val, test, cfg_dataset)

    cfg_model = cfg['MODEL']
    model = build_network_yaml(cfg_model, input_features=train.NPOLY)
    nparams = count_params(model)
    logging.info("Number of parameters: {}".format(nparams))

    t = Training(model, cfg, train, val, test, xscaler, yscaler)

    model = t.train_model()
    t.model_eval()

    os.rename(os.path.join(MODEL_FOLDER, "checkpoint.pt"), os.path.join(MODEL_FOLDER, MODEL + '.pt'))
