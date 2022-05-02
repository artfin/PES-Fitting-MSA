import torch.nn

import collections
import json
import logging
import random
import os
import sys
import time
import yaml

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from dataset import PolyDataset
from build_model import build_network_yaml

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

def preprocess_dataset(train, val, test, cfg_preprocess):
    if cfg_preprocess['NORMALIZE'] == 'std':
        xscaler = StandardScaler()
        yscaler = StandardScaler()
    else:
        raise ValueError("unreachable")

    train.X = torch.from_numpy(xscaler.fit_transform(train.X))
    val.X   = torch.from_numpy(xscaler.transform(val.X))
    test.X  = torch.from_numpy(xscaler.transform(test.X))

    train.y = torch.from_numpy(yscaler.fit_transform(train.y))
    val.y   = torch.from_numpy(yscaler.transform(val.y))
    test.y  = torch.from_numpy(yscaler.transform(test.y))

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


class WRMSELoss_Boltzmann(torch.nn.Module):
    def __init__(self, e_factor):
        super().__init__()
        self.e_factor = e_factor

    def forward(self, y, y_pred):
        w = torch.exp(-y * self.e_factor)
        w = w / w.max()
        wmse = (w * (y - y_pred)**2).mean()

        return torch.sqrt(wmse)

class WRMSELoss_Ratio(torch.nn.Module):
    def __init__(self, dwt=1.0):
        super().__init__()
        self.dwt = dwt

    def forward(self, y, y_pred):
        ymin = y.min()
        w = self.dwt / (self.dwt + y - ymin)
        wmse = (w * (y - y_pred)**2).mean()

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

    def __call__(self, score, model, xscaler, yscaler, meta_info):
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

class Training:
    def __init__(self, model, cfg, train, val, test, xscaler, yscaler):
        self.model = model
        self.cfg_solver = cfg['SOLVER']

        self.train = train
        self.val   = val
        self.test  = test

        self.xscaler = xscaler
        self.yscaler = yscaler

        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.loss_fn = self.build_loss()

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

    def build_optimizer(self):
        cfg_optimizer = self.cfg_solver['OPTIMIZER']

        lr               = cfg_optimizer.get('LR', 1.0)
        tolerance_grad   = cfg_optimizer.get('TOLERANCE_GRAD', 1e-14)
        tolerance_change = cfg_optimizer.get('TOLERANCE_CHANGE', 1e-14)
        max_iter         = cfg_optimizer.get('MAX_ITER', 100)

        if cfg_optimizer['NAME'] == 'LBFGS':
            optimizer = torch.optim.LBFGS(self.model.parameters(), lr=lr, line_search_fn='strong_wolfe', tolerance_grad=tolerance_grad,
                                          tolerance_change=tolerance_change, max_iter=max_iter)
        else:
            raise ValuerError("unreachable")

        logging.info("Build optimizer: {}".format(optimizer))

        return optimizer

    def build_loss(self):
        cfg_loss = self.cfg_solver['LOSS']
        if cfg_loss['NAME'] == 'WRMSE' and cfg_loss['WEIGHT_TYPE'] == 'Boltzmann':
            eff_temperature = cfg_loss.get('EFFECTIVE_TEMPERATURE', 2000.0)
            loss_fn = WRMSELoss_Boltzmann(e_factor=self.yscaler.scale_ / eff_temperature)
        elif cfg_loss['NAME'] == 'WRMSE' and cfg_loss['WEIGHT_TYPE'] == 'Ratio':
            dwt = cfg_loss.get('dwt', 1.0)
            loss_fn = WRMSELoss_Ratio(dwt=dwt)
        else:
            raise ValueError("unreachable")

        logging.info("Build loss function:")
        logging.info(" WEIGHT_TYPE:           {}".format(cfg_loss['WEIGHT_TYPE']))

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

    def get_metric(self, loss):
        loss_value = loss.detach().item()
        logging.info("loss_value: {}".format(loss_value))

        cfg_loss = self.cfg_solver['LOSS']
        if cfg_loss['NAME'] == 'WRMSE':
            descaler = self.yscaler.scale_[0]
            return loss_value * descaler
        else:
            raise ValueError("unreachable")

    def reset_weights(self):
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                logging.info(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()

    def train_model(self):
        start = time.time()

        MAX_EPOCHS = self.cfg_solver['MAX_EPOCHS']
        for epoch in range(MAX_EPOCHS):
            self.train_epoch(epoch)

            with torch.no_grad():
                self.model.eval()

                pred_val = self.model(self.val.X)
                loss_val = self.loss_fn(self.val.y, pred_val)
                self.metric_val = self.get_metric(loss_val)

            self.scheduler.step(self.metric_val)
            current_lr = self.optimizer.param_groups[0]['lr']
            logging.info("Current learning rate: {:.2e}".format(current_lr))

            logging.info("Epoch: {}; metric train: {:.3f} cm-1; metric_val: {:.3f} cm-1".format(
                epoch, self.metric_train, self.metric_val
            ))
            end = time.time()
            logging.info("Elapsed time: {:.0f}s\n".format(end - start))

            self.es(self.metric_val, self.model, self.xscaler, self.yscaler, meta_info=self.meta_info)
            if self.es.status:
                logging.info("Invoking early stop.")
                break

        if self.metric_val < self.es.best_score:
            self.es.save_checkpoint(self.model, self.xscaler, self.yscaler, meta_info=self.meta_info)

        logging.info("\nReloading best model from the last checkpoint")
        self.reset_weights()
        checkpoint = torch.load(self.chk_path)
        self.model.load_state_dict(checkpoint["model"])

        return self.model

    def train_epoch(self, epoch):
        def closure():
            self.optimizer.zero_grad()
            y_pred = self.model(self.train.X)
            loss = self.loss_fn(self.train.y, y_pred)
            loss.backward()
            return loss

        self.model.train()
        self.optimizer.step(closure)
        loss_train = closure()

        self.metric_train = self.get_metric(loss_train)

    def model_eval(self):
        with torch.no_grad():
            self.model.eval()

            pred_train        = self.model(self.train.X)
            loss_train        = self.loss_fn(self.train.y, pred_train)
            self.metric_train = self.get_metric(loss_train)

            pred_val        = self.model(self.val.X)
            loss_val        = self.loss_fn(self.val.y, pred_val)
            self.metric_val = self.get_metric(loss_val)

            pred_test        = self.model(self.test.X)
            loss_test        = self.loss_fn(self.test.y, pred_test)
            self.metric_test = self.get_metric(loss_test)

        logging.info("Model evaluation after training:")
        logging.info("Train      RMSE: {:.2f} cm-1".format(self.metric_train))
        logging.info("Validation RMSE: {:.2f} cm-1".format(self.metric_val))
        logging.info("Test       RMSE: {:.2f} cm-1".format(self.metric_test))


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(levelname)s] %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    MODEL_FOLDER = "models/rigid/L1/L1-tanh/";
    log_path = os.path.join(MODEL_FOLDER, "logs.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)

    cfg_path = os.path.join(MODEL_FOLDER, "config.yaml")
    with open(cfg_path, mode="r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.info(exc)

    logging.info("loaded configuration file from {}".format(cfg_path))

    cfg_dataset = cfg['DATASET']
    logging.info("Loading training dataset from TRAIN_DATA_PATH={}".format(cfg_dataset['TRAIN_DATA_PATH']))
    logging.info("Loading validation dataset from VAL_DATA_PATH={}".format(cfg_dataset['VAL_DATA_PATH']))
    logging.info("Loading testing dataset from TEST_DATA_PATH={}".format(cfg_dataset['TEST_DATA_PATH']))
    train = PolyDataset.from_pickle(cfg_dataset['TRAIN_DATA_PATH'])
    val   = PolyDataset.from_pickle(cfg_dataset['VAL_DATA_PATH'])
    test  = PolyDataset.from_pickle(cfg_dataset['TEST_DATA_PATH'])

    xscaler, yscaler = preprocess_dataset(train, val, test, cfg_dataset)
    logging.info(" xscaler.mean:  {}".format(xscaler.mean_))
    logging.info(" xscaler.scale: {}".format(xscaler.scale_))
    logging.info(" yscaler.mean:  {}".format(yscaler.mean_))
    logging.info(" yscaler.scale: {}".format(yscaler.scale_))

    cfg_model = cfg['MODEL']
    model = build_network_yaml(cfg_model, input_features=train.NPOLY)

    t = Training(model, cfg, train, val, test, xscaler, yscaler)
    model = t.train_model()
    t.model_eval()

