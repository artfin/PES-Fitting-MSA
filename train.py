import logging
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import LBFGS, Adam

from dataset import PolyDataset

torch.manual_seed(42)
np.random.seed(42)

# taken from https://github.com/hjmshi/PyTorch-LBFGS 
# from LBFGS import LBFGS, FullBatchLBFGS

# https://github.com/amirgholami/adahessian
# from adahessian import Adahessian

HTOCM = 2.194746313702e5

class StandardScaler:
    def fit(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x):
        c = torch.clone(x)
        c -= self.mean
        c /= self.std
        return torch.nan_to_num(c, nan=1.0)

class IdentityScaler:
    def fit(self, x):
        pass

    def transform(self, x):
        return x

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, ypred, y):
        return torch.sqrt(self.mse(ypred, y))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def count_params(model):
    nparams = 0
    for name, param in model.named_parameters():
        params = torch.tensor(param.size())
        nparams += torch.prod(params, 0)

    return nparams

def perform_lstsq(X, y, show_results=False):
    coeff = torch.linalg.lstsq(X, y, driver='gelss').solution
    y_pred = X @ coeff

    if show_results:
        NCONFIGS = y.size()[0]
        for n in range(NCONFIGS):
            print("{} \t {} \t {}".format(
                y[n].item() * HTOCM, y_pred[n].item() * HTOCM, (y[n] - y_pred[n]).item() * HTOCM
            ))

    loss = nn.MSELoss()
    rmse = torch.sqrt(loss(y, y_pred))
    rmse *= HTOCM
    print("(lstsq) RMSE: {} cm-1".format(rmse))

class FCNet(nn.Module):
    # Jun Li, Bin Jiang, and Hua Guo
    # J. Chem. Phys. 139, 204103 (2013); https://doi.org/10.1063/1.4832697
    # Suggest using Tanh activation function and 2 hidden layers
    def __init__(self, NPOLY, activation=nn.Tanh):
        super().__init__()

        self.NPOLY      = NPOLY
        self.activation = activation()

        # LinearRegressor model
        #self.layers = nn.Sequential(
        #    nn.Linear(self.NPOLY, 1, bias=False)
        #)

        self.layers = nn.Sequential(
            nn.Linear(self.NPOLY, 10),
            activation(),
            nn.Linear(10, 10),
            activation(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = x.view(-1, self.NPOLY)
        return self.layers(x)

def show_energy_distribution(y):
    energies = y.numpy() * HTOCM

    plt.figure(figsize=(10, 10))
    plt.title("Energy distribution")
    plt.xlabel(r"Energy, cm^{-1}")
    plt.ylabel(r"Density")

    plt.xlim((-500.0, 500.0))

    plt.hist(energies, bins=500)
    plt.show()

def show_feature_distribution(X, idx):
    if isinstance(X, np.ndarray):
        feature = X[:, idx]
    elif isinstance(X, torch.tensor):
        feature = X.numpy()[:, idx]
    else:
        raise ValueError("Unknown type")

    plt.figure(figsize=(10, 10))
    plt.title("Invariant polynomial distribution")
    plt.xlabel(r"Polynomial value")
    plt.ylabel(r"Density")

    plt.hist(feature, bins=500)
    plt.show()


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

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)

        elif score < self.best_score and (self.best_score - score) > self.tol:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)

        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.status = True

        logging.debug("Best validation RMSE: {:.2f}; current validation RMSE: {:.2f}".format(self.best_score, score))
        logging.debug("ES counter: {}; ES patience: {}".format(self.counter, self.patience))

    def save_checkpoint(self, model):
        logging.debug("Saving the checkpoint")
        torch.save(model.state_dict(), self.chk_path)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    wdir     = './H2-H2O'
    order    = "3"
    symmetry = "2 2 1"

    dataset = PolyDataset(wdir=wdir, config_fname='points.dat', order=order, symmetry=symmetry)

    ids = np.random.permutation(len(dataset))
    val_start  = int(len(dataset) * 0.8)
    test_start = int(len(dataset) * 0.9)
    train_ids = ids[:val_start]
    val_ids = ids[val_start:test_start]
    test_ids = ids[test_start:]

    X, y             = dataset.X, dataset.y
    X_train, y_train = X[train_ids], y[train_ids]
    X_val, y_val     = X[val_ids], y[val_ids]
    X_test, y_test   = X[test_ids],  y[test_ids]

    logging.info("Train size      = {}".format(y_train.size()))
    logging.info("Validation size = {}".format(y_val.size()))
    logging.info("Test size       = {}".format(y_test.size()))

    SCALE_OPTIONS = [None, "std"] # ? "minmax"
    scale_params = {
        "Xscale" : "std",
        "yscale" : "std"
    }

    assert scale_params["Xscale"] in SCALE_OPTIONS
    if scale_params["Xscale"] == "std":
        xscaler = StandardScaler()
    elif scale_params["Xscale"] == None:
        xscaler = IdentityScaler()
    else:
        raise ValueError("unreachable")

    assert scale_params["yscale"] in SCALE_OPTIONS
    if scale_params["yscale"] == "std":
        yscaler = StandardScaler()
    elif scale_params["yscale"] == None:
        yscaler = IdentityScaler()
    else:
        raise ValueError("unreachable")

    xscaler.fit(X_train)
    Xtr       = xscaler.transform(X)
    Xtr_train = xscaler.transform(X_train)
    Xtr_val   = xscaler.transform(X_val)
    Xtr_test  = xscaler.transform(X_test)

    yscaler.fit(y_train)
    ytr       = yscaler.transform(y)
    ytr_train = yscaler.transform(y_train)
    ytr_val   = yscaler.transform(y_val)
    ytr_test  = yscaler.transform(y_test)

    if scale_params["yscale"] == "std":
        rmse_descaler = torch.linalg.norm(yscaler.std)
    elif scale_params["yscale"] == None:
        rmse_descaler = 1.0
    else:
        raise ValueError("unreachable")

    #show_feature_distribution(X_train, idx=0)
    #show_energy_distribution(y)

    logging.info("matrix least-squares problem: (raw X, raw Y)")
    perform_lstsq(X_train, y_train)

    model = FCNet(NPOLY=dataset.NPOLY).double()
    nparams = count_params(model)
    logging.info("number of parameters: {}".format(nparams))

    optimizer = LBFGS(model.parameters(), lr=1.0, line_search_fn='strong_wolfe', tolerance_grad=1e-14, tolerance_change=1e-14, max_iter=100)
    #optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if isinstance(optimizer, LBFGS):
        logging.info("LBFGS optimizer is selected")
        OPTIM_TYPE = "lbfgs"
    elif isinstance(optimizer, Adam):
        logging.info("Adam optimizer is selected")
        OPTIM_TYPE = "adam"
    else:
        raise ValueError("Unknown optimizer is chosen.")

    METRIC_TYPES = ['RMSE', 'MSE']
    METRIC_TYPE = 'MSE'
    assert METRIC_TYPE in METRIC_TYPES

    logging.info("METRIC_TYPE = {}".format(METRIC_TYPE))

    if METRIC_TYPE == 'MSE':
        metric = nn.MSELoss()
    elif METRIC_TYPE == 'RMSE':
        metric = RMSELoss()
    else:
        raise ValueError("unreachable")


    prev_best = None
    best_rmse_val = None

    SCHEDULER_PATIENCE = 5
    RMSE_TOL  = 0.1 # cm-1
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, threshold=RMSE_TOL, threshold_mode='abs', cooldown=2, patience=SCHEDULER_PATIENCE)

    #ES_score = 0
    #ES_patience = 10

    ES_START_EPOCH = 10
    CHK_PATH = 'checkpoint.pt'
    es = EarlyStopping(patience=10, tol=RMSE_TOL, chk_path=CHK_PATH)

    n_epochs = 500

    ################ START TRAINING #######################
    for epoch in range(n_epochs):
        if OPTIM_TYPE == "adam":
            optimizer.zero_grad()
            y_pred = model(Xtr_train)
            loss = metric(y_pred, ytr_train)
            loss.backward()
            optimizer.step()

        elif OPTIM_TYPE == "lbfgs":
            def closure():
                optimizer.zero_grad()
                y_pred = model(Xtr_train)
                loss = metric(y_pred, ytr_train)
                loss.backward()
                return loss

            optimizer.step(closure)
            loss = closure()

        with torch.no_grad():
            pred_val = model(Xtr_val)
            loss_val = metric(pred_val, ytr_val)

        if METRIC_TYPE == 'MSE':
            rmse_val   = torch.sqrt(loss_val) * rmse_descaler * HTOCM
            rmse_train = torch.sqrt(loss)     * rmse_descaler * HTOCM
        elif METRIC_TYPE == 'RMSE':
            rmse_val   = loss_val * rmse_descaler * HTOCM
            rmse_train = loss     * rmse_descaler * HTOCM
        else:
            raise ValueError("unreachable")

        scheduler.step(rmse_val)
        lr = get_lr(optimizer)
        logging.info("Current learning rate: {:.2e}".format(lr))

        if epoch > ES_START_EPOCH:
            es(rmse_val, model)

            if es.status:
                logging.info("Invoking early stop")
                break

        logging.info("Epoch: {}; train RMSE: {:.2f} cm-1; validation RMSE: {:.2f}\n".format(
            epoch, rmse_train, rmse_val
        ))

    ################ END TRAINING #######################

    model_params = torch.load(CHK_PATH)
    model.load_state_dict(model_params)
    logging.info("\nReloading best model from the last checkpoint...")

    with torch.no_grad():
        pred_val = model(Xtr_val)
        loss_val = metric(pred_val, ytr_val)

        pred_test = model(Xtr_test)
        loss_test = metric(pred_test, ytr_test)

        pred_full = model(Xtr)
        loss_full = metric(pred_full, ytr)

        if METRIC_TYPE == 'MSE':
            rmse_val  = torch.sqrt(loss_val) * rmse_descaler * HTOCM
            rmse_test = torch.sqrt(loss_test) * rmse_descaler * HTOCM
            rmse_full = torch.sqrt(loss_full) * rmse_descaler * HTOCM
        elif METRIC_TYPE == 'RMSE':
            rmse_val  = loss_val * rmse_descaler * HTOCM
            rmse_test = loss_test * rmse_descaler * HTOCM
            rmse_full = loss_full * rmse_descaler * HTOCM

        logging.info("Final evaluation:")
        logging.info("Validation RMSE: {:.2f} cm-1".format(rmse_val))
        logging.info("Test RMSE:       {:.2f} cm-1".format(rmse_test))
        logging.info("Full RMSE:       {:.2f} cm-1".format(rmse_full))

    #model_fname = "NN_{}_{}.pt".format(order, symmetry)
    #model_path  = os.path.join(wdir, model_fname)
    #logging.info("saving model to {}...".format(model_path))

    #torch.save(model.state_dict(), model_path)
