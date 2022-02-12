import logging
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torch.nn as nn
from torch.optim import LBFGS, Adam

from dataset import PolyDataset

torch.manual_seed(42)
np.random.seed(42)

# taken from https://github.com/hjmshi/PyTorch-LBFGS 
# from LBFGS import LBFGS, FullBatchLBFGS

# https://github.com/amirgholami/adahessian
# from adahessian import Adahessian

HTOCM = 2.194746313702e5

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
            nn.Linear(self.NPOLY, 20),
            activation(),
            nn.Linear(20, 20),
            activation(),
            nn.Linear(20, 1)
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

class IdentityScaler:
    def fit(self, x):
        pass

    def transform(self, x):
        return x

class StandardScaler:
    def fit(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x):
        c = torch.clone(x)
        c -= self.mean
        c /= self.std
        return torch.nan_to_num(c, nan=1.0)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

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

    assert scale_params["yscale"] in SCALE_OPTIONS
    if scale_params["yscale"] == "std":
        yscaler = StandardScaler()
    elif scale_params["yscale"] == None:
        yscaler = IdentityScaler()

    xscaler.fit(X_train)
    Xtr_train = xscaler.transform(X_train)
    Xtr_val   = xscaler.transform(X_val)
    Xtr_test  = xscaler.transform(X_test)

    yscaler.fit(y_train)
    ytr_train = yscaler.transform(y_train)
    ytr_val   = yscaler.transform(y_val)
    ytr_test  = yscaler.transform(y_test)

    if scale_params["yscale"] == "std":
        rmse_descaler = torch.linalg.norm(yscaler.std)
    elif scale_params["yscale"] == None:
        rmse_descaler = 1.0

    print("rmse_descaler: {}".format(rmse_descaler))

    #show_feature_distribution(X_train, idx=0)
    #show_energy_distribution(y)

    perform_lstsq(X_train, y_train)
    perform_lstsq(Xtr_train, y_train)
    perform_lstsq(Xtr_train, ytr_train)

    model = FCNet(NPOLY=dataset.NPOLY).double()
    nparams = count_params(model)
    logging.info("number of parameters: {}".format(nparams))

    optimizer = LBFGS(model.parameters(), lr=1.0, line_search_fn='strong_wolfe', tolerance_grad=1e-14, tolerance_change=1e-14, max_iter=100)
    #optimizer = optim.Adam(model.parameters(), lr=1e-3)

    metrics = nn.MSELoss()

    if isinstance(optimizer, LBFGS):
        logging.info("LBFGS optimizer is selected")
        optim_type = "lbfgs"
    elif isinstance(optimizer, Adam):
        logging.info("Adam optimizer is selected")
        optim_type = "adam"
    else:
        raise ValueError("Unknown optimizer is chosen.")

    n_epochs = 10
    for epoch in range(n_epochs):
        if optim_type == "adam":
            optimizer.zero_grad()
            y_pred = model(Xtr_train)

            # the use of RMSE instead of MSE GREATLY improves the convergence 
            loss = torch.sqrt(metrics(y_pred, ytr_train))
            loss.backward()
            optimizer.step()

        elif optim_type == "lbfgs":
            def closure():
                optimizer.zero_grad()
                y_pred = model(Xtr_train)

                # the use of RMSE instead of MSE GREATLY improves the convergence 
                loss = torch.sqrt(metrics(y_pred, ytr_train))
                loss.backward()
                return loss

            optimizer.step(closure)
            loss = closure()

        with torch.no_grad():
            pred_val = model(Xtr_val)
            loss_val = metrics(pred_val, ytr_val)
            rmse_val = torch.sqrt(loss_val) * rmse_descaler * HTOCM

        rmse_train = loss.item() * rmse_descaler * HTOCM
        print("Epoch: {}; train RMSE: {:.10f} cm-1; validation RMSE: {:.10f}".format(
            epoch, rmse_train, rmse_val
        ))

    #model_fname = "NN_{}_{}.pt".format(order, symmetry)
    #model_path  = os.path.join(wdir, model_fname)
    #logging.info("saving model to {}...".format(model_path))

    #torch.save(model.state_dict(), model_path)
