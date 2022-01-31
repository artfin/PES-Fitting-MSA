import logging
import numpy as np
import os

import torch
import torch.nn as nn

from model import FCNet
from dataset import PolyDataset
from dataset import HTOCM

torch.manual_seed(42)
np.random.seed(42)

from tqdm import tqdm
import torch.optim as optim

# taken from https://github.com/hjmshi/PyTorch-LBFGS 
# from LBFGS import LBFGS, FullBatchLBFGS

# https://github.com/amirgholami/adahessian
from adahessian import Adahessian

def train(model, X_train, y_train, X_test, y_test, opt_type='adam', epochs=100):
    # in case the structure of the model changes
    model.train()

    metric = nn.MSELoss()

    if opt_type == 'lbfgs':
        # Standard version from torch.optim
        optimizer = optim.LBFGS(model.parameters(), lr=1.5, max_iter=20, max_eval=None, tolerance_grad=1e-14, tolerance_change=1e-14, history_size=10)
    elif opt_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
    elif opt_type == 'adahessian':
        logging.info('Adahessian optimizer is chosen')
        optimizer = Adahessian(model.parameters(), lr=1.0, betas=(0.9, 0.999),
                               eps=1e-4, weight_decay=0.0, hessian_power=1.0)
    elif opt_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=1e-4)
    else:
        raise NotImplementedError()

    train_losses, test_losses = [], []

    dprint = epochs // 20

    for epoch in tqdm(range(epochs)):
        def closure():
            optimizer.zero_grad()
            y_pred = model(X_train)

            # try RMSE instead MSE!
            if opt_type == 'adahessian':
                loss = torch.sqrt(metric(y_pred, y_train))
                loss.backward(create_graph=True)
            else:
                loss = torch.sqrt(metric(y_pred, y_train))
                loss.backward()

            return loss

        optimizer.step(closure)

        train_rmse = closure()
        #train_losses.append(train_loss.item())

        with torch.no_grad():
            y_pred = model(X_test)
            test_loss = metric(y_pred, y_test)
            test_rmse = np.sqrt(test_loss.item())

        if epoch % dprint == 0:
            print("Epoch: {}; train rmse: {:.10f}; test rmse: {:.10f}".format(
                epoch, train_rmse * HTOCM, test_rmse * HTOCM
            ))

    return train_losses, test_losses

def count_params(model):
    nparams = 0
    for name, param in model.named_parameters():
        params = torch.tensor(param.size())
        nparams += torch.prod(params, 0)

    return nparams


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
    train_size = int(0.8 * len(dataset))
    test_size  = len(dataset) - train_size
    logging.info("Train size = {}".format(train_size))
    logging.info("Test size  = {}".format(test_size))

    train_ids = ids[:train_size]
    test_ids  = ids[train_size:]

    X, y             = dataset.X, dataset.y
    X_train, y_train = X[train_ids], y[train_ids]
    X_test, y_test   = X[test_ids], y[test_ids]

    # scaling
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    # first polynomial is a constant => std = 0.0
    std[0] = 1.0

    X_train = (X_train - mean) / std
    X_test  = (X_test - mean) / std

    n_layers   = 2
    activation = nn.Tanh
    init_form  = "uniform"
    logging.info("Creating a fully connected neural network:")
    logging.info("    n_layers   = {}".format(n_layers))
    logging.info("    activation = {}".format(activation))
    logging.info("    init_form  = {}".format(init_form))

    model = FCNet(NPOLY=dataset.NPOLY, n_layers=n_layers, activation=activation, init_form=init_form)
    model.double() # use double precision for model parameters 

    nparams = count_params(model)
    logging.info("number of parameters: {}".format(nparams))

    train_loss, test_loss = train(model, X_train, y_train, X_test, y_test,
                                  opt_type='adahessian', epochs=10000)

    model_fname = "NN_{}_{}.pt".format(order, symmetry)
    model_path  = os.path.join(wdir, model_fname)
    logging.info("saving model to {}...".format(model_path))

    torch.save(model.state_dict(), model_path)

