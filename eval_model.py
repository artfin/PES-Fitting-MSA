import logging
import numpy as np
import os

import torch
import torch.nn as nn

from model import FCNet
from dataset import PolyDataset
from dataset import HTOCM

# very important to fix the seed !
torch.manual_seed(42)
np.random.seed(42)

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    wdir     = "./H2-H2O"
    order    = "3"
    symmetry = "2 2 1"

    model_fname = "NN_{}_{}.pt".format(order, symmetry)
    model_path  = os.path.join(wdir, model_fname)
    logging.info("loading model from {}...".format(model_path))
    state_dict = torch.load(model_path)

    NPOLY      = 102
    n_layers   = 2
    activation = nn.Tanh
    init_form  = "uniform"
    model = FCNet(NPOLY=NPOLY, n_layers=n_layers, activation=activation, init_form=init_form)
    model.double()

    model.load_state_dict(state_dict)
    model.eval()

    wdir     = './H2-H2O'
    order    = "3"
    symmetry = "2 2 1"
    dataset  = PolyDataset(wdir=wdir, config_fname='points.dat', order=order, symmetry=symmetry)

    ids = np.random.permutation(len(dataset))
    train_size = int(0.8 * len(dataset))
    test_size  = len(dataset) - train_size
    logging.info("Test size  = {}".format(test_size))

    train_ids = ids[:train_size]
    test_ids  = ids[train_size:]

    test_ids         = ids[train_size:]
    X, y             = dataset.X, dataset.y
    X_train, y_train = X[train_ids], y[train_ids]
    X_test, y_test   = X[test_ids], y[test_ids]

    # scaling
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    # first polynomial is a constant => std = 0.0
    std[0] = 1.0

    X_test = (X_test - mean) / std

    y_pred = model(X_test)
    y_pred = y_pred.detach().numpy()


    print("Prediction \t Test \t Diff")
    for ypred_val, ytest_val in zip(y_pred, y_test):
        diff = ypred_val[0] - ytest_val[0]
        print(" {:.8f} \t {:.8f} \t {:.8f}".format(ypred_val[0] * HTOCM, ytest_val[0] * HTOCM, diff * HTOCM))
