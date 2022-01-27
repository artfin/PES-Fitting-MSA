import logging
import numpy as np
import os

import torch
import torch.nn as nn

from model import FCNet
from dataset import PolyDataset
from dataset import HTOCM

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
    dataset  = PolyDataset(wdir=wdir, config_fname='test.dat', order=order, symmetry=symmetry)

    mean = np.loadtxt("_mean.txt")
    std  = np.loadtxt("_std.txt")

    X, y = dataset.X, dataset.y
    X = (X - mean) / std

    print(X)

    y_pred = model(X)

    print("y_pred = {0:.8f}".format(y_pred.item() * HTOCM))
    print("y      = {0:.8f}".format(y.item() * HTOCM))




