# about activation functions:
# https://neurohive.io/ru/osnovy-data-science/activation-functions/ 

import logging
import numpy as np
import os

import torch
import torch.nn as nn

from model import FCNet
from dataset import PolyDataset

torch.manual_seed(42)
np.random.seed(42)

def validate(model, X_test, y_test, criterion):
    # in case the structure of the model changes
    model.eval()

    cumloss = 0.0
    with torch.no_grad():
        y_pred = model(X_test)
        loss = criterion(y_pred, y_test)
        cumloss += loss

    sz = X_test.shape[0]
    return cumloss / sz

from tqdm import tqdm
import torch.optim as optim

def train(model, X_train, y_train, X_test, y_test, epochs=20):
    # in case the structure of the model changes
    model.train()

    criterion = nn.MSELoss()

    # TODO: also try SGD
    # optimizer = optim.SGD(model.parameters(), lr=1e-4)

    # weight decay???
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    train_losses, test_losses = [], []

    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        y_pred = model(X_train)
        train_loss = criterion(y_pred, y_train)
        train_losses.append(train_loss.item())

        train_loss.backward()
        optimizer.step()

        test_loss = validate(model, X_test, y_test, criterion)
        test_losses.append(test_loss)

        if epoch % 50 == 0:
            print("Epoch: {}; train loss: {:.10f}; test loss: {:.10f}".format(
                epoch, train_loss.item(), test_loss
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
    # a hack to avoid dividing by zero: 
    # first polynomial is a constant => std = 0.0
    std[0] = 1.0

    np.savetxt("_mean.txt", mean)
    np.savetxt("_std.txt", std)

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

    train_loss, test_loss = train(model, X_train, y_train, X_test, y_test, epochs=5000)

    model_fname = "NN_{}_{}.pt".format(order, symmetry)
    model_path  = os.path.join(wdir, model_fname)
    logging.info("saving model to {}...".format(model_path))

    torch.save(model.state_dict(), model_path)

