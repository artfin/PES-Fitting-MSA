import logging
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torch.nn as nn

from util import IdentityScaler, StandardScaler
from util import RMSELoss

HTOCM = 2.194746313702e5

def define_model(architecture, NPOLY):
    """
    architecture:
        tuple (h0, h1, ...)
        It implies the following MLP architecture:
            nn.Linear(NPOLY, h0)
            nn.Tanh()
            nn.Linear(h0, h1)
            nn.Tanh()
            ...
            nn.Tanh()
            nn.Linear(hk, 1)
    """
    layers = []

    in_features  = NPOLY
    # to allow constructing LinearRegressor with architecture=() 
    out_features = NPOLY

    for i in range(len(architecture)):
        out_features = architecture[i]
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.Tanh())

        in_features = out_features

    layers.append(nn.Linear(out_features, 1))
    model = nn.Sequential(*layers)

    sigmoid_gain = torch.nn.init.calculate_gain("tanh")
    for child in model.children():
        if isinstance(child, nn.Linear):
            for _ in range(0, len(layers)):
                torch.nn.init.xavier_uniform_(child.weight, gain=sigmoid_gain)
                if child.bias is not None:
                    torch.nn.init.zeros_(child.bias)

    return model

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    dataset_path = "CH4-N2/dataset.pt"
    print("Loading data from dataset_path = {}".format(dataset_path))
    d = torch.load(dataset_path)
    X, y = d["X"], d["y"]

    NPOLY = X.size()[1]
    model = define_model(architecture=(10, 10), NPOLY=NPOLY)
    model.double()
    model_params = torch.load("checkpoint.pt")
    model.load_state_dict(model_params)

    d = torch.load("CH4-N2/scaler_params.pt")
    X_mean, X_std = d["X_mean"], d["X_std"]
    y_mean, y_std = d["y_mean"], d["y_std"]

    logging.info("Loaded scaler parameters:")
    #logging.info("X_mean: {}; X_std: {}".format(X_mean, X_std))
    logging.info("Y_mean: {}; Y_std: {}".format(y_mean, y_std))

    xscaler = StandardScaler()
    xscaler.fit(X)
    Xtr = xscaler.transform(X, mean=X_mean, std=X_std)

    ytr_pred = model(Xtr)
    y_pred   = ytr_pred * y_std + y_mean

    RMSE = RMSELoss()

    MAX_ENERGY = y.max().item() * HTOCM # cm-1
    print("MAXIMUM ENERGY: {}".format(y.max().item() * HTOCM))
    idx = y.numpy() < MAX_ENERGY / HTOCM
    calc_energy = y.numpy()[idx] * HTOCM
    fit_energy  = y_pred.detach().numpy()[idx] * HTOCM
    print("RMSE(calc_energy, fit_energy): {}".format(RMSE(torch.tensor(calc_energy), torch.tensor(fit_energy))))

    MAX_ENERGY = 3000.0 # cm-1
    idx = y.numpy() < MAX_ENERGY / HTOCM
    calc_energy = y.numpy()[idx] * HTOCM
    fit_energy  = y_pred.detach().numpy()[idx] * HTOCM
    print("RMSE(calc_energy, fit_energy): {}".format(RMSE(torch.tensor(calc_energy), torch.tensor(fit_energy))))

    abs_error = calc_energy - fit_energy

    plt.figure(figsize=(10, 10))
    plt.scatter(calc_energy, abs_error, marker='o', facecolors='none', color='k')

    plt.xlim((-500.0, MAX_ENERGY))
    plt.ylim((-25.0, 25.0))

    plt.xlabel("Energy, cm-1")
    plt.ylabel("Absolute error, cm-1")

    plt.show()


