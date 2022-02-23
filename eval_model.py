import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import time

import torch
import torch.nn as nn

from util import IdentityScaler, StandardScaler
from util import RMSELoss

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

plt.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams['font.serif'] = 'Times'

latex_params = {
    "pgf.texsystem": "pdflatex",
    'figure.titlesize' : 'large',
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": 'Times',
    "font.monospace": [],
    "axes.labelsize": 18,
    "font.size": 18,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    #"text.latex.preamble": [
    #    r"\usepackage[utf8]{inputenc}",    # use utf8 fonts 
    #    r"\usepackage[detect-all]{siunitx}",
    #]
}
mpl.rcParams.update(latex_params)

HTOCM = 2.194746313702e5
Boltzmann = 1.380649e-23      # SI: J / K
Hartree = 4.3597447222071e-18 # SI: J
HkT = Hartree/Boltzmann       # to use as:  -V[a.u.]*`HkT`/T

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

def retrieve_checkpoint(folder, fname, NPOLY):
    fpath = os.path.join(folder, fname)
    logging.info("Retrieving checkpoint from fpath={}".format(fpath))

    checkpoint = torch.load(fpath)
    arch = checkpoint["architecture"]

    model = define_model(architecture=arch, NPOLY=NPOLY)
    model.double()
    model.load_state_dict(checkpoint["model"])

    xscaler = StandardScaler.from_precomputed(mean=checkpoint["X_mean"], std=checkpoint["X_std"])
    yscaler = StandardScaler.from_precomputed(mean=checkpoint["y_mean"], std=checkpoint["y_std"])

    return model, xscaler, yscaler

def summarize_optuna_run(optuna_folder, NPOLY):
    model_names = [f for f in os.listdir(optuna_folder) if os.path.isfile(os.path.join(optuna_folder, f))]

    for model_name in sorted(model_names):
        model, xscaler, yscaler = retrieve_checkpoint(folder=optuna_folder, fname=model_name, NPOLY=NPOLY)
        rmse_descaler = yscaler.std.item()

        Xtr = xscaler.transform(X)
        ytr = yscaler.transform(y)

        with torch.no_grad():
            pred = model(Xtr)

        RMSE = RMSELoss()
        rmse_full = RMSE(ytr, pred) * rmse_descaler * HTOCM
        logging.info("model: {}; full dataset RMSE: {} cm-1".format(model_name, rmse_full))


def load_dataset(folder, fname):
    fpath = os.path.join(folder, fname)
    logging.info("Loading dataset from fpath={}".format(fpath))

    d = torch.load(fpath)
    X, y = d["X"], d["y"]

    return X, y

def load_published():
    fname = "CH4-N2/published-pes/symm-adapted-published-pes-opt1.txt"
    data = np.loadtxt(fname)
    return data[:,1], data[:,2]

def plot_rmse_from_checkpoint(folder, fname, X, y):
    NPOLY = X.size()[1]
    model, xscaler, yscaler = retrieve_checkpoint(folder=folder, fname=fname, NPOLY=NPOLY)

    Xtr = xscaler.transform(X)
    ytr_pred = model(Xtr)

    y_mean, y_std = yscaler.mean, yscaler.std
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

    calc, published_fit = load_published()
    published_abs_error = calc - published_fit

    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    plt.scatter(calc_energy, abs_error, s=20, marker='o', facecolors='none', color='k', lw=0.5, label='NN')
    plt.scatter(calc, published_abs_error, s=20, marker='o', facecolors='none', color='r', lw=0.5, label='published')

    plt.xlim((-500.0, MAX_ENERGY))
    plt.ylim((-50.0, 50.0))

    plt.xlabel(r"Energy, cm$^{-1}$")
    plt.ylabel(r"Absolute error, cm$^{-1}$")

    ax.xaxis.set_major_locator(plt.MultipleLocator(500.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(100.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(10.0))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1.0))

    ax.tick_params(axis='x', which='major', width=1.0, length=6.0)
    ax.tick_params(axis='x', which='minor', width=0.5, length=3.0)
    ax.tick_params(axis='y', which='major', width=1.0, length=6.0)
    ax.tick_params(axis='y', which='minor', width=0.5, length=3.0)

    #plt.savefig("abs-error-distribution.png", format="png", dpi=300)
    plt.legend(fontsize=14)

    plt.show()

def timeit_model(model, X):
    ncycles = 100
    start = time.time()

    with torch.no_grad():
        for k in range(ncycles):
            pred = model(X)

    end = time.time()

    cycle_t = (end - start) / ncycles
    logging.info("Total execution time:     {} s".format(end - start))
    logging.info("Execution time per cycle: {} mcs".format(cycle_t * 1e6))

    npoints = X.size()[0]
    print("npoints: {}".format(npoints))
    point_t = cycle_t / npoints
    logging.info("Execution time per point: {} mcs".format(point_t * 1e6))


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    X, y = load_dataset("CH4-N2", "dataset.pt")
    NPOLY = X.size()[1]


    ###
    #yscaler = StandardScaler()
    #yscaler.fit(y)
    #ytr = yscaler.transform(y)

    #EMIN = torch.abs(y.min())
    #r = 0.005
    #w = r / (r + y + EMIN)
    #we = torch.exp(-y * HkT / 2000.0)
    #f = yscaler.std * HTOCM / 2000.0
    #w = torch.exp(-ytr * f)
    #w = w / w.max()

    #plt.figure(figsize=(10, 10))

    #plt.scatter(y * HTOCM, w, color='k', s=10, facecolors='none')
    #plt.scatter(y * HTOCM, we, color='r', s=10)

    #plt.xlim((-400, 2000.0))

    #plt.show()
    ###


    ############## 
    #summarize_optuna_run(optuna_folder="optuna-run-8991813c-ecb4-4c93-a3bf-0160a83a81a2", NPOLY=NPOLY)
    ############## 

    ############## 
    plot_rmse_from_checkpoint(folder=".", fname="checkpoint.pt", X=X, y=y)
    #plot_rmse_from_checkpoint(folder=".", fname="checkpoint_linreg.pt", X=X, y=y)
    #plot_rmse_from_checkpoint(folder="optuna-run-8991813c-ecb4-4c93-a3bf-0160a83a81a2", fname="model-2.pt", X=X, y=y)
    ############## 

    ############## 
    #model, xscaler, _ = retrieve_checkpoint(folder="optuna-run-98b0dd87-51ad-42b7-86b5-de7301440bce", fname="model-2.pt", NPOLY=NPOLY)
    #timeit_model(model, xscaler.transform(X))
    ############## 

    ############## 
    #model, _, _ = retrieve_checkpoint(folder="optuna-run-98b0dd87-51ad-42b7-86b5-de7301440bce", fname="model-2.pt", NPOLY=NPOLY)
    #export_torchscript(fname="test.pt", model=model, NPOLY=NPOLY)
    ############## 

