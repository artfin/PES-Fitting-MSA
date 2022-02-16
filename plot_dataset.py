import os
import logging
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

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
    "font.size": 10,
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

def load_dataset(folder, fname):
    fpath = os.path.join(folder, fname)
    logging.info("Loading dataset from fpath={}".format(fpath))

    d = torch.load(fpath)
    X, y = d["X"], d["y"]

    return X, y

def show_train_val_test_energy_distribution(X, y):
    X_train, y_train, X_val, y_val, X_test, y_test, _, _ = split_train_val_test(X, y, scale_params={"Xscale": None, "yscale": None})

    plt.figure(figsize=(10, 10))
    plt.title("Energy distribution")
    plt.xlabel(r"Energy, cm^{-1}")
    plt.ylabel(r"Density")

    plt.xlim((-500.0, 2000.0))

    # density=True:
    # displays a probability density: each bin displays the bin's raw count divided by 
    # the total number of counts and the bin width, so that the area under the histogram
    # integrates to 1
    nbins = 500
    plt.hist(y_train.numpy() * HTOCM, bins=nbins, density=True, lw=3, fc=(0, 0, 1, 0.5), label='train')
    plt.hist(y_val.numpy()   * HTOCM, bins=nbins, density=True, lw=3, fc=(1, 0, 0, 0.5), label='val')
    plt.hist(y_test.numpy()  * HTOCM, bins=nbins, density=True, lw=3, fc=(0, 1, 0, 0.5), label='test')
    plt.legend(fontsize=14)
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

def show_energy_distribution(y, xlim=(-500, 500)):
    energies = y.numpy() * HTOCM

    plt.figure(figsize=(10, 10))

    ax = plt.gca()
    plt.xlabel(r"Energy, cm$^{-1}$")
    plt.ylabel(r"Number of points")

    plt.xlim(xlim)

    plt.hist(energies, bins=500, width=150.0, color='#88B04B')

    plt.yscale('log')
    plt.ylim((1e2, 6e4))

    ax.yaxis.set_major_formatter(ScalarFormatter())

    ax.xaxis.set_major_locator(plt.MultipleLocator(500.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(100.0))

    ax.tick_params(axis='x', which='major', width=1.0, length=6.0)
    ax.tick_params(axis='x', which='minor', width=0.5, length=3.0)
    ax.tick_params(axis='y', which='major', width=1.0, length=6.0)
    ax.tick_params(axis='y', which='minor', width=0.5, length=3.0)

    plt.show()

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    #X, y = load_dataset("CH4-N2", "dataset.pt")
    #show_energy_distribution(y, xlim=(-300, 2950))


