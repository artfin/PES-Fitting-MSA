import collections
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn
import time
import yaml

from sklearn.preprocessing import StandardScaler
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from build_model import build_network_yaml
from dataset import PolyDataset
from genpip import cmdstat, cl

import pathlib
BASEDIR = pathlib.Path(__file__).parent.parent.resolve()

plt.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams['font.serif'] = 'Times'

latex_params = {
    "pgf.texsystem": "pdflatex",
    'figure.titlesize' : 'large',
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": 'Times',
    "font.monospace": [],
    "axes.labelsize": 21,
    "font.size": 21,
    "legend.fontsize": 21,
    "xtick.labelsize": 21,
    "ytick.labelsize": 21,
}
mpl.rcParams.update(latex_params)
plt.style.use('dark_background')

Boltzmann = 1.380649e-23      # SI: J / K
Hartree = 4.3597447222071e-18 # SI: J
HkT = Hartree/Boltzmann       # to use as:  -V[a.u.]*`HkT`/T

#def retrieve_checkpoint(folder, fname):
#    fpath = os.path.join(folder, fname)
#    logging.info("Retrieving checkpoint from fpath={}".format(fpath))
#
#    checkpoint = torch.load(fpath)
#    arch = checkpoint["architecture"]
#    NPOLY = checkpoint["meta_info"]["NPOLY"]
#
#    activation = checkpoint["activation"]
#    activation_instance = globals()[activation]()
#
#    model = define_model(architecture=arch, NPOLY=NPOLY, activation=activation_instance)
#    model.double()
#    model.load_state_dict(checkpoint["model"])
#
#    print(model)
#
#    model.eval()
#
#    xscaler = StandardScaler.from_precomputed(mean=checkpoint["X_mean"], std=checkpoint["X_std"])
#    yscaler = StandardScaler.from_precomputed(mean=checkpoint["y_mean"], std=checkpoint["y_std"])
#    meta_info = checkpoint["meta_info"]
#
#    return model, xscaler, yscaler, meta_info

def summarize_optuna_run(optuna_folder):
    model_names = [f for f in os.listdir(optuna_folder) if os.path.isfile(os.path.join(optuna_folder, f))]

    for model_name in sorted(model_names):
        model, xscaler, yscaler, _ = retrieve_checkpoint(folder=optuna_folder, fname=model_name)
        rmse_descaler = yscaler.std.item()

        Xtr = xscaler.transform(X)
        ytr = yscaler.transform(y)

        with torch.no_grad():
            pred = model(Xtr)

        RMSE = RMSELoss()
        rmse_full = RMSE(ytr, pred) * rmse_descaler
        logging.info("model: {}; full dataset RMSE: {} cm-1".format(model_name, rmse_full))


def load_dataset(fpath):
    logging.info("Loading dataset from fpath={}".format(fpath))
    d = torch.load(fpath)
    X, y = d["X"], d["y"]
    return X, y

def load_published():
    fname = "CH4-N2/published-pes/symm-adapted-published-pes-opt1.txt"
    data = np.loadtxt(fname)
    return data[:,1], data[:,2]

def plot_errors_from_checkpoint(evaluator, train, val, test, figpath=None):
    #inf_poly = torch.zeros(1, NPOLY, dtype=torch.double)
    #inf_poly[0, 0] = 1.0
    #inf_poly = xscaler.transform(inf_poly)
    #inf_pred = model(inf_poly)
    #inf_pred = inf_pred * y_std + y_mean
    #print("inf_pred: {}".format(inf_pred))
    #inf_pred = torch.ones(len(X), 1, dtype=torch.double) * inf_pred

    error_train = (train.y - evaluator(train.X)).detach().numpy()
    error_val   = (val.y - evaluator(val.X)).detach().numpy()
    error_test  = (test.y - evaluator(test.X)).detach().numpy()

    calc, published_fit = load_published()
    published_abs_error = calc - published_fit

    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    plt.scatter(train.y, error_train, s=20, marker='o', facecolors='none', color='#FF6F61', lw=0.5, label='train')
    plt.scatter(val.y,   error_val,  s=20, marker='o', facecolors='none', color='#6CD4FF', lw=0.5, label='val')
    plt.scatter(test.y,  error_test, s=20, marker='o', facecolors='none', color='#88B04B', lw=0.5, label='test')
    #plt.scatter(train.y, error_train, s=20, marker='o', facecolors='none', color='#FF6F61', lw=0.5, label='train')
    #plt.scatter(val.y,   error_val,  s=20, marker='o', facecolors='none', color='#FF6F61', lw=0.5, label='val')
    #plt.scatter(test.y,  error_test, s=20, marker='o', facecolors='none', color='#FF6F61', lw=0.5, label='test')
    plt.scatter(calc, published_abs_error, s=20, marker='o', facecolors='none', color='#CFBFF7', lw=0.5, label='Symmetry-adapted angular basis')

    plt.xlim((-200.0, 2000.0))
    plt.ylim((-30.0, 30.0))

    plt.xlabel(r"Energy, cm$^{-1}$")
    plt.ylabel(r"Absolute error, cm$^{-1}$")

    #ax.xaxis.set_major_locator(plt.MultipleLocator(500.0))
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(100.0))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(10.0))
    #ax.yaxis.set_minor_locator(plt.MultipleLocator(2.0))

    #ax.tick_params(axis='x', which='major', width=1.0, length=6.0)
    #ax.tick_params(axis='x', which='minor', width=0.5, length=3.0)
    #ax.tick_params(axis='y', which='major', width=1.0, length=6.0)
    #ax.tick_params(axis='y', which='minor', width=0.5, length=3.0)

    #lgnd = plt.legend(fontsize=18)
    #lgnd.legendHandles[0].set_lw(1.5)
    #lgnd.legendHandles[1].set_lw(1.5)

    if figpath is not None:
        plt.savefig(figpath, format="png", dpi=300)

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

def trim_png(figname):
    cl('convert {0} -trim +repage {0}'.format(figname))


class Evaluator:
    def __init__(self, model, xscaler, yscaler, meta_info):
        self.model = model
        self.xscaler = xscaler
        self.yscaler = yscaler
        self.meta_info = meta_info

    def __call__(self, X):
        self.model.eval()
        Xtr = self.xscaler.transform(X)
        ytr = self.model(torch.from_numpy(Xtr))
        return self.yscaler.inverse_transform(ytr.detach().numpy())


def retrieve_checkpoint(cfg):
    chk_path = os.path.join(cfg["OUTPUT_PATH"], "checkpoint.pt")
    checkpoint = torch.load(chk_path, map_location=torch.device('cpu'))
    meta_info = checkpoint["meta_info"]

    cfg_model = cfg['MODEL']
    model = build_network_yaml(cfg_model, input_features=meta_info["NPOLY"])
    model.load_state_dict(checkpoint["model"])

    xscaler        = StandardScaler()
    xscaler.mean_  = checkpoint['X_mean']
    xscaler.scale_ = checkpoint['X_std']

    yscaler        = StandardScaler()
    yscaler.mean_  = checkpoint['y_mean']
    yscaler.scale_ = checkpoint['y_std']

    evaluator = Evaluator(model, xscaler, yscaler, meta_info)
    return evaluator


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    MODEL_FOLDER = os.path.join(BASEDIR, "models", "rigid", "L1", "L1-tanh")

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
    train = PolyDataset.from_pickle(os.path.join(BASEDIR, cfg_dataset['TRAIN_DATA_PATH']))
    val   = PolyDataset.from_pickle(os.path.join(BASEDIR, cfg_dataset['VAL_DATA_PATH']))
    test  = PolyDataset.from_pickle(os.path.join(BASEDIR, cfg_dataset['TEST_DATA_PATH']))

    evaluator = retrieve_checkpoint(cfg)

    figpath = os.path.join(BASEDIR, cfg['OUTPUT_PATH'], "errors.png")
    plot_errors_from_checkpoint(evaluator, train, val, test, figpath=figpath)


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
    #summarize_optuna_run(optuna_folder="optuna-run-8991813c-ecb4-4c93-a3bf-0160a83a81a2")
    ############## 

    ############## 
    #figname = "abs-error-distribution.png"
    #plot_errors_from_checkpoint(folder=".", fname="checkpoint.pt", X=X, y=y) #, figname=figname)
    #trim_png(figname)
    #plot_rmse_from_checkpoint(folder=".", fname="checkpoint_linreg.pt", X=X, y=y)
    #plot_rmse_from_checkpoint(folder="optuna-run-8991813c-ecb4-4c93-a3bf-0160a83a81a2", fname="model-2.pt", X=X, y=y)
    ############## 

    ############## 
    #model, xscaler, _ = retrieve_checkpoint(folder="optuna-run-98b0dd87-51ad-42b7-86b5-de7301440bce", fname="model-2.pt")
    #timeit_model(model, xscaler.transform(X))
    ############## 
