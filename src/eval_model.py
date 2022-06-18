import argparse
import collections
from itertools import combinations
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
from train_model import load_dataset

import pathlib
BASEDIR = pathlib.Path(__file__).parent.parent.resolve()

import sys
sys.path.insert(0, os.path.join(BASEDIR, "external", "pes"))
from pybind_ch4 import Poten_CH4

plt.style.use('science')

plt.rcParams.update({
    "font.family": "serif",
    "font.serif" : ["Times"],
    'figure.titlesize' : "Large",
    "axes.labelsize" : 21,
    "xtick.labelsize" : 18,
    "ytick.labelsize" : 18,
})

Boltzmann = 1.380649e-23      # SI: J / K
Hartree   = 4.3597447222071e-18 # SI: J
HkT       = Hartree/Boltzmann       # to use as:  -V[a.u.]*`HkT`/T
BOHRTOANG = 0.529177249

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


def trim_png(figname):
    cl('convert {0} -trim +repage {0}'.format(figname))

def retrieve_checkpoint(cfg, chk_fname="checkpoint.pt"):
    chk_path = os.path.join(cfg["OUTPUT_PATH"], chk_fname)
    checkpoint = torch.load(chk_path, map_location=torch.device('cpu'))
    meta_info = checkpoint["meta_info"]

    cfg_model = cfg['MODEL']
    model = build_network_yaml(cfg_model, input_features=meta_info["NPOLY"])
    model.load_state_dict(checkpoint["model"])

    nparams = 0
    for name, param in model.named_parameters():
        params = torch.tensor(param.size())
        nparams += torch.prod(params, 0)

    logging.info("Number of parameters: {}".format(nparams))

    xscaler        = StandardScaler()
    xscaler.mean_  = checkpoint['X_mean']
    xscaler.scale_ = checkpoint['X_std']

    yscaler        = StandardScaler()
    yscaler.mean_  = checkpoint['y_mean']
    yscaler.scale_ = checkpoint['y_std']

    evaluator = Evaluator(model, xscaler, yscaler, meta_info)
    return evaluator

def load_published():
    fname = "CH4-N2/published-pes/symm-adapted-published-pes-opt1.txt"
    data = np.loadtxt(fname)
    return data[:,1], data[:,2]

def plot_errors_from_checkpoint(evaluator, train, val, test, EMAX, ylim, ylocators, figpath=None, add_reference_pes=True):
    error_train = (train.y - evaluator(train.X)).detach().numpy()
    error_val   = (val.y - evaluator(val.X)).detach().numpy()
    error_test  = (test.y - evaluator(test.X)).detach().numpy()

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)

    if add_reference_pes:
        calc, published_fit = load_published()
        published_abs_error = calc - published_fit
        plt.scatter(calc, published_abs_error, s=20, marker='o', facecolors='none', color='#CFBFF7', lw=1.0, label='Symmetry-adapted angular basis')

    plt.scatter(train.y, error_train, s=20, marker='o', facecolors='none', color='#FF6F61', lw=1.0, label='train')
    plt.scatter(val.y,   error_val,  s=20, marker='o', facecolors='none', color='#6CD4FF', lw=1.0, label='val')
    plt.scatter(test.y,  error_test, s=20, marker='o', facecolors='none', color='#88B04B', lw=1.0, label='test')

    plt.xlim((-200.0, EMAX))
    plt.ylim(ylim)

    plt.xlabel(r"Energy, cm$^{-1}$")
    plt.ylabel(r"Absolute error, cm$^{-1}$")

    if np.isclose(EMAX, 2000.0):
        ax.xaxis.set_major_locator(plt.MultipleLocator(500.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(100.0))
    elif np.isclose(EMAX, 10000.0):
        ax.xaxis.set_major_locator(plt.MultipleLocator(2000.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1000.0))
    else:
        raise NotImplementedError

    ax.yaxis.set_major_locator(plt.MultipleLocator(ylocators[0]))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(ylocators[1]))

    ax.tick_params(axis='x', which='major', width=1.0, length=6.0)
    ax.tick_params(axis='x', which='minor', width=0.5, length=3.0)
    ax.tick_params(axis='y', which='major', width=1.0, length=6.0)
    ax.tick_params(axis='y', which='minor', width=0.5, length=3.0)

    lgnd = plt.legend(fontsize=18)
    for element in lgnd.legendHandles:
        element.set_lw(1.5)

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


def plot_errors_for_files(cfg_dataset, evaluator, xyz_paths, EMAX, labels, ylim, ylocators, figpath=None):
    wdir = "datasets/external"

    known_options = ('ORDER', 'SYMMETRY', 'TYPE', 'INTRAMOLECULAR_TO_ZERO', 'PURIFY', 'NORMALIZE', 'ENERGY_LIMIT')
    for option in cfg_dataset.keys():
        assert option in known_options, "Unknown option: {}".format(option)

    order        = cfg_dataset['ORDER']
    symmetry     = cfg_dataset['SYMMETRY']
    typ          = cfg_dataset['TYPE'].lower()
    intramz      = cfg_dataset.get('INTRAMOLECULAR_TO_ZERO', False)
    purify       = cfg_dataset.get('PURIFY', False)

    assert order in (3, 4, 5)
    assert typ in ('rigid', 'nonrigid', 'nonrigid-clip')

    res_blocks = []
    for block in xyz_paths:
        pd = PolyDataset(wdir=wdir, xyz_file=block["xyz_path"], limit_file=block["limits_path"], order=order,
                         symmetry=symmetry, intramz=intramz, purify=purify)

        intermol_energy = evaluator(pd.X)
        error = (pd.y - intermol_energy).detach().numpy()

        r = np.hstack((intermol_energy, error))
        res_blocks.append(r)

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)

    colors = ['#FF6F61', '#6CD4FF', '#88B04B', '#575D90']
    zorder = 4

    for res_block, color, label in zip(res_blocks, colors, labels):
        ind = res_block[:,0] < EMAX
        plt.scatter(res_block[:,0][ind], res_block[:,1][ind], s=20.0, color=color, facecolor='none', lw=1.0, 
                    label=label, zorder=zorder)
        zorder = zorder - 1

    plt.xlim((-200.0, EMAX))
    plt.ylim(ylim)

    plt.xlabel(r"Energy, cm$^{-1}$")
    plt.ylabel(r"Absolute error, cm$^{-1}$")

    if np.isclose(EMAX, 2000.0):
        ax.xaxis.set_major_locator(plt.MultipleLocator(500.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(100.0))
    elif np.isclose(EMAX, 10000.0):
        ax.xaxis.set_major_locator(plt.MultipleLocator(2000.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1000.0))
    else:
        raise NotImplementedError

    ax.yaxis.set_major_locator(plt.MultipleLocator(ylocators[0]))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(ylocators[1]))

    ax.tick_params(axis='x', which='major', width=1.0, length=6.0)
    ax.tick_params(axis='x', which='minor', width=0.5, length=3.0)
    ax.tick_params(axis='y', which='major', width=1.0, length=6.0)
    ax.tick_params(axis='y', which='minor', width=0.5, length=3.0)

    plt.legend(fontsize=14)

    if figpath is not None:
        plt.savefig(figpath, format="png", dpi=300)

    plt.show()


def show_model_evaluation(evaluator, train, val, test, emax):
    mean_diff, max_diff = [], []

    for sampling_set in [train, val, test]:
        pred = torch.from_numpy(evaluator(sampling_set.X))

        ind = (sampling_set.y < emax).nonzero()[:,0]
        ys = sampling_set.y[ind]
        preds = pred[ind]
        diff = torch.abs(ys - preds)

        mean = torch.mean(diff)
        maxx = torch.max(diff)

        mean_diff.append(mean)
        max_diff.append(maxx)

    logging.info("[< {:.0f} cm-1] MEAN DIFFERENCE: (train) {:.3f} \t (val) {:.3f} \t (test) {:.3f}".format(emax, *mean_diff))
    logging.info("[< {:.0f} cm-1] MAX  DIFFERENCE: (train) {:.3f} \t (val) {:.3f} \t (test) {:.3f}".format(emax, *max_diff))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", required=True, type=str,
                        help="path to folder with YAML configuration file")
    parser.add_argument("--model_name",   required=True, type=str,
                        help="the name of the YAML configuration file [without extension]")
    parser.add_argument("--EMAX",         required=True, type=float,
                        help="maximum value of the energy range over which model should be evaluated")
    parser.add_argument("--learning_overview", required=False, type=str2bool, default=False,
                        help="whether to create on overview over train/val/test sets [False]")
    parser.add_argument("--ch4_overview",  required=False, type=str2bool, default=False,
                        help="whether to create an overview over CH4 energies [False]")
    parser.add_argument("--add_reference_pes", required=False, type=str2bool, default=False,
                        help="whether to add errors of reference potential on plots [False]")

    args = parser.parse_args()

    MODEL_FOLDER = os.path.join(BASEDIR, args.model_folder)
    assert os.path.isdir(MODEL_FOLDER), "Path to folder is invalid: {}".format(MODEL_FOLDER)
    logging.info("MODEL_FOLDER: {}".format(MODEL_FOLDER))

    MODEL = args.model_name
    cfg_fpath = os.path.join(MODEL_FOLDER, MODEL + ".yaml")
    assert os.path.isfile(cfg_fpath), "YAML configuration file does not exist at {}".format(cfg_fpath)
    logging.info("MODEL: {}".format(MODEL))

    logging.info("Values of optional parameters:")
    logging.info("  learning_overview: {}".format(args.learning_overview))
    logging.info("  ch4_overview:      {}".format(args.ch4_overview))
    logging.info("  add_reference:     {}".format(args.add_reference_pes))

    logging.info("EMAX: {}".format(args.EMAX))

    chk_fpath = os.path.join(MODEL_FOLDER, MODEL + ".pt")
    assert os.path.isfile(chk_fpath), "File with model weights (.pt) does not exist at {}".format(chk_fpath)

    with open(cfg_fpath, mode="r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.info(exc)

    logging.info("loaded configuration file from {}".format(cfg_fpath))

    # in order to plot ALL the points of the RIGID dataset
    # instead of only the clipped part of it
    cfg_dataset = cfg['DATASET']
    if cfg_dataset['TYPE'] == 'NONRIGID-CLIP':
        cfg_dataset['TYPE'] = 'NONRIGID'

    evaluator = retrieve_checkpoint(cfg, chk_fpath)

    if args.learning_overview:
        train, val, test = load_dataset(cfg_dataset)

        show_model_evaluation(evaluator, train, val, test, args.EMAX)

        if cfg_dataset['TYPE'] == 'NONRIGID':
            ylim      = (-20.0, 20.0)
            ylocators = (5.0, 1.0)
        else:
            raise NotImplementedError

        errors_png = os.path.join(MODEL_FOLDER, MODEL + "-EMAX={}.png".format(args.EMAX))
        logging.info("errors_png: {}".format(errors_png))

        plot_errors_from_checkpoint(evaluator, train, val, test, args.EMAX, ylim=ylim, ylocators=ylocators,
                                    figpath=errors_png, add_reference_pes=args.add_reference_pes)

    if args.ch4_overview:
        assert cfg_dataset['TYPE'] == 'NONRIGID'

        from make_dataset import RAW_DATASET_PATHS
        xyz_paths = RAW_DATASET_PATHS["NONRIGID"]

        ylim      = (-20.0, 20.0)
        ylocators = (5.0, 1.0)

        labels = [r"CH$_4$: eq", r"CH$_4$: 0--1000 cm$^{-1}$", r"CH$_4$: 1000--2000 cm$^{-1}$", r"CH$_4$: 2000--3000 cm$^{-1}$"]

        overview_png = os.path.join(MODEL_FOLDER, MODEL + "-ch4-overview.png")
        logging.info("overview_png: {}".format(overview_png))

        plot_errors_for_files(cfg_dataset, evaluator, xyz_paths, args.EMAX, labels, ylim, ylocators, figpath=overview_png)
