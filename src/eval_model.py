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
from train_model import load_dataset, load_cfg

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
KCALTOCM  = 349.757

class Evaluator:
    def __init__(self, model, xscaler, yscaler, meta_info):
        self.model = model
        self.xscaler = xscaler
        self.yscaler = yscaler
        self.meta_info = meta_info

    def energy(self, X):
        self.model.eval()

        Xtr = self.xscaler.transform(X)

        with torch.no_grad():
            ytr = self.model(torch.from_numpy(Xtr))

        return self.yscaler.inverse_transform(ytr.detach().numpy())

    def forces(self, dataset):
        Xtr = self.xscaler.transform(dataset.X)
        Xtr = torch.from_numpy(Xtr)

        Xtr.requires_grad = True

        y_pred = self.model(Xtr)
        dEdp   = torch.autograd.grad(outputs=y_pred, inputs=Xtr, grad_outputs=torch.ones_like(y_pred), retain_graph=True, create_graph=True)[0]

        Xtr.requires_grad = False

        # take into account normalization of polynomials
        # now we have derivatives of energy w.r.t. to polynomials
        x_scale = torch.from_numpy(self.xscaler.scale_)
        dEdp = torch.div(dEdp, x_scale)

        # force = -dE/dx = -\sigma(E) * dE/d(poly) * d(poly)/dx
        # `torch.einsum` throws a Runtime error without an explicit conversion to Double 
        dEdx = torch.einsum('ij,ijk -> ik', dEdp.double(), dataset.dX.double())

        # take into account normalization of model energy
        y_scale = torch.from_numpy(self.yscaler.scale_)
        dEdx = -torch.mul(dEdx, y_scale)

        return dEdx


def trim_png(figname):
    cl('convert {0} -trim +repage {0}'.format(figname))

def retrieve_checkpoint(cfg, chk_fpath):
    checkpoint = torch.load(chk_fpath, map_location=torch.device('cpu'))
    meta_info = checkpoint["meta_info"]

    if cfg['TYPE'] == 'ENERGY':
        model = build_network_yaml(cfg['MODEL'], input_features=meta_info["NPOLY"], output_features=1)
    elif cfg['TYPE'] == 'DIPOLE':
        model = build_network_yaml(cfg['MODEL'], input_features=meta_info["NPOLY"], output_features=3)

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
    fname = "datasets/raw/symm-adapted-published-pes-opt1.txt"
    data = np.loadtxt(fname)
    return data[:,1], data[:,2]

def plot_errors_from_checkpoint(evaluator, train, val, test, EMAX, ylim, ylocators, figpath=None, add_reference_pes=True):
    error_train = (train.y - evaluator.energy(train.X)).detach().numpy()
    error_val   = (val.y   - evaluator.energy(val.X)).detach().numpy()
    error_test  = (test.y  - evaluator.energy(test.X)).detach().numpy()

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)

    plt.scatter(train.y, error_train, s=20, marker='o', facecolors='none', color='#FF6F61', lw=1.0, label='train')
    #plt.scatter(val.y, error_val, s=20, marker='o', facecolors='none', color='#FF6F61', lw=1.0)
    #plt.scatter(test.y, error_test, s=20, marker='o', facecolors='none', color='#FF6F61', lw=1.0)
    plt.scatter(val.y,   error_val,  s=20, marker='o', facecolors='none', color='#6CD4FF', lw=1.0, label='val')
    plt.scatter(test.y,  error_test, s=20, marker='o', facecolors='none', color='#88B04B', lw=1.0, label='test')

    if add_reference_pes:
        calc, published_fit = load_published()
        published_abs_error = calc - published_fit
        plt.scatter(calc, published_abs_error, s=20, marker='o', facecolors='none', color='#CFBFF7', lw=1.0, label='Symmetry-adapted angular basis', zorder=5)

    EMIN = min(train.y)
    plt.xlim((EMIN, EMAX))
    plt.ylim(ylim)

    plt.xlabel(r"Energy, cm$^{-1}$")
    plt.ylabel(r"Absolute error, cm$^{-1}$")

    #if np.isclose(EMAX, 2000.0):
    #    ax.xaxis.set_major_locator(plt.MultipleLocator(500.0))
    #    ax.xaxis.set_minor_locator(plt.MultipleLocator(100.0))
    #elif np.isclose(EMAX, 10000.0):
    #    ax.xaxis.set_major_locator(plt.MultipleLocator(2000.0))
    #    ax.xaxis.set_minor_locator(plt.MultipleLocator(1000.0))
    #elif np.isclose(EMAX, 50000.0):
    #    ax.xaxis.set_major_locator(plt.MultipleLocator(5000.0))
    #    ax.xaxis.set_minor_locator(plt.MultipleLocator(1000.0))
    #else:
    #    raise NotImplementedError

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


def plot_errors_for_files(cfg_dataset, evaluator, EMAX, labels, ylim, ylocators, figpath=None):
    res_blocks = []
    c_rmse = 0.0
    c_points = 0.0

    file_paths = cfg_dataset['SOURCE']

    for file_path in file_paths:
        pd = PolyDataset(wdir=cfg_dataset['EXTERNAL_FOLDER'], typ='ENERGY', file_path=file_path, order=cfg_dataset['ORDER'], symmetry=cfg_dataset['SYMMETRY'],
                         load_forces=False, atom_mapping=cfg_dataset['ATOM_MAPPING'], variables=cfg_dataset['VARIABLES'], purify=cfg_dataset['PURIFY'])

        intermol_energy = evaluator.energy(pd.X)
        error = (pd.y - intermol_energy).detach().numpy()

        ind = (pd.y < EMAX).nonzero()[:,0]
        ys = pd.y[ind]
        preds = intermol_energy[ind]

        _mse = torch.mean((ys - preds) * (ys - preds))
        _rmse = torch.sqrt(_mse)
        c_rmse += _rmse.item() * ys.size()[0]
        c_points += ys.size()[0]

        logging.info("[< {:.0f} cm-1] file: {}; RMSE: {:.3f}".format(EMAX, file_path, _rmse))

        r = np.hstack((intermol_energy, error))
        res_blocks.append(r)

    c_rmse = c_rmse / c_points
    logging.info("Cumulative RMSE: {:.3f}".format(c_rmse))

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
    else: #np.isclose(EMAX, 10000.0):
        ax.xaxis.set_major_locator(plt.MultipleLocator(2000.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1000.0))

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

def model_evaluation_forces(evaluator, train, val, test, emax):
    natoms = train.NATOMS

    rmse = []

    for sampling_set in [train, val, test]:
        if sampling_set is None:
            rmse.append(0.0)
            continue

        ind   = (sampling_set.y < emax).nonzero()[:,0]

        fs          = sampling_set.dy[ind]
        forces_pred = evaluator.forces(sampling_set)
        preds       = forces_pred[ind].reshape(-1, natoms, 3)

        df = fs - preds
        _mse = torch.mean(torch.einsum('ijk,ijk->i', df, df)) / (3.0 * natoms)
        _rmse = torch.sqrt(_mse)
        rmse.append(_rmse)

    rmse_kcal_mol_A = [ff / KCALTOCM / BOHRTOANG for ff in rmse]

    logging.info("[< {:.0f} cm-1] RMSE FORCE: (train) {:.5f} \t (val) {:.5f} \t (test) {:.5f} cm-1/bohr".format(emax, *rmse))
    logging.info("[< {:.0f} cm-1] RMSE FORCE: (train) {:.5f} \t (val) {:.5f} \t (test) {:.5f} kcal/mol/A".format(emax, *rmse_kcal_mol_A))

def model_evaluation_energy(evaluator, train, val, test, emax, add_reference_pes=False):
    mean_diff, max_diff = [], []
    mse, rmse = [], []

    for sampling_set in [train, val, test]:
        if sampling_set is None:
            mean_diff.append(0.0)
            max_diff.append(0.0)
            rmse.append(0.0)
            continue

        pred = evaluator.energy(sampling_set.X)
        pred = torch.from_numpy(pred)

        ind = (sampling_set.y < emax).nonzero()[:,0]
        ys = sampling_set.y[ind]
        preds = pred[ind]

        diff = torch.abs(ys - preds)

        mean = torch.mean(diff)
        maxx = torch.max(diff)
        mean_diff.append(mean)
        max_diff.append(maxx)

        _mse = torch.mean((ys - preds) * (ys - preds))
        _rmse = torch.sqrt(_mse)
        mse.append(_mse)
        rmse.append(_rmse)

    min_energy = train.y.min()
    max_energy = train.y.max()
    logging.info(" (train) ENERGY RANGE: {:.3f} - {:.3f} cm-1".format(min_energy, max_energy))

    mean_diff_kcal_mol = [ff / KCALTOCM for ff in mean_diff]
    rmse_kcal_mol = [ff/ KCALTOCM for ff in rmse]

    logging.info("[< {:.0f} cm-1] MEAN DIFFERENCE: (train) {:.3f} \t (val) {:.3f} \t (test) {:.3f} cm-1".format(emax, *mean_diff))
    logging.info("[< {:.0f} cm-1] MEAN DIFFERENCE: (train) {:.3f} \t (val) {:.3f} \t (test) {:.3f} kcal/mol".format(emax, *mean_diff_kcal_mol))

    logging.info("[< {:.0f} cm-1] MAX  DIFFERENCE: (train) {:.3f} \t (val) {:.3f} \t (test) {:.3f}".format(emax, *max_diff))
    logging.info("[< {:.0f} cm-1] RMSE: (train) {:.3f} \t (val) {:.3f} \t (test) {:.3f} cm-1".format(emax, *rmse))
    logging.info("[< {:.0f} cm-1] RMSE: (train) {:.5f} \t (val) {:.5f} \t (test) {:.5f} kcal/mol".format(emax, *rmse_kcal_mol))

    if add_reference_pes:
        calc, published_fit = load_published()
        calc = torch.from_numpy(calc)
        published_fit = torch.from_numpy(published_fit)

        ind = (calc < emax).nonzero()[:,0]
        calc          = calc[ind]
        published_fit = published_fit[ind]

        rmse_published = torch.sqrt(torch.mean((calc - published_fit) * (calc - published_fit)))
        logging.info("[< {:.0f} cm-1] RMSE: (published) {:.3f}".format(emax, rmse_published))

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
                        help="the name of the YAML configuration file without extension")
    parser.add_argument("--chk_name",     required=False, type=str,
                        help="the name of the general checkpoint file without extension [default: same as model_name]")
    parser.add_argument("--EMAX",         required=True, type=float,
                        help="maximum value of the energy range over which model should be evaluated")
    parser.add_argument("--energy_overview", required=False, type=str2bool, default=False,
                        help="whether to create an overview of errors in energies over train/val/test sets [False]")
    parser.add_argument("--forces_overview", required=False, type=str2bool, default=False,
                        help="whether to create an overview of errors in forces over trian/val/test sets [False]")
    parser.add_argument("--ch4_overview",  required=False, type=str2bool, default=False,
                        help="whether to create an overview over CH4 energies [False]")
    parser.add_argument("--add_reference_pes", required=False, type=str2bool, default=False,
                        help="whether to add errors of reference potential on plots [False]")
    parser.add_argument("--save", required=False, type=str2bool, default=False,
                        help="whether to save the produced PNG to default generated path [False]")
    parser.add_argument("--test_file" , required=False, type=str, default="",
                        help="the name of the file with configurations to evaluate model on")

    args = parser.parse_args()

    MODEL_FOLDER = os.path.join(BASEDIR, args.model_folder)
    MODEL_NAME   = args.model_name

    assert os.path.isdir(MODEL_FOLDER), "Path to folder is invalid: {}".format(MODEL_FOLDER)

    cfg_path = os.path.join(MODEL_FOLDER, MODEL_NAME + ".yaml")
    assert os.path.isfile(cfg_path), "YAML configuration file does not exist at {}".format(cfg_path)

    logging.info("Values of optional parameters:")
    logging.info("  chk_name:          {}".format(args.chk_name))
    logging.info("  energy_overview:   {}".format(args.energy_overview))
    logging.info("  forces_overview:   {}".format(args.forces_overview))
    logging.info("  ch4_overview:      {}".format(args.ch4_overview))
    logging.info("  add_reference:     {}".format(args.add_reference_pes))
    logging.info("  EMAX:              {}".format(args.EMAX))

    if args.chk_name is not None:
        chk_path = os.path.join(MODEL_FOLDER, args.chk_name + ".pt")
    else:
        chk_path = os.path.join(MODEL_FOLDER, MODEL_NAME + ".pt")
    assert os.path.isfile(chk_path), "File with model weights (.pt) does not exist at {}".format(chk_path)

    cfg = load_cfg(cfg_path)
    logging.info("loaded configuration file from {}".format(cfg_path))

    assert 'TYPE' in cfg
    assert cfg['TYPE'] in ('ENERGY', 'DIPOLE')

    cfg_dataset = cfg['DATASET']
    # in order to plot ALL the points of the RIGID dataset
    # instead of only the clipped part of it
    #if cfg_dataset['TYPE'] == 'NONRIGID-CLIP':
    #    cfg_dataset['TYPE'] = 'NONRIGID'

    evaluator = retrieve_checkpoint(cfg, chk_path)

    if args.energy_overview:
        if args.test_file:
            assert os.path.isfile(args.test_file), "File with test configurations does exist at {}".format(args.test_file)

            cfg_dataset.setdefault("ATOM_MAPPING", False)
            cfg_dataset.setdefault("PURIFY", False)

            dataset = PolyDataset(wdir=cfg_dataset['EXTERNAL_FOLDER'], typ='ENERGY', file_path=args.test_file, order=cfg_dataset['ORDER'], symmetry=cfg_dataset['SYMMETRY'],
                                  load_forces=False, atom_mapping=cfg_dataset['ATOM_MAPPING'], variables=cfg_dataset['VARIABLES'], purify=cfg_dataset['PURIFY'])

            model_evaluation_energy(evaluator, dataset, None, None, args.EMAX, args.add_reference_pes)
        else:
            train, val, test = load_dataset(cfg_dataset, cfg['TYPE'])

            model_evaluation_energy(evaluator, train, val, test, args.EMAX, args.add_reference_pes)

            ylim      = (-50.0, 50.0)
            ylocators = (10.0, 5.0)

            errors_png = None
            if args.save:
                errors_png = os.path.join(MODEL_FOLDER, MODEL_NAME + "-EMAX={}.png".format(args.EMAX))
                logging.info("errors_png: {}".format(errors_png))

            plot_errors_from_checkpoint(evaluator, train, val, test, args.EMAX, ylim=ylim, ylocators=ylocators,
                                        figpath=errors_png, add_reference_pes=args.add_reference_pes)

    if args.forces_overview:
        if args.test_file:
            assert os.path.isfile(args.test_file), "File with test configurations does exist at {}".format(args.test_file)

            cfg_dataset.setdefault("ATOM_MAPPING", False)
            cfg_dataset.setdefault("PURIFY", False)

            dataset = PolyDataset(wdir=cfg_dataset['EXTERNAL_FOLDER'], typ='ENERGY', file_path=args.test_file, order=cfg_dataset['ORDER'], symmetry=cfg_dataset['SYMMETRY'],
                                  load_forces=True, atom_mapping=cfg_dataset['ATOM_MAPPING'], variables=cfg_dataset['VARIABLES'], purify=cfg_dataset['PURIFY'])

            model_evaluation_forces(evaluator, dataset, None, None, args.EMAX)
        else:
            train, val, test = load_dataset(cfg_dataset, cfg['TYPE'])
            model_evaluation_forces(evaluator, train, val, test, args.EMAX)

    if args.ch4_overview:
        ylim      = (-20.0, 20.0)
        ylocators = (5.0, 1.0)

        labels = [r"CH$_4$: eq", r"CH$_4$: 0--1000 cm$^{-1}$", r"CH$_4$: 1000--2000 cm$^{-1}$", r"CH$_4$: 2000--3000 cm$^{-1}$"]

        overview_png = None
        if args.save:
            overview_png = os.path.join(MODEL_FOLDER, MODEL_NAME + "-ch4-overview.png")
            logging.info("overview_png: {}".format(overview_png))

        plot_errors_for_files(cfg_dataset, evaluator, args.EMAX, labels, ylim, ylocators, figpath=overview_png)
