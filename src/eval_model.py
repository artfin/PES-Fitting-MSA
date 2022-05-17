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

    #plt.scatter(calc, published_abs_error, s=20, marker='o', facecolors='none', color='#CFBFF7', lw=0.5, label='Symmetry-adapted angular basis')
    plt.scatter(train.y, error_train, s=20, marker='o', facecolors='none', color='#FF6F61', lw=1.0, label='train')
    plt.scatter(val.y,   error_val,  s=20, marker='o', facecolors='none', color='#6CD4FF', lw=1.0, label='val')
    plt.scatter(test.y,  error_test, s=20, marker='o', facecolors='none', color='#88B04B', lw=1.0, label='test')
    #plt.scatter(train.y, error_train, s=20, marker='o', facecolors='none', color='#FF6F61', lw=0.5, label='train')
    #plt.scatter(val.y,   error_val,  s=20, marker='o', facecolors='none', color='#FF6F61', lw=0.5, label='val')
    #plt.scatter(test.y,  error_test, s=20, marker='o', facecolors='none', color='#FF6F61', lw=0.5, label='test')

    plt.xlim((-200.0, 2000.0))
    plt.ylim((-5.0, 5.0))

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

    lgnd = plt.legend(fontsize=18)
    lgnd.legendHandles[0].set_lw(1.5)
    lgnd.legendHandles[1].set_lw(1.5)
    lgnd.legendHandles[2].set_lw(1.5)

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

from collections import namedtuple
XYZConfig = namedtuple('Config', ['atoms', 'energy'])

class EvalFile:
    a0     = 2.0 # bohrs
    NATOMS = 7
    NDIS   = 21

    def __init__(self, evaluator, fpath):
        self.evaluator = evaluator
        self.fpath = fpath

        self.xyz_configs = self.load_xyz(fpath)
        self.setup_fortran_procs()

        self.set_intermolecular_to_zero = False
        logging.info("preparing atomic distances..")
        self.yij = self.make_yij(self.xyz_configs)
        logging.info("Done.")

    def make_yij(self, xyz_configs):
        if self.set_intermolecular_to_zero:
            logging.info("setting intermolecular Morse variables to zero")

        NCONFIGS = len(xyz_configs)

        yij = np.zeros((NCONFIGS, self.NDIS), order="F")

        for n in range(NCONFIGS):
            c = xyz_configs[n]

            k = 0
            for i, j in combinations(range(self.NATOMS), 2):
                # CH4-N2
                if self.set_intermolecular_to_zero:
                    if i == 0 and j == 1: yij[n, k] = 0.0; k = k + 1; continue; # H1 H2 
                    if i == 0 and j == 2: yij[n, k] = 0.0; k = k + 1; continue; # H1 H3
                    if i == 0 and j == 3: yij[n, k] = 0.0; k = k + 1; continue; # H1 H4
                    if i == 1 and j == 2: yij[n, k] = 0.0; k = k + 1; continue; # H2 H3
                    if i == 1 and j == 3: yij[n, k] = 0.0; k = k + 1; continue; # H2 H4
                    if i == 2 and j == 3: yij[n, k] = 0.0; k = k + 1; continue; # H3 H4
                    if i == 0 and j == 6: yij[n, k] = 0.0; k = k + 1; continue; # H1 C
                    if i == 1 and j == 6: yij[n, k] = 0.0; k = k + 1; continue; # H2 C
                    if i == 2 and j == 6: yij[n, k] = 0.0; k = k + 1; continue; # H3 C
                    if i == 3 and j == 6: yij[n, k] = 0.0; k = k + 1; continue; # H4 C
                    if i == 4 and j == 5: yij[n, k] = 0.0; k = k + 1; continue; # N1 N2

                yij[n, k] = np.linalg.norm(c.atoms[i] - c.atoms[j])
                yij[n][k] = np.exp(-yij[n, k] / self.a0)
                k = k + 1

        return yij

    def make_histogram_CH4_energy_vs_error(self):
        pes = Poten_CH4(libpath=os.path.join(BASEDIR, "external", "pes", "xy4.so"))

        NMON = 2892
        NPOLY = 650
        logging.info("USING NMON={}".format(NMON))
        logging.info("USING NPOLY={}".format(NPOLY))

        x = np.zeros((self.NDIS, 1))
        m = np.zeros((NMON, 1))
        p = np.zeros((NPOLY, 1))

        NCONFIGS = len(self.xyz_configs)

        error           = np.zeros((NCONFIGS, 1))
        intermol_energy = np.zeros((NCONFIGS, 1))
        ch4_energy      = np.zeros((NCONFIGS, 1))

        for n in range(0, NCONFIGS):
            x = self.yij[n, :].copy()
            self.evmono(x, m)
            self.evpoly(m, p)

            pred               = evaluator(p.reshape((1, NPOLY)))[0][0]
            intermol_energy[n] = self.xyz_configs[n].energy
            error[n]           = pred - intermol_energy[n]

            xyz_config = self.xyz_configs[n]
            CH4_config = np.array([
                *xyz_config.atoms[6, :] * BOHRTOANG, # C
                *xyz_config.atoms[0, :] * BOHRTOANG, # H1
                *xyz_config.atoms[1, :] * BOHRTOANG, # H2
                *xyz_config.atoms[2, :] * BOHRTOANG, # H3
                *xyz_config.atoms[3, :] * BOHRTOANG, # H4
            ])

            ch4_energy[n] = pes.eval(CH4_config)
            print(n)

        plt.figure(figsize=(10, 10))
        ax = plt.subplot(1, 1, 1)

        plt.scatter(intermol_energy, error, color='#88B04B', facecolor='none', lw=1.5)

        plt.xlim((-200.0, 2000.0))
        plt.ylim((-15.0, 15.0))

        plt.title(r"CH$_4$: 1000 -- 2000 cm$^{-1}$")
        plt.xlabel(r"Intermolecular energy, cm$^{-1}$")
        plt.ylabel(r"$\Delta$ E, cm$^{-1}$")

        ax.xaxis.set_major_locator(plt.MultipleLocator(500.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(100.0))
        ax.yaxis.set_major_locator(plt.MultipleLocator(5.0))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(1.0))

        ax.tick_params(axis='x', which='major', width=1.0, length=6.0)
        ax.tick_params(axis='x', which='minor', width=0.5, length=3.0)
        ax.tick_params(axis='y', which='major', width=1.0, length=6.0)
        ax.tick_params(axis='y', which='minor', width=0.5, length=3.0)

        plt.show()


    def setup_fortran_procs(self):
        import ctypes as ct
        F_LIBNAME = os.path.join(BASEDIR, "datasets", "external", "f_basis_4_2_1_4.so")
        logging.info("Loading and setting up Fortran procedures from LIBNAME: {}".format(F_LIBNAME))
        basislib = ct.CDLL(F_LIBNAME)

        self.evmono = basislib.c_evmono
        self.evpoly = basislib.c_evpoly

        from numpy.ctypeslib import ndpointer
        self.evmono.argtypes = [ndpointer(ct.c_double, flags="F_CONTIGUOUS"), ndpointer(ct.c_double, flags="F_CONTIGUOUS")]
        self.evpoly.argtypes = [ndpointer(ct.c_double, flags="F_CONTIGUOUS"), ndpointer(ct.c_double, flags="F_CONTIGUOUS")]

    def load_xyz(self, fpath):
        nlines = sum(1 for line in open(fpath, mode='r'))
        NATOMS = int(open(fpath, mode='r').readline())
        logging.info("detected NATOMS = {}".format(NATOMS))

        NCONFIGS = nlines // (NATOMS + 2)
        logging.info("detected NCONFIGS = {}".format(NCONFIGS))

        xyz_configs = []
        with open(fpath, mode='r') as inp:
            for i in range(NCONFIGS):
                line = inp.readline()
                energy = float(inp.readline())

                atoms = np.zeros((NCONFIGS, 3))
                for natom in range(NATOMS):
                    words = inp.readline().split()
                    atoms[natom, :] = list(map(float, words[1:]))

                c = XYZConfig(atoms=atoms, energy=energy)
                xyz_configs.append(c)

        return xyz_configs



if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    #MODEL_FOLDER = os.path.join(BASEDIR, "models", "rigid", "exp11")
    #MODEL_FOLDER = os.path.join(BASEDIR, "models", "rigid", "L1", "L1-2")

    #MODEL_FOLDER = os.path.join(BASEDIR, "models", "nonrigid", "L2", "L2-3")
    #MODEL_FOLDER = os.path.join(BASEDIR, "models", "nonrigid", "L1", "L1-2")
    #MODEL_FOLDER = os.path.join(BASEDIR, "models", "nonrigid", "L2", "L2-2-silu")
    #MODEL_FOLDER = os.path.join(BASEDIR, "models", "nonrigid", "L2", "L2-6-no-reg")

    #MODEL_FOLDER = os.path.join(BASEDIR, "models", "nonrigid", "L2", "L2-1-no-reg")
    #MODEL_FOLDER = os.path.join(BASEDIR, "models", "nonrigid", "L2", "L2-1-L1-lambda=1e-10")
    #MODEL_FOLDER = os.path.join(BASEDIR, "models", "nonrigid", "L2", "L2-1-L1-lambda=1e-9")
    #MODEL_FOLDER = os.path.join(BASEDIR, "models", "nonrigid", "L2", "L2-1-L1-lambda=1e-8")
    #MODEL_FOLDER = os.path.join(BASEDIR, "models", "nonrigid", "L2", "L2-1-L1-lambda=1e-7")
    #MODEL_FOLDER = os.path.join(BASEDIR, "models", "nonrigid", "L2", "L2-1-L1-lambda=1e-6")
    #MODEL_FOLDER = os.path.join(BASEDIR, "models", "nonrigid", "L2", "L2-1-L1-lambda=1e-5")
    #MODEL_FOLDER = os.path.join(BASEDIR, "models", "nonrigid", "L2", "L2-1-L1-lambda=1e-4")

    #MODEL_FOLDER = os.path.join(BASEDIR, "models", "nonrigid", "L2", "L2-7-L1-lambda=1e-8")
    #MODEL_FOLDER = os.path.join(BASEDIR, "models", "nonrigid", "L2", "L2-11")
    MODEL_FOLDER = os.path.join(BASEDIR, "models", "nonrigid", "L2", "L2-12")

    cfg_path = os.path.join(MODEL_FOLDER, "config.yaml")
    with open(cfg_path, mode="r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.info(exc)

    logging.info("loaded configuration file from {}".format(cfg_path))

    evaluator = retrieve_checkpoint(cfg, chk_fname="checkpoint.pt")

    #ef = EvalFile(evaluator, fpath=os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=0-1000-N2=0-1000.xyz"))
    ef = EvalFile(evaluator, fpath=os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=1000-2000-N2=0-1000.xyz"))
    #ef = EvalFile(evaluator, fpath=os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=2000-3000-N2=0-1000.xyz"))
    ef.make_histogram_CH4_energy_vs_error()

    assert False

    cfg_dataset = cfg['DATASET']
    train, val, test = load_dataset(cfg_dataset)

    pred_train = torch.from_numpy(evaluator(train.X))

    ind = (train.y < 2000.0).nonzero()[:,0]
    yf = train.y[ind]
    predf = pred_train[ind]
    diff = torch.abs(yf - predf)

    mean_diff = torch.mean(diff)
    max_diff = torch.max(diff)
    logging.info("[< 2000 cm-1] mean diff: {}".format(mean_diff))
    logging.info("[< 2000 cm-1] max diff: {}".format(max_diff))

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
