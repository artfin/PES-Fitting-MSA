import os
import logging
import torch
import numpy as np
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from sklearn.model_selection import train_test_split
from genpip import cmdstat, cl

import pathlib
BASEDIR = pathlib.Path(__file__).parent.parent.resolve()

import sys
sys.path.insert(0, os.path.join(BASEDIR, "external"))
from pybind_example import Poten_CH4
from nitrogen_morse import Poten_N2

plt.style.use('science')

plt.rcParams.update({
    "font.family": "serif",
    "font.serif" : ["Times"],
    'figure.titlesize' : "Large",
    "axes.labelsize" : 21,
    "xtick.labelsize" : 18,
    "ytick.labelsize" : 18,
})

BOHRTOANG = 0.529177249

SCALE_OPTIONS = [None, "std"]

def load_dataset(folder, fname):
    fpath = os.path.join(folder, fname)
    logging.info("Loading dataset from fpath={}".format(fpath))

    d = torch.load(fpath)
    X, y = d["X"], d["y"]

    return X, y

def show_split_energy_distribution(y_train, y_val, y_test):
    plt.figure(figsize=(10, 10))
    plt.title("Energy distribution")
    plt.xlabel(r"Energy, cm$^{-1}$")
    plt.ylabel(r"Density")

    # density=True:
    # displays a probability density: each bin displays the bin's raw count divided by 
    # the total number of counts and the bin width, so that the area under the histogram
    # integrates to 1
    nbins = 500
    plt.hist(y_train.numpy() * HTOCM, bins=nbins, density=True, lw=3, fc=(0, 0, 1, 0.5), label='train')
    plt.hist(y_val.numpy()   * HTOCM, bins=nbins, density=True, lw=3, fc=(1, 0, 0, 0.5), label='val')
    plt.hist(y_test.numpy()  * HTOCM, bins=nbins, density=True, lw=3, fc=(0, 1, 0, 0.5), label='test')

    plt.yscale('log')
    plt.ylim((1e-5, 1e-2))
    plt.xlim((-300.0, 2000.0))

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



from dataclasses import dataclass
from typing import List
import itertools

@dataclass
class XYZConfig:
    atoms  : np.array
    energy : float

class XYZPlotter:
    def __init__(self, *fpaths):
        self.xyz_configs = list(itertools.chain.from_iterable(self.load_xyz(fpath) for fpath in fpaths))

    def make_histogram_R(self, figpath=None):
        NCONFIGS = len(self.xyz_configs)
        R = np.zeros((NCONFIGS, 1))

        for k, xyz_config in enumerate(self.xyz_configs):
            N2_center = 0.5 * (xyz_config.atoms[4, :] + xyz_config.atoms[5, :])
            R[k] = np.linalg.norm(N2_center - xyz_config.atoms[6, :])

        R = R * BOHRTOANG

        plt.figure(figsize=(10, 10))
        plt.hist(R, color='#88B04B', bins='auto')

        if figpath is not None:
            assert False

        plt.show()

    def make_histogram_N2_energy(self, figpath=None):
        NCONFIGS = len(self.xyz_configs)
        N2_energy = np.zeros((NCONFIGS, 1))

        for k, xyz_config in enumerate(self.xyz_configs):
            l_N2 = np.linalg.norm(xyz_config.atoms[4, :] - xyz_config.atoms[5, :]) * BOHRTOANG
            N2_energy[k] = Poten_N2(l_N2)

        plt.figure(figsize=(10, 10))
        ax = plt.subplot(1, 1, 1)

        plt.hist(N2_energy, color='#88B04B', bins='auto')

        plt.xlim((0.0, 700.0))

        if figpath is not None:
            assert False

        plt.show()

    def make_histogram_CH4_energy(self, figpath=None):
        pes = Poten_CH4(libpath=os.path.join(BASEDIR, "external", "obj", "xy4.so"))

        NCONFIGS = len(self.xyz_configs)
        ch4_energy = np.zeros((NCONFIGS, 1))

        for k, xyz_config in enumerate(self.xyz_configs):
            CH4_config = np.array([
                *xyz_config.atoms[6, :] * BOHRTOANG, # C
                *xyz_config.atoms[0, :] * BOHRTOANG, # H1
                *xyz_config.atoms[1, :] * BOHRTOANG, # H2
                *xyz_config.atoms[2, :] * BOHRTOANG, # H3
                *xyz_config.atoms[3, :] * BOHRTOANG, # H4
            ])

            ch4_energy[k] = pes.eval(CH4_config)

        plt.figure(figsize=(10, 10))
        ax = plt.subplot(1, 1, 1)

        plt.hist(ch4_energy, color='#88B04B', bins='auto')

        if figpath is not None:
            assert False

        plt.show()

    def make_histogram_NN_distance(self, figpath=None):
        NCONFIGS = len(self.xyz_configs)
        NN_dist = np.zeros((NCONFIGS, 1))

        for k, xyz_config in enumerate(self.xyz_configs):
            NN_dist[k] = np.linalg.norm(xyz_config.atoms[4, :] - xyz_config.atoms[5, :])

        NN_dist = NN_dist * BOHRTOANG

        plt.figure(figsize=(10, 10))
        ax = plt.subplot(1, 1, 1)

        NN_ref_dist = 1.09768 # A

        plt.hist(NN_dist, color='#88B04B', bins='auto')
        plt.axvline(NN_ref_dist, color='#FF6F61', linewidth=3)

        if figpath is not None:
            assert False

        plt.show()

    def make_histogram_CH_distance(self, figpath=None):
        NCONFIGS = len(self.xyz_configs)
        CH_dist = np.zeros((NCONFIGS * 4, 1))

        k = 0
        for xyz_config in self.xyz_configs:
            r1 = np.linalg.norm(xyz_config.atoms[0, :] - xyz_config.atoms[6, :])
            r2 = np.linalg.norm(xyz_config.atoms[1, :] - xyz_config.atoms[6, :])
            r3 = np.linalg.norm(xyz_config.atoms[2, :] - xyz_config.atoms[6, :])
            r4 = np.linalg.norm(xyz_config.atoms[3, :] - xyz_config.atoms[6, :])

            CH_dist[k]     = r1
            CH_dist[k + 1] = r2
            CH_dist[k + 2] = r3
            CH_dist[k + 3] = r4
            k = k + 4

        CH_dist = CH_dist * BOHRTOANG

        CH_ref_dist = 1.08601

        plt.figure(figsize=(10, 10))
        ax = plt.subplot(1, 1, 1)

        plt.hist(CH_dist, color='#88B04B', bins='auto')
        plt.axvline(CH_ref_dist, color='#FF6F61', linewidth=3)

        plt.title("May 03; NCONFIGS={}".format(NCONFIGS))
        plt.xlabel(r'C--H distance, \AA')

        plt.xlim((1.04, 1.14))

        ax.xaxis.set_major_locator(plt.MultipleLocator(0.03))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.01))
        ax.yaxis.set_major_locator(plt.MultipleLocator(200.0))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(100.0))

        ax.tick_params(axis='x', which='major', width=1.0, length=6.0)
        ax.tick_params(axis='x', which='minor', width=0.5, length=3.0)
        ax.tick_params(axis='y', which='major', width=1.0, length=6.0)
        ax.tick_params(axis='y', which='minor', width=0.5, length=3.0)

        if figpath is not None:
            logging.info("Saving figure to figpath={}".format(figpath))
            plt.savefig(figpath, format="png", dpi=300)
            self.trim_png(figpath)

        plt.show()

    def make_histogram_HCH_angle(self, figpath=None):
        NCONFIGS = len(self.xyz_configs)
        HCH_angles = np.zeros((NCONFIGS * 6, 1))

        k = 0
        for xyz_config in self.xyz_configs:
            H1 = xyz_config.atoms[0, :] / np.linalg.norm(xyz_config.atoms[0, :] - xyz_config.atoms[6, :])
            H2 = xyz_config.atoms[1, :] / np.linalg.norm(xyz_config.atoms[1, :] - xyz_config.atoms[6, :])
            H3 = xyz_config.atoms[2, :] / np.linalg.norm(xyz_config.atoms[2, :] - xyz_config.atoms[6, :])
            H4 = xyz_config.atoms[3, :] / np.linalg.norm(xyz_config.atoms[3, :] - xyz_config.atoms[6, :])

            HCH_angles[k]     = np.arccos(np.dot(H1, H2))
            HCH_angles[k + 1] = np.arccos(np.dot(H1, H3))
            HCH_angles[k + 2] = np.arccos(np.dot(H1, H4))
            HCH_angles[k + 3] = np.arccos(np.dot(H2, H3))
            HCH_angles[k + 4] = np.arccos(np.dot(H2, H4))
            HCH_angles[k + 5] = np.arccos(np.dot(H3, H4))
            k = k + 6

        HCH_angles = HCH_angles * 180.0 / np.pi

        HCH_ref_angle = np.arccos(-1.0/3.0) / np.pi * 180.0

        plt.figure(figsize=(10, 10))
        ax = plt.subplot(1, 1, 1)

        plt.hist(HCH_angles, color='#88B04B', bins='auto')
        plt.axvline(HCH_ref_angle, color='#FF6F61', linewidth=3)

        plt.title("May 03; NCONFIGS={}".format(NCONFIGS))
        plt.xlabel(r"$\angle$ HCH angle")

        ax.xaxis.set_major_locator(plt.MultipleLocator(10.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(5.0))
        ax.yaxis.set_major_locator(plt.MultipleLocator(200.0))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(100.0))

        ax.tick_params(axis='x', which='major', width=1.0, length=6.0)
        ax.tick_params(axis='x', which='minor', width=0.5, length=3.0)
        ax.tick_params(axis='y', which='major', width=1.0, length=6.0)
        ax.tick_params(axis='y', which='minor', width=0.5, length=3.0)

        if figpath is not None:
            logging.info("Saving figure to figpath={}".format(figpath))
            plt.savefig(figpath, format="png", dpi=300)

        plt.show()

    def make_histogram_energy(self, figpath=None):
        NCONFIGS = len(self.xyz_configs)

        energy = np.asarray([xyz_config.energy for xyz_config in self.xyz_configs])

        plt.figure(figsize=(10, 10))
        ax = plt.subplot(1, 1, 1)

        #bins = [-200.0, -150.0, -100.0, -50.0, 0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0]
        #r = ([-200.0, -150.0], [-150.0, -100.0])

        bins = list(100.0 * x for x in range(-3, 21, 1)) + [10000.0]
        hist, bind_edges = np.histogram(energy, bins)

        plt.bar(range(len(hist)), hist, width=0.8, color='#88B04B')

        ax.set_xticks([0.5 + i for i, _ in enumerate(hist)])
        ax.set_xticklabels(['{}'.format(int(bins[i+1])) for i, _ in enumerate(hist[:-1])] + [''])
        ax.tick_params(axis='x', which='major', labelsize=15, rotation=45)

        plt.xlabel(r"Energy, cm$^{-1}$")

        ax.tick_params(axis='x', which='major', width=1.0, length=6.0)
        ax.tick_params(axis='x', which='minor', width=0.5, length=3.0)
        ax.tick_params(axis='y', which='major', width=1.0, length=6.0)
        ax.tick_params(axis='y', which='minor', width=0.5, length=3.0)

        if figpath is not None:
            plt.savefig(figpath, format='png', dpi=300)

        plt.show()

    def load_xyz(self, fpath):
        nlines = sum(1 for line in open(fpath, mode='r'))
        NATOMS = int(open(fpath, mode='r').readline())
        if hasattr(self, 'NATOMS'):
            assert self.NATOMS == NATOMS
            logging.info("NATOMS is consistent.")
        else:
            self.NATOMS = NATOMS
            logging.info("Setting NATOMS attribute.")

        logging.info("detected NATOMS = {}".format(self.NATOMS))

        NCONFIGS = nlines // (self.NATOMS + 2)
        logging.info("detected NCONFIGS = {}".format(NCONFIGS))

        xyz_configs = []
        with open(fpath, mode='r') as inp:
            for i in range(NCONFIGS):
                line = inp.readline()
                energy = float(inp.readline())

                atoms = np.zeros((NCONFIGS, 3))
                for natom in range(self.NATOMS):
                    words = inp.readline().split()
                    atoms[natom, :] = np.fromiter(map(float, words[1:]), dtype=np.float64)

                c = XYZConfig(atoms=atoms, energy=energy)
                xyz_configs.append(c)

        return xyz_configs

    def trim_png(self, figname):
        cl('convert {0} -trim +repage {0}'.format(figname))



if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    xyz_paths = [
        os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=0-1000-N2=0-1000.xyz"),
        os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=1000-2000-N2=0-1000.xyz"),
    ]

    plotter = XYZPlotter(*xyz_paths)
    #plotter.make_histogram_N2_energy()
    plotter.make_histogram_CH4_energy()
    #plotter.make_histogram_energy()

    #plotter.make_histogram_R()
    #plotter.make_histogram_NN_distance()
    #plotter.make_histogram_CH_distance()
    #plotter.make_histogram_CH_distance(figpath=os.path.join(BASEDIR, "datasets", "raw", "C-H-histogram.png"))

    #plotter.make_histogram_HCH_angle(figpath=os.path.join(BASEDIR, "datasets", "raw", "HCH-histogram.png"))
    #plotter.make_energy_distribution()

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    #logging.info("X_train.size(): {}".format(X_train.size()))
    #logging.info("X_val.size():   {}".format(X_val.size()))
    #logging.info("X_test.size():  {}".format(X_test.size()))

    #X_train, X_test, y_train, y_test = chi_split(X, y, test_size=0.2, nbins=20)
    #X_val, X_test, y_val, y_test     = chi_split(X_test, y_test, test_size=0.5, nbins=20)
    #show_split_energy_distribution(y_train, y_val, y_test)
