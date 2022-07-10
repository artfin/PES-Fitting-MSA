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
sys.path.insert(0, os.path.join(BASEDIR, "external", "pes"))
from pybind_ch4 import Poten_CH4
from nitrogen_morse import Poten_N2

plt.style.use('science')

plt.rcParams.update({
    "font.family": "serif",
    "font.serif" : ["Times"],
    'figure.titlesize' : "Large",
    "axes.labelsize" : 24,
    "xtick.labelsize" : 21,
    "ytick.labelsize" : 21,
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

from dataset import XYZConfig

class XYZPlotter:
    def __init__(self, *fpaths):
        self.xyz_configs = []
        for fpath in fpaths:
            if fpath.endswith(".xyz"):
                assert False
                from dataset import load_xyz
                self.xyz_configs.extend(load_xyz(fpath))
            elif fpath.endswith(".npz"):
                from dataset import load_npz
                natoms, nconfigs, xyz_configs = load_npz(fpath)
                self.xyz_configs.extend(xyz_configs)
            else:
                raise ValueError("unknown format")

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

        plt.xlim((0.0, 1000.0))

        plt.xlabel(r'E(N$_2$), cm$^{-1}$')
        plt.ylabel(r'\# configurations')

        ax.xaxis.set_major_locator(plt.MultipleLocator(200.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(100.0))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1000.0))
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

    def make_histogram_CH4_energy(self, figpath=None):
        pes = Poten_CH4(libpath=os.path.join(BASEDIR, "external", "pes", "xy4.so"))

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

        plt.xlim((0.0, 3000.0))

        plt.xlabel(r'E(CH$_4$), cm$^{-1}$')
        plt.ylabel(r'\# configurations')

        ax.xaxis.set_major_locator(plt.MultipleLocator(500.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(100.0))
        ax.yaxis.set_major_locator(plt.MultipleLocator(500.0))
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

        plt.xlabel(r'N--N distance, \AA')
        plt.ylabel(r'\# configurations')

        plt.xlim((1.05, 1.15))

        ax.xaxis.set_major_locator(plt.MultipleLocator(0.01))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.005))
        ax.yaxis.set_major_locator(plt.MultipleLocator(500.0))
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

        plt.xlabel(r'C--H distance, \AA')
        plt.ylabel(r'\# configurations')

        plt.xlim((1.04, 1.14))

        ax.xaxis.set_major_locator(plt.MultipleLocator(0.03))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.01))
        ax.yaxis.set_major_locator(plt.MultipleLocator(500.0))
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

        plt.xlabel(r"$\angle$ HCH")
        plt.ylabel(r'\# configurations')

        plt.xlim((85.0, 135.0))

        ax.xaxis.set_major_locator(plt.MultipleLocator(10.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(5.0))
        ax.yaxis.set_major_locator(plt.MultipleLocator(500.0))
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

    def make_histogram_asymptotic_energy(self, figpath=None):
        NCONFIGS = len(self.xyz_configs)
        energy = np.asarray([xyz_config.energy for xyz_config in self.xyz_configs])

        plt.figure(figsize=(10, 10))
        ax = plt.subplot(1, 1, 1)

        bins = list(x/2.0 for x in range(10, 20))
        hist, bind_edges = np.histogram(energy, bins)

        plt.bar(range(len(hist)), hist, width=0.8, color='#88B04B')

        ax.set_xticks([0.5 + i for i, _ in enumerate(hist)])
        ax.set_xticklabels(['{}'.format(bins[i+1]) for i, _ in enumerate(hist)])
        ax.tick_params(axis='x', which='major', labelsize=15)

        ax.tick_params(axis='x', which='major', width=1.0, length=6.0)
        ax.tick_params(axis='x', which='minor', width=0.5, length=3.0)
        ax.tick_params(axis='y', which='major', width=1.0, length=6.0)
        ax.tick_params(axis='y', which='minor', width=0.5, length=3.0)

        plt.xlabel(r"Asymptotic energy, cm$^{-1}$")
        plt.ylabel(r'\# configurations')

        if figpath is not None:
            logging.info("Saving figure to figpath={}".format(figpath))
            plt.savefig(figpath, format="png", dpi=300)
            self.trim_png(figpath)

        plt.show()


    def make_histogram_energy(self, figpath=None):
        NCONFIGS = len(self.xyz_configs)
        energy = np.asarray([xyz_config.energy for xyz_config in self.xyz_configs])

        plt.figure(figsize=(10, 10))
        ax = plt.subplot(1, 1, 1)


        #bins = list(500.0 * x for x in range(0, 31, 1)) + [20000.0]
        bins = list(500.0 * x for x in range(0, 21, 1))
        hist, bind_edges = np.histogram(energy, bins)

        plt.bar(range(len(hist)), hist, width=0.8, color='#88B04B')

        ax.set_xticks([0.5 + i for i, _ in enumerate(hist)])
        #ax.set_xticklabels(['{}'.format(int(bins[i+1])) for i, _ in enumerate(hist[:-1])] + [r"$>$ 2000.0"])
        ax.set_xticklabels(['{}'.format(int(bins[i+1])) for i, _ in enumerate(hist)])
        ax.tick_params(axis='x', which='major', labelsize=15, rotation=45)

        plt.xlabel(r"Intermolecular energy, cm$^{-1}$")
        plt.ylabel(r'\# configurations')

        plt.yscale('log')

        ax.tick_params(axis='x', which='major', width=1.0, length=6.0)
        ax.tick_params(axis='x', which='minor', width=0.5, length=3.0)
        ax.tick_params(axis='y', which='major', width=1.0, length=6.0)
        ax.tick_params(axis='y', which='minor', width=0.5, length=3.0)

        if figpath is not None:
            logging.info("Saving figure to figpath={}".format(figpath))
            plt.savefig(figpath, format="png", dpi=300)
            self.trim_png(figpath)

        plt.show()

    def trim_png(self, figname):
        cl('convert {0} -trim +repage {0}'.format(figname))


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    #fpaths = [
    #    os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=0-1000-N2=0-1000-LIMITS.xyz"),
    #    os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=1000-2000-N2=0-1000-LIMITS.xyz"),
    #    os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=2000-3000-N2=0-1000-LIMITS.xyz"),
    #]

    fpaths = [
        os.path.join(BASEDIR, "datasets/raw/ethanol_ccsd_t/ethanol_ccsd_t-train.npz"),
    ]

    plotter = XYZPlotter(*fpaths)
    plotter.make_histogram_energy()

    #plotter.make_histogram_asymptotic_energy(figpath=os.path.join(BASEDIR, "datasets", "raw", "asymptotic-energy.png"))

    #plotter.make_histogram_CH4_energy(figpath=os.path.join(BASEDIR, "datasets", "raw", "CH4-energy.png"))
    #plotter.make_histogram_N2_energy(figpath=os.path.join(BASEDIR, "datasets", "raw", "N2-energy.png"))
    #plotter.make_histogram_energy(figpath=os.path.join(BASEDIR, "datasets", "raw", "intermolecular-energy.png"))

    #plotter.make_histogram_CH_distance(figpath=os.path.join(BASEDIR, "datasets", "raw", "C-H-histogram.png"))
    #plotter.make_histogram_HCH_angle(figpath=os.path.join(BASEDIR, "datasets", "raw", "HCH-histogram.png"))
    #plotter.make_histogram_NN_distance(figpath=os.path.join(BASEDIR, "datasets", "raw", "NN-distance.png"))

    #plotter.make_histogram_R()

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    #logging.info("X_train.size(): {}".format(X_train.size()))
    #logging.info("X_val.size():   {}".format(X_val.size()))
    #logging.info("X_test.size():  {}".format(X_test.size()))

    #X_train, X_test, y_train, y_test = chi_split(X, y, test_size=0.2, nbins=20)
    #X_val, X_test, y_val, y_test     = chi_split(X_test, y_test, test_size=0.5, nbins=20)
    #show_split_energy_distribution(y_train, y_val, y_test)
