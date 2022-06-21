import ctypes as ct
import logging
import json
import numpy as np
from numpy.ctypeslib import ndpointer
import os
import re

from itertools import combinations

from genpip import cl

import torch
from torch.utils.data import Dataset

from dataclasses import dataclass
from typing import List, Optional

# parameter inside the `yij` exp(...)
a0 = 2.0 # bohrs
POLYNOMIAL_LIB = "MSA"

@dataclass
class XYZConfig:
    atoms  : np.array
    energy : float

    def check(self):
        C  = self.atoms[6, :]
        H1 = self.atoms[0, :]
        H2 = self.atoms[1, :]
        H3 = self.atoms[2, :]
        H4 = self.atoms[3, :]

        r1 = np.linalg.norm(C - H1)
        r2 = np.linalg.norm(C - H2)
        r3 = np.linalg.norm(C - H3)
        r4 = np.linalg.norm(C - H4)
        rr = np.array([r1, r2, r3, r4])

        MIN_CH = 1.88 # BOHR
        MAX_CH = 2.5  # BOHR
        assert np.all(rr > MIN_CH) and np.all(rr < MAX_CH), "Check the distance units; Angstrom instead of Bohrs are suspected"

@dataclass
class PolyDataset_t:
    NATOMS       : int
    NMON         : int
    NPOLY        : int
    symmetry     : str
    order        : str
    X            : torch.Tensor
    y            : torch.Tensor
    mask         : Optional[List[int]] = None
    energy_limit : float = None
    intramz      : bool  = False
    purify       : bool  = False


def load_xyz(fpath):
    nlines = sum(1 for line in open(fpath, mode='r'))
    NATOMS = int(open(fpath, mode='r').readline())
    NCONFIGS = nlines // (NATOMS + 2)

    xyz_configs = []
    with open(fpath, mode='r') as inp:
        for i in range(NCONFIGS):
            line = inp.readline()
            energy = float(inp.readline())

            atoms = np.zeros((NATOMS, 3))
            for natom in range(NATOMS):
                words = inp.readline().split()
                atoms[natom, :] = list(map(float, words[1:]))

            c = XYZConfig(atoms=atoms, energy=energy)
            c.check()
            xyz_configs.append(c)

    return NATOMS, NCONFIGS, xyz_configs

def write_xyz(xyz_path, xyz_configs):
    natoms = xyz_configs[0].atoms.shape[0]

    with open(xyz_path, mode='w') as fd:
        for xyz_config in xyz_configs:
            fd.write("    {}\n".format(natoms))
            fd.write("    {:.10f}\n".format(xyz_config.energy))
            fd.write("H \t {:.10f} \t {:.10f} \t {:.10f}\n".format(xyz_config.atoms[0, 0], xyz_config.atoms[0, 1], xyz_config.atoms[0, 2]))
            fd.write("H \t {:.10f} \t {:.10f} \t {:.10f}\n".format(xyz_config.atoms[1, 0], xyz_config.atoms[1, 1], xyz_config.atoms[1, 2]))
            fd.write("H \t {:.10f} \t {:.10f} \t {:.10f}\n".format(xyz_config.atoms[2, 0], xyz_config.atoms[2, 1], xyz_config.atoms[2, 2]))
            fd.write("H \t {:.10f} \t {:.10f} \t {:.10f}\n".format(xyz_config.atoms[3, 0], xyz_config.atoms[3, 1], xyz_config.atoms[3, 2]))
            fd.write("N \t {:.10f} \t {:.10f} \t {:.10f}\n".format(xyz_config.atoms[4, 0], xyz_config.atoms[4, 1], xyz_config.atoms[4, 2]))
            fd.write("N \t {:.10f} \t {:.10f} \t {:.10f}\n".format(xyz_config.atoms[5, 0], xyz_config.atoms[5, 1], xyz_config.atoms[5, 2]))
            fd.write("C \t {:.10f} \t {:.10f} \t {:.10f}\n".format(xyz_config.atoms[6, 0], xyz_config.atoms[6, 1], xyz_config.atoms[6, 2]))

class PolyDataset(Dataset):
    def __init__(self, wdir, xyz_file, limit_file=None, order=None, symmetry=None, intramz=False, purify=False):
        self.wdir = wdir
        logging.info("working directory: {}".format(self.wdir))

        self.order    = order
        self.symmetry = symmetry
        self.intramz  = intramz
        self.purify   = purify

        logging.info("Loading configurations from xyz_file: {}".format(xyz_file))
        NATOMS, NCONFIGS, xyz_configs = load_xyz(xyz_file)

        self.NATOMS   = NATOMS
        self.NDIS     = self.NATOMS * (self.NATOMS - 1) // 2
        self.NCONFIGS = NCONFIGS

        logging.info("  NATOMS = {}".format(self.NATOMS))
        logging.info("  NCONFIGS = {}".format(NCONFIGS))

        if limit_file is not None:
            logging.info("Loading configurations from limit_file: {}".format(limit_file))
            NATOMS, NCONFIGS, limit_configs = load_xyz(limit_file)
            assert NATOMS   == self.NATOMS
            assert NCONFIGS == self.NCONFIGS

            for n in range(NCONFIGS):
                xyz_config   = xyz_configs[n]
                limit_config = limit_configs[n]

                np.testing.assert_almost_equal(xyz_config.atoms[0, :], limit_config.atoms[0, :])
                np.testing.assert_almost_equal(xyz_config.atoms[1, :], limit_config.atoms[1, :])
                np.testing.assert_almost_equal(xyz_config.atoms[2, :], limit_config.atoms[2, :])
                np.testing.assert_almost_equal(xyz_config.atoms[3, :], limit_config.atoms[3, :])
                np.testing.assert_almost_equal(xyz_config.atoms[6, :], limit_config.atoms[6, :])

                xyz_NN   = np.linalg.norm(xyz_config.atoms[4, :]   - xyz_config.atoms[5, :])
                limit_NN = np.linalg.norm(limit_config.atoms[4, :] - limit_config.atoms[5, :])
                np.testing.assert_almost_equal(xyz_NN, limit_NN)

                xyz_configs[n].energy -= limit_configs[n].energy

            logging.info("Subtracted asymptotic energies")

        self.prepare_poly_lib()
        if self.purify:
            if self.intramz:
                raise ValueError("Conflicting options: PURIFY and INTRAMZ")

            self.make_purify_mask(xyz_configs)

        X, y = self.prepare_dataset_from_configs(xyz_configs)

        if self.purify:
            self.NPOLY = self.purify_mask.sum()
            X = X[:, self.purify_mask.astype(np.bool)]

        if POLYNOMIAL_LIB == "MSA":
            self.mask = X.abs().sum(dim=0).bool().numpy().astype(int)
            logging.info("Applying non-zero mask. Selecting {} polynomials out of {} initially...".format(
                self.mask.sum(), len(self.mask)
            ))

            #nonzero_index = self.mask.nonzero()[0].tolist()
            #logging.info("indices of non-zero polynomials: {}".format(nonzero_index))
            #logging.info(json.dumps(nonzero_index))

            self.NPOLY = self.mask.sum()
            X = X[:, self.mask.astype(np.bool)]
            logging.info("Final size of the X-array: {}".format(X.size()))

        self.X, self.y = X, y


    def prepare_poly_lib(self):
        logging.info("Preparing dynamic library to compute invariant polynomials: ")
        logging.info("  [GLOBAL] POLYNOMIAL_LIB={}".format(POLYNOMIAL_LIB))

        if POLYNOMIAL_LIB == "MSA":
            stub       = '_{}_{}'.format(self.symmetry.replace(' ', '_'), self.order)
            MONO_fname = os.path.join(self.wdir, 'MOL' + stub + '.MONO')
            POLY_fname = os.path.join(self.wdir, 'MOL' + stub + '.POLY')
            self.NMON  = sum(1 for line in open(MONO_fname))
            self.NPOLY = sum(1 for line in open(POLY_fname))

            self.F_LIBNAME = os.path.join(self.wdir, 'f_basis' + stub + '.so')
            if not os.path.isfile(self.F_LIBNAME):
                from genpip import compile_dlib
                compile_dlib(self.order, self.symmetry, self.wdir)

            assert os.path.isfile(self.F_LIBNAME), "No F_LIBRARY={} found".format(self.F_LIBNAME)
            self.setup_fortran_procs()

        elif POLYNOMIAL_LIB == "MSA":
            stub       = '_{}_{}'.format(self.symmetry.replace(' ', '_'), self.order)
            self.C_LIBNAME = os.path.join(self.wdir, 'c_basis' + stub + '.so')
            assert os.path.isfile(self.C_LIBNAME), "No C_LIBRARY={} found".format(self.C_LIBNAME)
            self.setup_c_procs()

            C_LIBNAME_CODE = os.path.join(self.wdir, 'c_basis' + stub + '.cc')
            with open(C_LIBNAME_CODE, mode='r') as fp:
                lines = "".join(fp.readlines())

            pattern    = "double p\[(\d+)\]"
            found      = re.findall(pattern, lines)
            self.NPOLY = int(found[0])
            self.NMON  = None

        else:
            raise ValueError("unreachable")

        logging.info(" [NUMBER OF MONOMIALS]   NMON  = {}".format(self.NMON))
        logging.info(" [NUMBER OF POLYNOMIALS] NPOLY = {}".format(self.NPOLY))

    def eval_poly(self, x):
        p = np.zeros((self.NPOLY, 1))
        if POLYNOMIAL_LIB == "MSA":
            m = np.zeros((self.NMON, 1))

            self.evmono(x, m)
            self.evpoly(m, p)

            assert ~np.isnan(np.sum(m)), "There are NaN values in monomials produced by Fortran_evmono"
            assert ~np.isnan(np.sum(p)), "There are NaN values in polynomials produced by Fortran_evpoly"
            assert np.max(p) < 1e10, "There are suspicious values of polynomials produced by Fortran_evpoly"

        elif POLYNOMIAL_LIB == "CUSTOM":
            self.evpoly(x, p)

            assert ~np.isnan(np.sum(p)), "There are NaN values in polynomials produced by C_evpoly"
            assert np.max(p) < 1e10, "There are suspicious values of polynomials produced by C_evpoly"
        else:
            raise ValueError("unreachable")

        return p

    def make_purify_mask(self, xyz_configs):
        # Calculate interatomic distances for each configuration 
        # and zero out intermolecular coordinates 
        logging.info("Preparing interatomic distances.")
        yij = self.make_yij(xyz_configs, intermz=True)
        logging.info("Done.")

        NCONFIGS = len(xyz_configs)
        poly = np.zeros((NCONFIGS, self.NPOLY))
        x = np.zeros((self.NDIS, 1))

        logging.info("Computing polynomials...")
        for n in range(0, NCONFIGS):
            x = yij[n, :].copy()
            p = self.eval_poly(x)
            poly[n, :] = p.reshape((self.NPOLY, )).copy()

        logging.info("Done.")

        X = torch.from_numpy(poly)
        self.purify_mask = 1 - X.abs().sum(dim=0).bool().numpy().astype(int)
        logging.info("Selecting {} purified polynomials out of {} initially...".format(
            self.purify_mask.sum(), len(self.purify_mask)
        ))

        #purify_index = self.purify_mask.nonzero()[0].tolist()
        #logging.info("indices of purified polynomials: {}".format(purify_index))
        #logging.info(json.dumps(purify_index))


    def prepare_dataset_from_configs(self, xyz_configs):
        logging.info("preparing interatomic distances..")
        yij = self.make_yij(xyz_configs, intramz=self.intramz, intermz=False)
        logging.info("Done.")

        NCONFIGS = len(xyz_configs)
        poly = np.zeros((NCONFIGS, self.NPOLY))
        x = np.zeros((self.NDIS, 1))

        logging.info("Computing polynomials...")
        for n in range(0, NCONFIGS):
            x = yij[n, :].copy()
            p = self.eval_poly(x)
            poly[n, :] = p.reshape((self.NPOLY, )).copy()

        logging.info("Done.")

        energies = np.zeros((NCONFIGS, 1))
        for ind, xyz_config in enumerate(xyz_configs):
            energies[ind] = xyz_config.energy

        # NOTE: LR-model was added here
        #    if lr_model is not None:
        #        energies[ind] -= lr_model(ind)

        X = torch.from_numpy(poly)
        y = torch.from_numpy(energies)

        return X, y


    @classmethod
    def from_pickle(cls, path):
        dict = torch.load(path)
        return cls.from_dict(dict)

    @classmethod
    def from_dict(cls, dict):
        return PolyDataset_t(**dict)

    def __len__(self):
        return len(self.y)

    def make_yij(self, xyz_configs, intramz=False, intermz=False):
        logging.info("Constructing an array of interatomic distances with options:")
        logging.info(" [INTRAMOLECULAR COORDINATES=ZERO] INTRAMZ={}".format(intramz))
        logging.info(" [INTERMOLECULAR COORDINATES=ZERO] INTERMZ={}".format(intermz))

        NCONFIGS = len(xyz_configs)
        yij = np.zeros((NCONFIGS, self.NDIS), order="F")

        for n in range(NCONFIGS):
            c = xyz_configs[n]

            k = 0
            for i, j in combinations(range(self.NATOMS), 2):
                # CH4-N2
                if intramz:
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

                if intermz:
                    if i == 0 and j == 4: yij[n, k] = 0.0; k = k + 1; continue; # H1 N1
                    if i == 0 and j == 5: yij[n, k] = 0.0; k = k + 1; continue; # H1 N2
                    if i == 1 and j == 4: yij[n, k] = 0.0; k = k + 1; continue; # H2 N1
                    if i == 1 and j == 5: yij[n, k] = 0.0; k = k + 1; continue; # H2 N2
                    if i == 2 and j == 4: yij[n, k] = 0.0; k = k + 1; continue; # H3 N1
                    if i == 2 and j == 5: yij[n, k] = 0.0; k = k + 1; continue; # H3 N2
                    if i == 3 and j == 4: yij[n, k] = 0.0; k = k + 1; continue; # H4 N1
                    if i == 3 and j == 5: yij[n, k] = 0.0; k = k + 1; continue; # H4 N2
                    if i == 4 and j == 6: yij[n, k] = 0.0; k = k + 1; continue; # N1 C
                    if i == 5 and j == 6: yij[n, k] = 0.0; k = k + 1; continue; # N2 C

                yij[n, k] = np.linalg.norm(c.atoms[i] - c.atoms[j])
                yij[n][k] = np.exp(-yij[n, k] / a0)
                k = k + 1

        return yij

    def setup_fortran_procs(self):
        logging.info("Loading and setting up Fortran procedures from LIBNAME: {}".format(self.F_LIBNAME))
        basislib = ct.CDLL(self.F_LIBNAME)

        self.evmono = basislib.c_evmono
        self.evpoly = basislib.c_evpoly

        self.evmono.argtypes = [ndpointer(ct.c_double, flags="F_CONTIGUOUS"), ndpointer(ct.c_double, flags="F_CONTIGUOUS")]
        self.evpoly.argtypes = [ndpointer(ct.c_double, flags="F_CONTIGUOUS"), ndpointer(ct.c_double, flags="F_CONTIGUOUS")]

    def setup_c_procs(self):
        logging.info("Loading and setting up C procedures from LIBNAME: {}".format(self.C_LIBNAME))
        basislib = ct.CDLL(self.C_LIBNAME)

        self.evpoly = basislib.evpoly
        self.evpoly.argtypes = [ndpointer(ct.c_double, flags="C_CONTIGUOUS"), ndpointer(ct.c_double, flags="C_CONTIGUOUS")]
