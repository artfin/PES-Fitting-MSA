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

from collections import namedtuple
XYZConfig = namedtuple('Config', ['atoms', 'energy'])

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PolyDataset_t:
    NATOMS   : int
    NMON     : int
    NPOLY    : int
    symmetry : str
    order    : str
    X        : torch.Tensor
    y        : torch.Tensor
    mask     : Optional[List[int]] = None

# parameter inside the `yij` exp(...)
a0 = 2.0 # bohrs
POLYNOMIAL_LIB = "CUSTOM"

class PolyDataset(Dataset):
    def __init__(self, wdir, xyz_file, order, symmetry, set_intermolecular_to_zero=True): #, lr_model=None):
        self.wdir = wdir
        logging.info("working directory: {}".format(self.wdir))

        self.order = order
        self.symmetry = symmetry
        self.set_intermolecular_to_zero = set_intermolecular_to_zero

        logging.info("Loading configurations from xyz_file: {}".format(xyz_file))

        xyz_configs = self.load_xyz(xyz_file)
        X, y = self.prepare_dataset_from_configs(xyz_configs)

        if POLYNOMIAL_LIB == "MSA":
            self.mask = X.abs().sum(dim=0).bool().numpy().astype(int)
            logging.info("Applying non-zero mask. Selecting {} polynomials out of {} initially...".format(
                self.mask.sum(), len(self.mask)
            ))

            nonzero_index = self.mask.nonzero()[0].tolist()
            logging.info("indices of non-zero polynomials: {}".format(nonzero_index))
            logging.info(json.dumps(nonzero_index))

            self.NPOLY = self.mask.sum()
            X = X[:, self.mask.astype(np.bool)]

            logging.info("New size of the X-array: {}".format(X.size()))

        self.X, self.y = X, y

    def prepare_dataset_from_configs(self, xyz_configs):
        logging.info("preparing atomic distances..")

        self.NDIS = self.NATOMS * (self.NATOMS - 1) // 2

        yij = self.make_yij(xyz_configs)
        logging.info("Done.")

        if POLYNOMIAL_LIB == "MSA":
            logging.info("using MSA dynamic library to compute invariant polynomials")

            stub       = '_{}_{}'.format(self.symmetry.replace(' ', '_'), self.order)
            MONO_fname = os.path.join(self.wdir, 'MOL' + stub + '.MONO')
            POLY_fname = os.path.join(self.wdir, 'MOL' + stub + '.POLY')
            self.NMON  = sum(1 for line in open(MONO_fname))
            self.NPOLY = sum(1 for line in open(POLY_fname))
            logging.info("detected NMON  = {}".format(self.NMON))
            logging.info("detected NPOLY = {}".format(self.NPOLY))

            self.F_LIBNAME = os.path.join(self.wdir, 'f_basis' + stub + '.so')
            self.setup_fortran_procs()

            x = np.zeros((self.NDIS, 1))
            m = np.zeros((self.NMON, 1))
            p = np.zeros((self.NPOLY, 1))
        elif POLYNOMIAL_LIB == "CUSTOM":
            logging.info("using custom C dynamic library to compute invariant polynomials")

            stub       = '_{}_{}'.format(self.symmetry.replace(' ', '_'), self.order)
            self.C_LIBNAME = os.path.join(self.wdir, 'c_basis' + stub + '.so')

            if not os.path.isfile(self.C_LIBNAME):
                logging.info("compiling custom C dynamic library")
                assert False
                import resource
                print(resource.getrlimit(resource.RLIMIT_AS))
                #fpath = os.path.join(wdir, fname)
                st = cl('gcc -c -fPIC c_basis_4_2_1_4.cc -o c_basis_4_2_1_4.o')
                logging.info(st.stdout)
                st = cl('gcc -shared -o c_basis_4_2_1_4.so c_basis_4_2_1_4.o')
                logging.info(st.stdout)

                #import subprocess
                #subprocess.call(['sh', os.path.join(self.wdir, 'build_shared.sh'), self.order])
                #subprocess.call('sh datasets/external/build_shared.sh 4')


            assert os.path.isfile(self.C_LIBNAME)
            self.setup_c_procs()

            C_LIBNAME_CODE = os.path.join(self.wdir, 'c_basis' + stub + '.cc')
            with open(C_LIBNAME_CODE, mode='r') as fp:
                lines = "".join(fp.readlines())

            pattern = "double p\[(\d+)\]"
            found = re.findall(pattern, lines)
            self.NPOLY = int(found[0])
            self.NMON = None

            x = np.zeros((self.NDIS,  1))
            p = np.zeros((self.NPOLY, 1))
        else:
            raise ValueError("unreachable")

        NCONFIGS = len(xyz_configs)
        poly = np.zeros((NCONFIGS, self.NPOLY))

        logging.info("Computing polynomials...")

        for n in range(0, NCONFIGS):
            x = yij[n, :].copy()

            if POLYNOMIAL_LIB == "MSA":
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

            poly[n, :] = p.reshape((self.NPOLY, )).copy()

        logging.info("Done.")

        energies = np.zeros((NCONFIGS, 1))
        for ind, xyz_config in enumerate(xyz_configs):
            energies[ind] = xyz_config.energy
        # NOTE: LR-model is to added here
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
                    atoms[natom, :] = list(map(float, words[1:]))

                c = XYZConfig(atoms=atoms, energy=energy)
                xyz_configs.append(c)

        return xyz_configs

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

                # H2-H2O (makes the fitting significantly worse)
                #if i == 0 and j == 1: yij[n, k] = 0.0; k = k + 1; continue # H1 H2 (H2)
                #if i == 2 and j == 3: yij[n, k] = 0.0; k = k + 1; continue # H1 H2 (H2O)
                #if i == 2 and j == 4: yij[n, k] = 0.0; k = k + 1; continue # H1 O  (H2O)
                #if i == 3 and j == 4: yij[n, k] = 0.0; k = k + 1; continue # H2 O  (H2O)
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
