import ctypes as ct
import logging
import numpy as np
import os

from itertools import combinations

import torch
from torch.utils.data import Dataset

from collections import namedtuple
Config = namedtuple('Config', ['atoms', 'energy'])

# Physical constants
BOHRTOANG = 0.52917721067
HTOCM     = 2.194746313702e5

# parameter inside the `yij` exp(...)
a0        = 2.0

class PolyDataset(Dataset):
    def __init__(self, wdir, config_fname, order, symmetry):
        self.wdir = wdir
        logging.info("working directory: {}".format(self.wdir))

        self.config_fname = os.path.join(self.wdir, config_fname)
        logging.info("configuration file: {}".format(self.config_fname))
        logging.info("loading configurations...")
        self.configs = self.load()

        logging.info("preparing the coordinates..")
        self.NDIS = self.NATOMS * (self.NATOMS - 1) // 2
        yij       = self.make_yij(self.configs)
        logging.info("Done.")

        stub       = '_{}_{}'.format(symmetry.replace(' ', '_'), order)
        MONO_fname = os.path.join(wdir, 'MOL' + stub + '.MONO')
        POLY_fname = os.path.join(wdir, 'MOL' + stub + '.POLY')
        self.NMON  = sum(1 for line in open(MONO_fname))
        self.NPOLY = sum(1 for line in open(POLY_fname))
        logging.info("detected NMON  = {}".format(self.NMON))
        logging.info("detected NPOLY = {}".format(self.NPOLY))

        self.LIBNAME = os.path.join(self.wdir, 'basis' + stub + '.so')
        self.setup_fortran_procs()

        x = np.zeros((self.NDIS, 1), order="F")
        m = np.zeros((self.NMON, 1), order="F")
        p = np.zeros((self.NPOLY, 1), order="F")
        poly = np.zeros((self.NCONFIGS, self.NPOLY))

        logging.info("Preparing the polynomials...")

        for n in range(0, self.NCONFIGS):
            x = yij[n, :].copy()

            x_ptr = x.ctypes.data_as(ct.POINTER(ct.c_double))
            m_ptr = m.ctypes.data_as(ct.POINTER(ct.c_double))
            self.evmono(x_ptr, m_ptr)

            p_ptr = p.ctypes.data_as(ct.POINTER(ct.c_double))
            self.evpoly(m_ptr, p_ptr)

            assert ~np.isnan(np.sum(m)), "There are NaN values in monomials produced by Fortran_evmono"
            assert ~np.isnan(np.sum(p)), "There are NaN values in polynomials produced by Fortran_evpoly"

            assert np.max(p) < 1e10, "There are suspicious values of polynomials produced by Fortran_evpoly"

            poly[n, :] = p.reshape((self.NPOLY, )).copy()

        logging.info("Done.")

        energies = np.asarray([c.energy for c in self.configs]).reshape((self.NCONFIGS, 1)) # (NCONFIGS,) -> (NCONFIGS, 1)

        self.X = torch.from_numpy(poly)
        self.y = torch.from_numpy(energies)

    def __len__(self):
        return len(self.y)

    def load(self):
        nlines = sum(1 for line in open(self.config_fname))

        self.NATOMS = int(open(self.config_fname).readline())
        logging.info("detected NATOMS = {}".format(self.NATOMS))

        self.NCONFIGS = nlines // (self.NATOMS + 2)
        logging.info("detected NCONFIGS = {}".format(self.NCONFIGS))

        configs = []
        with open(self.config_fname, mode='r') as inp:
            for i in range(self.NCONFIGS):
                line = inp.readline()
                energy = float(inp.readline())

                atoms = np.zeros((self.NATOMS, 3))
                for natom in range(self.NATOMS):
                    words = inp.readline().split()
                    atoms[natom, :] = list(map(float, words[1:]))

                c = Config(atoms=atoms, energy=energy)
                configs.append(c)

        return configs

    def make_yij(self, configs):
        ZERO_YIJ = True
        if ZERO_YIJ:
            logging.info("[-- NOTICE --] Morse variables with respect to intermolecular distances are zero.") 

        yij = np.zeros((self.NCONFIGS, self.NDIS), order="F")

        for n in range(self.NCONFIGS):
            c = configs[n]

            k = 0
            for i, j in combinations(range(self.NATOMS), 2):
                # CH4-N2
                if ZERO_YIJ:
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

                # H2-H2O (makes the fitting significantly worse; check the order=3 polynomials)
                #if i == 0 and j == 1: yij[n, k] = 0.0; k = k + 1; continue # H1 H2 (H2)
                #if i == 2 and j == 3: yij[n, k] = 0.0; k = k + 1; continue # H1 H2 (H2O)
                #if i == 2 and j == 4: yij[n, k] = 0.0; k = k + 1; continue # H1 O  (H2O)
                #if i == 3 and j == 4: yij[n, k] = 0.0; k = k + 1; continue # H2 O  (H2O)
                yij[n, k] = np.linalg.norm(c.atoms[i] - c.atoms[j])
                yij[n][k] /= BOHRTOANG
                yij[n][k] = np.exp(-yij[n, k] / a0)
                k = k + 1

        return yij

    def setup_fortran_procs(self):
        basislib = ct.CDLL(self.LIBNAME)

        self.evmono = basislib.c_evmono
        self.evmono.argtypes = [ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]

        self.evpoly = basislib.c_evpoly
        self.evpoly.argtypes = [ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
