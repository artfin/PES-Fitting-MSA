# MD17 dataset:
# http://quantum-machine.org/gdml/#datasets

import ctypes as ct
import logging
import json
import numpy as np
import os
import re

from itertools import combinations

from genpip import cl

import torch
from torch.utils.data import Dataset

from dataclasses import dataclass
from typing import List, Optional

BOHRTOANG = 0.529177249
KCALTOCM  = 349.757

# parameter inside the `yij` exp(...)
a0 = 2.0 # bohrs
POLYNOMIAL_LIB = "MSA"

SYMBOLS = ('H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F',)

@dataclass
class XYZConfig:
    coords  : np.array
    z       : np.array
    energy  : float
    forces  : Optional[np.array]

    def check(self):
        # TODO: check some interatomic distnaces (C-C, C-H and others to be in specific range)
        assert False

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
        #MAX_CH = 2.5  # BOHR
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
    dX           : Optional[torch.Tensor]
    dy           : Optional[torch.Tensor]
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

            coords = np.zeros((NATOMS, 3))
            z = np.zeros((NATOMS, 1))

            for natom in range(NATOMS):
                words = inp.readline().split()

                coords[natom, :] = list(map(float, words[1:]))
                charge = SYMBOLS.index(words[0]) + 1
                z[natom] = charge

            c = XYZConfig(coords=coords, z=z, energy=energy)
            #c.check()
            xyz_configs.append(c)

    return NATOMS, NCONFIGS, xyz_configs

def write_xyz(xyz_path, xyz_configs):
    natoms = xyz_configs[0].coords.shape[0]

    with open(xyz_path, mode='w') as fd:
        for xyz_config in xyz_configs:
            fd.write("    {}\n".format(natoms))
            fd.write("    {:.10f}\n".format(xyz_config.energy))
            for k in range(natoms):
                charge = xyz_config.z[k]
                symbol = SYMBOLS[charge - 1]
                fd.write("{} \t {:.10f} \t {:.10f} \t {:.10f}\n".format(symbol, xyz_config.coords[k, 0], xyz_config.coords[k, 1], xyz_config.coords[k, 2]))

def print_molpro_format(xyz_config):
    natoms = xyz_config.coords.shape[0]

    for k in range(natoms):
        charge = xyz_config.z[k]
        symbol = SYMBOLS[charge - 1]
        print(f"{k+1}, {symbol}{k+1},, {xyz_config.coords[k, 0]:.10f}, {xyz_config.coords[k, 1]:.10f}, {xyz_config.coords[k, 2]:.10f}")


def load_npz(fpath, load_forces=False):
    fd = np.load(fpath)

    if "theory" in fd:
        logging.info("  Theory: {}".format(fd['theory']))

    assert(fd['E'].shape[0] == fd['R'].shape[0])
    NCONFIGS = fd['E'].shape[0]
    NATOMS   = fd['R'].shape[1]
    z        = fd['z']

    energy_min = min(fd['E'])[0]
    energy_max = max(fd['E'])[0]
    logging.info("ENERGY_MIN: {}".format(energy_min))
    logging.info("ENERGY_MAX: {}".format(energy_max))

    # ETHANOL CCSD(T)
    #energy_min = -97095.38950894916
    # ETHANOL DFT (full)
    energy_min = -97208.40600498248

    xyz_configs = []
    for ind, (coords, energy) in enumerate(zip(fd['R'], fd['E'])):
        # TODO:
        # THIS DATASET HAS DISTANCES IN ANGSTROM AND ENERGY IN KCAL/MOL
        # MAYBE HAVE THIS INFO IN .NPZ FILE, PARSE AND TRANSFORM ACCORDINGLY?

        coords_bohr    = coords / BOHRTOANG
        energy_cm      = (energy[0] - energy_min) * KCALTOCM

        if load_forces:
            forces = fd['F'][ind]
            forces_cm_bohr = forces * BOHRTOANG * KCALTOCM
        else:
            forces_cm_bohr = None

        xyz_configs.append(
            XYZConfig(
                coords=coords_bohr,
                z=z,
                energy=energy_cm,
                forces=forces_cm_bohr
            )
        )

    return NATOMS, NCONFIGS, xyz_configs

def write_npz(npz_path, xyz_configs):
    from random import shuffle
    shuffle(xyz_configs)
    xyz_configs = xyz_configs[:10000]

    natoms   = xyz_configs[0].coords.shape[0]
    nconfigs = len(xyz_configs)

    energy = np.array([xyz_config.energy for xyz_config in xyz_configs]).reshape((nconfigs, 1))
    z = xyz_configs[0].z
    R = np.asarray([xyz_config.coords for xyz_config in xyz_configs])
    F = np.asarray([xyz_config.forces for xyz_config in xyz_configs])

    np.savez(npz_path, E=energy, z=z, R=R, F=F)


class PolyDataset(Dataset):
    def __init__(self, wdir, file_path, order=None, symmetry=None, use_forces=False, intramz=False, purify=False):
        self.wdir = wdir
        logging.info("working directory: {}".format(self.wdir))

        self.order      = order
        self.symmetry   = symmetry
        self.use_forces = use_forces
        self.intramz    = intramz
        self.purify     = purify

        logging.info("Loading configurations from file_path: {}".format(file_path))
        if file_path.endswith(".xyz"):
            NATOMS, NCONFIGS, xyz_configs = load_xyz(file_path)
            #write_npz("ch4-n2-rigid.npz", xyz_configs)
        elif file_path.endswith(".npz"):
            NATOMS, NCONFIGS, xyz_configs = load_npz(file_path, use_forces)
            #write_npz("ethanol_dft-10000.npz", xyz_configs)
        else:
            raise ValueError("Unrecognized file format.")

        self.NATOMS   = NATOMS
        self.NDIS     = self.NATOMS * (self.NATOMS - 1) // 2
        self.NCONFIGS = NCONFIGS
        logging.info("  NATOMS = {}".format(self.NATOMS))
        logging.info("  NCONFIGS = {}".format(NCONFIGS))

        self.prepare_poly_lib()

        if self.purify:
            if self.intramz:
                raise ValueError("Conflicting options: PURIFY and INTRAMZ")

            self.make_purify_mask(xyz_configs)

        #  X: values of polynomials
        # dX: derivatives of polynomials w.r.t. cartesian coordinates
        #  y: values of energies
        # dy: derivatives of energies w.r.t. cartesian coordinates [a.k.a forces]
        if self.use_forces:
            self.X, self.dX, self.y, self.dy = self.prepare_dataset_with_derivatives_from_configs(xyz_configs)
        else:
            self.X, self.y   = self.prepare_dataset_from_configs(xyz_configs)
            self.dX, self.dy = None, None

        # TODO: create mask explicitly when `purify` and `intramz` options are selected
        if self.purify:
            self.NPOLY = self.purify_mask.sum()
            X = X[:, self.purify_mask.astype(np.bool)]
            assert False

        if self.intramz:
            assert False

        #if POLYNOMIAL_LIB == "MSA":
        #    self.mask = X.abs().sum(dim=0).bool().numpy().astype(int)
        #    logging.info("Applying non-zero mask. Selecting {} polynomials out of {} initially...".format(
        #        self.mask.sum(), len(self.mask)
        #    ))

        #    #nonzero_index = self.mask.nonzero()[0].tolist()
        #    #logging.info("indices of non-zero polynomials: {}".format(nonzero_index))
        #    #logging.info(json.dumps(nonzero_index))

        #    self.NPOLY = self.mask.sum()
        #    X = X[:, self.mask.astype(np.bool)]
        #    logging.info("Final size of the X-array: {}".format(X.size()))

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
                from genpip import compile_basis_dlib
                compile_basis_dlib(self.order, self.symmetry, self.wdir)

            assert os.path.isfile(self.F_LIBNAME), "No F_BASIS_LIBRARY={} found".format(self.F_LIBNAME)

            if self.use_forces:
                self.F_DER_LIBNAME = os.path.join(self.wdir, 'f_gradbasis' + stub + '.so')
                if not os.path.isfile(self.F_DER_LIBNAME):
                    from genpip import compile_derivatives_dlib
                    compile_derivatives_dlib(self.order, self.symmetry, self.wdir)

                assert os.path.isfile(self.F_DER_LIBNAME), "No F_DERIVATIVES_LIBRARY={} found".format(self.F_DER_LIBNAME)

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
        p = np.zeros((self.NPOLY, ))
        if POLYNOMIAL_LIB == "MSA":
            m = np.zeros((self.NMON, ))

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


    def prepare_dataset_with_derivatives_from_configs(self, xyz_configs):
        logging.info("Preparing interatomic distances and their derivatives.")
        yij  = self.make_yij(xyz_configs, intramz=self.intramz, intermz=False)
        drdx = self.make_drdx(xyz_configs, intramz=self.intramz, intermz=False)
        logging.info("Done.")


        NCONFIGS = len(xyz_configs)
        poly             = np.zeros((NCONFIGS, self.NPOLY))
        poly_derivatives = np.zeros((NCONFIGS, self.NPOLY, 3 * self.NATOMS))

        # omit second number for the arrays to be considered one-dimensional
        m  = np.zeros((self.NMON, ))
        dm = np.zeros((self.NMON, ), order="F")
        p  = np.zeros((self.NPOLY, ))
        dp = np.zeros((self.NPOLY, ))

        if POLYNOMIAL_LIB == "MSA":
            for n in range(0, NCONFIGS):
                yij_c  = yij[n, :].copy()
                drdx_c = np.asfortranarray(drdx[n, :, :].copy())

                self.evmono(yij_c, m)
                self.evpoly(m, p)
                poly[n, :] = p.reshape((self.NPOLY, )).copy()

                for nd in range(3 * self.NATOMS):
                    # I guess we have to take into account here that fortran is 1-based
                    # `nd` is passed by value into fortran code: `nd` -> `nd` + 1
                    self.devmono(drdx_c, dm, m, nd + 1)
                    self.devpoly(dm, p, dp)
                    poly_derivatives[n, :, nd] = dp

        elif POLYNOMIAL_LIB == "CUSTOM":
            assert False
        else:
            assert False

        energies = np.zeros((NCONFIGS, 1))
        for ind, xyz_config in enumerate(xyz_configs):
            energies[ind] = xyz_config.energy

        forces = np.zeros((NCONFIGS, self.NATOMS, 3))
        for ind, xyz_config in enumerate(xyz_configs):
            forces[ind, :, :] = xyz_config.forces

        X  = torch.from_numpy(poly)
        dX = torch.from_numpy(poly_derivatives)
        y  = torch.from_numpy(energies)
        dy = torch.from_numpy(forces)

        return X, dX, y, dy

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

    def make_drdx(self, xyz_configs, intramz=False, intermz=False):
        NCONFIGS = len(xyz_configs)
        drdx = np.zeros((NCONFIGS, self.NATOMS * 3, self.NDIS), order="F")

        k = 1
        for n in range(NCONFIGS):
            c = xyz_configs[n]

            k = 0
            for i, j in combinations(range(self.NATOMS), 2):
                if intramz:
                    assert False

                if intermz:
                    assert False

                dr      = c.coords[i] - c.coords[j]
                dr_norm = np.linalg.norm(dr)

                drdx[n, 3*i:3*i + 3, k] =  dr / dr_norm
                drdx[n, 3*j:3*j + 3, k] = -dr / dr_norm

                k = k + 1

        return drdx

    def make_yij(self, xyz_configs, intramz=False, intermz=False):
        """
            intramz: True -> intramolecular distances are set to zero
                             (interatomic distances between atoms within one monomer)
            intermz: True -> intermolecular distances are set to zero
                    This option is used for basis purification to find polynomials
                    that do not vanish when monomers are infinitely separated.
        """

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
                    # TODO: get rid of this by introducing the concept of monomer
                    #       formalize the notion of intra- and intermolecular coordinates
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
                    # TODO: get rid of this by introducing the concept of monomer
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

                yij[n, k] = np.linalg.norm(c.coords[i] - c.coords[j])
                yij[n][k] = np.exp(-yij[n, k] / a0)
                k = k + 1

        return yij

    def setup_fortran_procs(self):
        logging.info("Loading and setting up Fortran procedures for basis evaluation from LIBNAME: {}".format(self.F_LIBNAME))
        basislib = ct.CDLL(self.F_LIBNAME)

        self.evmono = basislib.c_evmono
        self.evpoly = basislib.c_evpoly

        from numpy.ctypeslib import ndpointer
        self.evmono.argtypes = [ndpointer(ct.c_double, ndim=1, flags="F"), ndpointer(ct.c_double, ndim=1, flags="F")]
        self.evpoly.argtypes = [ndpointer(ct.c_double, ndim=1, flags="F"), ndpointer(ct.c_double, ndim=1, flags="F")]

        if self.use_forces:
            logging.info("Loading and setting up Fortran procedures for derivatives evaluation from LIBNAME: {}".format(self.F_DER_LIBNAME))
            derlib = ct.CDLL(self.F_DER_LIBNAME)

            self.devmono = derlib.c_devmono
            self.devpoly = derlib.c_devpoly

            self.devmono.argtypes = [ndpointer(ct.c_double, ndim=2, flags="F"), ndpointer(ct.c_double, ndim=1, flags="F"),
                                     ndpointer(ct.c_double, ndim=1, flags="F"), ct.c_int]

            self.devpoly.argtypes = [ndpointer(ct.c_double, ndim=1, flags="F"), ndpointer(ct.c_double, ndim=1, flags="F"),
                                     ndpointer(ct.c_double, ndim=1, flags="F")]

    def setup_c_procs(self):
        logging.info("Loading and setting up C procedures from LIBNAME: {}".format(self.C_LIBNAME))
        basislib = ct.CDLL(self.C_LIBNAME)

        self.evpoly = basislib.evpoly

        from numpy.ctypeslib import ndpointer
        self.evpoly.argtypes = [ndpointer(ct.c_double, flags="C_CONTIGUOUS"), ndpointer(ct.c_double, flags="C_CONTIGUOUS")]
