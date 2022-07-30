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
#POLYNOMIAL_LIB = "MSA"
POLYNOMIAL_LIB = "CUSTOM"

SYMBOLS = ('H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F',)

# Optional actually means that variable is either T or None
# so to make optional you have to specify None as default value
@dataclass
class XYZConfig:
    coords  : np.array
    z       : np.array
    energy  : Optional[float]    = None
    dipole  : Optional[np.array] = None
    forces  : Optional[np.array] = None

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
class XYZConfigPair:
    coords1 : np.array
    coords2 : np.array
    z1      : np.array
    z2      : np.array
    energy  : np.array

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


def load_npz(fpath, load_forces=False):
    fd = np.load(fpath)

    if "theory" in fd:
        logging.info("  Theory: {}".format(fd['theory']))

    energy_min = min(fd['E'])[0]
    energy_max = max(fd['E'])[0]
    logging.info("    Energy range: {} - {} cm-1".format(energy_min, energy_max))

    if fd['nmol'] == 1:
        assert(fd['E'].shape[0] == fd['R'].shape[0])
        NCONFIGS = fd['E'].shape[0]
        NATOMS   = fd['R'].shape[1]
        z        = fd['z']

        xyz_configs = []
        if load_forces:
            for coords, energy, forces in zip(fd['R'], fd['E'], fd['F']):
                #coords_bohr    = coords / BOHRTOANG
                #energy_cm      = (energy[0] - energy_min) * KCALTOCM
                #forces_cm_bohr = forces * BOHRTOANG * KCALTOCM

                xyz_configs.append(
                    XYZConfig(
                        coords=coords,
                        z=z,
                        energy=energy,
                        forces=forces
                    )
                )
        else:
            for coords, energy in zip(fd['R'], fd['E']):
                #coords_bohr    = coords / BOHRTOANG
                #energy_cm      = (energy[0] - energy_min) * KCALTOCM

                xyz_configs.append(
                    XYZConfig(
                        coords=coords,
                        z=z,
                        energy=energy,
                        forces=None
                    )
                )
    elif fd['nmol'] == 2:
        NCONFIGS = fd['E'].shape[0]
        z1, z2   = fd['z1'], fd['z2']
        NATOMS   = z1.shape[0] + z2.shape[0]

        assert fd['R1'].shape[0] == NCONFIGS
        assert fd['R2'].shape[0] == NCONFIGS

        xyz_configs = [XYZConfigPair(coords1=c1, coords2=c2, z1=z1, z2=z2, energy=energy)
                       for c1, c2, energy in zip(fd['R1'], fd['R2'], fd['E'])]

    return NATOMS, NCONFIGS, xyz_configs

def load_xyz_with_energy(fpath):
    nlines = sum(1 for line in open(fpath, mode='r'))
    NATOMS = int(open(fpath, mode='r').readline())
    NCONFIGS = nlines // (NATOMS + 2)

    xyz_configs = []
    with open(fpath, mode='r') as inp:
        for i in range(NCONFIGS):
            line = inp.readline()

            energy = float(inp.readline())
            coords = np.zeros((NATOMS, 3))
            z      = np.zeros((NATOMS, 1))

            for natom in range(NATOMS):
                words = inp.readline().split()

                coords[natom, :] = list(map(float, words[1:]))
                charge           = SYMBOLS.index(words[0]) + 1
                z[natom]         = charge

            c = XYZConfig(coords=coords, z=z, energy=energy)
            #c.check()
            xyz_configs.append(c)

    return NATOMS, NCONFIGS, xyz_configs

def load_xyz_with_dipole(fpath):
    nlines = sum(1 for line in open(fpath, mode='r'))
    NATOMS = int(open(fpath, mode='r').readline())
    NCONFIGS = nlines // (NATOMS + 2)

    xyz_configs = []
    with open(fpath, mode='r') as inp:
        for i in range(NCONFIGS):
            line = inp.readline()

            dipole = np.fromiter(map(float, inp.readline().split()), dtype=np.float32)
            coords = np.zeros((NATOMS, 3))
            z = np.zeros((NATOMS, 1))

            for natom in range(NATOMS):
                words = inp.readline().split()

                coords[natom, :] = list(map(float, words[1:]))
                charge           = SYMBOLS.index(words[0]) + 1
                z[natom]         = charge

            c = XYZConfig(coords=coords, z=z, dipole=dipole)
            #c.check()
            xyz_configs.append(c)

    return NATOMS, NCONFIGS, xyz_configs

def write_xyz(xyz_path, xyz_configs, prop={"energy", "dipole"}):
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


def write_npz(npz_path, xyz_configs):
    from random import shuffle
    shuffle(xyz_configs)
    xyz_configs = xyz_configs[:50000]

    natoms   = xyz_configs[0].coords.shape[0]
    nconfigs = len(xyz_configs)

    energy = np.array([xyz_config.energy for xyz_config in xyz_configs]).reshape((nconfigs, 1))
    z = xyz_configs[0].z
    R = np.asarray([xyz_config.coords for xyz_config in xyz_configs])
    F = np.asarray([xyz_config.forces for xyz_config in xyz_configs])

    np.savez(npz_path, E=energy, z=z, R=R, F=F)


class PolyDataset(Dataset):
    def __init__(self, wdir, typ, file_path, order=None, symmetry=None, load_forces=False, atom_mapping=None, intramz=False, purify=False):
        self.wdir = wdir
        logging.info("working directory: {}".format(self.wdir))

        self.typ          = typ
        self.order        = order
        self.symmetry     = symmetry
        self.load_forces  = load_forces
        self.atom_mapping = atom_mapping
        self.intramz      = intramz
        self.purify       = purify

        logging.info("Loading configurations from file_path: {}".format(file_path))
        if file_path.endswith(".xyz"):
            if load_forces:
                logging.error("STORING/LOADING FORCES IN XYZ FILE IS NOT SUPPORTED FOR NOW")
                assert False
            if self.typ == 'ENERGY':
                NATOMS, NCONFIGS, self.xyz_configs = load_xyz_with_energy(file_path)
            elif self.typ == 'DIPOLE':
                NATOMS, NCONFIGS, self.xyz_configs = load_xyz_with_dipole(file_path)
            else:
                assert False

            #write_npz("ch4-n2-rigid.npz", xyz_configs)
        elif file_path.endswith(".npz"):
            if self.typ == 'ENERGY':
                NATOMS, NCONFIGS, self.xyz_configs = load_npz(file_path, load_forces)
            else:
                assert False

            #write_npz("ethanol_dft-50000.npz", self.xyz_configs)
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
        #  y: values of energies | dipoles
        # dy: (minus) derivatives of energies w.r.t. cartesian coordinates [a.k.a forces]

        if self.load_forces:
            self.X, self.dX, self.y, self.dy = self.prepare_dataset_with_derivatives_from_configs()
        else:
            if self.typ == 'ENERGY':
                self.X, self.y = self.prepare_dataset_with_energies_from_configs()
            elif self.typ == 'DIPOLE':
                self.X, self.y = self.prepare_dataset_with_dipoles_from_configs()
            else:
                assert False

            self.dX, self.dy = None, None

        # TODO: create mask explicitly when `purify` and `intramz` options are selected
        if self.purify:
            assert False
            self.NPOLY = self.purify_mask.sum()
            X = X[:, self.purify_mask.astype(np.bool)]

        if self.intramz:
            self.mask = self.X.abs().sum(dim=0).bool().numpy().astype(int)
            logging.info("Applying non-zero mask. Selecting {} polynomials out of {} initially...".format(
                self.mask.sum(), len(self.mask)
            ))

            nonzero_index = self.mask.nonzero()[0].tolist()
            #logging.info("indices of non-zero polynomials: {}".format(nonzero_index))
            #logging.info(json.dumps(nonzero_index))

            stub = '_{}_{}'.format(self.symmetry.replace(' ', '_'), self.order)
            basis_file = os.path.join(self.wdir, 'MOL' + stub + '.BAS')

            from genproc import load_polynomials_from_bas, save_polynomials_to_bas
            poly = load_polynomials_from_bas(basis_file)

            poly_masked = [poly[k] for k in nonzero_index]
            basis_file_masked = os.path.join(self.wdir, 'MOL' + stub + '_intramz.BAS')
            save_polynomials_to_bas(basis_file_masked, poly_masked)
            logging.info("Saving .BAS file for the basis set to: {}".format(basis_file_masked))

            self.NPOLY = self.mask.sum()
            self.X = self.X[:, self.mask.astype(np.bool)]
            logging.info("Final size of the X-array: {}".format(self.X.size()))


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
                from genpip import compile_basis_dlib_f
                compile_basis_dlib_f(self.order, self.symmetry, self.wdir)

            assert os.path.isfile(self.F_LIBNAME), "No F_BASIS_LIBRARY={} found".format(self.F_LIBNAME)

            if self.load_forces:
                self.F_DER_LIBNAME = os.path.join(self.wdir, 'f_gradbasis' + stub + '.so')
                if not os.path.isfile(self.F_DER_LIBNAME):
                    from genpip import compile_derivatives_dlib_f
                    compile_derivatives_dlib_f(self.order, self.symmetry, self.wdir)

                assert os.path.isfile(self.F_DER_LIBNAME), "No F_DERIVATIVES_LIBRARY={} found".format(self.F_DER_LIBNAME)

            self.setup_fortran_procs()

        elif POLYNOMIAL_LIB == "CUSTOM":
            stub       = '_{}_{}'.format(self.symmetry.replace(' ', '_'), self.order)
            self.C_LIBNAME = os.path.join(self.wdir, 'c_basis' + stub + '.so')
            if not os.path.isfile(self.C_LIBNAME):
                from genpip import compile_basis_dlib_c
                compile_basis_dlib_c(self.order, self.symmetry, self.wdir)

            assert os.path.isfile(self.C_LIBNAME), "No C_LIBRARY={} found".format(self.C_LIBNAME)

            if self.load_forces:
                self.C_DER_LIBNAME = os.path.join(self.wdir, 'c_jac' + stub + '.so')
                if not os.path.isfile(self.C_DER_LIBNAME):
                    from genpip import compile_derivatives_dlib_c
                    compile_derivatives_dlib_c(self.order, self.symmetry, self.wdir)

                assert os.path.isfile(self.C_DER_LIBNAME), "No C_DERIVATIVES_LIBRARY={} found".format(self.C_DER_LIBNAME)

            self.setup_c_procs()

            C_LIB = os.path.join(self.wdir, 'c_basis' + stub + '.cc')
            assert os.path.isfile(C_LIB)
            with open(C_LIB, mode='r') as fp:
                lines = "".join(fp.readlines())

            pattern    = "p\[(\d+)\]"
            found      = re.findall(pattern, lines)
            assert found

            self.NPOLY = max(list(map(int, found))) + 1
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


    def prepare_dataset_with_derivatives_from_configs(self):
        logging.info("Preparing interatomic distances and their derivatives.")
        yij  = self.make_yij(self.xyz_configs, intramz=self.intramz, intermz=False)
        drdx = self.make_drdx(self.xyz_configs, intramz=self.intramz, intermz=False)
        logging.info("Done.")

        NCONFIGS = len(self.xyz_configs)
        poly             = np.zeros((NCONFIGS, self.NPOLY))
        poly_derivatives = np.zeros((NCONFIGS, self.NPOLY, 3 * self.NATOMS))

        if POLYNOMIAL_LIB == "MSA":
            # omit second number for the arrays to be considered one-dimensional
            p  = np.zeros((self.NPOLY, ))
            dp = np.zeros((self.NPOLY, ))
            m  = np.zeros((self.NMON, ), order="F")
            dm = np.zeros((self.NMON, ), order="F")

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
            dpdy    = np.zeros((self.NDIS, self.NPOLY))
            dpdy_pp = (dpdy.__array_interface__['data'][0] + np.arange(dpdy.shape[0]) * dpdy.strides[0]).astype(np.uintp)

            p = np.zeros((self.NPOLY, ))

            for n in range(0, NCONFIGS):
                _yij  = yij[n, :].copy()
                _dydr = -1.0 / a0 * np.diag(_yij)
                _drdx = drdx[n, :, :].copy()

                self.evpoly(_yij, p)
                self.c_jac_dpdy(dpdy_pp, _yij)

                dpdx = _drdx @ _dydr @ dpdy

                poly[n, :]          = p.copy()
                poly_derivatives[n] = dpdx.T

        else:
            assert False

        energies = np.zeros((NCONFIGS, 1))
        for ind, xyz_config in enumerate(self.xyz_configs):
            energies[ind] = xyz_config.energy

        forces = np.zeros((NCONFIGS, self.NATOMS, 3))
        for ind, xyz_config in enumerate(self.xyz_configs):
            forces[ind, :, :] = xyz_config.forces

        X  = torch.from_numpy(poly)
        dX = torch.from_numpy(poly_derivatives)
        y  = torch.from_numpy(energies)
        dy = torch.from_numpy(forces)

        return X, dX, y, dy

    def prepare_dataset_with_dipoles_from_configs(self):
        logging.info("preparing interatomic distances..")
        yij = self.make_yij(self.xyz_configs, intramz=self.intramz, intermz=False)
        logging.info("Done.")

        NCONFIGS = len(self.xyz_configs)
        poly = np.zeros((NCONFIGS, self.NPOLY))
        x = np.zeros((self.NDIS, 1))

        logging.info("Computing polynomials...")
        for n in range(0, NCONFIGS):
            x = yij[n, :].copy()
            p = self.eval_poly(x)
            poly[n, :] = p.reshape((self.NPOLY, )).copy()

        logging.info("Done.")

        dipoles = np.zeros((NCONFIGS, 3))
        for ind, xyz_config in enumerate(self.xyz_configs):
            dipoles[ind] = xyz_config.dipole

        X = torch.from_numpy(poly)
        y = torch.from_numpy(dipoles)

        return X, y

    def prepare_dataset_with_energies_from_configs(self):
        logging.info("preparing interatomic distances..")
        yij = self.make_yij(self.xyz_configs, intramz=self.intramz, intermz=False)
        logging.info("Done.")

        NCONFIGS = len(self.xyz_configs)
        poly = np.zeros((NCONFIGS, self.NPOLY))
        x = np.zeros((self.NDIS, 1))

        logging.info("Computing polynomials...")
        for n in range(0, NCONFIGS):
            x = yij[n, :].copy()
            p = self.eval_poly(x)
            poly[n, :] = p.reshape((self.NPOLY, )).copy()

        logging.info("Done.")

        energies = np.zeros((NCONFIGS, 1))
        for ind, xyz_config in enumerate(self.xyz_configs):
            energies[ind] = xyz_config.energy

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

        if intramz:
            for n in range(NCONFIGS):
                c = xyz_configs[n]

                coords = []
                iter1, iter2 = 0, 0
                for mapping in self.atom_mapping:
                    (monomer, z), = mapping.items()
                    if monomer == 0:
                        assert c.z1[iter1] == z
                        coords.append(c.coords1[iter1])
                        iter1 = iter1 + 1
                    elif monomer == 1:
                        assert c.z2[iter2] == z
                        coords.append(c.coords2[iter2])
                        iter2 = iter2 + 1

                k = 0
                for i, j in combinations(range(self.NATOMS), 2):
                    (monomer_i, _), = self.atom_mapping[i].items()
                    (monomer_j, _), = self.atom_mapping[j].items()

                    if monomer_i == monomer_j:
                        yij[n, k] = 0.0
                    else:
                        yij[n, k] = np.linalg.norm(coords[i] - coords[j])
                        yij[n][k] = 1.0 / yij[n, k]**6
                        #yij[n][k] = np.exp(-yij[n, k] / a0)

                    k = k + 1

            return yij

        if intermz:
            assert False

            # TODO: get rid of this by introducing the concept of monomer
            #if i == 0 and j == 4: yij[n, k] = 0.0; k = k + 1; continue; # H1 N1
            #if i == 0 and j == 5: yij[n, k] = 0.0; k = k + 1; continue; # H1 N2
            #if i == 1 and j == 4: yij[n, k] = 0.0; k = k + 1; continue; # H2 N1
            #if i == 1 and j == 5: yij[n, k] = 0.0; k = k + 1; continue; # H2 N2
            #if i == 2 and j == 4: yij[n, k] = 0.0; k = k + 1; continue; # H3 N1
            #if i == 2 and j == 5: yij[n, k] = 0.0; k = k + 1; continue; # H3 N2
            #if i == 3 and j == 4: yij[n, k] = 0.0; k = k + 1; continue; # H4 N1
            #if i == 3 and j == 5: yij[n, k] = 0.0; k = k + 1; continue; # H4 N2
            #if i == 4 and j == 6: yij[n, k] = 0.0; k = k + 1; continue; # N1 C
            #if i == 5 and j == 6: yij[n, k] = 0.0; k = k + 1; continue; # N2 C

        if isinstance(self.xyz_configs[0], XYZConfig):
            for n in range(NCONFIGS):
                c = xyz_configs[n]

                k = 0
                for i, j in combinations(range(self.NATOMS), 2):
                    yij[n, k] = np.linalg.norm(c.coords[i] - c.coords[j])
                    yij[n][k] = np.exp(-yij[n, k] / a0)
                    k = k + 1

        if isinstance(xyz_config[0], XYZConfigPair):
            assert False

        return yij

    def setup_fortran_procs(self):
        logging.info("Loading and setting up Fortran procedures for basis evaluation from LIBNAME: {}".format(self.F_LIBNAME))
        basislib = ct.CDLL(self.F_LIBNAME)

        self.evmono = basislib.c_evmono
        self.evpoly = basislib.c_evpoly

        from numpy.ctypeslib import ndpointer
        self.evmono.argtypes = [ndpointer(ct.c_double, ndim=1, flags="F"), ndpointer(ct.c_double, ndim=1, flags="F")]
        self.evpoly.argtypes = [ndpointer(ct.c_double, ndim=1, flags="F"), ndpointer(ct.c_double, ndim=1, flags="F")]

        if self.load_forces:
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
        proc_name = 'evpoly_{}_{}'.format(self.symmetry.replace(' ', '_'), self.order)
        self.evpoly = getattr(basislib, proc_name)

        from numpy.ctypeslib import ndpointer
        self.evpoly.argtypes = [ndpointer(ct.c_double, flags="C"), ndpointer(ct.c_double, flags="C")]

        if self.load_forces:
            logging.info("Loading and setting up C procedures for derivatives evaluation from LIBNAME: {}".format(self.C_DER_LIBNAME))
            derlib = ct.CDLL(self.C_DER_LIBNAME)
            proc_name = 'evpoly_jac_{}_{}'.format(self.symmetry.replace(' ', '_'), self.order)
            self.c_jac_dpdy = getattr(derlib, proc_name)

            # double** should be passed as an array of type np.uintp
            # see https://stackoverflow.com/questions/22425921/pass-a-2d-numpy-array-to-c-using-ctypes
            pp = ndpointer(dtype=np.uintp, ndim=1, flags="C") # double**
            self.c_jac_dpdy.argtypes = [pp, ndpointer(ct.c_double, ndim=1, flags="C")]
            self.c_jac_dpdy.restype  = None


