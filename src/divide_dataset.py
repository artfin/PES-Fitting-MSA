import os
import sys
sys.path.insert(0, "external")
from pybind_example import Poten_CH4
from nitrogen_morse import Poten_N2

import itertools
import logging
import numpy as np

import pathlib
BASEDIR = pathlib.Path(__file__).parent.parent.resolve()

from dataclasses import dataclass

BOHRTOANG = 0.529177249

@dataclass
class XYZConfig:
    atoms  : np.array
    energy : float

def load_xyz(fpath):
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
                atoms[natom, :] = np.fromiter(map(float, words[1:]), dtype=np.float64)

            c = XYZConfig(atoms=atoms, energy=energy)
            xyz_configs.append(c)

    return xyz_configs


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fpaths = [
        os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-1.xyz"),
        os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-2.xyz"),
    ]

    xyz_configs = list(itertools.chain.from_iterable(load_xyz(fpath) for fpath in fpaths))

    ch4_pes = Poten_CH4(libpath=os.path.join(BASEDIR, "external", "obj", "xy4.so"))

    EMIN_CH4 = 2000.0
    EMAX_CH4 = 3000.0

    EMIN_N2 = 0.0
    EMAX_N2 = 1000.0

    out_fpath = os.path.join(BASEDIR, "CH4-N2-EN-NONRIGID-CH4={0:.0f}-{1:.0f}-N2={2:.0f}-{3:.0f}.xyz".format(EMIN_CH4, EMAX_CH4, EMIN_N2, EMAX_N2))
    logging.info("Writing found configurations to {}".format(out_fpath))

    fd = open(out_fpath, "w")

    k = 0
    for xyz_config in xyz_configs:
        CH4_config = np.array([
            *xyz_config.atoms[6, :] * BOHRTOANG, # C
            *xyz_config.atoms[0, :] * BOHRTOANG, # H1
            *xyz_config.atoms[1, :] * BOHRTOANG, # H2
            *xyz_config.atoms[2, :] * BOHRTOANG, # H3
            *xyz_config.atoms[3, :] * BOHRTOANG, # H4
        ])

        N2_len = np.linalg.norm(xyz_config.atoms[4, :] - xyz_config.atoms[5, :]) * BOHRTOANG
        N2_energy = Poten_N2(N2_len)

        CH4_energy = ch4_pes.eval(CH4_config)
        if CH4_energy > EMIN_CH4 and CH4_energy < EMAX_CH4 and N2_energy > EMIN_N2 and N2_energy < EMAX_N2:
            fd.write("   7\n")
            fd.write("   {:.16f}\n".format(xyz_config.energy))
            fd.write("H {:.10f} {:.10f} {:.10f}\n".format(xyz_config.atoms[0, 0], xyz_config.atoms[0, 1], xyz_config.atoms[0, 2]))
            fd.write("H {:.10f} {:.10f} {:.10f}\n".format(xyz_config.atoms[1, 0], xyz_config.atoms[1, 1], xyz_config.atoms[1, 2]))
            fd.write("H {:.10f} {:.10f} {:.10f}\n".format(xyz_config.atoms[2, 0], xyz_config.atoms[2, 1], xyz_config.atoms[2, 2]))
            fd.write("H {:.10f} {:.10f} {:.10f}\n".format(xyz_config.atoms[3, 0], xyz_config.atoms[3, 1], xyz_config.atoms[3, 2]))
            fd.write("N {:.10f} {:.10f} {:.10f}\n".format(xyz_config.atoms[4, 0], xyz_config.atoms[4, 1], xyz_config.atoms[4, 2]))
            fd.write("N {:.10f} {:.10f} {:.10f}\n".format(xyz_config.atoms[5, 0], xyz_config.atoms[5, 1], xyz_config.atoms[5, 2]))
            fd.write("C {:.10f} {:.10f} {:.10f}\n".format(xyz_config.atoms[6, 0], xyz_config.atoms[6, 1], xyz_config.atoms[6, 2]))
            k = k + 1

    fd.close()
    logging.info("Found {} configs.".format(k))

