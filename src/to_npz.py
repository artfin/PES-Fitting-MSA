import numpy as np
import math

from dataset import load_xyz_with_energy, load_xyz_with_dipole, load_npz, XYZConfig

def write_npz_co2_ar(npz_path, natoms, nconfigs, xyz_configs):
    energy = np.array([xyz_config.energy for xyz_config in xyz_configs]).reshape((nconfigs, 1))
    z1 = np.vstack((xyz_configs[0].z[0], xyz_configs[0].z[1], xyz_configs[0].z[2]))
    z2 = np.vstack((xyz_configs[0].z[3]))

    R1, R2 = [], []
    for xyz_config in xyz_configs:
        CO2 = xyz_config.coords[0:3, :]
        Ar  = [xyz_config.coords[3, :]]
        R1.append(CO2)
        R2.append(Ar)

    R1 = np.asarray(R1)
    R2 = np.asarray(R2)

    np.savez(npz_path, nmol=2, E=energy, z1=z1, z2=z2, R1=R1, R2=R2)

def write_npz_h2o_h2o(npz_path, natoms, nconfigs, xyz_configs):
    energy = np.array([xyz_config.energy for xyz_config in xyz_configs]).reshape((nconfigs, 1))
    z1 = np.vstack((xyz_configs[0].z[0], xyz_configs[0].z[1], xyz_configs[0].z[2]))
    z2 = np.vstack((xyz_configs[0].z[3], xyz_configs[0].z[4], xyz_configs[0].z[5]))

    R1, R2 = [], []
    for xyz_config in xyz_configs:
        H2O_1 = np.vstack((xyz_config.coords[0, :], xyz_config.coords[1, :], xyz_config.coords[2, :]))
        H2O_2 = np.vstack((xyz_config.coords[3, :], xyz_config.coords[4, :], xyz_config.coords[5, :]))
        R1.append(H2O_1)
        R2.append(H2O_2)

    R1 = np.asarray(R1)
    R2 = np.asarray(R2)

    np.savez(npz_path, nmol=2, E=energy, z1=z1, z2=z2, R1=R1, R2=R2)

def write_npz_ch4_n2(npz_path, natoms, nconfigs, xyz_configs):
    energy = np.array([xyz_config.energy for xyz_config in xyz_configs]).reshape((nconfigs, 1))
    z1 = np.vstack((xyz_configs[0].z[0:4], xyz_configs[0].z[6]))
    z2 = np.asarray(xyz_configs[0].z[4:6])

    R1, R2 = [], []
    for xyz_config in xyz_configs:
        CH4 = np.vstack((xyz_config.coords[0:4, :], xyz_config.coords[6, :]))
        N2  = xyz_config.coords[4:6]
        R1.append(CH4)
        R2.append(N2)

    R1 = np.asarray(R1)
    R2 = np.asarray(R2)

    np.savez(npz_path, nmol=2, E=energy, z1=z1, z2=z2, R1=R1, R2=R2)

def write_npz_ch4_n2_dipole(npz_path, natoms, nconfigs, xyz_configs):
    energy = np.array([xyz_config.energy for xyz_config in xyz_configs]).reshape((nconfigs, 1))
    dipole = np.array([xyz_config.dipole for xyz_config in xyz_configs]).reshape((nconfigs, 3))
    z1 = np.vstack((xyz_configs[0].z[0:4], xyz_configs[0].z[6]))
    z2 = np.asarray(xyz_configs[0].z[4:6])

    R1, R2 = [], []
    for xyz_config in xyz_configs:
        CH4 = np.vstack((xyz_config.coords[0:4, :], xyz_config.coords[6, :]))
        N2  = xyz_config.coords[4:6]
        R1.append(CH4)
        R2.append(N2)

    R1 = np.asarray(R1)
    R2 = np.asarray(R2)

    import random
    nconfigs = len(xyz_configs)
    ind = random.choices(range(nconfigs), k=100)
    np.savez(npz_path, nmol=2, E=energy[ind], D=dipole[ind], z1=z1, z2=z2, R1=R1[ind], R2=R2[ind])
    #np.savez(npz_path, nmol=2, E=energy, D=dipole, z1=z1, z2=z2, R1=R1, R2=R2)

def write_npz_n2_ar(npz_path, natoms, nconfigs, xyz_configs):
    energy = np.array([xyz_config.energy for xyz_config in xyz_configs]).reshape((nconfigs, 1))
    z1 = np.asarray(xyz_configs[0].z[0:2])
    z2 = np.asarray(xyz_configs[0].z[2])

    R1, R2 = [], []
    for xyz_config in xyz_configs:
        N2 = xyz_config.coords[:2, :]
        Ar = [xyz_config.coords[2, :]]
        R1.append(N2)
        R2.append(Ar)

    R1 = np.asarray(R1)
    R2 = np.asarray(R2)

    np.savez(npz_path, nmol=2, E=energy, z1=z1, z2=z2, R1=R1, R2=R2)

def prepare_n2_ar_dipole_deriv_file():
    natoms, nconfigs1, xyz_configs_en = load_xyz_with_energy("datasets/raw/n2-ar/N2-AR-EN-TQ5-CBS-RIGID.xyz")
    natoms, nconfigs2, xyz_configs_dip = load_xyz_with_dipole("datasets/raw/n2-ar/N2-AR-DIP-PREP-DERIVATIVE-2.xyz")
    nconfigs = nconfigs1

    def find_config_with_n2_length(configs, c, l_n2):
        curr_n2_len = np.linalg.norm(c.coords[0, :] - c.coords[1, :])
        c_to_find = c.coords.copy()
        c_to_find[0, :] *= l_n2 / curr_n2_len
        c_to_find[1, :] *= l_n2 / curr_n2_len

        for cc in configs:
            if np.allclose(cc.coords, c_to_find):
                return cc

        return None

    #z1 = np.vstack((xyz_configs_en[0].z[0:2]))
    z1 = np.vstack((xyz_configs_en[0].z[0:2], -1, -1))
    z2 = np.asarray(xyz_configs_en[0].z[2])

    R1, R2 = [], []
    energy, dipole = [], []

    from tqdm import tqdm
    for ind, c in enumerate(xyz_configs_en):
        cc_contr = find_config_with_n2_length(xyz_configs_dip, c, l_n2=2.077567491)
        cc_elong = find_config_with_n2_length(xyz_configs_dip, c, l_n2=2.079567491)

        if cc_contr is None or cc_elong is None:
            print(c)
            assert False

        delta = 0.001 # bohr
        dip_derivative = (cc_elong.dipole - cc_contr.dipole) / (2.0 * delta)
        dip_derivative[1] = 0.0
        #print("dip_elong: {}; dip_contr: {}; dip_derivative: {}".format(cc_elong.dipole, cc_contr.dipole, dip_derivative))

        R = c.coords[2, 2]
        l_n2 = np.linalg.norm(c.coords[0, :])
        theta = math.atan2(c.coords[0, 0] / l_n2, c.coords[0, 2] / l_n2)

        #if abs(theta - 1.338) < 0.05:
        #    print("{:.5f} {:.5f} {:.6f} {:.6f}".format(R, theta / np.pi * 180.0, dip_derivative[0], dip_derivative[2]))

        #if abs(R - 7.0) < 0.05:
        #    print("{:.5f} {:.5f} {:.6f} {:.6f}".format(R, theta, dip_derivative[0], dip_derivative[2]))

        R1.append([
            [c.coords[0, 0], c.coords[0, 1],  c.coords[0, 2]],
            [c.coords[1, 0], c.coords[1, 1],  c.coords[1, 2]],
            [c.coords[0, 2], c.coords[0, 1], -c.coords[0, 0]],
            [c.coords[1, 2], c.coords[1, 1], -c.coords[1, 0]],
        ])

        R2.append([c.coords[2, :]])
        energy.append(c.energy)
        dipole.append(dip_derivative)

    energy = np.asarray(energy).reshape((nconfigs, 1))
    dipole = np.asarray(dipole).reshape((nconfigs, 3))

    npz_path = "N2-AR-EN-DIP-DERIVATIVE-RIGID.npz"
    print("Writing NPZ file to file={}".format(npz_path))
    np.savez(npz_path, nmol=2, E=energy, D=dipole, z1=z1, z2=z2, R1=R1, R2=R2)

    import os
    nbytes = os.path.getsize(npz_path)
    print("Succesfully written {} bytes = {:.3g}K = {:.3g}M".format(nbytes, nbytes/1024, nbytes/1024**2))


def prepare_n2_ar_npz_file():
    natoms, nconfigs1, xyz_configs_en = load_xyz_with_energy("datasets/raw/n2-ar/N2-AR-EN-TQ5-CBS.xyz")
    natoms, nconfigs2, xyz_configs_dip = load_xyz_with_dipole("datasets/raw/n2-ar/N2-AR-DIP-VTZ-RIGID.xyz")
    #assert nconfigs1 == nconfigs2, "nconfigs1: {}; nconfigs2: {}".format(nconfigs1, nconfigs2)
    nconfigs = nconfigs2

    def find_config(configs, c):
        for cc in configs:
            if np.allclose(cc.coords, c.coords):
                return cc

        return None

    #z1 = np.vstack((xyz_configs_en[0].z[0:2]))
    z1 = np.vstack((xyz_configs_en[0].z[0:2], -1, -1))
    z2 = np.asarray(xyz_configs_en[0].z[2])

    R1, R2 = [], []
    energy, dipole = [], []

    from tqdm import tqdm
    for ind, c in enumerate(tqdm(xyz_configs_dip)):
        cc = find_config(xyz_configs_en, c)
        if cc is None:
            print(c)
            assert False

        # additional points
        R1.append([
            [c.coords[0, 0], c.coords[0, 1],  c.coords[0, 2]],
            [c.coords[1, 0], c.coords[1, 1],  c.coords[1, 2]],
            [c.coords[0, 2], c.coords[0, 1], -c.coords[0, 0]],
            [c.coords[1, 2], c.coords[1, 1], -c.coords[1, 0]],
        ])

        #R1.append(c.coords[:2, :])
        R2.append([c.coords[2, :]])
        energy.append(cc.energy)
        dipole.append(c.dipole)

    energy = np.asarray(energy).reshape((nconfigs, 1))
    dipole = np.asarray(dipole).reshape((nconfigs, 3))

    npz_path = "N2-AR-EN-DIP-RIGID.npz"
    print("Writing NPZ file to file={}".format(npz_path))
    np.savez(npz_path, nmol=2, E=energy, D=dipole, z1=z1, z2=z2, R1=R1, R2=R2)

    import os
    nbytes = os.path.getsize(npz_path)
    print("Succesfully written {} bytes = {:.3g}K = {:.3g}M".format(nbytes, nbytes/1024, nbytes/1024**2))

def prepare_h2_ar_npz_file():
    natoms, nconfigs, xyz_configs = load_xyz_with_dipole("datasets/raw/h2-ar/H2-AR-DIP-KALUGINA-EFFQUAD.xyz")
    dipole = np.array([xyz_config.dipole for xyz_config in xyz_configs]).reshape((nconfigs, 3))
    energy = np.array([0.0 for xyz_config in xyz_configs]).reshape((nconfigs, 1))

    z1 = np.asarray(xyz_configs[0].z[0:4])
    z2 = np.asarray(xyz_configs[0].z[4])

    R1, R2 = [], []
    for xyz_config in xyz_configs:
        H2 = xyz_config.coords[:4, :]
        Ar = [xyz_config.coords[4, :]]
        R1.append(H2)
        R2.append(Ar)

    R1 = np.asarray(R1)
    R2 = np.asarray(R2)

    npz_path = "H2-AR-EN-DIP-KALUGINA-EFFQUAD.npz"
    print("Writing NPZ file to file={}".format(npz_path))
    np.savez(npz_path, nmol=2, E=energy, D=dipole, z1=z1, z2=z2, R1=R1, R2=R2)

    import os
    nbytes = os.path.getsize(npz_path)
    print("Succesfully written {} bytes = {:.3g}K = {:.3g}M".format(nbytes, nbytes/1024, nbytes/1024**2))


if __name__ == "__main5__":
    fname = "datasets/raw/CH4-N2-DIPOLES.xyz"
    natoms, nconfigs, xyz_configs1 = load_xyz_with_dipole(fname)

    fname = "datasets/raw/ch4-n2/CH4-N2-EN-RIGID-CORRECTED.xyz"
    natoms, nconfig, xyz_configs2 = load_xyz_with_energy(fname)

    for n in range(nconfigs):
        if not np.allclose(xyz_configs1[n].coords, xyz_configs2[n].coords):
            print(n)
            print(xyz_configs1[n].coords)
            print(xyz_configs2[n].coords)
            assert False

        xyz_configs1[n].energy = xyz_configs2[n].energy

    xyz_configs = xyz_configs1
    write_npz_ch4_n2_dipole("CH4-N2-EN-DIP-RIGID.npz", natoms, nconfigs, xyz_configs)

if __name__ == "__main10__":
    from dataset import load_npz
    from random import sample

    fpath = "datasets/raw/ethanol/ethanol_dft.npz"
    natoms, nconfigs, xyz_configs = load_npz(fpath, load_forces=True)

    print(natoms, nconfigs)
    #print(xyz_configs)

    #seq = sample(xyz_configs, k=50000)
    nconfigs = 25000
    seq = xyz_configs[:nconfigs]

    E = [c.energy for c in seq]
    F = [c.forces for c in seq]
    R = [c.coords for c in seq]
    z = seq[0].z

    npz_path = "ethanol_dft-{}.npz".format(nconfigs)
    np.savez(npz_path, nmol=1, E=E, F=F, R=R, z=z)
    print("Saving {} configs to npz_path={}".format(nconfigs, npz_path))

if __name__ == "__main5__":
    fname = "datasets/raw/n2-ar/N2-AR-EN-F12b+MBF.xyz"
    natoms, nconfigs, xyz_configs = load_xyz_with_energy(fname)

    write_npz_n2_ar("N2-AR-EN-F12b+MBF.npz", natoms, nconfigs, xyz_configs)

if __name__ == "__main2__":
    fname = "datasets/raw/CH4-N2-EN-RIGID-10000-CORRECTED.xyz"
    natoms, nconfigs, xyz_configs = load_xyz_with_energy(fname)

    write_npz_ch4_n2("CH4-N2-EN-RIGID-10000-CORRECTED.npz", natoms, nconfigs, xyz_configs)

if __name__ == "__main12__":
    fname = "datasets/raw/h2o-h2o/h2o-h2o.xyz"
    natoms, nconfigs, xyz_configs = load_xyz_with_energy(fname)

    write_npz_h2o_h2o("h2o-h2o.npz", natoms, nconfigs, xyz_configs)


if __name__ == "__main__":
    filepath = "datasets/raw/co2-ar/CCSD-T.xyz"
    natoms, nconfigs, xyz_configs = load_xyz_with_energy(filepath)

    write_npz_co2_ar("CCSD-T.npz", natoms, nconfigs, xyz_configs)


###################################################################################3

import argparse
import logging
import sys

SYMBOLS = ('H', 'He',\
           'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',\
           'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar')

if __name__ == "__main11__":
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.handlers = []

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.INFO)

    xyz_path_en = "datasets/raw/n2-ar/N2-AR-EN-F12b+MBF.xyz"
    xyz_path_dip = "datasets/raw/n2-ar/N2-AR-DIP-F12b+MBF.xyz"

    logging.info("XYZ file with energies: {}".format(xyz_path_en))
    logging.info("XYZ file with dipoles:  {}".format(xyz_path_dip))

    natoms, nconfigs, xyz_en_configs = load_xyz_with_energy(xyz_path_en)
    natoms, nconfigs, xyz_dip_configs = load_xyz_with_dipole(xyz_path_dip)

    z1 = np.array([[7.0], [7.0], [-1.0], [-1.0]])
    z2 = np.array([[18.0]])

    R1, R2 = [], []
    energy, dipole = [], []
    nconfigs = 0

    SELECT_SHORT_RANGE = True

    for xyze, xyzd in zip(xyz_en_configs, xyz_dip_configs):
        assert np.allclose(xyze.coords, xyzd.coords, atol=1e-15)

        mon1 = xyze.coords[0:2, :]
        mon2 = [xyze.coords[2, :]]

        if SELECT_SHORT_RANGE:
            if mon2[0][2] > 10.0:
                logging.info("Long-range point: R = {}".format(mon2[0][2]))
                continue
        else:
            if mon2[0][2] <= 10.0:
                logging.info("Short-range point: R = {}".format(mon2[0][2]))
                continue

        R1.append([
            [mon1[0, 0], mon1[0, 1],  mon1[0, 2]],
            [mon1[1, 0], mon1[1, 1],  mon1[1, 2]],
            [mon1[0, 2], mon1[0, 1], -mon1[0, 0]],
            [mon1[1, 2], mon1[1, 1], -mon1[1, 0]],
        ])

        R2.append(mon2)

        energy.append(xyze.energy)
        dipole.append(xyzd.dipole)
        nconfigs += 1

    R1 = np.asarray(R1)
    R2 = np.asarray(R2)

    energy = np.array(energy).reshape((nconfigs, 1))
    dipole = np.array(dipole).reshape((nconfigs, 3))

    if SELECT_SHORT_RANGE:
        npz_path = "datasets/raw/n2-ar/N2-AR-EN-DIP-F12b+MBF-EFFQUAD-SHORT.npz"
    else:
        npz_path = "datasets/raw/n2-ar/N2-AR-EN-DIP-F12b+MBF-EFFQUAD-LONG.npz"

    np.savez(npz_path, nmol=2, E=energy, D=dipole, z1=z1, z2=z2, R1=R1, R2=R2)

    import os
    nbytes = os.path.getsize(npz_path)
    logging.info(f"Succesfully written nconfigs={nconfigs} to {npz_path} :: {nbytes} bytes = {nbytes/1024:.3g}K = {nbytes/1024**2:.3g}M")


