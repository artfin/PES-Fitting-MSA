import numpy as np

from dataset import load_xyz_with_energy, load_xyz_with_dipole, load_npz, XYZConfig

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


if __name__ == "__main__":
    natoms, nconfigs1, xyz_configs_en = load_xyz_with_energy("datasets/raw/n2-ar/N2-AR-EN-TQ5-CBS.xyz")
    natoms, nconfigs2, xyz_configs_dip = load_xyz_with_dipole("datasets/raw/N2-AR-DIP-VTZ.xyz")
    assert nconfigs1 == nconfigs2
    nconfigs = nconfigs1

    def find_config(configs, c):
        for cc in configs:
            if np.allclose(cc.coords, c.coords):
                return cc

        return None

    z1 = np.vstack((xyz_configs_en[0].z[0:2]))
    z2 = np.asarray(xyz_configs_en[0].z[2])

    R1, R2 = [], []
    energy, dipole = [], []

    from tqdm import tqdm
    for ind, c in enumerate(tqdm(xyz_configs_en)):
        cc = find_config(xyz_configs_dip, c)
        if cc is None:
            print(c)
            assert False

        R1.append(c.coords[:2, :])
        R2.append([c.coords[2, :]])
        energy.append(c.energy)
        dipole.append(cc.dipole)

    energy = np.asarray(energy).reshape((nconfigs, 1))
    dipole = np.asarray(dipole).reshape((nconfigs, 3))

    npz_path = "N2-AR-EN-DIP-NONRIGID.npz"
    print("Writing NPZ file to file={}".format(npz_path))
    np.savez("N2-AR-EN-DIP-NONRIGID.npz", nmol=2, E=energy, D=dipole, z1=z1, z2=z2, R1=R1, R2=R2)

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

if __name__ == "__main4__":
    from dataset import load_npz
    from random import sample

    fpath = "datasets/raw/ethanol/ethanol_dft.npz"
    natoms, nconfigs, xyz_configs = load_npz(fpath, load_forces=True)

    print(natoms, nconfigs)
    #print(xyz_configs)

    #seq = sample(xyz_configs, k=50000)
    nconfigs = 100
    seq = xyz_configs[:nconfigs]

    E = [c.energy for c in seq]
    F = [c.forces for c in seq]
    R = [c.coords for c in seq]
    z = seq[0].z

    npz_path = "ethanol_dft-{}.npz".format(nconfigs)
    np.savez(npz_path, nmol=1, E=E, F=F, R=R, z=z)
    print("Saving {} configs to npz_path={}".format(nconfigs, npz_path))

if __name__ == "__main3__":
    fname = "datasets/raw/n2-ar/N2-AR-EN-VQZ.xyz"
    natoms, nconfigs, xyz_configs = load_xyz_with_energy(fname)

    write_npz_n2_ar("N2-AR-EN-VQZ.npz", natoms, nconfigs, xyz_configs)

if __name__ == "__main2__":
    fname = "datasets/raw/CH4-N2-EN-RIGID-10000-CORRECTED.xyz"
    natoms, nconfigs, xyz_configs = load_xyz_with_energy(fname)

    write_npz_ch4_n2("CH4-N2-EN-RIGID-10000-CORRECTED.npz", natoms, nconfigs, xyz_configs)



