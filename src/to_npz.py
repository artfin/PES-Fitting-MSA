import numpy as np

from dataset import load_xyz_with_energy

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



