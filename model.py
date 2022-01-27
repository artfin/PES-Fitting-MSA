# TODO 1: apply normalization to polynomial values and energies

import logging
import os

import ctypes as ct
import numpy as np

# Physical constants
BOHRTOANG = 0.52917721067
HTOCM     = 2.194746313702e5

# parameter inside the `yij` exp(...)
a0        = 2.0

from itertools import combinations
from collections import namedtuple
Config = namedtuple('Config', ['atoms', 'energy'])

import torch
from torch.utils.data import Dataset, DataLoader, random_split

torch.manual_seed(42)
np.random.seed(42)

class PolyDataset(Dataset):
    def __init__(self, wdir, config_fname, order, symmetry):
        self.wdir = wdir
        logging.info("working directory: {}".format(self.wdir))

        self.config_fname = os.path.join(self.wdir, config_fname)
        logging.info("configuration file: {}".format(self.config_fname))
        logging.info("loading configurations...")
        configs = self.load()

        logging.info("preparing the coordinates..")
        self.NDIS = self.NATOMS * (self.NATOMS - 1) // 2
        yij       = self.make_yij(configs)
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
            poly[n, :] = p.reshape((self.NPOLY, )).copy()

        logging.info("Done.")

        energies = np.asarray([c.energy for c in configs]).reshape((self.NCONFIGS, 1)) # (NCONFIGS,) -> (NCONFIGS, 1)

        #poly_mean = np.mean(poly, axis=0)
        #poly_std  = np.std(poly, axis=0)
        #poly_stat = np.vstack((poly_mean, poly_std)).T
        #np.savetxt("poly_stats.txt", poly_stat)

        self.X = poly
        self.y = energies

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X[idx], self.y[idx]]

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
        yij = np.zeros((self.NCONFIGS, self.NDIS), order="F")

        for n in range(self.NCONFIGS):
            c = configs[n]

            k = 0
            for i, j in combinations(range(self.NATOMS), 2):
                #if i == 0 and j == 1: yij[n, k] = 0.0; k = k + 1; continue; # H1 H2 
                #if i == 0 and j == 2: yij[n, k] = 0.0; k = k + 1; continue; # H1 H3
                #if i == 0 and j == 3: yij[n, k] = 0.0; k = k + 1; continue; # H1 H4
                #if i == 1 and j == 2: yij[n, k] = 0.0; k = k + 1; continue; # H2 H3
                #if i == 1 and j == 3: yij[n, k] = 0.0; k = k + 1; continue; # H2 H4
                #if i == 2 and j == 3: yij[n, k] = 0.0; k = k + 1; continue; # H3 H4
                #if i == 0 and j == 6: yij[n, k] = 0.0; k = k + 1; continue; # H1 C
                #if i == 1 and j == 6: yij[n, k] = 0.0; k = k + 1; continue; # H2 C
                #if i == 2 and j == 6: yij[n, k] = 0.0; k = k + 1; continue; # H3 C
                #if i == 3 and j == 6: yij[n, k] = 0.0; k = k + 1; continue; # H4 C
                #if i == 4 and j == 5: yij[n, k] = 0.0; k = k + 1; continue; # N1 N2

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


# Jun Li, Bin Jiang, and Hua Guo
# J. Chem. Phys. 139, 204103 (2013); https://doi.org/10.1063/1.4832697
# Suggest using Tanh activation function and 2 hidden layers

import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

class FCNet(nn.Module):
    def __init__(self, NPOLY, n_layers=2, activation=nn.Tanh, init_form="uniform"):
        super().__init__()

        self.NPOLY      = NPOLY
        self.n_layers   = n_layers
        self.activation = activation()
        self.init_form  = init_form

        layers = [
            nn.Linear(self.NPOLY, 20), self.activation,
            nn.Linear(20, 20),    self.activation,
            nn.Linear(20, 1)
        ]

        self.layers = nn.Sequential(*layers)

        if isinstance(self.activation, nn.ReLU):
            self.init_kaiming(activation_str="relu")
        elif isinstance(self.activation, nn.Tanh):
            self.init_xavier(activation_str="tanh")
        elif isinstance(self.activation, nn.Sigmoid):
            self.init_xavier(activation_str="sigmoid")
        else:
            raise NotImplementedError()

    def forward(self, x):
        x = x.view(-1, self.NPOLY)
        return self.layers(x)

    def init_xavier(self, activation_str):
        sigmoid_gain = nn.init.calculate_gain(activation_str)
        for child in self.layers.children():
            if isinstance(child, nn.Linear):
                for _ in range(0, self.n_layers - 1):
                    if self.init_form == "normal":
                        nn.init.xavir_normal_(child.weight, gain=sigmoid_gain)
                        if child.bias is not None:
                            nn.init.zeros_(child.bias)
                    elif self.init_form == "uniform":
                        nn.init.xavier_uniform_(child.weight, gain=sigmoid_gain)
                        if child.bias is not None:
                            nn.init.zeros_(child.bias)

                    else:
                        raise NotImplementedError()

    def init_kaiming(self, activation_str):
        raise NotImplementedError()


def validate(model, validation_loader, criterion):
    # in case the structure of the model changes
    model.eval()

    cumloss = 0.0
    with torch.no_grad():
        for poly_batch, energy_batch in validation_loader:
            output = model(poly_batch)
            loss = criterion(output, energy_batch)
            cumloss += loss

    return cumloss / len(validation_loader)


def train(model, trainset, testset, epochs=20):
    # in case the structure of the model changes
    model.train()

    bs = 1000
    train_loader = DataLoader(trainset, batch_size=bs, shuffle=True)
    test_loader  = DataLoader(testset,  batch_size=bs, shuffle=True)

    criterion = nn.MSELoss()

    # TODO: also try SGD
    # optimizer = optim.SGD(model.parameters(), lr=1e-4)
    # weight decay???
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    for epoch in tqdm(range(epochs)):
        for poly_batch, energy_batch in train_loader:
            optimizer.zero_grad()
            output = model(poly_batch)
            loss = criterion(output, energy_batch)
            loss.backward()
            optimizer.step()

        val_loss = validate(model, test_loader, criterion)
        print("Epoch: {}; Loss: {:.10f}; Val loss: {:.10f}".format(
            epoch, loss.item(), val_loss
        ))

def count_params(model):
    nparams = 0
    for name, param in model.named_parameters():
        params = torch.tensor(param.size())
        nparams += torch.prod(params, 0)

    return nparams


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    dataset = PolyDataset(wdir='./H2-H2O', config_fname='points.dat', order="3", symmetry="2 2 1")

    # TODO: introduce validation set
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size])

    logging.info("Train size = {}".format(train_size))
    logging.info("Test size  = {}".format(test_size))

    n_layers   = 2
    activation = nn.Tanh
    init_form  = "uniform"
    logging.info("Creating a fully connected neural network:")
    logging.info("    n_layers   = {}".format(n_layers))
    logging.info("    activation = {}".format(activation))
    logging.info("    init_form  = {}".format(init_form))

    model = FCNet(NPOLY=dataset.NPOLY, n_layers=n_layers, activation=activation, init_form=init_form)
    model.double() # use double precision for model parameters 

    nparams = count_params(model)
    print("Number of parameters: {}".format(nparams))

    train(model, trainset, testset, epochs=100)
