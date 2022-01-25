# TODO 1: apply normalization to polynomial values and energies
# TODO 2: zero out intermolecular `yij`-coordinates (see .cpp) 


import logging

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

# TODO: parse from data files
NCONFIGS = 71610

# MAX ORD = 4
# NMON    = 2892
# NPOLY   = 650

# MAX ORD = 2
NMON  = 140
NPOLY = 32

class PolyDataset(Dataset):
    # molecular system parameters
    NATOMS = 7
    NDIS = NATOMS * (NATOMS - 1) // 2

    def __init__(self):
        logging.info("Loading configurations...")

        CONFIG_FNAME = './ch4-n2-energies.xyz'
        configs = self.load(CONFIG_FNAME)

        logging.info("Preparing the coordinates..")
        yij = self.make_yij(configs)
        logging.info("Done.")

        self.setup_fortran_procs()

        x = np.zeros((self.NDIS, 1), order="F")
        m = np.zeros((NMON, 1), order="F")
        p = np.zeros((NPOLY, 1), order="F")
        poly = np.zeros((NCONFIGS, NPOLY))

        logging.info("Preparing the polynomials...")

        for n in range(0, NCONFIGS):
            x = yij[n, :].copy()

            x_ptr = x.ctypes.data_as(ct.POINTER(ct.c_double))
            m_ptr = m.ctypes.data_as(ct.POINTER(ct.c_double))
            self.evmono(x_ptr, m_ptr)

            p_ptr = p.ctypes.data_as(ct.POINTER(ct.c_double))
            self.evpoly(m_ptr, p_ptr)
            poly[n, :] = p.reshape((NPOLY, )).copy()

        logging.info("Done.")

        energies = np.asarray([c.energy for c in configs]).reshape((NCONFIGS, 1))

        self.X = poly
        self.y = energies

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X[idx], self.y[idx]]

    def load(self, fname):
        configs = []
        with open(fname, mode='r') as inp:
            for i in range(NCONFIGS):
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
        yij = np.zeros((NCONFIGS, self.NDIS), order="F")

        for n in range(NCONFIGS):
            c = configs[n]

            k = 0
            for i, j in combinations(range(self.NATOMS), 2):
                yij[n, k] = np.linalg.norm(c.atoms[i] - c.atoms[j])
                yij[n][k] /= BOHRTOANG
                yij[n][k] = np.exp(-yij[n, k]/ a0)
                k = k + 1

        return yij

    def setup_fortran_procs(self):
        LIBNAME = './basis-maxord-2.so'
        basislib = ct.CDLL(LIBNAME)

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
    def __init__(self, n_layers=2, activation=nn.Tanh, init_form="uniform"):
        super().__init__()

        self.n_layers = n_layers
        self.activation = activation()
        self.init_form = init_form

        layers = [
            nn.Linear(NPOLY, 30), self.activation,
            nn.Linear(30, 70),    self.activation,
            nn.Linear(70, 1)
        ]

        #for _ in range(0, self.n_layers - 1):
        #    layers.append(nn.Linear(40, 40))
        #    layers.append(self.activation)
        #layers.append(nn.Linear(40, 1))

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
        x = x.view(-1, NPOLY)
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

    bs = 310
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
    logging.basicConfig(level=logging.INFO)

    dataset = PolyDataset()

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

    model = FCNet(n_layers=n_layers, activation=activation, init_form=init_form)
    model.double() # use double precision for model parameters 

    nparams = count_params(model)
    print("Number of parameters: {}".format(nparams))

    train(model, trainset, testset, epochs=20)
