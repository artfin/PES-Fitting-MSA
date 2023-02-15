import itertools
import collections
import json
import logging
import numpy as np
import os

import sklearn.model_selection
import torch

from dataset import PolyDataset

import pathlib
BASEDIR = pathlib.Path(__file__).parent.parent.resolve()

import pprint
pp = pprint.PrettyPrinter(indent=4)

def prepare_qmodel_structure(sg):
    """
    partial charge model structure:
        qmodel = {
            symmetry group [string] -> indices of atoms
        }
    """
    qmodel_s = {}
    def add_sg(sg, natom):
        sg = " ".join([c for c in sg])
        if sg in qmodel_s:
            qmodel_s[sg].append(natom)
        else:
            qmodel_s[sg] = [natom]

    if " " in sg:
        sg = sg.replace(" ", "")

    # current atom
    ca = 0

    for ind in range(len(sg)):
        n = int(sg[ind])

        if n == 1:
            add_sg(sg, ca)
            ca = ca + 1
        elif n > 1:
            # sgc -- symmetry group for current atom's partial charge
            sgc = sg[:ind] + "1" * n + sg[(ind + 1):]
            for k in range(n):
                add_sg(sgc, ca)
                ca = ca + 1
        else:
            assert False, "unreachable"

    return qmodel_s

def make_dataset_fpaths(cfg_dataset):
    def flatten(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    # keywords to be considered to make a hash
    keywords = ('SOURCE', 'LOAD_FORCES', 'ENERGY_LIMIT', 'NORMALIZE', 'ANCHOR_POSITIONS', 'ORDER', 'SYMMETRY', 'VARIABLES', 'PURIFY')
    d = flatten({kw : cfg_dataset[kw] for kw in keywords})

    from hashlib import sha1
    _hash = sha1(repr(sorted(d.items())).encode('utf-8')).hexdigest() # sorting dictionary makes the hash consistent across runs (since dictionary is unordered collection)

    train_fpath = os.path.join(cfg_dataset['INTERIM_FOLDER'], f"{cfg_dataset['NAME']}-train-{_hash}.pk")
    val_fpath   = os.path.join(cfg_dataset['INTERIM_FOLDER'], f"{cfg_dataset['NAME']}-val-{_hash}.pk")
    test_fpath  = os.path.join(cfg_dataset['INTERIM_FOLDER'], f"{cfg_dataset['NAME']}-test-{_hash}.pk")

    return train_fpath, val_fpath, test_fpath

def rot_z(ang):
    """
    rotation matrix around OZ
    """
    return np.array([
        [np.cos(ang), -np.sin(ang), 0.0],
        [np.sin(ang),  np.cos(ang), 0.0],
        [        0.0,          0.0, 1.0],
    ])

def rot_y(ang):
    """
    rotation matrix around OY
    """
    return np.array([
        [ np.cos(ang), 0.0, np.sin(ang)],
        [         0.0, 1.0,         0.0],
        [-np.sin(ang), 0.0, np.cos(ang)],
    ])

def make_dataset(cfg_dataset, dataset_fpaths):
    # default split function [dataset] -> [train, val, test] 
    data_split = sklearn.model_selection.train_test_split

    if cfg_dataset['LOAD_FORCES']:
        if len(cfg_dataset['SOURCE']) > 1:
            logging.info("Stratification is not implemented for `use_forces=True`")
            assert False

        dataset = PolyDataset(wdir=cfg_dataset['EXTERNAL_FOLDER'], typ=cfg_dataset['TYPE'], file_path=cfg_dataset['SOURCE'][0],
                              order=cfg_dataset['ORDER'], load_forces=True, symmetry=cfg_dataset['SYMMETRY'],
                              atom_mapping=cfg_dataset['ATOM_MAPPING'], variables=cfg_dataset['VARIABLES'], purify=cfg_dataset['PURIFY'])

        NCONFIGS = dataset.X.size()[0]
        indices = list(range(NCONFIGS))

        train_ind, test_ind = data_split(indices, random_state=42, test_size=0.2)
        val_ind, test_ind   = data_split(test_ind, random_state=42, test_size=0.5)

        X_train, y_train = dataset.X[train_ind, :], dataset.y[train_ind]
        X_val, y_val     = dataset.X[val_ind,   :], dataset.y[val_ind]
        X_test, y_test   = dataset.X[test_ind,  :], dataset.y[test_ind]

        print("Memory size of dX: {} GB", dataset.dX.element_size() * dataset.dX.nelement() / (1024 * 1024 * 1024))

        dX_train, dy_train = dataset.dX[train_ind, :, :], dataset.dy[train_ind, :, :]
        dX_val, dy_val     = dataset.dX[val_ind,   :, :], dataset.dy[val_ind,   :, :]
        dX_test, dy_test   = dataset.dX[test_ind,  :, :], dataset.dy[test_ind,  :, :]

        logging.info("Saving train dataset to: {}".format(dataset_fpaths["train"]))
        torch.save(dict(NATOMS=dataset.NATOMS, NMON=dataset.NMON, NPOLY=dataset.NPOLY, symmetry=cfg_dataset['SYMMETRY'], order=dataset.order,
                        energy_limit=None, variables=cfg_dataset['VARIABLES'], purify=cfg_dataset['PURIFY'], X=X_train, y=y_train,
                        dX=dX_train, dy=dy_train, xyz_ordered=None, grm=None), dataset_fpaths["train"])

        logging.info("Saving val dataset to: {}".format(dataset_fpaths["val"]))
        torch.save(dict(NATOMS=dataset.NATOMS, NMON=dataset.NMON, NPOLY=dataset.NPOLY, symmetry=cfg_dataset['SYMMETRY'], order=dataset.order,
                        energy_limit=None, variables=cfg_dataset['VARIABLES'], purify=cfg_dataset['PURIFY'], X=X_val, y=y_val,
                        dX=dX_val, dy=dy_val, xyz_ordered=None, grm=None), dataset_fpaths["val"])

        logging.info("Saving test dataset to: {}".format(dataset_fpaths["val"]))
        torch.save(dict(NATOMS=dataset.NATOMS, NMON=dataset.NMON, NPOLY=dataset.NPOLY, symmetry=cfg_dataset['SYMMETRY'], order=dataset.order,
                        energy_limit=None, variables=cfg_dataset['VARIABLES'], purify=cfg_dataset['PURIFY'], X=X_test, y=y_test,
                        dX=dX_test, dy=dy_test, xyz_ordered=None, grm=None), dataset_fpaths["test"])

    elif cfg_dataset['TYPE'] == 'DIPOLE':
        if len(cfg_dataset['SOURCE']) > 1:
            logging.info("Stratification is not implemented for `typ=dipole`")
            assert False

        anchor_pos = cfg_dataset['ANCHOR_POSITIONS']
        assert len(anchor_pos) == 3

        dataset = PolyDataset(wdir=cfg_dataset['EXTERNAL_FOLDER'], typ=cfg_dataset['TYPE'], file_path=cfg_dataset['SOURCE'][0],
                              order=cfg_dataset['ORDER'], load_forces=False, symmetry=cfg_dataset['SYMMETRY'],
                              atom_mapping=cfg_dataset['ATOM_MAPPING'], variables=cfg_dataset['VARIABLES'], purify=cfg_dataset['PURIFY'],
                              anchor_pos=cfg_dataset['ANCHOR_POSITIONS'])


        nconfigs = len(dataset.xyz_configs)
        y = np.zeros((nconfigs, 4))

        for k in range(nconfigs):
            xyz_config = dataset.xyz_configs[k]
            y[k, :] = np.concatenate((xyz_config.energy, xyz_config.dipole))

        dataset.y = torch.from_numpy(y)
        print(dataset.y.size())
        print(dataset.y)

        indices = list(range(nconfigs))
        train_ind, test_ind = data_split(indices, random_state=42, test_size=0.2)
        val_ind, test_ind   = data_split(test_ind, random_state=42, test_size=0.5)

        X_train, y_train = dataset.X[train_ind, :], dataset.y[train_ind]
        X_val, y_val     = dataset.X[val_ind,   :], dataset.y[val_ind]
        X_test, y_test   = dataset.X[test_ind,  :], dataset.y[test_ind]

        xyz_configs_train = [dataset.xyz_configs[ind] for ind in train_ind]
        xyz_configs_val   = [dataset.xyz_configs[ind] for ind in val_ind]
        xyz_configs_test  = [dataset.xyz_configs[ind] for ind in test_ind]

        grm_train = dataset.grm[train_ind]
        grm_val = dataset.grm[val_ind]
        grm_test = dataset.grm[test_ind]

        logging.info("Saving train dataset to: {}".format(dataset_fpaths["train"]))
        torch.save(dict(NATOMS=dataset.NATOMS, NMON=dataset.NMON, NPOLY=dataset.NPOLY, symmetry=cfg_dataset['SYMMETRY'], order=dataset.order,
                        energy_limit=None, variables=cfg_dataset['VARIABLES'], purify=cfg_dataset['PURIFY'], X=X_train, y=y_train,
                        dX=None, dy=None, xyz_ordered=None, grm=grm_train), dataset_fpaths["train"])

        logging.info("Saving val dataset to: {}".format(dataset_fpaths["val"]))
        torch.save(dict(NATOMS=dataset.NATOMS, NMON=dataset.NMON, NPOLY=dataset.NPOLY, symmetry=cfg_dataset['SYMMETRY'], order=dataset.order,
                        energy_limit=None, variables=cfg_dataset['VARIABLES'], purify=cfg_dataset['PURIFY'], X=X_val, y=y_val,
                        dX=None, dy=None, xyz_ordered=None, grm=grm_val), dataset_fpaths["val"])

        logging.info("Saving test dataset to: {}".format(dataset_fpaths["val"]))
        torch.save(dict(NATOMS=dataset.NATOMS, NMON=dataset.NMON, NPOLY=dataset.NPOLY, symmetry=cfg_dataset['SYMMETRY'], order=dataset.order,
                        energy_limit=None, variables=cfg_dataset['VARIABLES'], purify=cfg_dataset['PURIFY'], X=X_test, y=y_test,
                        dX=None, dy=None, xyz_ordered=None, grm=grm_test), dataset_fpaths["test"])

    elif cfg_dataset['TYPE'] == 'DIPOLEC':
        if len(cfg_dataset['SOURCE']) > 1:
            logging.info("Stratification is not implemented for `typ=dipolec`")
            assert False

        dataset = PolyDataset(wdir=cfg_dataset['EXTERNAL_FOLDER'], typ='DIPOLEC', file_path=cfg_dataset['SOURCE'][0],
                              order=cfg_dataset['ORDER'], load_forces=False, symmetry=cfg_dataset['SYMMETRY'],
                              atom_mapping=cfg_dataset['ATOM_MAPPING'], variables=cfg_dataset['VARIABLES'], purify=cfg_dataset['PURIFY'])

        nconfigs = len(dataset.xyz_configs)
        y = np.zeros((nconfigs, 2))

        for k in range(nconfigs):
            xyz_config = dataset.xyz_configs[k]
            y[k, :] = np.array([xyz_config.energy[0], xyz_config.dipole[0]])
            #y[k, :] = np.concatenate((xyz_config.energy[0], xyz_config.dipole[0]))

        dataset.y = torch.from_numpy(y)

        indices = list(range(dataset.NCONFIGS))
        train_ind, test_ind = data_split(indices, random_state=42, test_size=0.2)
        val_ind, test_ind   = data_split(test_ind, random_state=42, test_size=0.5)

        X_train, y_train = dataset.X[train_ind, :], dataset.y[train_ind]
        X_val, y_val     = dataset.X[val_ind,   :], dataset.y[val_ind]
        X_test, y_test   = dataset.X[test_ind,  :], dataset.y[test_ind]

        logging.info("Saving train dataset to: {}".format(dataset_fpaths["train"]))
        torch.save(dict(NATOMS=dataset.NATOMS, NMON=dataset.NMON, NPOLY=dataset.NPOLY, symmetry=cfg_dataset['SYMMETRY'], order=dataset.order,
                        energy_limit=None, variables=cfg_dataset['VARIABLES'], purify=cfg_dataset['PURIFY'], X=X_train, y=y_train,
                        dX=None, dy=None, xyz_ordered=None, grm=None), dataset_fpaths["train"])

        logging.info("Saving val dataset to: {}".format(dataset_fpaths["val"]))
        torch.save(dict(NATOMS=dataset.NATOMS, NMON=dataset.NMON, NPOLY=dataset.NPOLY, symmetry=cfg_dataset['SYMMETRY'], order=dataset.order,
                        energy_limit=None, variables=cfg_dataset['VARIABLES'], purify=cfg_dataset['PURIFY'], X=X_val, y=y_val,
                        dX=None, dy=None, xyz_ordered=None, grm=None), dataset_fpaths["val"])

        logging.info("Saving test dataset to: {}".format(dataset_fpaths["val"]))
        torch.save(dict(NATOMS=dataset.NATOMS, NMON=dataset.NMON, NPOLY=dataset.NPOLY, symmetry=cfg_dataset['SYMMETRY'], order=dataset.order,
                        energy_limit=None, variables=cfg_dataset['VARIABLES'], purify=cfg_dataset['PURIFY'], X=X_test, y=y_test,
                        dX=None, dy=None, xyz_ordered=None, grm=None), dataset_fpaths["test"])

    elif cfg_dataset['TYPE'] == 'DIPOLEQ':
        if len(cfg_dataset['SOURCE']) > 1:
            logging.info("Stratification is not implemented for `typ=dipoleq`")
            assert False

        qmodel_structure = prepare_qmodel_structure(cfg_dataset['SYMMETRY'])
        logging.info("Q-MODEL STRUCTURE:")
        pp.pprint(qmodel_structure)

        datasets = [
            PolyDataset(wdir=cfg_dataset['EXTERNAL_FOLDER'], typ='DIPOLEQ', file_path=cfg_dataset['SOURCE'][0],
                        order=cfg_dataset['ORDER'], load_forces=False, symmetry=sg,
                        atom_mapping=cfg_dataset['ATOM_MAPPING'], variables=cfg_dataset['VARIABLES'], purify=cfg_dataset['PURIFY'])
            for sg in qmodel_structure.keys()
        ]

        dataset = datasets[0]
        nconfigs = len(dataset.xyz_configs)
        assert all(len(dataset.xyz_configs) == nconfigs for dataset in datasets)

        y = torch.zeros((nconfigs, 4))
        for k in range(nconfigs):
            xyz_config = dataset.xyz_configs[k]
            y[k, :] = torch.cat((
                torch.from_numpy(xyz_config.energy),
                torch.from_numpy(xyz_config.dipole)
            ))

        NPOLYs = [dataset.NPOLY for dataset in datasets]
        NPOLYs_acc = list(itertools.accumulate(NPOLYs))

        NPOLY_TOTAL = sum(NPOLYs)
        logging.info("NPOLY total: {}".format(NPOLY_TOTAL))

        X = torch.zeros((nconfigs, NPOLY_TOTAL))
        for dataset, i1, i2 in zip(datasets, [0, *NPOLYs_acc], [*NPOLYs_acc, NPOLY_TOTAL]):
            X[:, i1:i2] = dataset.X

        indices = list(range(nconfigs))
        train_ind, test_ind = data_split(indices, random_state=42, test_size=0.2)
        val_ind, test_ind   = data_split(test_ind, random_state=42, test_size=0.5)

        X_train, y_train, c_train = X[train_ind, :], y[train_ind], dataset.xyz_ordered[train_ind]
        X_val, y_val, c_val       = X[val_ind, :],   y[val_ind],   dataset.xyz_ordered[val_ind]
        X_test, y_test, c_test    = X[test_ind, :],  y[test_ind],  dataset.xyz_ordered[test_ind]

        logging.info("Saving train dataset to: {}".format(dataset_fpaths["train"]))
        torch.save(dict(NATOMS=dataset.NATOMS, NMON=dataset.NMON, NPOLY=NPOLYs, symmetry=qmodel_structure, order=dataset.order,
                        energy_limit=None, variables=cfg_dataset['VARIABLES'], purify=cfg_dataset['PURIFY'], X=X_train, y=y_train,
                        dX=None, dy=None, xyz_ordered=c_train, grm=None), dataset_fpaths["train"])

        logging.info("Saving val dataset to: {}".format(dataset_fpaths["val"]))
        torch.save(dict(NATOMS=dataset.NATOMS, NMON=dataset.NMON, NPOLY=NPOLYs, symmetry=qmodel_structure, order=dataset.order,
                        energy_limit=None, variables=cfg_dataset['VARIABLES'], purify=cfg_dataset['PURIFY'], X=X_val, y=y_val,
                        dX=None, dy=None, xyz_ordered=c_val, grm=None), dataset_fpaths["val"])

        logging.info("Saving test dataset to: {}".format(dataset_fpaths["val"]))
        torch.save(dict(NATOMS=dataset.NATOMS, NMON=dataset.NMON, NPOLY=NPOLYs, symmetry=qmodel_structure, order=dataset.order,
                        energy_limit=None, variables=cfg_dataset['VARIABLES'], purify=cfg_dataset['PURIFY'], X=X_test, y=y_test,
                        dX=None, dy=None, xyz_ordered=c_test, grm=None), dataset_fpaths["test"])

    elif cfg_dataset['TYPE'] == 'ENERGY':
        GLOBAL_SET      = False
        GLOBAL_NATOMS   = None
        GLOBAL_NMON     = None
        GLOBAL_NPOLY    = None
        GLOBAL_MASK     = None

        Xs, ys = [], []

        # Adding the label for stratification during splitting
        labels = []
        label = 0

        for file_path in cfg_dataset['SOURCE']:
            dataset = PolyDataset(wdir=cfg_dataset['EXTERNAL_FOLDER'], typ=cfg_dataset['TYPE'], file_path=file_path, order=cfg_dataset['ORDER'], symmetry=cfg_dataset['SYMMETRY'],
                                  load_forces=cfg_dataset['LOAD_FORCES'], atom_mapping=cfg_dataset['ATOM_MAPPING'], variables=cfg_dataset['VARIABLES'], purify=cfg_dataset['PURIFY'])

            if GLOBAL_SET:
                assert GLOBAL_NATOMS == dataset.NATOMS
                assert GLOBAL_NMON   == dataset.NMON
                assert GLOBAL_NPOLY  == dataset.NPOLY

                if hasattr(dataset, "mask"):
                    np.testing.assert_equal(GLOBAL_MASK, dataset.mask)
            else:
                GLOBAL_NATOMS = dataset.NATOMS
                GLOBAL_NMON   = dataset.NMON
                GLOBAL_NPOLY  = dataset.NPOLY
                if hasattr(dataset, "mask"):
                    GLOBAL_MASK = dataset.mask

            Xs.append(dataset.X)
            ys.append(dataset.y)

            labels.extend([label] * len(dataset.y))
            label = label + 1

        X = torch.cat(tuple(Xs))
        y = torch.cat(tuple(ys))

        if len(cfg_dataset['SOURCE']) > 1:
            logging.info("Performing stratified splitting.")

        # We need to put the labels into X array because we divide it 2 times:
        #           full -> (train, test) -> (train, val, test)
        labels = torch.Tensor(labels).reshape((len(labels), 1))
        X = torch.cat((X, labels), dim=1)

        X_train, X_test, y_train, y_test = data_split(X, y, test_size=0.2, random_state=42, stratify=X[:, -1])
        X_val, X_test, y_val, y_test = data_split(X_test, y_test, test_size=0.5, random_state=42, stratify=X_test[:, -1])

        logging.info("X_train.size(): {}".format(X_train.size()))
        logging.info("X_val.size(): {}".format(X_val.size()))
        logging.info("X_test.size(): {}".format(X_test.size()))

        sz = y_train.size()[0] + y_val.size()[0] + y_test.size()[0]
        logging.info("Total number of configurations: {}".format(sz))

        labels = list(map(int, labels.squeeze().tolist()))
        for label_typ in set(labels):
            p_train = sum(X_train[:, -1] == label_typ) / X_train.size()[0] * 100.0
            p_val   = sum(X_val[:, -1] == label_typ) / X_val.size()[0] * 100.0
            p_test  = sum(X_test[:, -1] == label_typ) / X_test.size()[0] * 100.0
            logging.info("[label_typ={}] train: {:.2f}%; val: {:.2f}%; test: {:.2f}%".format(label_typ, p_train, p_val, p_test))

        X_train = X_train[:, :-1]
        X_val   = X_val[:, :-1]
        X_test  = X_test[:, :-1]

        logging.info("Saving train dataset to: {}".format(dataset_fpaths["train"]))
        torch.save(dict(NATOMS=dataset.NATOMS, NMON=dataset.NMON, NPOLY=dataset.NPOLY, symmetry=cfg_dataset['SYMMETRY'], order=cfg_dataset['ORDER'],
                        energy_limit=cfg_dataset['ENERGY_LIMIT'], variables=cfg_dataset['VARIABLES'], purify=cfg_dataset['PURIFY'], 
                        X=X_train, y=y_train, dX=None, dy=None, xyz_ordered=None, grm=None), dataset_fpaths["train"])

        logging.info("Saving val dataset to: {}".format(dataset_fpaths["val"]))
        torch.save(dict(NATOMS=dataset.NATOMS, NMON=dataset.NMON, NPOLY=dataset.NPOLY, symmetry=cfg_dataset['SYMMETRY'], order=cfg_dataset['ORDER'],
                        energy_limit=cfg_dataset['ENERGY_LIMIT'], variables=cfg_dataset['VARIABLES'], purify=cfg_dataset['PURIFY'],
                        X=X_val, y=y_val, dX=None, dy=None, xyz_ordered=None, grm=None), dataset_fpaths["val"])

        logging.info("Saving test dataset to: {}".format(dataset_fpaths["test"]))
        torch.save(dict(NATOMS=dataset.NATOMS, NMON=dataset.NMON, NPOLY=dataset.NPOLY, symmetry=cfg_dataset['SYMMETRY'], order=cfg_dataset['ORDER'],
                        energy_limit=cfg_dataset['ENERGY_LIMIT'], variables=cfg_dataset['VARIABLES'], purify=cfg_dataset['PURIFY'],
                        X=X_test, y=y_test, dX=None, dy=None, xyz_ordered=None, grm=None), dataset_fpaths["test"])
    else:
        assert False, "unreachable"

    #if cfg_dataset['ENERGY_LIMIT'] is not None:
    #    # TODO:
    #    # Don't know if this whole idea of capping the max energy is worthy 
    #    # Need to look more into this
    #    assert False
    #    logging.info("Placing an energy limit train/val sets; moving rejected points to test set")

    #    indl, indm = (y_train < energy_limit).nonzero()[:, 0], (y_train >= energy_limit).nonzero()[:, 0]
    #    y_test = torch.cat((y_test, y_train[indm]))
    #    X_test = torch.cat((X_test, X_train[indm]))
    #    X_train, y_train = X_train[indl], y_train[indl]

    #    indl, indm = (y_val < energy_limit).nonzero()[:, 0], (y_val >= energy_limit).nonzero()[:, 0]
    #    y_test = torch.cat((y_test, y_val[indm]))
    #    X_test = torch.cat((X_test, X_val[indm]))
    #    X_val, y_val  = X_val[indl], y_val[indl]

    #print("Size of training dataset: {}".format(X_train.size()))
    #train_index = [torch.where((dataset.X == X_train[k]).all(dim=1))[0].item() for k in range(10)] #range(X_train.size()[0])]
    #print("Indices of training elements: {}".format(train_index))
    #train_index_fname = os.path.join(DATASETS_INTERIM, BASENAME + "-train-index.json")
    #with open(train_index_fname, 'w') as fp:
    #    json.dump(train_index, fp=fp)
