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

#RAW_DATASET_PATHS = {
#    "RIGID" : [
#        {
#            "xyz_path"    : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-RIGID.xyz"),
#            "limits_path" : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-RIGID-LIMITS.xyz"),
#        }
#    ],
#    "NONRIGID" : [{
#            "xyz_path"    : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-RIGID.xyz"),
#            "limits_path" : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-RIGID-LIMITS.xyz"),
#        },
#        {
#            "xyz_path"    : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=0-1000-N2=0-1000.xyz"),
#            "limits_path" : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=0-1000-N2=0-1000-LIMITS.xyz"),
#        },
#        {
#            "xyz_path"    : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=1000-2000-N2=0-1000.xyz"),
#            "limits_path" : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=1000-2000-N2=0-1000-LIMITS.xyz"),
#        },
#        {
#            "xyz_path"    : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=2000-3000-N2=0-1000.xyz"),
#            "limits_path" : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=2000-3000-N2=0-1000-LIMITS.xyz"),
#        }
#    ],
#    "NONRIGID-CLIP" : [
#        {
#            "xyz_path"    : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-RIGID-10000.xyz"),
#            "limits_path" : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-RIGID-10000-LIMITS.xyz"),
#        },
#        {
#            "xyz_path"    : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=0-1000-N2=0-1000.xyz"),
#            "limits_path" : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=0-1000-N2=0-1000-LIMITS.xyz"),
#        },
#        {
#            "xyz_path"    : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=1000-2000-N2=0-1000.xyz"),
#            "limits_path" : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=1000-2000-N2=0-1000-LIMITS.xyz"),
#        },
#        {
#            "xyz_path"    : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=2000-3000-N2=0-1000.xyz"),
#            "limits_path" : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=2000-3000-N2=0-1000-LIMITS.xyz"),
#        }
#    ]
#}

class JSONNumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return super(JSONNumpyEncoder, self).default(obj)

def save_json(d, fpath):
    with open(fpath, mode='w') as fp:
        json.dump(d, cls=JSONNumpyEncoder, fp=fp)

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
    else:
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

            if cfg_dataset['TYPE'] == 'DIPOLE':
                anchor_pos = tuple(map(int, cfg_dataset['ANCHOR_POSITIONS'].split()))
                assert anchor_pos[0] in range(dataset.NATOMS)
                assert anchor_pos[1] in range(dataset.NATOMS)

                for xyz_config in dataset.xyz_configs:
                    anchor = xyz_config.coords[anchor_pos[0]] - xyz_config.coords[anchor_pos[1]]

                    x, y, z = anchor[0], anchor[1], anchor[2]
                    alpha = np.arctan2(-y, x)
                    beta  = np.arctan2(-np.sqrt(x**2 + y**2), z)

                    S = rot_y(beta) @ rot_z(alpha)
                    xyz_config.coords = xyz_config.coords @ S.T
                    xyz_config.dipole = S @ xyz_config.dipole

                    rot_anchor = S @ anchor
                    assert np.isclose(rot_anchor / np.linalg.norm(rot_anchor), np.array([0.0, 0.0, 1.0])).all()

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

        dX_train, dX_val, dX_test = None, None, None
        dy_train, dy_val, dy_test = None, None, None

    if cfg_dataset['ENERGY_LIMIT'] is not None:
        # TODO:
        # Don't know if this whole idea of capping the max energy is worthy 
        # Need to look more into this
        assert False
        logging.info("Placing an energy limit train/val sets; moving rejected points to test set")

        indl, indm = (y_train < energy_limit).nonzero()[:, 0], (y_train >= energy_limit).nonzero()[:, 0]
        y_test = torch.cat((y_test, y_train[indm]))
        X_test = torch.cat((X_test, X_train[indm]))
        X_train, y_train = X_train[indl], y_train[indl]

        indl, indm = (y_val < energy_limit).nonzero()[:, 0], (y_val >= energy_limit).nonzero()[:, 0]
        y_test = torch.cat((y_test, y_val[indm]))
        X_test = torch.cat((X_test, X_val[indm]))
        X_val, y_val  = X_val[indl], y_val[indl]

    #print("Size of training dataset: {}".format(X_train.size()))
    #train_index = [torch.where((dataset.X == X_train[k]).all(dim=1))[0].item() for k in range(10)] #range(X_train.size()[0])]
    #print("Indices of training elements: {}".format(train_index))
    #train_index_fname = os.path.join(DATASETS_INTERIM, BASENAME + "-train-index.json")
    #with open(train_index_fname, 'w') as fp:
    #    json.dump(train_index, fp=fp)

    train_fpath = dataset_fpaths["train"]
    val_fpath   = dataset_fpaths["val"]
    test_fpath  = dataset_fpaths["test"]

    dict_pk = dict(
        NATOMS=dataset.NATOMS,
        NMON=dataset.NMON,
        NPOLY=dataset.NPOLY,
        symmetry=cfg_dataset['SYMMETRY'],
        order=cfg_dataset['ORDER'],
        energy_limit=cfg_dataset['ENERGY_LIMIT'],
        variables=cfg_dataset['VARIABLES'],
        purify=cfg_dataset['PURIFY'],
        X=X_train,
        y=y_train,
        dX=dX_train,
        dy=dy_train
    )

    logging.info("Saving training dataset to: {}".format(train_fpath))
    torch.save(dict_pk, train_fpath)

    dict_pk  = dict(
        NATOMS=dataset.NATOMS,
        NMON=dataset.NMON,
        NPOLY=dataset.NPOLY,
        symmetry=cfg_dataset['SYMMETRY'],
        order=cfg_dataset['ORDER'],
        energy_limit=cfg_dataset['ENERGY_LIMIT'],
        variables=cfg_dataset['VARIABLES'],
        purify=cfg_dataset['PURIFY'],
        X=X_val,
        y=y_val,
        dX=dX_val,
        dy=dy_val
    )
    logging.info("Saving validation dataset to: {}".format(val_fpath))
    torch.save(dict_pk, val_fpath)

    dict_pk  = dict(
        NATOMS=dataset.NATOMS,
        NMON=dataset.NMON,
        NPOLY=dataset.NPOLY,
        symmetry=cfg_dataset['SYMMETRY'],
        order=cfg_dataset['ORDER'],
        energy_limit=cfg_dataset['ENERGY_LIMIT'],
        variables=cfg_dataset['VARIABLES'],
        purify=cfg_dataset['PURIFY'],
        X=X_test,
        y=y_test,
        dX=dX_test,
        dy=dy_test
    )
    logging.info("Saving testing dataset to: {}".format(test_fpath))
    torch.save(dict_pk, test_fpath)
