import json
import logging
import numpy as np
import os

import sklearn.model_selection
import torch

from dataset import PolyDataset

import pathlib
BASEDIR = pathlib.Path(__file__).parent.parent.resolve()

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

def make_dataset_fpaths(order, symmetry, typ, energy_limit, intramz, purify):
    dataset_folder = os.path.join(BASEDIR, "datasets", "interim")

    if energy_limit is not None:
        enlim_str = "-enlim={:.0f}".format(energy_limit)
    else:
        enlim_str = ""

    if intramz:
        intramz_str = "-intramz=true"
    else:
        intramz_str = ""

    if purify:
        purify_str = "-purify=true"
    else:
        purify_str = ""

    symmetry_str = symmetry.replace(' ', '_')
    train_fpath = os.path.join(dataset_folder, f"poly_{symmetry_str}_{order}-train-{typ}{enlim_str}{intramz_str}{purify_str}.pk")
    val_fpath   = os.path.join(dataset_folder, f"poly_{symmetry_str}_{order}-val-{typ}{enlim_str}{intramz_str}{purify_str}.pk")
    test_fpath  = os.path.join(dataset_folder, f"poly_{symmetry_str}_{order}-test-{typ}{enlim_str}{intramz_str}{purify_str}.pk")

    return train_fpath, val_fpath, test_fpath

def make_dataset(order, symmetry, typ, **kwargs):
    wdir = "datasets/external"

    intramz      = kwargs.get('intramz', False)
    purify       = kwargs.get('purify', False)
    energy_limit = kwargs.get("energy_limit", None)

    dataset_postfix = '-' + typ
    if typ == 'rigid':
        xyz_paths = [
            {
                "xyz_path"    : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-RIGID.xyz"),
                "limits_path" : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-RIGID-LIMITS.xyz"),
            }
        ]
    elif typ == 'nonrigid':
        xyz_paths = [
            {
                "xyz_path"    : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-RIGID.xyz"),
                "limits_path" : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-RIGID-LIMITS.xyz"),
            },
            {
                "xyz_path"    : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=0-1000-N2=0-1000.xyz"),
                "limits_path" : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=0-1000-N2=0-1000-LIMITS.xyz"),
            },
            {
                "xyz_path"    : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=1000-2000-N2=0-1000.xyz"),
                "limits_path" : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=1000-2000-N2=0-1000-LIMITS.xyz"),
            },
            {
                "xyz_path"    : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=2000-3000-N2=0-1000.xyz"),
                "limits_path" : os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=2000-3000-N2=0-1000-LIMITS.xyz"),
            }
        ]

    GLOBAL_SET      = False
    GLOBAL_NATOMS   = None
    GLOBAL_NMON     = None
    GLOBAL_NPOLY    = None
    GLOBAL_MASK     = None

    Xs, ys = [], []

    # Adding the label for stratification during splitting
    labels = []
    label = 0

    for block in xyz_paths:
        dataset = PolyDataset(wdir=wdir, xyz_file=block["xyz_path"], limit_file=block["limits_path"], order=order,
                              symmetry=symmetry, intramz=intramz, purify=purify)

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

    labels = torch.Tensor(labels).reshape((len(labels), 1))
    X = torch.cat((X, labels), dim=1)

    #######################################################################
    # splitting polynomials into train/val/test and saving
    #######################################################################
    train_fpath, val_fpath, test_fpath = make_dataset_fpaths(order, symmetry, typ, energy_limit, intramz, purify)

    logging.info("Performing stratified splitting.")
    data_split = sklearn.model_selection.train_test_split

    X_train, X_test, y_train, y_test = data_split(X, y, test_size=0.2, random_state=42, stratify=X[:, -1])
    X_val, X_test, y_val, y_test = data_split(X_test, y_test, test_size=0.5, random_state=42, stratify=X_test[:, -1])

    if energy_limit is not None:
        logging.info("Placing an energy limit train/val sets; moving rejected points to test set")

        indl, indm = (y_train < energy_limit).nonzero()[:, 0], (y_train >= energy_limit).nonzero()[:, 0]
        y_test = torch.cat((y_test, y_train[indm]))
        X_test = torch.cat((X_test, X_train[indm]))
        X_train, y_train = X_train[indl], y_train[indl]

        indl, indm = (y_val < energy_limit).nonzero()[:, 0], (y_val >= energy_limit).nonzero()[:, 0]
        y_test = torch.cat((y_test, y_val[indm]))
        X_test = torch.cat((X_test, X_val[indm]))
        X_val, y_val  = X_val[indl], y_val[indl]

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

    #print("Size of training dataset: {}".format(X_train.size()))
    #train_index = [torch.where((dataset.X == X_train[k]).all(dim=1))[0].item() for k in range(10)] #range(X_train.size()[0])]
    #print("Indices of training elements: {}".format(train_index))
    #train_index_fname = os.path.join(DATASETS_INTERIM, BASENAME + "-train-index.json")
    #with open(train_index_fname, 'w') as fp:
    #    json.dump(train_index, fp=fp)

    dict_pk = dict(
        NATOMS=GLOBAL_NATOMS,
        NMON=GLOBAL_NMON,
        NPOLY=GLOBAL_NPOLY,
        symmetry=symmetry,
        order=order,
        energy_limit=energy_limit,
        intramz=intramz,
        purify=purify,
        X=X_train,
        y=y_train
    )

    logging.info("Saving training dataset to: {}".format(train_fpath))
    torch.save(dict_pk, train_fpath)

    X_val = X_val[:, :-1]
    dict_pk  = dict(
        NATOMS=GLOBAL_NATOMS,
        NMON=GLOBAL_NMON,
        NPOLY=GLOBAL_NPOLY,
        symmetry=symmetry,
        order=order,
        energy_limit=energy_limit,
        intramz=intramz,
        purify=purify,
        X=X_val,
        y=y_val
    )
    logging.info("Saving validation dataset to: {}".format(val_fpath))
    torch.save(dict_pk, val_fpath)

    X_test = X_test[:, :-1]
    dict_pk  = dict(
        NATOMS=GLOBAL_NATOMS,
        NMON=GLOBAL_NMON,
        NPOLY=GLOBAL_NPOLY,
        symmetry=symmetry,
        order=order,
        energy_limit=energy_limit,
        intramz=intramz,
        purify=purify,
        X=X_test,
        y=y_test
    )
    logging.info("Saving testing dataset to: {}".format(test_fpath))
    torch.save(dict_pk, test_fpath)

if __name__ == "__main__":
    pass
