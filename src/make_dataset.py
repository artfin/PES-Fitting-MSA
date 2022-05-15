import json
import logging
import numpy as np
import os

import sklearn.model_selection
import torch

from dataset import PolyDataset

import pathlib
BASEDIR = pathlib.Path(__file__).parent.parent.resolve()

DATASET_POSTFIX = "-nonrigid"
XYZ_PATHS = [
    os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-RIGID.xyz"),
    os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=0-1000-N2=0-1000.xyz"),
    os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=1000-2000-N2=0-1000.xyz"),
    os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-NONRIGID-CH4=2000-3000-N2=0-1000.xyz"),
]

#DATASET_POSTFIX = "-rigid"
#XYZ_PATHS = {
#    os.path.join(BASEDIR, "datasets", "raw", "CH4-N2-EN-RIGID.xyz"),
#}

order     = "4"
wdir      = "datasets/external"
symmetry  = "4 2 1"

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

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    GLOBAL_SET      = False
    GLOBAL_NATOMS   = None
    GLOBAL_NMON     = None
    GLOBAL_NPOLY    = None
    GLOBAL_MASK     = None

    Xs, ys = [], []

    # Adding the label for stratification during splitting
    labels = []
    label = 0

    for xyz_path in XYZ_PATHS:
        dataset = PolyDataset(wdir=wdir, xyz_file=xyz_path, order=order, symmetry=symmetry, set_intermolecular_to_zero=False)

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

    DATASETS_INTERIM = "datasets/interim"
    BASENAME = "poly_{}_{}".format(symmetry.replace(" ", "_"), order)

    #######################################################################
    # splitting polynomials into train/val/test and saving
    #######################################################################

    logging.info("Performing stratified splitting.")
    data_split = sklearn.model_selection.train_test_split

    X_train, X_test, y_train, y_test = data_split(X, y, test_size=0.2, random_state=42, stratify=X[:, -1])
    X_val, X_test, y_val, y_test = data_split(X_test, y_test, test_size=0.5, random_state=42, stratify=X_test[:, -1])

    logging.info("Placing an energy limit train/val sets; moving rejected points to test set")
    ENERGY_LIMIT = 2000.0 # cm-1

    indl, indm = (y_train < ENERGY_LIMIT).nonzero()[:, 0], (y_train >= ENERGY_LIMIT).nonzero()[:, 0]
    y_test = torch.cat((y_test, y_train[indm]))
    X_test = torch.cat((X_test, X_train[indm]))
    X_train, y_train = X_train[indl], y_train[indl]

    indl, indm = (y_val < ENERGY_LIMIT).nonzero()[:, 0], (y_val >= ENERGY_LIMIT).nonzero()[:, 0]
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


    #print("Size of training dataset: {}".format(X_train.size()))
    #train_index = [torch.where((dataset.X == X_train[k]).all(dim=1))[0].item() for k in range(10)] #range(X_train.size()[0])]

    #print("Indeces of training elements: {}".format(train_index))
    #train_index_fname = os.path.join(DATASETS_INTERIM, BASENAME + "-train-index.json")
    #with open(train_index_fname, 'w') as fp:
    #    json.dump(train_index, fp=fp)


    X_train = X_train[:, :-1]
    dict_pk  = dict(NATOMS=GLOBAL_NATOMS, NMON=GLOBAL_NMON, NPOLY=GLOBAL_NPOLY,
                    symmetry=symmetry, order=order, X=X_train, y=y_train)
    train_interim_pk_fname = os.path.join(DATASETS_INTERIM, BASENAME + "-train{}.pk".format(DATASET_POSTFIX))
    torch.save(dict_pk, train_interim_pk_fname)


    X_val = X_val[:, :-1]
    dict_pk  = dict(NATOMS=GLOBAL_NATOMS, NMON=GLOBAL_NMON, NPOLY=GLOBAL_NPOLY,
                    symmetry=symmetry, order=order, X=X_val, y=y_val)
    val_interim_pk_fname = os.path.join(DATASETS_INTERIM, BASENAME + "-val{}.pk".format(DATASET_POSTFIX))
    torch.save(dict_pk, val_interim_pk_fname)

    X_test = X_test[:, :-1]
    dict_pk  = dict(NATOMS=GLOBAL_NATOMS, NMON=GLOBAL_NMON, NPOLY=GLOBAL_NPOLY,
                    symmetry=symmetry, order=order, X=X_test, y=y_test)
    test_interim_pk_fname = os.path.join(DATASETS_INTERIM, BASENAME + "-test{}.pk".format(DATASET_POSTFIX))
    torch.save(dict_pk, test_interim_pk_fname)
