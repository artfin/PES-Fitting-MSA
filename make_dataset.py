import json
import logging
import numpy as np
import os

import sklearn.model_selection
import torch

from dataset import PolyDataset

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

    order     = "4"
    wdir      = "datasets/external"
    symmetry  = "4 2 1"

    PATH_RIGID    = os.path.join("datasets", "raw", "CH4-N2-EN-RIGID.xyz")
    dataset_rigid = PolyDataset(wdir=wdir, xyz_file=PATH_RIGID, order=order, symmetry=symmetry, set_intermolecular_to_zero=True) #, lr_model=lr_model)

    PATH_NONRIGID = os.path.join("datasets", "raw", "CH4-N2-EN-NONRIGID.xyz")
    dataset_nonrigid = PolyDataset(wdir=wdir, xyz_file=PATH_NONRIGID, order=order, symmetry=symmetry, set_intermolecular_to_zero=True)

    assert dataset_rigid.NATOMS   == dataset_nonrigid.NATOMS
    assert dataset_rigid.NMON     == dataset_nonrigid.NMON
    assert dataset_rigid.NPOLY    == dataset_nonrigid.NPOLY
    assert dataset_rigid.symmetry == dataset_nonrigid.symmetry
    assert dataset_rigid.order    == dataset_nonrigid.order

    if hasattr(dataset_rigid, "mask"):
        assert dataset_rigid.mask == dataset_nonrigid.mask

    X = torch.cat((dataset_rigid.X, dataset_nonrigid.X))
    y = torch.cat((dataset_rigid.y, dataset_nonrigid.y))

    # labels: {0: RIGID, 1: NONRIGID}
    labels = [0] * len(dataset_rigid.y) + [1] * len(dataset_nonrigid.y)
    labels = torch.Tensor(labels).reshape((len(labels), 1))

    X = torch.cat((X, labels), dim=1)

    dict_pk  = dict(NATOMS=dataset_rigid.NATOMS, NMON=dataset_rigid.NMON, NPOLY=dataset_rigid.NPOLY,
                    symmetry=dataset_rigid.symmetry, order=dataset_rigid.order, X=X, y=y, labels=labels) 

    dict_json = dict(NATOMS=dataset_rigid.NATOMS, NMON=dataset_rigid.NMON, NPOLY=dataset_rigid.NPOLY,
                     symmetry=dataset_rigid.symmetry, order=dataset_rigid.order, X=X.numpy(), y=y.numpy(), labels=labels)

    if hasattr(dataset_rigid, "mask"):
        dict_pk.update({"mask" : dataset_rigid.mask})
        dict_json.update({"mask" : dataset_rigid.mask})

    DATASETS_INTERIM = "datasets/interim"
    BASENAME = "poly_{}_{}".format(symmetry.replace(" ", "_"), order)

    #######################################################################
    # Saving polynomials
    #######################################################################

    #interim_pk_fname = os.path.join(DATASETS_INTERIM, BASENAME + ".pk")
    #logging.info("saving polynomials (pickled) to {}".format(interim_pk_fname))
    #torch.save(dict_pk, interim_pk_fname)

    #interim_json_fname = os.path.join(DATASETS_INTERIM, BASENAME + ".json")
    #logging.info("saving polynomials (readable) to {}".format(interim_json_fname))
    #save_json(dict_json, interim_json_fname)

    #######################################################################
    # splitting polynomials into train/val/test and saving
    #######################################################################

    data_split = sklearn.model_selection.train_test_split

    X_train, X_test, y_train, y_test = data_split(X, y, test_size=0.2, random_state=42, stratify=X[:, -1])
    X_val, X_test, y_val, y_test = data_split(X_test, y_test, test_size=0.5, random_state=42, stratify=X_test[:, -1])

    logging.info("X_train.size(): {}".format(X_train.size()))
    logging.info("X_val.size(): {}".format(X_val.size()))
    logging.info("X_test.size(): {}".format(X_test.size()))

    logging.info("[Train] percentage of NONRIGID: {}%".format(sum(X_train[:, -1]) / X_train.size()[0] * 100.0))
    logging.info("[Val]   percentage of NONRIGID: {}%".format(sum(X_val[:, -1]) / X_val.size()[0] * 100.0))
    logging.info("[Test]  percentage of NONRIGID: {}%".format(sum(X_test[:, -1]) / X_test.size()[0] * 100.0))

    #print("Size of training dataset: {}".format(X_train.size()))
    #train_index = [torch.where((dataset.X == X_train[k]).all(dim=1))[0].item() for k in range(10)] #range(X_train.size()[0])]

    #print("Indeces of training elements: {}".format(train_index))
    #train_index_fname = os.path.join(DATASETS_INTERIM, BASENAME + "-train-index.json")
    #with open(train_index_fname, 'w') as fp:
    #    json.dump(train_index, fp=fp)

    X_train = X_train[:, :-1]
    dict_pk  = dict(NATOMS=dataset_rigid.NATOMS, NMON=dataset_rigid.NMON, NPOLY=dataset_rigid.NPOLY,
                    symmetry=dataset_rigid.symmetry, order=dataset_rigid.order, X=X_train, y=y_train)
    train_interim_pk_fname = os.path.join(DATASETS_INTERIM, BASENAME + "-train.pk")
    torch.save(dict_pk, train_interim_pk_fname)

    X_val = X_val[:, :-1]
    dict_pk  = dict(NATOMS=dataset_rigid.NATOMS, NMON=dataset_rigid.NMON, NPOLY=dataset_rigid.NPOLY,
                    symmetry=dataset_rigid.symmetry, order=dataset_rigid.order, X=X_val, y=y_val)
    val_interim_pk_fname = os.path.join(DATASETS_INTERIM, BASENAME + "-val.pk")
    torch.save(dict_pk, val_interim_pk_fname)

    X_test = X_test[:, :-1]
    dict_pk  = dict(NATOMS=dataset_rigid.NATOMS, NMON=dataset_rigid.NMON, NPOLY=dataset_rigid.NPOLY,
                    symmetry=dataset_rigid.symmetry, order=dataset_rigid.order, X=X_test, y=y_test)
    test_interim_pk_fname = os.path.join(DATASETS_INTERIM, BASENAME + "-test.pk")
    torch.save(dict_pk, test_interim_pk_fname)
