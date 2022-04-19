import json
import logging
import os

import sklearn.model_selection
import torch

from dataset import PolyDataset

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    order     = "5"
    wdir      = "datasets/external"
    symmetry  = "4 2 1"
    xyz_fname = "datasets/raw/ch4-n2-energies.xyz"
    dataset = PolyDataset(wdir=wdir, xyz_fname=xyz_fname, order=order, symmetry=symmetry, set_intermolecular_to_zero=True) #, lr_model=lr_model)

    DATASETS_INTERIM = "datasets/interim"
    BASENAME = "poly_{}_{}".format(symmetry.replace(" ", "_"), order)

    #######################################################################
    # Saving polynomials
    #######################################################################

    interim_pk_fname = os.path.join(DATASETS_INTERIM, BASENAME + ".pk")
    logging.info("saving polynomials (pickled) to {}".format(interim_pk_fname))
    dataset.save_pickle(interim_pk_fname)

    interim_json_fname = os.path.join(DATASETS_INTERIM, BASENAME + ".json")
    logging.info("saving polynomials (readable) to {}".format(interim_json_fname))
    dataset.save_json(interim_json_fname)

    #######################################################################
    # splitting polynomials into train/val/test and saving 
    #######################################################################

    data_split = sklearn.model_selection.train_test_split
    X_train, X_test, y_train, y_test = data_split(dataset.X, dataset.y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = data_split(X_test, y_test, test_size=0.5, random_state=42)

    print("Size of training dataset: {}".format(X_train.size()))
    train_index = [torch.where((dataset.X == X_train[k]).all(dim=1))[0].item() for k in range(10)] #range(X_train.size()[0])]

    print("Indeces of training elements: {}".format(train_index))
    train_index_fname = os.path.join(DATASETS_INTERIM, BASENAME + "-train-index.json")
    with open(train_index_fname, 'w') as fp:
        json.dump(train_index, fp=fp)

    dataset_train = dataset.make_dict()
    dataset_train.update({"X" : X_train, "y" : y_train})
    train_interim_pk_fname = os.path.join(DATASETS_INTERIM, BASENAME + "-train.pk")
    torch.save(dataset_train, train_interim_pk_fname)

    dataset_val = dataset.make_dict()
    dataset_val.update({"X" : X_val, "y" : y_val})
    val_interim_pk_fname = os.path.join(DATASETS_INTERIM, BASENAME + "-val.pk")
    torch.save(dataset_val, val_interim_pk_fname)

    dataset_test  = dataset.make_dict()
    dataset_test.update({"X" : X_test, "y" : y_test})
    test_interim_pk_fname = os.path.join(DATASETS_INTERIM, BASENAME + "-test.pk")
    torch.save(dataset_test, test_interim_pk_fname)
