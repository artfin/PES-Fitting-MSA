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

def make_dataset_fpaths(typ, order, symmetry, use_forces, energy_limit, intramz, purify, interim_folder):
    assert typ in ('energy', 'dipole')
    assert order in (1, 2, 3, 4, 5)
    assert use_forces in (True, False)

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

    if use_forces:
        forces_str = "-forces=true"
    else:
        forces_str = ""

    symmetry_str = symmetry.replace(' ', '_')
    train_fpath = os.path.join(interim_folder, f"{typ}-poly_{symmetry_str}_{order}{forces_str}-train{enlim_str}{intramz_str}{purify_str}.pk")
    val_fpath   = os.path.join(interim_folder, f"{typ}-poly_{symmetry_str}_{order}{forces_str}-val{enlim_str}{intramz_str}{purify_str}.pk")
    test_fpath  = os.path.join(interim_folder, f"{typ}-poly_{symmetry_str}_{order}{forces_str}-test{enlim_str}{intramz_str}{purify_str}.pk")

    return train_fpath, val_fpath, test_fpath

def make_dataset(source, typ, order, symmetry, dataset_fpaths, external_folder, use_forces, intramz, purify, energy_limit):
    # default split function [dataset] -> [train, val, test] 
    data_split = sklearn.model_selection.train_test_split

    if use_forces:
        if len(source) > 1:
            logging.info("Stratification is not implemented for `use_forces=True`")
            assert False

        dataset = PolyDataset(wdir=external_folder, file_path=source[0], order=order, use_forces=use_forces,
                              symmetry=symmetry, intramz=intramz, purify=purify)

        NCONFIGS = dataset.X.size()[0]
        indices = list(range(NCONFIGS))

        train_ind, test_ind = data_split(indices, test_size=0.2)
        val_ind, test_ind   = data_split(test_ind, test_size=0.5)

        X_train, y_train = dataset.X[train_ind, :], dataset.y[train_ind]
        X_val, y_val     = dataset.X[val_ind,   :], dataset.y[val_ind]
        X_test, y_test   = dataset.X[test_ind,  :], dataset.y[test_ind]

        import psutil
        print("Memory: ", psutil.virtual_memory())

        GB = 1024 * 1024 * 1024
        print("Memory size of dX: {} GB", dataset.dX.element_size() * dataset.dX.nelement() / GB)

        dX_train, dy_train = dataset.dX[train_ind, :, :], dataset.dy[train_ind, :, :]
        dX_val, dy_val     = dataset.dX[val_ind,   :, :], dataset.dy[val_ind,   :, :]
        dX_test, dy_test   = dataset.dX[test_ind,  :, :], dataset.dy[test_ind,  :, :]

        print("Memory: ", psutil.virtual_memory())
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

        for file_path in source:
            dataset = PolyDataset(wdir=external_folder, file_path=file_path, order=order, use_forces=use_forces,
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

        if len(source) > 1:
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

    if energy_limit is not None:
        # TODO:
        # Don't know if this whole idea is worth using in the future
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
        symmetry=symmetry,
        order=order,
        energy_limit=energy_limit,
        intramz=intramz,
        purify=purify,
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
        symmetry=symmetry,
        order=order,
        energy_limit=energy_limit,
        intramz=intramz,
        purify=purify,
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
        symmetry=symmetry,
        order=order,
        energy_limit=energy_limit,
        intramz=intramz,
        purify=purify,
        X=X_test,
        y=y_test,
        dX=dX_test,
        dy=dy_test
    )
    logging.info("Saving testing dataset to: {}".format(test_fpath))
    torch.save(dict_pk, test_fpath)
