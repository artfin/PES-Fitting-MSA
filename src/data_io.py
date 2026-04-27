import logging
import os
import random

import numpy as np
import torch
import yaml
from sklearn.preprocessing import StandardScaler

from config import TORCH_FLOAT
from dataset import PolyDataset
from make_dataset import make_dataset, make_dataset_fpaths
from build_model import build_network

import pathlib
BASEDIR = pathlib.Path(__file__).parent.parent.resolve()

from distributed import is_main_process

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

USE_WANDB = False
if USE_WANDB:
    import wandb

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------------------
# MGDA (Multiple Gradient Descent Algorithm) helpers
# ---------------------------------------------------------------------------

class IdentityScaler:
    def __init__(self):
        pass

    def fit_transform(self, x):
        self.mean_  = np.zeros((x.shape[1]))
        self.scale_ = np.ones((x.shape[1]))
        return np.asarray(x)

    def transform(self, y):
        return np.asarray(y)

def apply_scalers_on_dataset(train, val, test, xscaler, yscaler):
    try:
        train.X = torch.from_numpy(xscaler.transform(train.X)).to(TORCH_FLOAT)
        val.X   = torch.from_numpy(xscaler.transform(val.X)).to(TORCH_FLOAT)
        test.X  = torch.from_numpy(xscaler.transform(test.X)).to(TORCH_FLOAT)
    except ValueError:
        logging.error("[use_scalers_on_dataset] caught ValueError")
        val.X  = torch.empty((1, 1), dtype=TORCH_FLOAT)
        test.X = torch.empty((1, 1), dtype=TORCH_FLOAT)

    try:
        train.y = torch.from_numpy(yscaler.transform(train.y)).to(TORCH_FLOAT)
        val.y   = torch.from_numpy(yscaler.transform(val.y)).to(TORCH_FLOAT)
        test.y  = torch.from_numpy(yscaler.transform(test.y)).to(TORCH_FLOAT)
    except ValueError:
        logging.error("[use_scalers_on_dataset] caught ValueError")
        val.y = torch.empty(1, dtype=TORCH_FLOAT)
        test.y = torch.empty(1, dtype=TORCH_FLOAT)
def fit_scalers_to_train_dataset(train, cfg, X=None, y=None):
    """Fit scalers to training data.

    Args:
        train: Training dataset (used if X, y not provided)
        cfg: Dataset config with NORMALIZE setting
        X: Optional explicit X data (use for fitting on full data before sharding)
        y: Optional explicit y data (use for fitting on full data before sharding)
    """
    if cfg['NORMALIZE'] == 'std':
        xscaler = StandardScaler()
        yscaler = StandardScaler()
    elif cfg['NORMALIZE'] == 'std-none':
        xscaler = StandardScaler()
        yscaler = IdentityScaler()
    else:
        raise ValueError("unreachable")

    xscaler.fit(X if X is not None else train.X)
    yscaler.fit(y if y is not None else train.y)

    return xscaler, yscaler
def load_from_checkpoint(chk_path):
    state = torch.load(chk_path, map_location=torch.device(DEVICE), weights_only=False)
    assert state.get("model", None) is not None, "No 'model' field found in checkpoint loaded from {}".format(chk_path)
    assert state.get("X_mean", None) is not None, "No 'X_mean' field found in checkpoint loaded from {}".format(chk_path)
    assert state.get("X_std", None) is not None, "No 'X_std' field found in checkpoint loaded from {}".format(chk_path)
    assert state.get("y_mean", None) is not None, "No 'y_mean' field found in checkpoint loaded from {}".format(chk_path)
    assert state.get("y_std", None) is not None, "No 'y_std' field found in checkpoint loaded from {}".format(chk_path)
    assert state.get("meta_info", None) is not None, "No 'meta_info' field found in checkpoint loaded from {}".format(chk_path)

    shapes_of_loaded_weights = []
    for key, value in state['model'].items():
        if 'bias' in key: continue
        shapes_of_loaded_weights.append(value.shape[1])

    assert len(shapes_of_loaded_weights) >= 1

    # TODO: unhardcode activation function
    cfg_model = {
        "ACTIVATION": "SiLU",
    }

    model = build_network(
        cfg_model=cfg_model,
        hidden_dims=shapes_of_loaded_weights[1:],
        input_features=shapes_of_loaded_weights[0],
        output_features=1)

    model.load_state_dict(state['model'])

    logging.warning("Data scalers (xscaler & yscaler) are taken from the checkpoint")
    xscaler = StandardScaler()
    xscaler.mean_ = state["X_mean"]
    xscaler.scale_ = state["X_std"]

    yscaler = StandardScaler()
    yscaler.mean_ = state["y_mean"]
    yscaler.scale_ = state["y_std"]

    return model, xscaler, yscaler


def save_checkpoint(model, xscaler, yscaler, meta_info, chk_path):
    if not is_main_process():
        return
    logging.info("Saving the checkpoint.")

    checkpoint = {
        "model"        :  model.state_dict(),
        "X_mean"       :  xscaler.mean_,
        "X_std"        :  xscaler.scale_,
        "y_mean"       :  yscaler.mean_,
        "y_std"        :  yscaler.scale_,
        "meta_info"    :  meta_info,
    }
    torch.save(checkpoint, chk_path)
def setup_google_folder():
    assert os.path.exists('client_secrets.json')
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()

    drive = GoogleDrive(gauth)

    folderName = "PES-Fitting-MSA"

    folders = drive.ListFile(
        {'q': "title='" + folderName + "' and mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()

    for folder in folders:
        if folder['title'] == folderName:
            file = drive.CreateFile({'parents': [{'id': folder['id']}]})
            file.SetContentFile('README.md')
            file.Upload()

def load_cfg(cfg_path):
    with open(cfg_path, mode="r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.info(exc)

    known_groups = ('TYPE', 'DATASET', 'MODEL', 'LOSS', 'TRAINING', 'PRINT_PRECISION', 'PRETRAINED_MODEL_SETTINGS', 'REGULARIZATION', 'BATCH', 'DEBUG')
    for group in cfg.keys():
        assert group in known_groups, "Unknown group: {}".format(group)

    return cfg

def load_dataset(cfg_dataset, typ):
    from enum import Enum, auto
    class KeywordType:
        KEYWORD_OPTIONAL = auto()
        KEYWORD_REQUIRED = auto()

    KEYWORDS = [
        ('NAME', KeywordType.KEYWORD_REQUIRED, None), # `str` 
        # FILE ORGANIZATION 
        #  `list` : paths (relative to BASEDIR) to the .xyz/.npz files 
        ('SOURCE', KeywordType.KEYWORD_REQUIRED, None),
        #  `str`  : path to store pickled train/val/test datasets 
        ('INTERIM_FOLDER', KeywordType.KEYWORD_OPTIONAL, os.path.join(BASEDIR, "datasets", "interim")),
        #  `str`  : path to folder with files to compute invariant polynomials: .f90 to compute polynomials (and their derivatives) + .MONO + .POLY 
        ('EXTERNAL_FOLDER', KeywordType.KEYWORD_OPTIONAL, os.path.join(BASEDIR, "datasets", "external")),
        # DATA SELECTION 
        ('LOAD_GRADIENTS',  KeywordType.KEYWORD_OPTIONAL, False), # `bool` : whether to load gradients from dataset
        ('ENERGY_LIMIT', KeywordType.KEYWORD_OPTIONAL, None),  # `bool` : NOT SUPPORTED now -- set an upper bound on energies in the training dataset 
        # DATASET PREPROCESSING
        ('NORMALIZE',        KeywordType.KEYWORD_REQUIRED, None),  # `str`  : how to perform data normalization  
        ('ANCHOR_POSITIONS', KeywordType.KEYWORD_OPTIONAL, None),  # [REQUIRED for TYPE=dipole] `int`s : select atoms whose radius-vectors to use as basis 
        # PIP CONSTRUCTION
        ('ORDER',         KeywordType.KEYWORD_REQUIRED, None),  # `int`  : maximum order of PIPs 
        ('SYMMETRY',      KeywordType.KEYWORD_REQUIRED, None),  # `int`s : permutational symmetry of the molecule | molecular pair
        ('PURIFY',        KeywordType.KEYWORD_OPTIONAL, False), # `bool` : use purified basis of PIPs
        ('ATOM_MAPPING',  KeywordType.KEYWORD_OPTIONAL, False), # `list` : mapping atoms->monomer (which atom belongs to which monomer)
        ('VARIABLES' ,    KeywordType.KEYWORD_REQUIRED, None), # `dict` : mapping interatomic distances->polynomial variables 
        ('SHARDED',       KeywordType.KEYWORD_OPTIONAL, False), # `bool` : enable data sharding for distributed full-batch training
    ]

    from operator import itemgetter
    for keyword in cfg_dataset.keys():
        assert keyword in list(map(itemgetter(0), KEYWORDS)), "Unknown keyword: {}".format(keyword)

    for keyword, keyword_type, default_value in KEYWORDS:
        if keyword_type == KeywordType.KEYWORD_REQUIRED:
            assert keyword in cfg_dataset, "Required keyword {} is missing".format(keyword)
        elif keyword_type == KeywordType.KEYWORD_OPTIONAL:
            cfg_dataset.setdefault(keyword, default_value)

    if typ == 'DIPOLE':
        assert 'ANCHOR_POSITIONS' in cfg_dataset
        assert not cfg_dataset['LOAD_GRADIENTS']

    VARIABLES_BLOCK_REQUIRED = ('INTRAMOLECULAR', 'INTERMOLECULAR', 'EXP_LAMBDA')
    for keyword in VARIABLES_BLOCK_REQUIRED:
        assert keyword in cfg_dataset['VARIABLES']

    cfg_dataset['TYPE'] = typ

    if not os.path.isdir(cfg_dataset['INTERIM_FOLDER']):
        os.makedirs(cfg_dataset['INTERIM_FOLDER']) # can create nested directories

    if not os.path.isdir(cfg_dataset['EXTERNAL_FOLDER']):
        os.makedirs(cfg_dataset['EXTERNAL_FOLDER']) # can create nested directories

    logging.info("Dataset options:")
    for keyword, value in cfg_dataset.items():
        logging.info("{:>25}: \t {}".format(keyword, value))

    train_fpath, val_fpath, test_fpath = make_dataset_fpaths(cfg_dataset)
    if not os.path.isfile(train_fpath) or not os.path.isfile(val_fpath) or not os.path.isfile(test_fpath):
        logging.info("Invoking make_dataset to create polynomial dataset")

        # we suppose that paths in YAML configuration are relative to BASEDIR (repo folder)
        source = [os.path.join(BASEDIR, path) for path in cfg_dataset['SOURCE']]

        dataset_fpaths = {"train" : train_fpath, "val": val_fpath, "test" : test_fpath}
        make_dataset(cfg_dataset, dataset_fpaths)
    else:
        logging.info("Dataset found.")

    if USE_WANDB:
        wandb.config = {
           "type"   : typ,
           "name"   : cfg_dataset['NAME'],
           "source" : cfg_dataset['SOURCE'],
        }

    train = PolyDataset.from_pickle(train_fpath)
    assert train.energy_limit == cfg_dataset['ENERGY_LIMIT']
    assert train.purify       == cfg_dataset['PURIFY']
    logging.info("Loading training dataset: {}; len: {}".format(train_fpath, len(train.y)))

    # Robust scale of gradient components on the training split.
    # MAD of componentwise deviations from zero (gradients are ~zero-mean on
    # average). Used to auto-derive the Huber cutoff delta = ~2 * MAD.
    if train.dy is not None:
        comp = train.dy.reshape(-1).to(TORCH_FLOAT)
        mad = comp.abs().median().item()
        train.mad_grad_components = mad
        logging.info("Gradient-component MAD (training split): {:.6e} cm-1/Bohr "
                     "(#components={})".format(mad, comp.numel()))

    val   = PolyDataset.from_pickle(val_fpath)
    assert val.energy_limit == cfg_dataset['ENERGY_LIMIT']
    assert val.purify       == cfg_dataset['PURIFY']
    logging.info("Loading validation dataset: {}; len: {}".format(val_fpath, len(val.y)))

    test  = PolyDataset.from_pickle(test_fpath)
    assert test.energy_limit == cfg_dataset['ENERGY_LIMIT']
    assert test.purify       == cfg_dataset['PURIFY']
    logging.info("Loading testing dataset: {}; len: {}".format(test_fpath, len(test.y)))

    return train, val, test

