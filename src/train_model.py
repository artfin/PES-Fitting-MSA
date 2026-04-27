import argparse
import logging
import os
import sys

import torch

from data_io import load_cfg, load_dataset, seed_torch
from trainer import Training
from distributed import setup_distributed, cleanup

import pathlib
BASEDIR = pathlib.Path(__file__).parent.parent.resolve()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRINT_PRECISION = 3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder",  required=True, type=str, help="path to folder with YAML configuration file")
    parser.add_argument("--model_name",    required=True, type=str, help="the name of the YAML configuration file without extension")
    parser.add_argument("--log_name",      required=False, type=str, default=None, help="name of the logging file without extension")
    parser.add_argument("--chk_name",      required=False, type=str, default=None, help="name of the general checkpoint without extension")

    args = parser.parse_args()

    MODEL_FOLDER = os.path.join(BASEDIR, args.model_folder)
    MODEL_NAME   = args.model_name

    assert os.path.isdir(MODEL_FOLDER), "Path to folder is invalid: {}".format(MODEL_FOLDER)

    cfg_path = os.path.join(MODEL_FOLDER, MODEL_NAME + ".yaml")
    assert os.path.isfile(cfg_path), "YAML configuration file does not exist at {}".format(cfg_path)

    cfg = load_cfg(cfg_path)
    logging.info("loaded configuration file from {}".format(cfg_path))

    if 'PRINT_PRECISION' in cfg:
        PRINT_PRECISION = cfg['PRINT_PRECISION']

    if args.log_name is not None:
        log_path = os.path.join(MODEL_FOLDER, args.log_name + ".log")
    else:
        log_path = os.path.join(MODEL_FOLDER, MODEL_NAME + ".log")

    if args.chk_name is not None:
        chk_path = os.path.join(MODEL_FOLDER, args.chk_name + ".pt")
    else:
        chk_path = os.path.join(MODEL_FOLDER, MODEL_NAME + ".pt")

    if os.path.exists(log_path):
        os.remove(log_path)

    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.handlers = []

    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.INFO)

    seed_torch()
    if DEVICE.type == 'cuda':
        logging.info("CUDA Device found: {}".format(torch.cuda.get_device_name(0)))
        logging.info("Memory usage:")
        logging.info("Allocated: {} GB".format(round(torch.cuda.memory_allocated(0)/1024**3, 1)))
    else:
        logging.info("No CUDA Device Found. Using CPU")

    import psutil
    logging.info("[psutil] Memory status: \n {}".format(psutil.virtual_memory()))

    assert 'TYPE' in cfg
    typ = cfg['TYPE']
    assert typ in ('ENERGY', 'DIPOLE', 'DIPOLEQ', 'DIPOLEC')

    train, val, test = load_dataset(cfg['DATASET'], typ)

    USE_WANDB = False
    if USE_WANDB:
        cfg_dataset = cfg['DATASET']
        project_name = cfg_dataset['NAME'] + "-" + cfg['TYPE']
        import wandb
        wandb.init(project=project_name)
        wandb.config = {
           "type"   : typ,
           "name"   : cfg_dataset['NAME'],
           "source" : cfg_dataset['SOURCE'],
        }

    rank, world_size, local_rank = setup_distributed()

    t = Training(MODEL_FOLDER, MODEL_NAME, chk_path, cfg, train, val, test,
                 rank=rank, world_size=world_size, local_rank=local_rank)

    try:
        t.train_model()
        t.model_eval()
    finally:
        cleanup()
