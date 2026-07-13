import argparse
import logging
import os
import shutil
import sys
import time

import torch

from data_io import load_cfg, load_dataset, seed_torch
from trainers import get_trainer
from distributed import setup_distributed, cleanup
from distributed import is_main_process
from run_tracking import capture_provenance, collect_metrics, write_metrics

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

    MODEL_FOLDER = os.path.normpath(os.path.join(BASEDIR, args.model_folder))
    MODEL_NAME   = args.model_name

    assert os.path.isdir(MODEL_FOLDER), "Path to folder is invalid: {}".format(MODEL_FOLDER)

    cfg_path = os.path.join(MODEL_FOLDER, MODEL_NAME + ".yaml")
    assert os.path.isfile(cfg_path), "YAML configuration file does not exist at {}".format(cfg_path)

    cfg = load_cfg(cfg_path)
    logging.info("loaded configuration file from {}".format(cfg_path))

    if 'PRINT_PRECISION' in cfg:
        PRINT_PRECISION = cfg['PRINT_PRECISION']

    # Give each run its own self-contained directory under <model_folder>/runs/,
    # mirroring the layout produced by run_tracking/migrate.py: the config is
    # copied in and every artifact (log, checkpoint, provenance, metrics,
    # diagnostics, eval sidecars) is written there, stem-named. Re-point
    # --model_folder at the run dir with the same --model_name to resume/evaluate.
    # Skip re-nesting when --model_folder already sits inside a runs/ tree.
    if os.path.basename(os.path.dirname(MODEL_FOLDER)) != "runs":
        RUN_DIR = os.path.join(MODEL_FOLDER, "runs", MODEL_NAME)
        os.makedirs(RUN_DIR, exist_ok=True)
        run_cfg_path = os.path.join(RUN_DIR, MODEL_NAME + ".yaml")
        if os.path.abspath(run_cfg_path) != os.path.abspath(cfg_path):
            shutil.copy2(cfg_path, run_cfg_path)
        MODEL_FOLDER = RUN_DIR

    # Stem shared by the run's log and its provenance/metrics manifests, so the
    # tracking artifacts sit beside the log with a matching name.
    RUN_STEM = args.log_name if args.log_name is not None else MODEL_NAME

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

    logging.info("Run directory: {}".format(MODEL_FOLDER))

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

    # Capture run provenance (git SHA + working-tree diff, resolved config, env)
    # so this run is reproducible. Best-effort; only the main process writes,
    # which is why this follows setup_distributed() (is_main_process gates on rank).
    if is_main_process():
        capture_provenance(cfg, MODEL_FOLDER, RUN_STEM,
                           extra={"world_size": world_size})

    trainer = get_trainer(MODEL_FOLDER, MODEL_NAME, chk_path, cfg, train, val, test,
                          rank=rank, world_size=world_size, local_rank=local_rank)

    run_start = time.time()
    try:
        trainer.train_model()
        trainer.model_eval()
    finally:
        # Emit structured final metrics for the leaderboard/report. Best-effort;
        # runs even if training raised, so partial runs are still recorded.
        if is_main_process():
            metrics = collect_metrics(trainer, wall_time_s=time.time() - run_start)
            write_metrics(MODEL_FOLDER, RUN_STEM, metrics)
        cleanup()
