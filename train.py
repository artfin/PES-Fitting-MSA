import logging
import numpy as np
import pandas as pd
import os
from pathlib import Path
import random
import uuid

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import LBFGS, Adam

import optuna
from optuna.trial import TrialState

from dataset import PolyDataset
from util import IdentityScaler, StandardScaler
from util import RMSELoss

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

HTOCM = 2.194746313702e5

# scaling of X (polynomial values) and y (energies)
# TODO: check out several other scaling transformations (minmax, etc)
SCALE_OPTIONS = [None, "std"]
SCALE_PARAMS = {
    "Xscale" : "std",
    "yscale" : "std"
}

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def count_params(model):
    nparams = 0
    for name, param in model.named_parameters():
        params = torch.tensor(param.size())
        nparams += torch.prod(params, 0)

    return nparams

def perform_lstsq(X, y, show_results=False):
    X_train, y_train, X_val, y_val, X_test, y_test, _, _ = split_train_val_test(X, y, scale_params={"Xscale": None, "yscale": None})

    coeff = torch.linalg.lstsq(X_train, y_train, driver='gelss').solution

    pred_train = X_train @ coeff
    pred_val   = X_val   @ coeff
    pred_test  = X_test  @ coeff

    RMSE = RMSELoss()
    rmse_train = RMSE(y_train, pred_train) * HTOCM
    rmse_val   = RMSE(y_val, pred_val) * HTOCM
    rmse_test  = RMSE(y_test, pred_test) * HTOCM

    logging.info("A rundown for the least-squares regression [in matrix form]:")
    logging.info("  RMSE train      = {:.2f} cm-1".format(rmse_train))
    logging.info("  RMSE validation = {:.2f} cm-1".format(rmse_val))
    logging.info("  RMSE test       = {:.2f} cm-1".format(rmse_test))

    if show_results:
        MAX_ENERGY = 3000.0 # cm-1
        idx = y_train.numpy() < MAX_ENERGY / HTOCM
        calc_energy = y_train.numpy()[idx] * HTOCM
        fit_energy  = pred_train.numpy()[idx] * HTOCM

        abs_error = calc_energy - fit_energy

        plt.figure(figsize=(10, 10))
        plt.scatter(calc_energy, abs_error, marker='o', facecolors='none', color='k')

        plt.xlim((-500.0, MAX_ENERGY))
        plt.ylim((-100.0, 100.0))

        plt.xlabel("Energy, cm-1")
        plt.ylabel("Absolute error, cm-1")

        plt.show()
        #for n in range(nconfigs_train):
        #    print("{} \t {} \t {}".format(
        #        y[n].item() * HTOCM, y_pred[n].item() * HTOCM, (y[n] - y_pred[n]).item() * HTOCM
        #    ))


class EarlyStopping:
    def __init__(self, patience=10, tol=0.1, chk_path='checkpoint.pt'):
        """
        patience : how many epochs to wait after the last time the monitored quantity [validation loss] has improved
        tol:       minimum change in the monitored quantity to qualify as an improvement
        path:      path for the checkpoint to be saved to
        """
        self.patience = patience
        self.tol = tol
        self.chk_path = chk_path

        self.counter = 0
        self.best_score = None
        self.status = False

    def __call__(self, score, model, xscaler, yscaler):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, xscaler, yscaler)

        elif score < self.best_score and (self.best_score - score) > self.tol:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model, xscaler, yscaler)

        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.status = True

        logging.debug("Best validation RMSE: {:.2f}; current validation RMSE: {:.2f}".format(self.best_score, score))
        logging.debug("ES counter: {}; ES patience: {}".format(self.counter, self.patience))

    def save_checkpoint(self, model, xscaler, yscaler):
        logging.debug("Saving the checkpoint")

        architecture = [m.out_features for m in next(model.modules()) if isinstance(m, torch.nn.modules.linear.Linear)]
        architecture = tuple(architecture[:-1])

        checkpoint = {
            "model":        model.state_dict(),
            "architecture": architecture,
            "X_mean":       xscaler.mean,
            "X_std":        xscaler.std,
            "X_zero_idx":   xscaler.zero_idx,
            "y_mean":       yscaler.mean,
            "y_std":        yscaler.std,
            "y_zero_idx":   yscaler.zero_idx,
        }
        torch.save(checkpoint, self.chk_path)

def split_train_val_test(X, y, scale_params):
    """
    # TODO: implement energy-based splitting of dataset
    """
    sz = y.size()[0]

    ids = np.random.permutation(sz)
    val_start  = int(sz * 0.8)
    test_start = int(sz * 0.9)
    train_ids = ids[:val_start]
    val_ids = ids[val_start:test_start]
    test_ids = ids[test_start:]

    X_train, y_train = X[train_ids], y[train_ids]
    X_val, y_val     = X[val_ids], y[val_ids]
    X_test, y_test   = X[test_ids],  y[test_ids]

    assert scale_params["Xscale"] in SCALE_OPTIONS
    if scale_params["Xscale"] == "std":
        xscaler = StandardScaler()
    elif scale_params["Xscale"] == None:
        xscaler = IdentityScaler()
    else:
        raise ValueError("unreachable")

    assert scale_params["yscale"] in SCALE_OPTIONS
    if scale_params["yscale"] == "std":
        yscaler = StandardScaler()
    elif scale_params["yscale"] == None:
        yscaler = IdentityScaler()
    else:
        raise ValueError("unreachable")

    xscaler.fit(X_train)
    Xtr       = xscaler.transform(X)
    Xtr_train = xscaler.transform(X_train)
    Xtr_val   = xscaler.transform(X_val)
    Xtr_test  = xscaler.transform(X_test)

    yscaler.fit(y_train)
    ytr       = yscaler.transform(y)
    ytr_train = yscaler.transform(y_train)
    ytr_val   = yscaler.transform(y_val)
    ytr_test  = yscaler.transform(y_test)

    return Xtr_train, ytr_train, Xtr_val, ytr_val, Xtr_test, ytr_test, xscaler, yscaler

def optuna_define_model(trial, NPOLY):
    # TODO: maybe add a little Dropout as a means to counteract overfitting

    # we optimize the number of layers and hidden units
    n_layers = trial.suggest_int("n_layers", low=2, high=2)

    layers = []

    in_features = NPOLY
    for i in range(n_layers):
        out_features = trial.suggest_int("n_hidden_l{}".format(i), low=5, high=20)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.Tanh())
        #dropout = trial.suggest_float("droupout_l{}".format(i), low=0.01, high=0.1)
        #layers.append(nn.Dropout(dropout))

        in_features = out_features

    layers.append(nn.Linear(in_features, 1))
    model = nn.Sequential(*layers)

    sigmoid_gain = torch.nn.init.calculate_gain("tanh")
    for child in model.children():
        if isinstance(child, nn.Linear):
            for _ in range(0, len(layers)):
                torch.nn.init.xavier_uniform_(child.weight, gain=sigmoid_gain)
                if child.bias is not None:
                    torch.nn.init.zeros_(child.bias)

    return model

def define_model(architecture, NPOLY):
    """
    architecture:
        tuple (h0, h1, ...)
        It implies the following MLP architecture:
            nn.Linear(NPOLY, h0)
            nn.Tanh()
            nn.Linear(h0, h1)
            nn.Tanh()
            ...
            nn.Tanh()
            nn.Linear(hk, 1)
    """
    layers = []

    in_features  = NPOLY
    # to allow constructing LinearRegressor with architecture=() 
    out_features = NPOLY

    for i in range(len(architecture)):
        out_features = architecture[i]
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.Tanh())

        in_features = out_features

    layers.append(nn.Linear(out_features, 1))
    model = nn.Sequential(*layers)

    sigmoid_gain = torch.nn.init.calculate_gain("tanh")
    for child in model.children():
        if isinstance(child, nn.Linear):
            for _ in range(0, len(layers)):
                torch.nn.init.xavier_uniform_(child.weight, gain=sigmoid_gain)
                if child.bias is not None:
                    torch.nn.init.zeros_(child.bias)

    return model

def build_model(trial=None, architecture="optuna", dataset_path=None, optuna_run_folder=None):
    logging.info("Loading data from dataset_path = {}".format(dataset_path))
    d = torch.load(dataset_path)
    X, y = d["X"], d["y"]
    logging.info("Loaded the data.")

    X_train, y_train, X_val, y_val, X_test, y_test, xscaler, yscaler = split_train_val_test(X, y, scale_params=SCALE_PARAMS)
    X = xscaler.transform(X)
    y = yscaler.transform(y)

    if architecture == "optuna":
        chk_path = os.path.join(optuna_run_folder, "model-{}.pt".format(trial.number))
        logging.info("Saving current model to chk_path={}".format(chk_path))
    else:
        chk_path = "checkpoint.pt"

    if SCALE_PARAMS["yscale"] == "std":
        rmse_descaler = yscaler.std.item()

    elif SCALE_PARAMS["yscale"] == None:
        rmse_descaler = 1.0
    else:
        raise ValueError("unreachable")

    NPOLY = X_train.size()[1]
    if architecture == "optuna":
        model = optuna_define_model(trial, NPOLY)
    else:
        model = define_model(architecture, NPOLY)

    print("model: {}".format(model))
    model.double()

    optimizer = LBFGS(model.parameters(), lr=1.0, line_search_fn='strong_wolfe', tolerance_grad=1e-14, tolerance_change=1e-14, max_iter=100)

    METRIC_TYPES = ['RMSE', 'MSE']
    METRIC_TYPE = 'MSE'
    assert METRIC_TYPE in METRIC_TYPES

    logging.info("METRIC_TYPE = {}".format(METRIC_TYPE))

    if METRIC_TYPE == 'MSE':
        metric = nn.MSELoss()
    elif METRIC_TYPE == 'RMSE':
        metric = RMSELoss()
    else:
        raise ValueError("unreachable")

    SCHEDULER_PATIENCE = 10
    RMSE_TOL           = 0.5 # cm-1
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, threshold=RMSE_TOL, threshold_mode='abs', cooldown=2, patience=SCHEDULER_PATIENCE)

    ES_START_EPOCH = 10
    es = EarlyStopping(patience=15, tol=RMSE_TOL, chk_path=chk_path)

    MAX_EPOCHS = 500

    ################ START TRAINING #######################
    for epoch in range(MAX_EPOCHS):
        def closure():
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = metric(y_pred, y_train)
            loss.backward()
            return loss

        model.train()
        optimizer.step(closure)
        loss = closure()

        with torch.no_grad():
            model.eval()
            pred_val = model(X_val)
            loss_val = metric(pred_val, y_val)

        if METRIC_TYPE == 'MSE':
            rmse_val   = torch.sqrt(loss_val) * rmse_descaler * HTOCM
            rmse_train = torch.sqrt(loss)     * rmse_descaler * HTOCM
        elif METRIC_TYPE == 'RMSE':
            rmse_val   = loss_val * rmse_descaler * HTOCM
            rmse_train = loss     * rmse_descaler * HTOCM
        else:
            raise ValueError("unreachable")

        scheduler.step(rmse_val)
        lr = get_lr(optimizer)
        logging.info("Current learning rate: {:.2e}".format(lr))

        if epoch > ES_START_EPOCH:
            es(rmse_val, model, xscaler, yscaler)

            if es.status:
                logging.info("Invoking early stop")
                break

        logging.info("Epoch: {}; train RMSE: {:.2f} cm-1; validation RMSE: {:.2f}\n".format(
            epoch, rmse_train, rmse_val
        ))
    ################ END TRAINING #######################

    checkpoint = torch.load(chk_path)
    model.load_state_dict(checkpoint["model"])
    logging.info("\nReloading best model from the last checkpoint...")

    with torch.no_grad():
        pred_val = model(X_val)
        loss_val = metric(pred_val, y_val)

        pred_test = model(X_test)
        loss_test = metric(pred_test, y_test)

        pred_full = model(X)
        loss_full = metric(pred_full, y)

        if METRIC_TYPE == 'MSE':
            rmse_val  = torch.sqrt(loss_val)  * rmse_descaler * HTOCM
            rmse_test = torch.sqrt(loss_test) * rmse_descaler * HTOCM
            rmse_full = torch.sqrt(loss_full) * rmse_descaler * HTOCM
        elif METRIC_TYPE == 'RMSE':
            rmse_val  = loss_val  * rmse_descaler * HTOCM
            rmse_test = loss_test * rmse_descaler * HTOCM
            rmse_full = loss_full * rmse_descaler * HTOCM

        logging.info("Final evaluation:")
        logging.info("Validation RMSE: {:.2f} cm-1".format(rmse_val))
        logging.info("Test RMSE:       {:.2f} cm-1".format(rmse_test))
        logging.info("Full RMSE:       {:.2f} cm-1".format(rmse_full))

    return rmse_val


def optuna_neural_network_achitecture_search(dataset_path):
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize")

    uniq_id = uuid.uuid4()
    OPTUNA_RUN_FOLDER = "optuna-run-{}".format(uniq_id)
    Path(OPTUNA_RUN_FOLDER).mkdir(exist_ok=True)
    logging.info("Saving models to OPTUNA_RUN_FOLDER={}".format(OPTUNA_RUN_FOLDER))

    objective = lambda trial: build_model(trial, architecture="optuna", dataset_path=dataset_path, optuna_run_folder=OPTUNA_RUN_FOLDER)
    study.optimize(objective, n_trials=5, timeout=600)

    pruned_trials   = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logging.info("Study statistics:")
    logging.info("  Number of finished trials: {}".format(study.trials))
    logging.info("  Number of pruned trials:   {}".format(pruned_trials))
    logging.info("  Number of complete trials: {}".format(complete_trials))

    best_trial = study.best_trial
    logging.info("Best trial:")
    logging.info("  Best target value: {}".format(best_trial.value))
    logging.info("  Parameters:")

    for key, value in best_trial.params.items():
        logging.info("    {}: {}".format(key, value))

################ H2-H2O #######################
if __name__ == "__main2__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    wdir     = './H2-H2O'
    order    = "3"
    symmetry = "2 2 1"
    dataset = PolyDataset(wdir=wdir, config_fname="points.dat", order=order, symmetry=symmetry)

    X, y = dataset.X, dataset.y
    torch.save({"X" : X, "y" : y}, "H2-H2O/dataset.pt")

    perform_lstsq(X, y)

    optuna_neural_network_achitecture_search(dataset_path="H2-H2O/dataset.pt")

    ###
    #architecture = (10, 10)
    #build_model(trial=None, architecture=architecture, dataset_path="H2-H2O/dataset.pt")
    ###

    #show_feature_distribution(X_train, idx=0)
    #show_energy_distribution(y)

    #nparams = count_params(model)
    #logging.info("number of parameters: {}".format(nparams))

################ CH4-N2 #######################
if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    wdir     = "CH4-N2"
    order    = "4"
    symmetry = "4 2 1"
    dataset = PolyDataset(wdir=wdir, config_fname="ch4-n2-energies.xyz", order=order, symmetry=symmetry)

    X, y = dataset.X, dataset.y
    torch.save({"X" : X, "y" : y}, "CH4-N2/dataset.pt")

    dataset_path = "CH4-N2/dataset.pt"
    print("Loading data from dataset_path = {}".format(dataset_path))
    d = torch.load(dataset_path)
    X, y = d["X"], d["y"]

    #show_feature_distribution(X_train, idx=0)
    show_energy_distribution(y, xlim=(-400, 3000))
    #show_train_val_test_energy_distribution(X, y)

    perform_lstsq(X, y, show_results=False)

    #optuna_neural_network_achitecture_search(dataset_path="CH4-N2/dataset.pt")

    #### 
    #architecture = (10, 10)
    #build_model(trial=None, architecture=architecture, dataset_path="CH4-N2/dataset.pt")
    #### 

