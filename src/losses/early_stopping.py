import logging

from config import PRINT_TRAINING_STEPS, PRINT_PRECISION
from data_io import save_checkpoint
from distributed import is_main_process


class EarlyStopping:
    def __init__(self, patience, tol, chk_path):
        """
        patience : how many epochs to wait after the last time the monitored quantity [validation loss] has improved
        tol:       minimum change in the monitored quantity to qualify as an improvement
        path:      path for the checkpoint to be saved to
        """
        self.patience = patience
        self.tol      = tol
        self.chk_path = chk_path

        self.counter    = 0
        self.best_score = None
        self.status     = False

    def reset(self):
        self.counter    = 0
        self.best_score = None
        self.status     = False

    def __call__(self, epoch, score, model, xscaler, yscaler, meta_info):
        if self.best_score is None:
            self.best_score = score
            save_checkpoint(model, xscaler, yscaler, meta_info, self.chk_path)
        elif score < self.best_score and (self.best_score - score) > self.tol:
            self.best_score = score
            self.counter = 0
            save_checkpoint(model, xscaler, yscaler, meta_info, self.chk_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.status = True

        if epoch % PRINT_TRAINING_STEPS == 0:
            if is_main_process():
                logging.info("(Early Stopping) Best validation RMSE: {1:.{0}f}; current validation RMSE: {2:.{0}f}".format(PRINT_PRECISION, self.best_score, score))
                logging.info("(Early Stopping) counter: {}; patience: {}; tolerance: {}".format(self.counter, self.patience, self.tol))
