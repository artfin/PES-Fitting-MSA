import logging
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from config import TORCH_FLOAT
from build_model import build_network, QModel
from data_io import fit_scalers_to_train_dataset, apply_scalers_on_dataset, load_from_checkpoint, save_checkpoint
from losses import EarlyStopping, WMSELoss_TrustRegion_wgradients
from distributed import is_main_process, shard_dataset

import pathlib
BASEDIR = pathlib.Path(__file__).parent.parent.parent.resolve()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRINT_PRECISION = 3


def count_params(model):
    nparams = 0
    for name, param in model.named_parameters():
        params = torch.tensor(param.size())
        nparams += torch.prod(params, 0)
    return nparams

class TrainingBase:
    def __init__(self, model_folder, model_name, chk_path, cfg, train, val, test, rank=0, world_size=1, local_rank=0):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank

        cfg_dataset = cfg.get('DATASET', {})
        sharding_enabled = cfg_dataset.get('SHARDED', False) and self.world_size > 1

        # Save full training data for scaler fitting BEFORE sharding
        full_train_X = train.X
        full_train_y = train.y

        # Data sharding for distributed full-batch training
        if sharding_enabled:
            from distributed import shard_dataset
            train, dropped_train = shard_dataset(train, self.rank, self.world_size)
            val, dropped_val = shard_dataset(val, self.rank, self.world_size)
            test, dropped_test = shard_dataset(test, self.rank, self.world_size)
            if is_main_process():
                logging.info(f"Data sharding enabled: {len(train.y)} train / {len(val.y)} val / {len(test.y)} test per rank")
                total_dropped = dropped_train + dropped_val + dropped_test
                if total_dropped > 0:
                    logging.info(f"Dropped {total_dropped} samples to ensure equal shards")

        EVENTDIR = "runs"
        if not os.path.isdir(EVENTDIR):
            os.makedirs(EVENTDIR)

        self.model_name = model_name

        self.model_folder = model_folder

        self.cfg = cfg

        self.train = train
        self.val   = val
        self.test  = test

        cfg_model = cfg.get('MODEL', None)

        pretrained_model, pretrained_xscaler, pretrained_yscaler = self.load_pretrained_model_if_configured()
        if pretrained_model is not None:
            self.model = pretrained_model
            self.xscaler = pretrained_xscaler
            self.yscaler = pretrained_yscaler

            logging.info("Applying xscaler and yscaler loaded from pretrained model on the new dataset") 
            apply_scalers_on_dataset(self.train, self.val, self.test, self.xscaler, self.yscaler)

            if cfg_model is not None:
                print("\n")
                logging.warning("Configuration provided within the MODEL is going to be ignored! The configuration of the pretrained model will be retained.\n")
        else:
            if self.cfg['TYPE'] == 'ENERGY':
                self.model = build_network(cfg_model, hidden_dims=cfg['MODEL']['HIDDEN_DIMS'], input_features=train.NPOLY, output_features=1)
            elif self.cfg['TYPE'] == 'DIPOLE':
                self.model = build_network(cfg_model, hidden_dims=cfg['MODEL']['HIDDEN_DIMS'][0], input_features=train.NPOLY, output_features=3)
            elif self.cfg['TYPE'] == 'DIPOLEQ':
                self.model = QModel(cfg_model, input_features=train.NPOLY, output_features=[len(natoms) for natoms in train.symmetry.values()])
            elif self.cfg['TYPE'] == 'DIPOLEC':
                self.model = build_network(cfg_model, input_features=3 * train.NATOMS, output_features=1)
            else:
                assert False, "unreachable"

            logging.info("Fitting scalers to full training dataset (before sharding)...\n")
            self.xscaler, self.yscaler = fit_scalers_to_train_dataset(train, cfg['DATASET'], X=full_train_X, y=full_train_y)
            apply_scalers_on_dataset(self.train, self.val, self.test, self.xscaler, self.yscaler)

        logging.info("Using the NN model structured as {}".format(self.model))
        nparams = count_params(self.model)
        logging.info("Number of parameters: {}".format(nparams))

        self.cfg_solver = cfg['TRAINING']
        self.grad_clip_norm = self.cfg_solver.get('GRAD_CLIP_NORM', None)

        self.cfg_loss = cfg['LOSS']
        self.loss_fn  = self.build_loss()
        self.loss_fn.set_scale(self.yscaler.mean_, self.yscaler.scale_)

        # Track when gradient training starts for progressive G_LAMBDA ramping
        if self.cfg_loss['USE_GRADIENTS'] and self.cfg_loss.get('USE_GRADIENTS_AFTER_EPOCH') is None:
            self.gradient_start_epoch = 0
        else:
            self.gradient_start_epoch = None

        self.cfg_regularization = cfg.get('REGULARIZATION', None)
        self.regularization = self.build_regularization()

        self.cfg_batch = self._parse_batch_cfg(cfg.get('BATCH', None))

        # Data sharding is only compatible with full-batch L-BFGS
        if cfg_dataset.get('SHARDED', False) and bool(self.cfg_batch.get('MULTIBATCH_ENABLED', False)):
            raise ValueError(
                "DATASET.SHARDED=true is incompatible with BATCH.MULTIBATCH_ENABLED=true. "
                "Data sharding only works with full-batch L-BFGS."
            )

        self.cfg_debug = cfg.get('DEBUG', {})

        self.chk_path = chk_path
        self.es = self.build_early_stopper()
        self.meta_info = {
            "NPOLY":    self.train.NPOLY,
            "NMON":     self.train.NMON,
            "NATOMS":   self.train.NATOMS,
            "symmetry": self.train.symmetry,
            "order":    self.train.order,
        }

        # Trust-region diagnostics state (lazy init in train_epoch).
        # _prev_trust_mask: bool tensor (N,) -- last epoch's membership
        # _prev_train_gradient_errors: float tensor (N,) -- per-config train gradient
        #     RMSE from the last validation pass; used to test the eviction signal
        # _trust_flip_count: int tensor (N,) -- cumulative # times each config
        #     has toggled in/out of the trust set across training
        # _trust_history_path: where to write per-epoch CSV summary
        self._prev_trust_mask = None
        self._prev_train_gradient_errors = None
        self._trust_flip_count = None
        self._trust_history_path = os.path.join(
            self.model_folder, "{}.trust_history.csv".format(model_name)
        )
        self._trust_history_initialized = False

        # Per-epoch gradient-loss contribution + phi histogram (active set only).
        self._gradient_diag_path = os.path.join(
            self.model_folder, "{}.gradient_diagnostics.csv".format(model_name)
        )
        self._gradient_diag_initialized = False

        # L-BFGS line-search telemetry (state inspected after optimizer.step).
        self._lbfgs_diag_path = os.path.join(
            self.model_folder, "{}.lbfgs_diagnostics.csv".format(model_name)
        )
        self._lbfgs_diag_initialized = False
        self._lbfgs_prev_n_iter = 0
        self._lbfgs_prev_func_evals = 0
        self._last_vendored_closure_eval = 0

        # Distributed per-rank diagnostics (local metrics before averaging)
        self._dist_diag_path = os.path.join(
            self.model_folder, "{}.distributed_diagnostics.csv".format(model_name)
        )
        self._dist_diag_initialized = False

        # MGDA (Multi-objective Gradient Descent Algorithm) state
        # Now uses GradNorm: normalized gradients + adaptive alpha from loss ratios
        self._mgda_alpha_ema = None  # EMA-smoothed alpha value
        self._mgda_energy_loss_ema = None  # EMA of energy loss for adaptive alpha
        self._mgda_gradient_loss_ema = None  # EMA of gradient loss for adaptive alpha
        self._mgda_diag_path = os.path.join(
            self.model_folder, "{}.mgda_diagnostics.csv".format(model_name)
        )
        self._mgda_diag_initialized = False

    def _log(self, msg):
        """Rank-0 only logging helper."""
        if is_main_process():
            logging.info(msg)
    def reset_weights(self):
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                logging.info(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()
    def load_pretrained_model_if_configured(self):
        self.cfg_pretrained_model_settings = self.cfg.get('PRETRAINED_MODEL_SETTINGS', None)
        if self.cfg_pretrained_model_settings is None: 
            return None, None, None 

        pretrained_source_path = self.cfg_pretrained_model_settings.get('SOURCE', None)
        assert pretrained_source_path is not None, "SOURCE path for pretrained model is not provided"

        pretrained_source_path = os.path.join(self.model_folder, pretrained_source_path)
        print("\n")
        logging.info("Looking for pretrained model (.pt) in {}".format(pretrained_source_path))
        model, xscaler, yscaler = load_from_checkpoint(pretrained_source_path)


        return model, xscaler, yscaler
    def continue_from_checkpoint(self, chkpath):
        assert os.path.exists(chkpath)

        self.reset_weights()
        checkpoint = torch.load(chkpath, map_location=torch.device(DEVICE))
        self.model.load_state_dict(checkpoint["model"])

        self.train_model()
    def model_eval(self):
        self.test.X = self.test.X.to(self.device)
        self.test.y = self.test.y.to(self.device)

        if self.test.dX is not None:
            self.test.dX = self.test.dX.to(self.device)
            self.test.dy = self.test.dy.to(self.device)

        # Calling model.eval() will change the behavior of some layers, 
        # such as nn.Dropout, which will be disabled, and nn.BatchNormXd, which will use the running stats during evaluation.
        self.model.eval()

        if self.cfg_loss['USE_GRADIENTS']:
            # Use memory-efficient gradient evaluation (no create_graph needed)
            train_y_pred, train_dy_pred = self.compute_gradients_eval(self.train)
            val_y_pred, val_dy_pred     = self.compute_gradients_eval(self.val)
            test_y_pred, test_dy_pred   = self.compute_gradients_eval(self.test)

            # Trust-region loss expects an extra trust_indices argument;
            # for final evaluation we evaluate gradients on the full dataset.
            if isinstance(self.loss_fn, WMSELoss_TrustRegion_wgradients):
                train_indices = torch.arange(len(self.train.y), device=self.device)
                val_indices   = torch.arange(len(self.val.y), device=self.device)
                test_indices  = torch.arange(len(self.test.y), device=self.device)

                loss_train_e, loss_train_g = self.loss_fn.forward_separate(self.train.y, train_y_pred, self.train.dy, train_dy_pred, train_indices)
                loss_val_e, loss_val_g     = self.loss_fn.forward_separate(self.val.y, val_y_pred, self.val.dy, val_dy_pred, val_indices)
                loss_test_e, loss_test_g   = self.loss_fn.forward_separate(self.test.y, test_y_pred, self.test.dy, test_dy_pred, test_indices)
            else:
                loss_train_e, loss_train_g = self.loss_fn.forward_separate(self.train.y, train_y_pred, self.train.dy, train_dy_pred)
                loss_val_e, loss_val_g     = self.loss_fn.forward_separate(self.val.y, val_y_pred, self.val.dy, val_dy_pred)
                loss_test_e, loss_test_g   = self.loss_fn.forward_separate(self.test.y, test_y_pred, self.test.dy, test_dy_pred)

            if self.world_size > 1:
                loss_train_e = reduce_mean(loss_train_e)
                loss_train_g = reduce_mean(loss_train_g)
                loss_val_e   = reduce_mean(loss_val_e)
                loss_val_g   = reduce_mean(loss_val_g)
                loss_test_e  = reduce_mean(loss_test_e)
                loss_test_g  = reduce_mean(loss_test_g)

            self._log("Model evaluation after training:")
            self._log("Train      loss: {1:.{0}f} cm-1; gradient loss: {2:.{0}f} cm-1/bohr".format(PRINT_PRECISION, loss_train_e, loss_train_g))
            self._log("Validation loss: {1:.{0}f} cm-1; gradient loss: {2:.{0}f} cm-1/bohr".format(PRINT_PRECISION, loss_val_e, loss_val_g))
            self._log("Test       loss: {1:.{0}f} cm-1; gradient loss: {2:.{0}f} cm-1/bohr".format(PRINT_PRECISION, loss_test_e, loss_test_g))

        elif self.cfg['TYPE'] == 'ENERGY':
            # To disable the gradient calculation, set the .requires_grad attribute of all parameters to False 
            # or wrap the forward pass into with torch.no_grad().
            with torch.no_grad():
                pred_train = self.model(self.train.X)
                loss_train = self.loss_fn(self.train.y, pred_train)

                pred_val   = self.model(self.val.X)
                loss_val   = self.loss_fn(self.val.y, pred_val)

                pred_test  = self.model(self.test.X)
                loss_test  = self.loss_fn(self.test.y, pred_test)

            if self.world_size > 1:
                loss_train = reduce_mean(loss_train)
                loss_val   = reduce_mean(loss_val)
                loss_test  = reduce_mean(loss_test)

            self._log("Model evaluation after training:")
            self._log("Train      loss: {1:.{0}f} cm-1".format(PRINT_PRECISION, loss_train))
            self._log("Validation loss: {1:.{0}f} cm-1".format(PRINT_PRECISION, loss_val))
            self._log("Test       loss: {1:.{0}f} cm-1".format(PRINT_PRECISION, loss_test))

        elif self.cfg['TYPE'] == 'DIPOLEQ':
            # To disable the gradient calculation, set the .requires_grad attribute of all parameters to False 
            # or wrap the forward pass into with torch.no_grad().
            with torch.no_grad():
                train_q_pred   = self.model(self.train.X)
                train_X_inf    = torch.zeros_like(self.train.X).cpu()
                train_X_inf_tr = torch.from_numpy(self.xscaler.transform(train_X_inf)).to(self.device)
                train_q_inf    = self.model(train_X_inf_tr)
                train_q_corr   = train_q_pred - train_q_inf
                dip_pred_train = torch.einsum('ijk,ij->ik', self.train.xyz_ordered.to(TORCH_FLOAT), train_q_corr)
                loss_train     = self.loss_fn(self.train.y, dip_pred_train)

                val_q_pred   = self.model(self.val.X)
                val_X_inf    = torch.zeros_like(self.val.X).cpu()
                val_X_inf_tr = torch.from_numpy(self.xscaler.transform(val_X_inf)).to(self.device)
                val_q_inf    = self.model(val_X_inf_tr)
                val_q_corr   = val_q_pred - val_q_inf
                dip_pred_val = torch.einsum('ijk,ij->ik', self.val.xyz_ordered.to(TORCH_FLOAT), val_q_corr)
                loss_val     = self.loss_fn(self.val.y, dip_pred_val)

                test_q_pred   = self.model(self.test.X)
                test_X_inf    = torch.zeros_like(self.test.X).cpu()
                test_X_inf_tr = torch.from_numpy(self.xscaler.transform(test_X_inf)).to(self.device)
                test_q_inf    = self.model(test_X_inf_tr)
                test_q_corr   = test_q_pred - test_q_inf
                dip_pred_test = torch.einsum('ijk,ij->ik', self.test.xyz_ordered.to(TORCH_FLOAT), test_q_corr)
                loss_test     = self.loss_fn(self.test.y, dip_pred_test)

            if self.world_size > 1:
                loss_train = reduce_mean(loss_train)
                loss_val   = reduce_mean(loss_val)
                loss_test  = reduce_mean(loss_test)

            self._log("Model evluation after training:")
            self._log("Train      loss: {1:{0}f}".format(PRINT_PRECISION, loss_train))
            self._log("Validation loss: {1:{0}f}".format(PRINT_PRECISION, loss_val))
            self._log("Test       loss: {1:{0}f}".format(PRINT_PRECISION, loss_test))

        else:
            assert False, "unreachable"

