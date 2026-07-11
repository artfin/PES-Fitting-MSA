import logging
import os
import random
import time
import timeit
from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from config import DEVICE, PRINT_TRAINING_STEPS, PRINT_PRECISION, USE_WANDB
from data_io import fit_scalers_to_train_dataset, apply_scalers_on_dataset, load_from_checkpoint, save_checkpoint
from losses import EarlyStopping
from regularization import L1Regularization, L2Regularization
from distributed import (
    is_main_process, shard_dataset,
    reduce_mean, sync_gradients, all_gather_scalar, barrier,
)
from .mixins import DiagnosticsMixin, DistributedDiagnosticsMixin

import sys
import pathlib
BASEDIR = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(BASEDIR / "vendor"))
from pytorch_lbfgs import LBFGS as HjmshiLBFGS, FullBatchLBFGS as HjmshiFullBatchLBFGS


def count_params(model):
    nparams = 0
    for name, param in model.named_parameters():
        params = torch.tensor(param.size())
        nparams += torch.prod(params, 0)
    return nparams


def flatten_gradients(model):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
        else:
            grads.append(torch.zeros_like(p).view(-1))
    return torch.cat(grads)


def set_gradients(model, flat_grad):
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.grad = flat_grad[offset:offset + numel].view_as(p)
        offset += numel


def compute_mgda_alpha(g_energy, g_gradient, alpha_min=0.0, alpha_max=1.0,
                       energy_loss=None, gradient_loss=None,
                       ema_energy_loss=None, ema_gradient_loss=None):
    eps = 1e-12
    norm_e = torch.norm(g_energy) + eps
    norm_g = torch.norm(g_gradient) + eps
    g_energy_norm = g_energy / norm_e
    g_gradient_norm = g_gradient / norm_g
    cos_sim = torch.dot(g_energy_norm, g_gradient_norm)
    if (energy_loss is not None and gradient_loss is not None and
        ema_energy_loss is not None and ema_gradient_loss is not None and
        ema_energy_loss > eps and ema_gradient_loss > eps):
        rel_energy = energy_loss / ema_energy_loss
        rel_gradient = gradient_loss / ema_gradient_loss
        if not isinstance(rel_energy, torch.Tensor):
            rel_energy = torch.tensor(rel_energy, device=g_energy.device)
        if not isinstance(rel_gradient, torch.Tensor):
            rel_gradient = torch.tensor(rel_gradient, device=g_gradient.device)
        alpha = rel_gradient / (rel_energy + rel_gradient + eps)
    else:
        alpha = torch.tensor(0.5, device=g_energy.device)
    alpha = torch.clamp(alpha, alpha_min, alpha_max)
    g_combined = alpha * g_energy_norm + (1 - alpha) * g_gradient_norm
    combined_norm = torch.norm(g_combined) + eps
    target_norm = 0.5 * (norm_e + norm_g)
    rescale_factor = torch.clamp(target_norm / combined_norm, max=10.0)
    g_combined = g_combined * rescale_factor
    return alpha, cos_sim, g_combined


class BaseTrainer(DiagnosticsMixin, DistributedDiagnosticsMixin, ABC):
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
            self.model = self.build_model()

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

    @abstractmethod
    def build_model(self) -> torch.nn.Module:
        """Construct and return the network (nn.Module)."""
        raise NotImplementedError

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
    @abstractmethod
    def model_eval(self) -> None:
        """Final train/val/test evaluation and logging."""
        raise NotImplementedError

    def build_regularization(self):
        if self.cfg_regularization is None:
            return None

        if self.cfg_regularization['NAME'] == 'L1':
            lambda_ = float(self.cfg_regularization['LAMBDA'])
            reg = L1Regularization(lambda_)
        elif self.cfg_regularization['NAME'] == 'L2':
            lambda_ = float(self.cfg_regularization['LAMBDA'])
            reg = L2Regularization(lambda_)
        else:
            raise ValueError("unreachable")

        return reg
    def _parse_batch_cfg(self, cfg_batch):
        defaults = {
            'MULTIBATCH_ENABLED':    False,
            'MODE':                  'multi_batch',   # 'multi_batch' | 'full_overlap'
            'BATCH_SIZE':            None,
            'OVERLAP_FRACTION':      0.25,            # used only in 'multi_batch'
            'RESHUFFLE_EACH_EPOCH':  True,
            'LR':                    1.0,
            'HISTORY_SIZE':          10,
            'LINE_SEARCH':           None,            # None|'None'|'Wolfe'|'Armijo'
            'DAMPING':               True,            # Powell damping for 'multi_batch'
            'DAMPING_EPS':           0.2,
            'SEED':                  42,
        }

        if cfg_batch is None:
            return defaults

        known = set(defaults.keys())
        for key in cfg_batch.keys():
            assert key in known, "[BATCH] unknown option: {}".format(key)

        out = dict(defaults)
        out.update(cfg_batch)

        if not out['MULTIBATCH_ENABLED']:
            return out

        assert out['MODE'] in ('multi_batch', 'full_overlap'), \
            "[BATCH] MODE must be 'multi_batch' or 'full_overlap', got {}".format(out['MODE'])
        assert out['BATCH_SIZE'] is not None and int(out['BATCH_SIZE']) > 0, \
            "[BATCH] BATCH_SIZE must be a positive integer when MULTIBATCH_ENABLED"
        out['BATCH_SIZE'] = int(out['BATCH_SIZE'])

        overlap = float(out['OVERLAP_FRACTION'])
        assert 0.0 < overlap < 0.5, \
            "[BATCH] OVERLAP_FRACTION must be in (0, 0.5), got {}".format(overlap)
        out['OVERLAP_FRACTION'] = overlap

        assert out['MODE'] != 'multi_batch', \
            "[BATCH] MODE='multi_batch' is disabled; use 'full_overlap' instead"

        if out['MODE'] == 'multi_batch':
            ls = out['LINE_SEARCH']
            assert ls in (None, 'None'), \
                "[BATCH] MODE='multi_batch' expects LINE_SEARCH=None (fixed steplength); got {}".format(ls)
        else:  # full_overlap
            ls = out['LINE_SEARCH']
            assert ls in ('Wolfe', 'Armijo'), \
                "[BATCH] MODE='full_overlap' requires LINE_SEARCH='Wolfe' or 'Armijo'; got {}".format(ls)

        assert self.cfg['TYPE'] == 'ENERGY', \
            "[BATCH] multi-batch L-BFGS is currently only supported for TYPE=ENERGY"

        assert self.cfg_loss.get('TRUST_THRESHOLD') is None, \
            "[BATCH] trust-region loss (TRUST_THRESHOLD) is not supported with multi-batch L-BFGS yet"

        assert float(self.cfg_loss.get('FOCAL_GAMMA', 0.0)) == 0.0, \
            "[BATCH] focal-EMA weighting (FOCAL_GAMMA>0) is not supported with multi-batch L-BFGS yet"

        opt_name = self.cfg_solver['OPTIMIZER']['NAME']
        assert opt_name == 'LBFGS', \
            "[BATCH] MULTIBATCH_ENABLED requires OPTIMIZER.NAME=LBFGS, got {}".format(opt_name)

        return out
    def build_optimizer(self, cfg_optimizer):
        if cfg_optimizer['NAME'] == 'LBFGS':
            lr               = cfg_optimizer.get('LR', 1.0)
            if self.world_size > 1:
                # Use vendored FullBatchLBFGS for distributed training.
                # torch.optim.LBFGS is not DDP-safe because its line search
                # resets parameters after trial evaluations, which breaks DDP's
                # asynchronous gradient reduction invariants.
                history_size = cfg_optimizer.get('HISTORY_SIZE', 100)
                line_search = cfg_optimizer.get('LINE_SEARCH', 'Wolfe')
                if line_search not in ['Armijo', 'Wolfe', 'None']:
                    raise ValueError(f"Invalid LINE_SEARCH: {line_search}. Must be 'Armijo', 'Wolfe', or 'None'")

                # Line search parameters (stored for passing to step())
                self._lbfgs_ls_options = {
                    'max_ls': cfg_optimizer.get('MAX_LS', 10),
                    'c1': cfg_optimizer.get('C1', 1e-4),
                    'c2': cfg_optimizer.get('C2', 0.9),
                    'eta': cfg_optimizer.get('ETA', 2.0),
                    'interpolate': cfg_optimizer.get('INTERPOLATE', True),
                    'ls_debug': cfg_optimizer.get('LS_DEBUG', False),
                }
                logging.info(f"Line search options: {self._lbfgs_ls_options}")

                optimizer = HjmshiFullBatchLBFGS(
                    self.model.parameters(),
                    lr=lr,
                    history_size=history_size,
                    line_search=line_search,
                )
                logging.info("Build optimizer: {} (distributed-aware, vendored FullBatchLBFGS)".format(optimizer))
            else:
                tolerance_grad   = cfg_optimizer.get('TOLERANCE_GRAD', 1e-14)
                tolerance_change = cfg_optimizer.get('TOLERANCE_CHANGE', 1e-14)
                max_iter         = cfg_optimizer.get('MAX_ITER', 100)

                optimizer        = torch.optim.LBFGS(self.model.parameters(), lr=lr, line_search_fn='strong_wolfe', tolerance_grad=tolerance_grad,
                                                     tolerance_change=tolerance_change, max_iter=max_iter)
                logging.info("Build optimizer: {}".format(optimizer))
        elif cfg_optimizer['NAME'] == 'Adam':
            lr           = cfg_optimizer.get('LR', 1e-3)
            weight_decay = cfg_optimizer.get('WEIGHT_DECAY', 0.0)
            weight_decay = cfg_optimizer.get('WEIGHT_DECAY', 0.0)
            optimizer    = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError("unreachable")

        logging.info("Build optimizer: {}".format(optimizer))

        return optimizer
    @abstractmethod
    def build_loss(self) -> torch.nn.Module:
        """Construct and return the loss function (self.loss_fn)."""
        raise NotImplementedError

    def build_scheduler(self):
        cfg_scheduler = self.cfg_solver['SCHEDULER']
        scheduler_name = cfg_scheduler['NAME']

        if scheduler_name == 'ReduceLROnPlateau':
            factor         = cfg_scheduler.get('LR_REDUCE_GAMMA', 0.1)
            threshold      = cfg_scheduler.get('THRESHOLD', 0.1)
            threshold_mode = cfg_scheduler.get('THRESHOLD_MODE', 'abs')
            patience       = cfg_scheduler.get('PATIENCE', 10)
            cooldown       = cfg_scheduler.get('COOLDOWN', 0)
            min_lr         = cfg_scheduler.get('MIN_LR', 1e-5)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor=factor, threshold=threshold, threshold_mode=threshold_mode,
                patience=patience, cooldown=cooldown, min_lr=min_lr)

            logging.info("Build scheduler:")
            logging.info(" NAME:            {}".format(scheduler_name))
            logging.info(" LR_REDUCE_GAMMA: {}".format(factor))
            logging.info(" THRESHOLD:       {}".format(threshold))
            logging.info(" THRESHOLD_MODE:  {}".format(threshold_mode))
            logging.info(" PATIENCE:        {}".format(patience))
            logging.info(" COOLDOWN:        {}".format(cooldown))
            logging.info(" MIN_LR:          {}\n".format(min_lr))

        elif scheduler_name == 'CosineAnnealingWarmRestarts':
            T_0     = cfg_scheduler.get('T_0', 100)
            T_mult  = cfg_scheduler.get('T_MULT', 2)
            eta_min = cfg_scheduler.get('ETA_MIN', 1e-6)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)

            logging.info("Build scheduler:")
            logging.info(" NAME:    {}".format(scheduler_name))
            logging.info(" T_0:     {} (epochs until first restart)".format(T_0))
            logging.info(" T_MULT:  {} (period multiplier after each restart)".format(T_mult))
            logging.info(" ETA_MIN: {}\n".format(eta_min))

        else:
            raise ValueError("Unknown scheduler: {}".format(scheduler_name))

        return scheduler
    def build_early_stopper(self):
        cfg_early_stopping = self.cfg_solver['EARLY_STOPPING']

        patience  = cfg_early_stopping.get('PATIENCE', 1000)
        tolerance = cfg_early_stopping.get('TOLERANCE', 0.1)

        return EarlyStopping(patience=patience, tol=tolerance, chk_path=self.chk_path)
    @abstractmethod
    def prepare_data_for_device(self) -> None:
        """Move dataset tensors to self.device."""
        raise NotImplementedError

    def train_model(self):
        try:
            # Set device based on mode
            if self.world_size > 1:
                self.device = torch.device(f"cuda:{self.local_rank}")
            else:
                self.device = DEVICE

            self.model = self.model.to(self.device)

            if self.cfg_solver.get('TORCH_COMPILE', False):
                self.model = torch.compile(self.model, mode='reduce-overhead')
                if is_main_process():
                    logging.info("Model compiled with torch.compile(mode='reduce-overhead')")

            # Wrap with DDP for distributed training (except for LBFGS which uses
            # explicit gradient sync to avoid race conditions with line search)
            opt_name = self.cfg_solver['OPTIMIZER']['NAME']
            if self.world_size > 1 and opt_name != 'LBFGS':
                self.model = DDP(self.model, device_ids=[self.local_rank])
                if is_main_process():
                    logging.info(f"Distributed training enabled: {self.world_size} GPUs (DDP)")
            elif self.world_size > 1:
                if is_main_process():
                    logging.info(f"Distributed training enabled: {self.world_size} GPUs (explicit gradient sync, no DDP)")

            # Initialize TensorBoard only on rank 0 to avoid event-file corruption.
            if is_main_process():
                log_dir = os.path.join("runs", self.model_name)
                self.writer = SummaryWriter(log_dir=log_dir)
            else:
                self.writer = None

            # nn.Module.to() moves parameters and buffers, but our loss classes
            # store plain Tensor attributes (e.g. self.dwt). Move them explicitly.
            def _move_plain_tensors(mod, device):
                for k, v in mod.__dict__.items():
                    if isinstance(v, torch.Tensor) and not isinstance(v, torch.nn.Module):
                        setattr(mod, k, v.to(device))
            _move_plain_tensors(self.loss_fn, self.device)
            if self.regularization is not None:
                _move_plain_tensors(self.regularization, self.device)

            multibatch = bool(self.cfg_batch.get('MULTIBATCH_ENABLED', False))

            self.prepare_data_for_device()
            self.loss_fn = self.loss_fn.to(self.device)


            if multibatch:
                self.optimizer = self._build_multibatch_optimizer()
                self._init_multibatch_sampler()
            else:
                self.optimizer = self.build_optimizer(self.cfg_solver['OPTIMIZER'])
            self.scheduler = self.build_scheduler()

            start = time.time()

            MAX_EPOCHS = self.cfg_solver['MAX_EPOCHS']

            for epoch in range(MAX_EPOCHS):
                # Called here (not inside train_epoch) so the per-epoch prologue
                # (gradient-inclusion switch, ramps, LBFGS reset) also reaches the
                # multibatch path, which USE_GRADIENTS_AFTER_EPOCH configs rely on.
                self.prepare_epoch(epoch)

                self._log("loss function: {}".format(self.loss_fn))

                if bool(self.cfg_batch.get('MULTIBATCH_ENABLED', False)):
                    self.train_epoch_multibatch(epoch, self.optimizer)
                else:
                    self.train_epoch(epoch, self.optimizer)

                # Step scheduler - ReduceLROnPlateau requires metric, CosineAnnealing does not
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(self.loss_val)
                else:
                    self.scheduler.step()

                if epoch % PRINT_TRAINING_STEPS == 0:
                    end = time.time()
                    self._log("Elapsed time: {:.0f}s\n".format(end - start))

                # writing all pending events to disk
                if self.writer is not None:
                    self.writer.flush()

                # pass loss values to EarlyStopping mechanism 
                self.es(epoch, self.loss_val, self.model, self.xscaler, self.yscaler, meta_info=self.meta_info)

                if self.es.status:
                    self._log("Invoking early stop.")
                    break

            if self.loss_val < self.es.best_score:
                save_checkpoint(self.model, self.xscaler, self.yscaler, self.meta_info, self.chk_path)

            self._log("\nReloading best model from the last checkpoint")

            self.reset_weights()
            checkpoint = torch.load(self.chk_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])

            if is_main_process() and getattr(self, 'writer', None) is not None:
                self.writer.close()
            return self.model
        except Exception:
            if is_main_process() and getattr(self, 'writer', None) is not None:
                self.writer.close()
            raise


    def prepare_epoch(self, epoch: int) -> None:
        """No-op; GradientTrainer overrides with the gradient prologue.
        Energy/dipole trainers need no per-epoch preparation."""
        pass

    @abstractmethod
    def compute_loss(self, separate: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass + training loss. Returns a tensor, or an (energy_loss, gradient_loss) tuple when separate=True."""
        raise NotImplementedError

    @abstractmethod
    def evaluate_and_log(self, epoch: int, current_lr: float) -> None:
        """Per-epoch train/val evaluation and logging; must set self.loss_val."""
        raise NotImplementedError

    def supports_mgda(self) -> bool:
        return False

    def train_epoch(self, epoch, optimizer):
        CLOSURE_CALL_COUNT = 0
        debug_closure = self.cfg_debug.get('CLOSURE', False)

        def closure():
            nonlocal CLOSURE_CALL_COUNT
            CLOSURE_CALL_COUNT = CLOSURE_CALL_COUNT + 1
            optimizer.zero_grad()
            loss = self.compute_loss()
            loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            # Synchronize loss across ranks so L-BFGS line search makes
            # identical decisions on every process.
            if self.world_size > 1:
                loss = reduce_mean(loss.detach())
            return loss

        def closure_no_backward():
            nonlocal CLOSURE_CALL_COUNT
            CLOSURE_CALL_COUNT = CLOSURE_CALL_COUNT + 1
            optimizer.zero_grad()
            loss = self.compute_loss()
            return loss

        # MGDA (Multi-objective Gradient Descent Algorithm) closure
        # Computes optimal combination of energy and gradient loss gradients
        # MGDA is only active when gradients are currently being used
        use_mgda = self.supports_mgda()
        mgda_alpha_min = self.cfg_loss.get('MGDA_ALPHA_MIN', 0.1)
        mgda_alpha_max = self.cfg_loss.get('MGDA_ALPHA_MAX', 0.9)
        mgda_ema_decay = self.cfg_loss.get('MGDA_EMA_DECAY', 0.9)
        _mgda_alpha_raw = [None]  # Mutable container for closure
        _mgda_alpha = [None]
        _mgda_cos_sim = [None]

        def closure_mgda():
            """MGDA+GradNorm closure: normalized gradients + adaptive alpha from loss ratios."""
            nonlocal CLOSURE_CALL_COUNT
            CLOSURE_CALL_COUNT = CLOSURE_CALL_COUNT + 1
            optimizer.zero_grad()

            # Compute separate losses
            if debug_closure:
                logging.info(f"[rank {self.rank}] closure_mgda: computing separate losses")
            energy_loss, gradient_loss = self.compute_loss(separate=True)

            # Update loss EMAs for adaptive alpha computation
            energy_loss_val = energy_loss.detach().item()
            gradient_loss_val = gradient_loss.detach().item()
            if debug_closure:
                logging.info(f"[rank {self.rank}] closure_mgda: e_loss={energy_loss_val:.4f}, g_loss={gradient_loss_val:.4f}")

            if self._mgda_energy_loss_ema is None:
                self._mgda_energy_loss_ema = energy_loss_val
                self._mgda_gradient_loss_ema = gradient_loss_val
            else:
                self._mgda_energy_loss_ema = (mgda_ema_decay * self._mgda_energy_loss_ema +
                                              (1 - mgda_ema_decay) * energy_loss_val)
                self._mgda_gradient_loss_ema = (mgda_ema_decay * self._mgda_gradient_loss_ema +
                                                (1 - mgda_ema_decay) * gradient_loss_val)

            # Backward pass for energy gradient
            if debug_closure:
                logging.info(f"[rank {self.rank}] closure_mgda: energy backward START")
            energy_loss.backward(retain_graph=True)
            g_energy = flatten_gradients(self.model)
            if debug_closure:
                logging.info(f"[rank {self.rank}] closure_mgda: energy backward DONE, g_energy norm={g_energy.norm().item():.4f}")

            # Backward pass for gradient loss gradient
            optimizer.zero_grad()
            if debug_closure:
                logging.info(f"[rank {self.rank}] closure_mgda: gradient backward START")
            gradient_loss.backward()
            g_gradient = flatten_gradients(self.model)
            if debug_closure:
                logging.info(f"[rank {self.rank}] closure_mgda: gradient backward DONE, g_gradient norm={g_gradient.norm().item():.4f}")

            # Sync gradients across ranks before computing weights
            if self.world_size > 1:
                if debug_closure:
                    logging.info(f"[rank {self.rank}] closure_mgda: all_reduce START")
                dist.all_reduce(g_energy, op=dist.ReduceOp.SUM)
                g_energy = g_energy / self.world_size
                dist.all_reduce(g_gradient, op=dist.ReduceOp.SUM)
                g_gradient = g_gradient / self.world_size
                if debug_closure:
                    logging.info(f"[rank {self.rank}] closure_mgda: all_reduce DONE")

            # Compute GradNorm weights: normalized gradients + adaptive alpha from loss ratios
            alpha_raw, cos_sim, combined_grad = compute_mgda_alpha(
                g_energy, g_gradient,
                mgda_alpha_min, mgda_alpha_max,
                energy_loss=energy_loss_val,
                gradient_loss=gradient_loss_val,
                ema_energy_loss=self._mgda_energy_loss_ema,
                ema_gradient_loss=self._mgda_gradient_loss_ema
            )

            # EMA smoothing of alpha to prevent oscillation
            if self._mgda_alpha_ema is None:
                alpha = alpha_raw
                self._mgda_alpha_ema = alpha.item()
            else:
                alpha = mgda_ema_decay * self._mgda_alpha_ema + (1 - mgda_ema_decay) * alpha_raw.item()
                self._mgda_alpha_ema = alpha
                alpha = torch.tensor(alpha, device=g_energy.device)

            # Store for diagnostics
            _mgda_alpha_raw[0] = alpha_raw.item()
            _mgda_alpha[0] = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
            _mgda_cos_sim[0] = cos_sim.item()

            # Set the combined normalized gradient
            set_gradients(self.model, combined_grad)

            # Gradient clipping on combined gradient
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            # Return combined loss for L-BFGS line search
            combined_loss = alpha * energy_loss.detach() + (1 - alpha) * gradient_loss.detach()
            if self.world_size > 1:
                combined_loss = reduce_mean(combined_loss)
            return combined_loss

        def closure_mgda_no_backward():
            """MGDA closure for line search (backward may be called by vendored LBFGS)."""
            nonlocal CLOSURE_CALL_COUNT
            CLOSURE_CALL_COUNT = CLOSURE_CALL_COUNT + 1
            if debug_closure: 
                logging.info(f"[rank {self.rank}] closure_mgda_no_backward: call #{CLOSURE_CALL_COUNT}")
            optimizer.zero_grad()
            energy_loss, gradient_loss = self.compute_loss(separate=True)
            # Use current EMA alpha for consistent loss evaluation
            alpha = self._mgda_alpha_ema if self._mgda_alpha_ema is not None else 0.5
            combined_loss = alpha * energy_loss + (1 - alpha) * gradient_loss
            if debug_closure:
                logging.info(f"[rank {self.rank}] closure_mgda_no_backward: done, loss={combined_loss.item():.4f}")
            return combined_loss

        # Calling model.train() will change the behavior of some layers such as nn.Dropout and nn.BatchNormXd
        self.model.train()

        # Reset focal weighting flag to allow one error_scale update per optimizer step
        # (prevents non-deterministic loss during LBFGS line search)
        if hasattr(self.loss_fn, 'reset_error_scale_flag'):
            self.loss_fn.reset_error_scale_flag()

        start_time = timeit.default_timer()
        if isinstance(optimizer, HjmshiFullBatchLBFGS):
            # Vendored FullBatchLBFGS for distributed training.
            if use_mgda:
                # MGDA mode: use MGDA closures that compute optimal gradient combination
                if debug_closure:
                    logging.info(f"[rank {self.rank}] vendored LBFGS (MGDA): about to call initial closure_mgda")
                loss = closure_mgda()  # This sets gradients via MGDA
                if debug_closure:
                    logging.info(f"[rank {self.rank}] vendored LBFGS (MGDA): initial closure_mgda done, loss={loss.item():.4f}")
                # Note: closure_mgda already syncs gradients and applies clipping
                # Build grad_sync closure that captures self.model
                def _grad_sync():
                    if debug_closure:
                        logging.info(f"[rank {self.rank}] _grad_sync: START")
                    sync_gradients(self.model)
                    if debug_closure:
                        logging.info(f"[rank {self.rank}] _grad_sync: DONE")
                options = {
                    'closure': closure_mgda_no_backward,
                    'current_loss': loss,
                    'grad_clip_norm': None,  # Already applied in closure_mgda
                    'loss_sync_fn': reduce_mean if self.world_size > 1 else None,
                    'grad_sync_fn': _grad_sync if self.world_size > 1 else None,
                }
            else:
                # Standard mode
                # Pre-compute loss & gradient at the current iterate.
                logging.debug(f"[rank {self.rank}] vendored LBFGS: zero_grad")
                optimizer.zero_grad()
                logging.debug(f"[rank {self.rank}] vendored LBFGS: closure_no_backward")
                loss = closure_no_backward()
                logging.debug(f"[rank {self.rank}] vendored LBFGS: backward (loss={loss.item():.4f})")
                loss.backward()
                # Explicit gradient sync - don't rely on DDP's implicit async sync
                if self.world_size > 1:
                    logging.debug(f"[rank {self.rank}] vendored LBFGS: sync_gradients START")
                    sync_gradients(self.model)
                    logging.debug(f"[rank {self.rank}] vendored LBFGS: sync_gradients DONE")
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                if self.world_size > 1:
                    logging.debug(f"[rank {self.rank}] vendored LBFGS: reduce_mean START")
                    loss = reduce_mean(loss.detach())
                    logging.debug(f"[rank {self.rank}] vendored LBFGS: reduce_mean DONE")
                # Build grad_sync closure that captures self.model
                def _grad_sync():
                    sync_gradients(self.model)
                options = {
                    'closure': closure_no_backward,
                    'current_loss': loss,
                    'grad_clip_norm': self.grad_clip_norm,
                    'loss_sync_fn': reduce_mean if self.world_size > 1 else None,
                    'grad_sync_fn': _grad_sync if self.world_size > 1 else None,
                }
            # Add line search options from config
            if hasattr(self, '_lbfgs_ls_options'):
                options.update(self._lbfgs_ls_options)
            if debug_closure:
                logging.info(f"[rank {self.rank}] vendored LBFGS: calling optimizer.step()")
            obj, grad_new, t, ls_step, closure_eval, grad_eval, desc_dir, fail = optimizer.step(options=options)
            if debug_closure:
                logging.info(f"[rank {self.rank}] vendored LBFGS: optimizer.step() done, evals={closure_eval}")
            self._last_vendored_closure_eval = closure_eval + 1  # +1 for the initial evaluation above
            CLOSURE_CALL_COUNT = self._last_vendored_closure_eval
            elapsed = timeit.default_timer() - start_time
            self._log("Optimizer makes step in {:.2f}s".format(elapsed))
            self._log("CLOSURE_CALL_COUNT = {}".format(CLOSURE_CALL_COUNT))
        else:
            # Non-vendored optimizer (e.g., torch.optim.LBFGS)
            if use_mgda:
                optimizer.step(closure_mgda)
            else:
                optimizer.step(closure)
            elapsed = timeit.default_timer() - start_time
            self._log("Optimizer makes step in {:.2f}s".format(elapsed))
            self._log("CLOSURE_CALL_COUNT = {}".format(CLOSURE_CALL_COUNT))

        current_lr = optimizer.param_groups[0]['lr']
        self._log("(optimizer) current lr: {}".format(current_lr))

        # LBFGS line-search telemetry (no-op for non-LBFGS optimizers).
        self.log_lbfgs_diagnostics(epoch, optimizer)

        # MGDA diagnostics logging
        if use_mgda and _mgda_alpha[0] is not None:
            self._log("(MGDA) alpha={:.4f} (raw={:.4f}), cos_sim={:.4f}".format(
                _mgda_alpha[0], _mgda_alpha_raw[0], _mgda_cos_sim[0]))
            self.log_mgda_diagnostics(epoch, _mgda_alpha[0], _mgda_alpha_raw[0], _mgda_cos_sim[0])

        # Calling model.eval() will change the behavior of some layers, 
        # such as nn.Dropout, which will be disabled, and nn.BatchNormXd, which will use the running stats during evaluation.
        self.model.eval()

        self.evaluate_and_log(epoch, current_lr)

