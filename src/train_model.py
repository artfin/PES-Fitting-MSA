import argparse
import collections
import json
import sys
import logging
import random
import os
import time
import timeit
import yaml

import torch.nn
from torch.utils.tensorboard import SummaryWriter

USE_WANDB = False
if USE_WANDB:
    import wandb

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from config import TORCH_FLOAT
from dataset import PolyDataset
from make_dataset import make_dataset, make_dataset_fpaths
from build_model import build_network, QModel

import pathlib
BASEDIR = pathlib.Path(__file__).parent.parent.resolve()

# Vendored hjmshi/PyTorch-LBFGS used for multi-batch + full-overlap modes.
sys.path.insert(0, str(BASEDIR / "vendor"))
from pytorch_lbfgs import LBFGS as HjmshiLBFGS, FullBatchLBFGS as HjmshiFullBatchLBFGS

from batching import FullOverlapSampler, MultiBatchSampler, DistributedFullOverlapSampler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from distributed import (
    setup_distributed, cleanup, is_distributed,
    is_main_process, get_rank, get_world_size, reduce_mean, reduce_min, barrier,
    sync_gradients, reduce_rmse, reduce_mae, reduce_sum, all_gather_scalar
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRINT_TRAINING_STEPS = 1
PRINT_PRECISION      = 3

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

def flatten_gradients(model):
    """Flatten all parameter gradients into a single 1D tensor."""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
        else:
            grads.append(torch.zeros_like(p).view(-1))
    return torch.cat(grads)


def set_gradients(model, flat_grad):
    """Set model parameter gradients from a flattened 1D tensor."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.grad = flat_grad[offset:offset + numel].view_as(p)
        offset += numel


def compute_mgda_alpha(g_energy, g_gradient, alpha_min=0.0, alpha_max=1.0,
                       energy_loss=None, gradient_loss=None,
                       ema_energy_loss=None, ema_gradient_loss=None):
    """
    Compute optimal convex combination weight using GradNorm principles.

    Combines two key ideas:
    1. Normalize gradients to unit vectors (removes magnitude bias that
       caused alpha to pin at alpha_min in pure MGDA)
    2. Adaptive alpha based on loss ratios - tasks falling behind get
       more weight to balance training rates

    Args:
        g_energy: flattened gradient vector for energy loss
        g_gradient: flattened gradient vector for gradient loss
        alpha_min: minimum weight for energy objective
        alpha_max: maximum weight for energy objective
        energy_loss: current energy loss value (for adaptive alpha)
        gradient_loss: current gradient loss value (for adaptive alpha)
        ema_energy_loss: EMA of energy loss (for relative comparison)
        ema_gradient_loss: EMA of gradient loss (for relative comparison)

    Returns:
        alpha: weight for energy gradient (1-alpha for gradient loss)
        cos_sim: cosine similarity between normalized gradients
        g_combined: combined gradient vector (using normalized gradients)
    """
    eps = 1e-12

    # Normalize gradients to unit vectors (GradNorm key idea #1)
    norm_e = torch.norm(g_energy) + eps
    norm_g = torch.norm(g_gradient) + eps
    g_energy_norm = g_energy / norm_e
    g_gradient_norm = g_gradient / norm_g

    # Cosine similarity between normalized gradients
    cos_sim = torch.dot(g_energy_norm, g_gradient_norm)

    # Adaptive alpha based on loss ratios (GradNorm key idea #2)
    # Task with higher relative loss (falling behind) gets more weight
    if (energy_loss is not None and gradient_loss is not None and
        ema_energy_loss is not None and ema_gradient_loss is not None and
        ema_energy_loss > eps and ema_gradient_loss > eps):

        # Relative loss: current / EMA (>1 means task is falling behind)
        rel_energy = energy_loss / ema_energy_loss
        rel_gradient = gradient_loss / ema_gradient_loss

        # Convert to tensors if needed
        if not isinstance(rel_energy, torch.Tensor):
            rel_energy = torch.tensor(rel_energy, device=g_energy.device)
        if not isinstance(rel_gradient, torch.Tensor):
            rel_gradient = torch.tensor(rel_gradient, device=g_energy.device)

        # alpha = rel_gradient / (rel_energy + rel_gradient)
        # If gradient loss is falling behind: rel_gradient > rel_energy -> alpha < 0.5
        # This gives more weight to gradient objective (1 - alpha)
        alpha = rel_gradient / (rel_energy + rel_gradient + eps)
    else:
        # Fallback: equal weighting when no loss history available
        alpha = torch.tensor(0.5, device=g_energy.device)

    # Clamp to bounds
    alpha = torch.clamp(alpha, alpha_min, alpha_max)

    # Combine NORMALIZED gradients (key difference from original MGDA)
    g_combined = alpha * g_energy_norm + (1 - alpha) * g_gradient_norm

    return alpha, cos_sim, g_combined


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


class L1Regularization(torch.nn.Module):
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = torch.tensor(lambda_).to(DEVICE)

    def __repr__(self):
        return "L1Regularization(lambda={})".format(self.lambda_.item())

    def forward(self, model):
        l1_norm = torch.tensor(0.).to(dtype=torch.float64, device=DEVICE)
        for p in model.parameters():
            l1_norm += p.abs().sum()

        return self.lambda_ * l1_norm

class L2Regularization(torch.nn.Module):
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = torch.tensor(lambda_).to(DEVICE)

    def __repr__(self):
        return "L2Regularization(lambda={})".format(self.lambda_.item())

    def forward(self, model):
        l2_norm = torch.tensor(0.).to(DEVICE)
        for p in model.parameters():
            l2_norm += (p**2).sum()
        return self.lambda_ * l2_norm

class WMSELoss_Boltzmann(torch.nn.Module):
    def __init__(self, Eref):
        super().__init__()
        self.Eref = torch.tensor(Eref).to(DEVICE)

        self.y_mean = None
        self.y_std  = None

    def set_scale(self, y_mean, y_std):
        self.y_mean = torch.FloatTensor(y_mean.tolist()).to(DEVICE)
        self.y_std  = torch.FloatTensor(y_std.tolist()).to(DEVICE)

    def __repr__(self):
        return "WMSELoss_Boltzmann(Eref={})".format(self.Eref)

    def forward(self, y, y_pred):
        assert self.y_mean is not None
        assert self.y_std is not None

        # descale energies
        yd      = y      * self.y_std + self.y_mean
        yd_pred = y_pred * self.y_std + self.y_mean

        w = torch.exp(-yd / self.Eref)
        w = w / w.max()

        wmse = (w * (yd - yd_pred)**2).mean()
        return wmse

class WRMSELoss_Boltzmann(torch.nn.Module):
    def __init__(self, Eref):
        super().__init__()
        self.Eref = torch.tensor(Eref).to(DEVICE)

        self.y_mean = None
        self.y_std  = None

    def set_scale(self, y_mean, y_std):
        self.y_mean = torch.FloatTensor(y_mean.tolist()).to(DEVICE)
        self.y_std  = torch.FloatTensor(y_std.tolist()).to(DEVICE)

    def __repr__(self):
        return "WRMSELoss_Boltzmann(Eref={})".format(self.Eref)

    def forward(self, y, y_pred):
        assert self.y_mean is not None
        assert self.y_std is not None

        # descale energies
        yd      = y      * self.y_std + self.y_mean
        yd_pred = y_pred * self.y_std + self.y_mean

        w = torch.exp(-yd / self.Eref)
        w = w / w.max()

        wmse = (w * (yd - yd_pred)**2).mean()
        return torch.sqrt(wmse)

class WMSELoss_Ratio(torch.nn.Module):
    """
    Weighted MSE loss with energy-based ratio weighting and optional focal weighting.

    Energy weighting: w_energy = dwt / (dwt + E - E_min)
        - Low-energy configs get higher weight

    Focal weighting (when focal_gamma > 0): w_focal = 1 + gamma * (|error| / error_scale)
        - All configs retain base weight (w_energy)
        - Hard examples (high error) get EXTRA weight
        - error_scale is tracked via exponential moving average (EMA)
        - This is more stable than down-weighting easy examples

    Combined: w_total = w_energy * w_focal
    """
    def __init__(self, dwt=1.0, focal_gamma=0.0, focal_ema_decay=0.95):
        super().__init__()
        self.dwt = torch.tensor(dwt).to(DEVICE)
        self.focal_gamma = focal_gamma
        self.focal_ema_decay = focal_ema_decay

        self.y_mean = None
        self.y_std  = None

        # EMA tracker for error scale (used in focal weighting)
        self.error_scale = None
        # Flag to ensure error_scale is only updated once per epoch (not during LBFGS line search)
        self._error_scale_updated_this_step = False

    def set_scale(self, y_mean, y_std):
        self.y_mean = torch.FloatTensor(y_mean.tolist()).to(DEVICE)
        self.y_std  = torch.FloatTensor(y_std.tolist()).to(DEVICE)

    def __repr__(self):
        return "WMSELoss_Ratio(dwt={}, focal_gamma={}, focal_ema_decay={})".format(
            self.dwt, self.focal_gamma, self.focal_ema_decay)

    def reset_error_scale_flag(self):
        """Call this at the start of each optimizer step to allow one error_scale update."""
        self._error_scale_updated_this_step = False

    def forward(self, y, y_pred):
        assert self.y_mean is not None
        assert self.y_std is not None

        # descale energies
        yd      = y      * self.y_std + self.y_mean
        yd_pred = y_pred * self.y_std + self.y_mean

        # Sync minimum across ranks for consistent weighting in distributed mode
        ymin = reduce_min(yd.min())

        # Energy-based weight
        w_energy = self.dwt / (self.dwt + yd - ymin)

        # Focal weighting (if enabled)
        if self.focal_gamma > 0:
            errors = (yd - yd_pred).abs()

            # Update error scale via EMA only ONCE per optimizer step
            # (not during LBFGS line search which calls forward() many times)
            if not self._error_scale_updated_this_step:
                current_max_error = errors.max().detach()
                if self.error_scale is None:
                    self.error_scale = current_max_error
                else:
                    self.error_scale = (self.focal_ema_decay * self.error_scale +
                                        (1 - self.focal_ema_decay) * current_max_error)
                self._error_scale_updated_this_step = True

            # Compute normalized errors
            error_normalized = errors / (self.error_scale + 1e-8)
            error_normalized = error_normalized.clamp(0, 1)

            # Additive focal weight: base weight 1 + extra for hard examples
            # This is more stable than multiplicative (which can zero out gradients)
            w_focal = 1.0 + self.focal_gamma * error_normalized

            # Combined weight
            w = w_energy * w_focal
        else:
            w = w_energy

        wmse = (w * (yd - yd_pred)**2).mean()

        return wmse

class WRMSELoss_Ratio_dipole(torch.nn.Module):
    def __init__(self, dwt=1.0):
        super().__init__()
        self.dwt    = torch.tensor(dwt).to(DEVICE)

        self.y_mean = None
        self.y_std  = None

    def set_scale(self, y_mean, y_std):
        self.y_mean = torch.FloatTensor(y_mean.tolist()).to(DEVICE)
        self.y_std  = torch.FloatTensor(y_std.tolist()).to(DEVICE)

    def __repr__(self):
        return "WRMSELoss_Ratio_dipole(dwt={})".format(self.dwt)

    def forward(self, y, y_pred):
        """
        y:      (E, dipx,      dipy,      dipz     )
        y_pred: (   dipx_pred, dipy_pred, dipz_pred)
        """
        assert self.y_mean is not None
        assert self.y_std is not None

        # descale
        dip_pred = y_pred * self.y_std[1:] + self.y_mean[1:]
        yd       = y      * self.y_std     + self.y_mean
        dip      = yd[:, 1:]
        en       = yd[:, 0]

        # Sync minimum across ranks for consistent weighting in distributed mode
        en_min = reduce_min(en.min())
        w  = self.dwt / (self.dwt + en - en_min)

        dd   = dip - dip_pred
        wdd  = torch.einsum('ij,i->ij', dd, w)
        wmse = torch.mean(torch.einsum('ij,ij->i', wdd, dd))

        #for k in range(10):
        #    print("dip: {}; dip_pred: {}".format(dip[k].detach().numpy(), dip_pred[k].detach().numpy()))

        return 1000.0 * torch.sqrt(wmse)
        #return 1000.0 * torch.mean(torch.abs(wdd))


class WMSELoss_Ratio_wgradients(torch.nn.Module):
    def __init__(self, natoms, dwt=1.0, g_lambda=1.0):
        super().__init__()
        self.natoms = natoms
        self.dwt    = torch.tensor(dwt).to(DEVICE)
        self.g_lambda = torch.tensor(g_lambda).to(DEVICE)

        self.en_mean = None
        self.en_std  = None

    def set_scale(self, en_mean, en_std):
        self.en_mean = torch.from_numpy(en_mean).to(DEVICE)
        self.en_std  = torch.from_numpy(en_std).to(DEVICE)

    def __repr__(self):
        return "WMSELoss_Ratio_wgradients(natoms={}, dwt={}, g_lambda={})".format(self.natoms, self.dwt, self.g_lambda)

    def forward(self, en, en_pred, gradients, gradients_pred):
        wmse_en, wmse_gradients = self.forward_separate(en, en_pred, gradients, gradients_pred)
        # Apply g_lambda weighting for backward compatibility (non-MGDA path)
        return wmse_en + self.g_lambda * wmse_gradients

    def descale_energies(self, en):
        return en * self.en_std + self.en_mean

    def forward_separate(self, en, en_pred, gradients, gradients_pred):
        assert self.en_mean is not None
        assert self.en_std is not None

        en_pred = en_pred.to(DEVICE)

        # descale energies
        # gradients are supposed to be already unnormalized
        _en      = self.descale_energies(en)
        _en_pred = self.descale_energies(en_pred)

        # Sync minimum across ranks for consistent weighting in distributed mode
        enmin   = reduce_min(_en.min())
        w       = self.dwt / (self.dwt + _en - enmin)
        w       = w.view(-1)
        wmse_en = (w * (_en - _en_pred)**2).mean()

        nconfigs = gradients.size()[0]

        gradients_pred = gradients_pred.reshape(nconfigs, self.natoms, 3)
        gradients    = gradients.reshape(nconfigs, self.natoms, 3)

        df = gradients - gradients_pred
        wdf = torch.einsum('ijk,i->ijk', df, w)
        # Return raw gradient loss (g_lambda applied in forward() for non-MGDA,
        # or MGDA computes its own optimal weighting)
        wmse_gradients = torch.einsum('ijk,ijk->', wdf, df) / (3.0 * self.natoms) / nconfigs

        return wmse_en, wmse_gradients


class WMSELoss_TrustRegion_wgradients(torch.nn.Module):
    """
    Memory-efficient trust region loss for gradient training with optional focal
    weighting and smooth (soft) trust boundaries.

    Gradients are pre-filtered before being passed to this loss (computed only
    for configs in the trust / active set). This avoids OOM by never computing
    gradients for configs outside the active set.

    Trust region uses soft boundaries with sigmoid weighting:
        phi(e_i) = sigmoid((trust_threshold - e_i) / soft_scale) in [0, 1]
    smoothly decaying with energy error. Gradient loss is normalized by the
    SUM of weights, so downweighting actually reduces a config's contribution
    rather than redistributing it.

    The "active set" passed in is the subset of configs whose phi exceeds
    soft_cutoff (a memory optimization only -- configs outside the active set
    contribute negligibly).

    Soft boundaries avoid the discontinuous "evasion" incentive of hard masks,
    where pushing a config across the threshold would discretely zero its
    gradient loss, breaking L-BFGS convergence.

    Focal weighting (when focal_gamma > 0) up-weights hard-to-fit configs.

    Expected inputs:
    - en, en_pred:           full energy tensors (all configs)
    - gradients_subset:      reference gradients for active-set configs
    - gradients_pred_subset: predicted gradients for active-set configs
    - trust_indices:         indices of active-set configs
    - gradient_weights:      per-config soft weights for the active set
    """
    def __init__(self, natoms, dwt=1.0, g_lambda=1.0, trust_threshold=100.0,
                 soft_scale=None, focal_gamma=0.0, focal_ema_decay=0.95,
                 huber_delta=None):
        super().__init__()
        self.natoms = natoms
        self.dwt = torch.tensor(dwt).to(DEVICE)
        self.g_lambda = torch.tensor(g_lambda).to(DEVICE)
        self.trust_threshold = trust_threshold
        self.soft_scale = soft_scale  # cm^-1; defaults to trust_threshold/4
        self.focal_gamma = focal_gamma
        self.focal_ema_decay = focal_ema_decay
        # Pseudo-Huber cutoff on per-component gradient residuals (cm^-1/Bohr).
        # None = pure MSE on gradients (backward-compatible).
        self.huber_delta = huber_delta

        self.en_mean = None
        self.en_std = None

        # EMA tracker for error scale (used in focal weighting)
        self.error_scale = None
        # Flag to ensure error_scale is only updated once per optimizer step
        # (not during LBFGS line search which calls forward() many times)
        self._error_scale_updated_this_step = False

    def set_scale(self, en_mean, en_std):
        self.en_mean = torch.from_numpy(en_mean).to(DEVICE)
        self.en_std = torch.from_numpy(en_std).to(DEVICE)

    def __repr__(self):
        return ("WMSELoss_TrustRegion_wgradients(natoms={}, dwt={}, g_lambda={}, "
                "trust_threshold={}, soft_scale={}, "
                "focal_gamma={}, focal_ema_decay={}, "
                "huber_delta={})").format(
            self.natoms, self.dwt, self.g_lambda, self.trust_threshold,
            self.soft_scale, self.focal_gamma, self.focal_ema_decay,
            self.huber_delta)

    @staticmethod
    def soft_phi(energy_errors, trust_threshold, soft_scale=None):
        """Return sigmoid soft trust factor phi(e_i) in [0, 1], detached.

        phi(e_i) = sigmoid((trust_threshold - e_i) / soft_scale)

        At e_i = trust_threshold: phi = 0.5
        For e_i << trust_threshold: phi -> 1
        For e_i >> trust_threshold: phi -> 0

        energy_errors: 1-D tensor of |E_pred - E_true| per config (cm^-1).
        """
        s = soft_scale if soft_scale is not None else (trust_threshold / 4.0)
        e = energy_errors.detach().clamp(min=0.0)
        return torch.sigmoid((trust_threshold - e) / s)

    def reset_error_scale_flag(self):
        """Call this at the start of each optimizer step to allow one error_scale update."""
        self._error_scale_updated_this_step = False

    def descale_energies(self, en):
        return en * self.en_std + self.en_mean

    def _compute_weights(self, _en, _en_pred):
        """Compute combined energy-based and focal weights."""
        # Sync minimum across ranks for consistent weighting in distributed mode
        enmin = reduce_min(_en.min())
        w_energy = self.dwt / (self.dwt + _en - enmin)
        w_energy = w_energy.view(-1)

        # Focal weighting (if enabled)
        if self.focal_gamma > 0:
            errors = (_en - _en_pred).abs().view(-1)

            # Update error scale via EMA only ONCE per optimizer step
            # (not during LBFGS line search which calls forward() many times)
            if not self._error_scale_updated_this_step:
                current_max_error = errors.max().detach()
                if self.error_scale is None:
                    self.error_scale = current_max_error
                else:
                    self.error_scale = (self.focal_ema_decay * self.error_scale +
                                        (1 - self.focal_ema_decay) * current_max_error)
                self._error_scale_updated_this_step = True

            # Compute normalized errors
            error_normalized = errors / (self.error_scale + 1e-8)
            error_normalized = error_normalized.clamp(0, 1)

            # Additive focal weight: base weight 1 + extra for hard examples
            # This is more stable than multiplicative (which can zero out gradients)
            w_focal = 1.0 + self.focal_gamma * error_normalized

            # Combined weight
            w = w_energy * w_focal
        else:
            w = w_energy

        return w

    def forward(self, en, en_pred, gradients_subset, gradients_pred_subset,
                trust_indices, gradient_weights):
        """
        Compute combined energy + gradient loss.
        Energy loss is computed on ALL configs, gradient loss only on the
        active (trust) subset.

        gradient_weights: 1-D tensor of soft phi values for the active subset
        (same length as trust_indices). Gradient loss is normalized by the SUM
        of these weights.
        """
        wmse_en, wmse_gradients = self.forward_separate(
            en, en_pred, gradients_subset, gradients_pred_subset,
            trust_indices, gradient_weights
        )
        # Apply g_lambda weighting for backward compatibility (non-MGDA path)
        return wmse_en + self.g_lambda * wmse_gradients

    def forward_separate(self, en, en_pred, gradients_subset, gradients_pred_subset,
                         trust_indices, gradient_weights):
        assert self.en_mean is not None
        assert self.en_std is not None

        en_pred = en_pred.to(DEVICE)

        # Descale energies for weight computation
        _en = self.descale_energies(en)
        _en_pred = self.descale_energies(en_pred)

        # Compute weights (energy-based + optional focal)
        w = self._compute_weights(_en, _en_pred)

        # Energy loss on ALL configs
        wmse_en = (w * (_en - _en_pred)**2).mean()

        # Gradient loss on active subset only
        n_in_trust = len(trust_indices)
        if n_in_trust > 0:
            # Combined per-config weight on the active subset:
            # w_trusted = w_energy * w_focal * phi(e_i)
            w_trusted = w[trust_indices] * gradient_weights

            # Reshape gradients
            gradients_subset = gradients_subset.reshape(n_in_trust, self.natoms, 3)
            gradients_pred_subset = gradients_pred_subset.reshape(n_in_trust, self.natoms, 3)

            # Per-config residual sum: MSE (sum of squared components) or
            # pseudo-Huber (bounded-influence sum per component) when delta set.
            df = gradients_subset - gradients_pred_subset
            if self.huber_delta is not None:
                # Pseudo-Huber: L_delta(r) = delta^2 * (sqrt(1 + (r/delta)^2) - 1)
                # Quadratic for |r| << delta, linear for |r| >> delta. Smooth
                # everywhere (L-BFGS-friendly). delta is derived from training
                # data (~2 * MAD) so no new user-facing knob.
                d = self.huber_delta
                per_config_sq = torch.sum(
                    d * d * (torch.sqrt(1.0 + (df / d) ** 2) - 1.0),
                    dim=(1, 2),
                )
            else:
                per_config_sq = torch.sum(df ** 2, dim=(1, 2))  # (n_in_trust,)

            contrib = w_trusted * per_config_sq               # (n_in_trust,)
            sq = contrib.sum()

            # Normalize by sum of soft weights so that downweighting a config
            # genuinely shrinks its contribution.
            denom = gradient_weights.sum().clamp(min=1.0) * 3.0 * self.natoms

            # Return raw gradient loss (g_lambda applied in forward() for non-MGDA,
            # or MGDA computes its own optimal weighting)
            wmse_gradients = sq / denom
        else:
            wmse_gradients = torch.tensor(0.0, device=DEVICE)

        return wmse_en, wmse_gradients

    def forward_energy_only(self, en, en_pred):
        """Fallback for when no configs are in trust region."""
        assert self.en_mean is not None
        assert self.en_std is not None

        en_pred = en_pred.to(DEVICE)

        _en = self.descale_energies(en)
        _en_pred = self.descale_energies(en_pred)

        # Compute weights (energy-based + optional focal)
        w = self._compute_weights(_en, _en_pred)

        wmse_en = (w * (_en - _en_pred)**2).mean()

        return wmse_en


class WRMSELoss_Ratio(torch.nn.Module):
    """
    Weighted RMSE loss with energy-based ratio weighting and optional focal weighting.
    Same as WMSELoss_Ratio but returns sqrt(WMSE).
    """
    def __init__(self, dwt=1.0, focal_gamma=0.0, focal_ema_decay=0.95):
        super().__init__()
        self.dwt = torch.tensor(dwt).to(DEVICE)
        self.focal_gamma = focal_gamma
        self.focal_ema_decay = focal_ema_decay

        self.y_mean = None
        self.y_std  = None

        # EMA tracker for error scale (used in focal weighting)
        self.error_scale = None
        # Flag to ensure error_scale is only updated once per epoch (not during LBFGS line search)
        self._error_scale_updated_this_step = False

    def set_scale(self, y_mean, y_std):
        self.y_mean = torch.FloatTensor(y_mean.tolist()).to(DEVICE)
        self.y_std  = torch.FloatTensor(y_std.tolist()).to(DEVICE)

    def __repr__(self):
        return "WRMSELoss_Ratio(dwt={}, focal_gamma={}, focal_ema_decay={})".format(
            self.dwt, self.focal_gamma, self.focal_ema_decay)

    def reset_error_scale_flag(self):
        """Call this at the start of each optimizer step to allow one error_scale update."""
        self._error_scale_updated_this_step = False

    def forward(self, y, y_pred):
        assert self.y_mean is not None
        assert self.y_std is not None

        # descale energies
        yd      = y      * self.y_std + self.y_mean
        yd_pred = y_pred * self.y_std + self.y_mean

        # Sync minimum across ranks for consistent weighting in distributed mode
        ymin = reduce_min(yd.min())

        # Energy-based weight
        w_energy = self.dwt / (self.dwt + yd - ymin)

        # Focal weighting (if enabled)
        if self.focal_gamma > 0:
            errors = (yd - yd_pred).abs()

            # Update error scale via EMA only ONCE per optimizer step
            if not self._error_scale_updated_this_step:
                current_max_error = errors.max().detach()
                if self.error_scale is None:
                    self.error_scale = current_max_error
                else:
                    self.error_scale = (self.focal_ema_decay * self.error_scale +
                                        (1 - self.focal_ema_decay) * current_max_error)
                self._error_scale_updated_this_step = True

            # Compute normalized errors
            error_normalized = errors / (self.error_scale + 1e-8)
            error_normalized = error_normalized.clamp(0, 1)

            # Additive focal weight: base weight 1 + extra for hard examples
            w_focal = 1.0 + self.focal_gamma * error_normalized

            # Combined weight
            w = w_energy * w_focal
        else:
            w = w_energy

        wmse = (w * (yd - yd_pred)**2).mean()

        return torch.sqrt(wmse)

class WMSELoss_PS(torch.nn.Module):
    """
    Weighted mean-squared error with
    weight factors suggested by Partridge and Schwenke
    H. Partridge, D. W. Schwenke, J. Chem. Phys. 106, 4618 (1997)
    """
    def __init__(self, Emax=2000.0):
        super().__init__()
        self.Emax   = torch.FloatTensor([Emax]).to(DEVICE)
        self.y_mean = None
        self.y_std  = None

    def set_scale(self, y_mean, y_std):
        self.y_mean = torch.FloatTensor(y_mean.tolist()).to(DEVICE)
        self.y_std  = torch.FloatTensor(y_std.tolist()).to(DEVICE)

    def __repr__(self):
        return "WMSELoss_PS(Emax={})".format(self.Emax)

    def forward(self, y, y_pred):
        assert self.y_mean is not None
        assert self.y_std is not None

        # descale energies
        yd      = y      * self.y_std + self.y_mean
        yd_pred = y_pred * self.y_std + self.y_mean

        Ehat = torch.max(yd, self.Emax.expand_as(yd))
        w = (torch.tanh(-6e-4 * (Ehat - self.Emax.expand_as(Ehat))) + 1.0) / 2.0 / Ehat
        w /= w.max()
        wmse = (w * (yd - yd_pred)**2).mean()

        return wmse

class WRMSELoss_PS(torch.nn.Module):
    """
    Weight factors of the form suggested by Partridge and Schwenke
    H. Partridge, D. W. Schwenke, J. Chem. Phys. 106, 4618 (1997)
    """
    def __init__(self, Emax=2000.0):
        super().__init__()
        self.Emax   = torch.FloatTensor([Emax]).to(DEVICE)
        self.y_mean = None
        self.y_std  = None

    def set_scale(self, y_mean, y_std):
        self.y_mean = torch.FloatTensor(y_mean.tolist()).to(DEVICE)
        self.y_std  = torch.FloatTensor(y_std.tolist()).to(DEVICE)

    def __repr__(self):
        return "WRMSELoss_PS(Emax={})".format(self.Emax)

    def forward(self, y, y_pred):
        assert self.y_mean is not None
        assert self.y_std is not None

        # descale energies
        yd      = y      * self.y_std + self.y_mean
        yd_pred = y_pred * self.y_std + self.y_mean

        N = 1e-4
        Ehat = torch.max(yd, self.Emax.expand_as(yd))
        w = (torch.tanh(-6e-4 * (Ehat - self.Emax.expand_as(Ehat))) + 1.002002002) / 2.002002002 / N / Ehat
        w /= w.max()
        wmse = (w * (yd - yd_pred)**2).mean()

        return torch.sqrt(wmse)

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


def count_params(model):
    nparams = 0
    for name, param in model.named_parameters():
        params = torch.tensor(param.size())
        nparams += torch.prod(params, 0)

    return nparams



class Training:
    def __init__(self, model_folder, model_name, ckh_path, cfg, train, val, test, rank=0, world_size=1, local_rank=0):
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
            if typ == 'ENERGY':
                self.model = build_network(cfg_model, hidden_dims=cfg['MODEL']['HIDDEN_DIMS'], input_features=train.NPOLY, output_features=1)
            elif typ == 'DIPOLE':
                self.model = build_network(cfg_model, hidden_dims=cfg['MODEL']['HIDDEN_DIMS'][0], input_features=train.NPOLY, output_features=3)
            elif typ == 'DIPOLEQ':
                self.model = QModel(cfg_model, input_features=train.NPOLY, output_features=[len(natoms) for natoms in train.symmetry.values()])
            elif typ == 'DIPOLEC':
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
        self.cfg_pretrained_model_settings = cfg.get('PRETRAINED_MODEL_SETTINGS', None)
        if self.cfg_pretrained_model_settings is None: 
            return None, None, None 

        pretrained_source_path = self.cfg_pretrained_model_settings.get('SOURCE', None)
        assert pretrained_source_path is not None, "SOURCE path for pretrained model is not provided"

        pretrained_source_path = os.path.join(self.model_folder, pretrained_source_path)
        print("\n")
        logging.info("Looking for pretrained model (.pt) in {}".format(pretrained_source_path))
        model, xscaler, yscaler = load_from_checkpoint(pretrained_source_path)


        return model, xscaler, yscaler

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
            optimizer    = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError("unreachable")

        logging.info("Build optimizer: {}".format(optimizer))

        return optimizer

    def build_loss(self):
        known_options = ('NAME', 'WEIGHT_TYPE', 'DWT', 'EREF', 'EMAX', 'USE_GRADIENTS', 'USE_GRADIENTS_AFTER_EPOCH', 'G_LAMBDA', 'G_LAMBDA_RAMP_EPOCHS', 'LAMBDA_Q', 'TRUST_THRESHOLD', 'TRUST_THRESHOLD_START', 'TRUST_THRESHOLD_RAMP_EPOCHS', 'TRUST_SOFT_SCALE', 'TRUST_SOFT_CUTOFF', 'GRADIENT_TRUST_THRESHOLD', 'GRADIENT_TRUST_SOFT_SCALE', 'FOCAL_GAMMA', 'FOCAL_EMA_DECAY', 'USE_HUBER_GRADIENT', 'HUBER_DELTA', 'USE_MGDA', 'MGDA_ALPHA_MIN', 'MGDA_ALPHA_MAX', 'MGDA_EMA_DECAY')
        for option in self.cfg_loss.keys():
            assert option.upper() in known_options, "[build_loss] unknown option: {}".format(option)

        # have all defaults in the same place and set them to configuration if the value is omitted in the YAML file
        self.cfg_loss.setdefault('LAMBDA_Q', 1.0e3)
        self.cfg_loss.setdefault('USE_GRADIENTS_AFTER_EPOCH', None)
        self.cfg_loss.setdefault('USE_GRADIENTS', False)
        self.cfg_loss.setdefault('G_LAMBDA_RAMP_EPOCHS', 0)
        self.cfg_loss.setdefault('TRUST_THRESHOLD_RAMP_EPOCHS', 0)
        self.cfg_loss.setdefault('TRUST_SOFT_SCALE', None)
        self.cfg_loss.setdefault('TRUST_SOFT_CUTOFF', 0.01)
        self.cfg_loss.setdefault('GRADIENT_TRUST_THRESHOLD', None)
        self.cfg_loss.setdefault('GRADIENT_TRUST_SOFT_SCALE', None)
        self.cfg_loss.setdefault('FOCAL_GAMMA', 0.0)
        self.cfg_loss.setdefault('FOCAL_EMA_DECAY', 0.95)
        self.cfg_loss.setdefault('USE_HUBER_GRADIENT', False)

        if self.cfg_loss['NAME'] == 'WRMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'Ratio' and self.cfg['TYPE'] == 'DIPOLE':
            dwt = self.cfg_loss.get('dwt', 1.0)
            loss_fn = WRMSELoss_Ratio_dipole(dwt=dwt)
        elif self.cfg_loss['NAME'] == 'WRMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'Ratio' and self.cfg['TYPE'] == 'DIPOLEQ':
            dwt = self.cfg_loss.get('dwt', 1.0)
            loss_fn = WRMSELoss_Ratio_dipole(dwt=dwt)
        elif self.cfg_loss['NAME'] == 'WRMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'Ratio' and self.cfg['TYPE'] == 'DIPOLEC':
            dwt = self.cfg_loss.get('dwt', 1.0)
            loss_fn = WRMSELoss_Ratio_dipole(dwt=dwt)

        elif self.cfg_loss['NAME'] == 'WRMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'Boltzmann' and not self.cfg_loss['USE_GRADIENTS']:
            Eref = self.cfg_loss.get('EREF', 2000.0)
            loss_fn = WRMSELoss_Boltzmann(Eref=Eref)
        elif self.cfg_loss['NAME'] == 'WMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'Boltzmann' and not self.cfg_loss['USE_GRADIENTS']:
            Eref = self.cfg_loss.get('EREF', 2000.0)
            loss_fn = WMSELoss_Boltzmann(Eref=Eref)

        elif self.cfg_loss['NAME'] == 'WRMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'Ratio' and not self.cfg_loss['USE_GRADIENTS']:
            dwt = self.cfg_loss.get('dwt', 1.0)
            focal_gamma = self.cfg_loss.get('FOCAL_GAMMA', 0.0)
            focal_ema_decay = self.cfg_loss.get('FOCAL_EMA_DECAY', 0.95)
            loss_fn = WRMSELoss_Ratio(dwt=dwt, focal_gamma=focal_gamma, focal_ema_decay=focal_ema_decay)
        elif self.cfg_loss['NAME'] == 'WMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'Ratio' and not self.cfg_loss['USE_GRADIENTS']:
            dwt = self.cfg_loss.get('dwt', 1.0)
            focal_gamma = self.cfg_loss.get('FOCAL_GAMMA', 0.0)
            focal_ema_decay = self.cfg_loss.get('FOCAL_EMA_DECAY', 0.95)
            loss_fn = WMSELoss_Ratio(dwt=dwt, focal_gamma=focal_gamma, focal_ema_decay=focal_ema_decay)

        elif self.cfg_loss['NAME'] == 'WRMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'PS' and not self.cfg_loss['USE_GRADIENTS']:
            Emax = self.cfg_loss.get('EMAX', 2000.0)
            loss_fn = WRMSELoss_PS(Emax=Emax)
        elif self.cfg_loss['NAME'] == 'WMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'PS' and not self.cfg_loss['USE_GRADIENTS']:
            Emax = self.cfg_loss.get('EMAX', 2000.0)
            loss_fn = WMSELoss_PS(Emax=Emax)


        elif self.cfg_loss['NAME'] == 'WMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'Ratio' and self.cfg_loss['USE_GRADIENTS']:
            dwt = self.cfg_loss.get('dwt', 1.0)
            g_lambda = self.cfg_loss.get('G_LAMBDA', 1.0)
            trust_threshold = self.cfg_loss.get('TRUST_THRESHOLD', None)
            focal_gamma = self.cfg_loss.get('FOCAL_GAMMA', 0.0)
            focal_ema_decay = self.cfg_loss.get('FOCAL_EMA_DECAY', 0.95)
            if trust_threshold is not None:
                # Use memory-efficient trust region loss with soft boundaries
                soft_scale = self.cfg_loss.get('TRUST_SOFT_SCALE', None)
                use_huber = self.cfg_loss.get('USE_HUBER_GRADIENT', False)
                huber_delta = None
                if use_huber:
                    # Check for explicit override first
                    huber_delta = self.cfg_loss.get('HUBER_DELTA', None)
                    if huber_delta is not None:
                        logging.info("Huber delta (explicit) = {:.6e}".format(huber_delta))
                    else:
                        mad = getattr(self.train, 'mad_grad_components', None)
                        assert mad is not None and mad > 0, (
                            "USE_HUBER_GRADIENT requires train.mad_grad_components; "
                            "available only for gradient-loaded datasets.")
                        # Huber 95%-efficiency constant at the normal is k = 1.345*sigma.
                        # For Gaussian, sigma ~= 1.4826 * MAD, so k ~= 1.994 * MAD.
                        huber_delta = 2.0 * float(mad)
                        logging.info("Huber delta (auto) = 2 * MAD = {:.6e}".format(huber_delta))
                loss_fn = WMSELoss_TrustRegion_wgradients(natoms=self.train.NATOMS, dwt=dwt, g_lambda=g_lambda,
                                                       trust_threshold=trust_threshold,
                                                       soft_scale=soft_scale,
                                                       focal_gamma=focal_gamma,
                                                       focal_ema_decay=focal_ema_decay,
                                                       huber_delta=huber_delta)
                # Log gradient trust settings (applied in compute_trust_mask)
                grad_trust_threshold = self.cfg_loss.get('GRADIENT_TRUST_THRESHOLD', None)
                if grad_trust_threshold is not None:
                    grad_trust_soft_scale = self.cfg_loss.get('GRADIENT_TRUST_SOFT_SCALE', None)
                    logging.info("Gradient trust threshold = {:.2f} cm-1/bohr (soft_scale={})".format(
                        grad_trust_threshold, grad_trust_soft_scale))
            else:
                loss_fn = WMSELoss_Ratio_wgradients(natoms=self.train.NATOMS, dwt=dwt, g_lambda=g_lambda)

        else:
            print(self.cfg_loss)
            raise ValueError("unreachable")

        logging.info("Build loss function: {}".format(loss_fn))

        return loss_fn

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

    def continue_from_checkpoint(self, chkpath):
        assert os.path.exists(chkpath)

        self.reset_weights()
        checkpoint = torch.load(chkpath, map_location=torch.device(DEVICE))
        self.model.load_state_dict(checkpoint["model"])

        self.train_model()


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

            if multibatch:
                # Keep the training set on CPU; each step copies only its batch
                # to the GPU (pin_memory makes the per-batch copy faster when CUDA).
                if torch.cuda.is_available():
                    self.train.X = self.train.X.pin_memory()
                    self.train.y = self.train.y.pin_memory()
                    if self.train.dX is not None:
                        self.train.dX = self.train.dX.pin_memory()
                        self.train.dy = self.train.dy.pin_memory()
                # Val stays on GPU for cheap eval.
                self.val.X = self.val.X.to(self.device)
                self.val.y = self.val.y.to(self.device)
            else:
                self.train.X = self.train.X.to(self.device)
                self.train.y = self.train.y.to(self.device)
                self.val.X = self.val.X.to(self.device)
                self.val.y = self.val.y.to(self.device)

            self.loss_fn = self.loss_fn.to(self.device)

            if self.cfg['TYPE'] == 'DIPOLE':
                self.train.grm = self.train.grm.to(self.device)
                self.val.grm   = self.val.grm.to(self.device)

            if self.cfg['TYPE'] == 'DIPOLEQ':
                self.train.xyz_ordered = self.train.xyz_ordered.to(self.device)
                self.val.xyz_ordered = self.val.xyz_ordered.to(self.device)
                self.test.xyz_ordered = self.test.xyz_ordered.to(self.device)

            if self.train.dX is not None and not multibatch:
                self.train.dX = self.train.dX.to(self.device)
                self.train.dy = self.train.dy.to(self.device)

                self.val.dX = self.val.dX.to(self.device)
                self.val.dy = self.val.dy.to(self.device)
            elif self.train.dX is not None and multibatch:
                # Only move validation gradient tensors; train stays on pinned CPU.
                self.val.dX = self.val.dX.to(self.device)
                self.val.dy = self.val.dy.to(self.device)


            if multibatch:
                self.optimizer = self._build_multibatch_optimizer()
                self._init_multibatch_sampler()
            else:
                self.optimizer = self.build_optimizer(self.cfg_solver['OPTIMIZER'])
            self.scheduler = self.build_scheduler()

            start = time.time()

            MAX_EPOCHS = self.cfg_solver['MAX_EPOCHS']

            for epoch in range(MAX_EPOCHS):
                # switch into mixed loss function: E + F
                if self.cfg_loss['USE_GRADIENTS_AFTER_EPOCH'] is not None and epoch == self.cfg_loss['USE_GRADIENTS_AFTER_EPOCH']:
                    self.cfg_loss['USE_GRADIENTS'] = True
                    self.loss_fn = self.build_loss().to(self.device)
                    self.loss_fn.set_scale(self.yscaler.mean_, self.yscaler.scale_)
                    self.gradient_start_epoch = epoch

                    self.es.reset()

                    # Reset L-BFGS curvature history. The stored (s_k, y_k) pairs
                    # describe the energy-only loss surface and produce degenerate
                    # search directions on the new energy+gradient surface, causing
                    # the Wolfe line search to return t=0 indefinitely.
                    if isinstance(self.optimizer, (torch.optim.LBFGS, HjmshiLBFGS, HjmshiFullBatchLBFGS)):
                        if isinstance(self.optimizer, torch.optim.LBFGS):
                            self.optimizer.state.clear()
                        else:
                            # vendored LBFGS / FullBatchLBFGS
                            state = self.optimizer.state['global_state']
                            state['n_iter'] = 0
                            state['curv_skips'] = 0
                            state['fail_skips'] = 0
                            state['H_diag'] = 1
                            state['fail'] = True
                            state['old_dirs'] = []
                            state['old_stps'] = []
                            if 'rho' in state:
                                state['rho'] = [None] * self.optimizer.param_groups[0]['history_size']
                            if 'alpha' in state:
                                state['alpha'] = [None] * self.optimizer.param_groups[0]['history_size']
                        self._lbfgs_prev_n_iter = 0
                        self._lbfgs_prev_func_evals = 0
                        self._log("Reset L-BFGS state at gradient inclusion (epoch {})".format(epoch))

                    # Reset LR to initial value so the optimizer has full step
                    # budget to explore the new loss landscape.
                    initial_lr = self.cfg_solver['OPTIMIZER'].get('LR', 0.1)
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = initial_lr
                    self.scheduler = self.build_scheduler()
                    self._log("Reset LR to {} and rebuilt scheduler at gradient inclusion".format(initial_lr))

                # Progressive G_LAMBDA ramp
                if self.cfg_loss['USE_GRADIENTS'] and self.cfg_loss.get('G_LAMBDA_RAMP_EPOCHS', 0) > 0:
                    ramp_epochs = self.cfg_loss['G_LAMBDA_RAMP_EPOCHS']
                    start_epoch = self.gradient_start_epoch if self.gradient_start_epoch is not None else 0
                    progress = (epoch - start_epoch) / ramp_epochs
                    progress = max(0.0, min(1.0, progress))
                    target_g_lambda = self.cfg_loss.get('G_LAMBDA', 1.0)
                    current_g_lambda = target_g_lambda * progress
                    self.loss_fn.g_lambda = torch.tensor(current_g_lambda).to(self.device)
                    if epoch % PRINT_TRAINING_STEPS == 0 or epoch == start_epoch or epoch == start_epoch + ramp_epochs:
                        self._log("G_LAMBDA ramp: epoch {}, progress {:.1%}, g_lambda = {:.4f}".format(epoch, progress, current_g_lambda))

                # Progressive trust-threshold annealing
                if self.cfg_loss['USE_GRADIENTS'] and self.cfg_loss.get('TRUST_THRESHOLD_RAMP_EPOCHS', 0) > 0:
                    ramp_epochs = self.cfg_loss['TRUST_THRESHOLD_RAMP_EPOCHS']
                    start_epoch = self.gradient_start_epoch if self.gradient_start_epoch is not None else 0
                    progress = (epoch - start_epoch) / ramp_epochs
                    progress = max(0.0, min(1.0, progress))
                    target_threshold = self.cfg_loss.get('TRUST_THRESHOLD', 50.0)
                    start_threshold = self.cfg_loss.get('TRUST_THRESHOLD_START', target_threshold)
                    self.current_trust_threshold = start_threshold + (target_threshold - start_threshold) * progress
                    if epoch % PRINT_TRAINING_STEPS == 0 or epoch == start_epoch or epoch == start_epoch + ramp_epochs:
                        self._log("Trust-threshold anneal: epoch {}, progress {:.1%}, threshold = {:.1f}".format(epoch, progress, self.current_trust_threshold))
                else:
                    self.current_trust_threshold = None

                # Periodic L-BFGS curvature reset. The combined energy+gradient
                # surface evolves as G_LAMBDA ramps; stale curvature pairs cause
                # the Wolfe line search to return t=0. Clearing the history gradients
                # steepest-descent restart and fresh curvature accumulation.
                lbfgs_reset_interval = self.cfg_solver['OPTIMIZER'].get(
                    'LBFGS_RESET_INTERVAL', 0
                )
                if (lbfgs_reset_interval > 0
                        and self.cfg_loss['USE_GRADIENTS']
                        and isinstance(self.optimizer, (torch.optim.LBFGS, HjmshiLBFGS, HjmshiFullBatchLBFGS))
                        and epoch > self.cfg_loss.get('USE_GRADIENTS_AFTER_EPOCH', 0)
                        and (epoch - self.cfg_loss.get('USE_GRADIENTS_AFTER_EPOCH', 0))
                            % lbfgs_reset_interval == 0):
                    if isinstance(self.optimizer, torch.optim.LBFGS):
                        self.optimizer.state.clear()
                    else:
                        state = self.optimizer.state['global_state']
                        state['n_iter'] = 0
                        state['curv_skips'] = 0
                        state['fail_skips'] = 0
                        state['H_diag'] = 1
                        state['fail'] = True
                        state['old_dirs'] = []
                        state['old_stps'] = []
                        if 'rho' in state:
                            state['rho'] = [None] * self.optimizer.param_groups[0]['history_size']
                        if 'alpha' in state:
                            state['alpha'] = [None] * self.optimizer.param_groups[0]['history_size']
                    self._lbfgs_prev_n_iter = 0
                    self._lbfgs_prev_func_evals = 0
                    self._log("Periodic L-BFGS state reset (epoch {})".format(epoch))

                    # Optionally reset LR to initial value on L-BFGS reset
                    if self.cfg_solver['OPTIMIZER'].get('LR_RESET_ON_LBFGS_RESET', False):
                        initial_lr = self.cfg_solver['OPTIMIZER'].get('LR', 0.1)
                        for pg in self.optimizer.param_groups:
                            pg['lr'] = initial_lr
                        self.scheduler = self.build_scheduler()
                        self._log("Reset LR to {} and rebuilt scheduler".format(initial_lr))

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

    def compute_gradients(self, dataset):
        Xtr = dataset.X

        Xtr.requires_grad = True

        y_pred = self.model(Xtr)
        dEdp   = torch.autograd.grad(outputs=y_pred, inputs=Xtr, grad_outputs=torch.ones_like(y_pred), retain_graph=True, create_graph=True)[0]

        Xtr.requires_grad = False

        # take into account normalization of polynomials
        # now we have derivatives of energy w.r.t. to polynomials
        x_scale = torch.from_numpy(self.xscaler.scale_).to(self.device)
        dEdp = torch.div(dEdp, x_scale)

        # gradient = dE/dx = \sigma(E) * dE/d(poly) * d(poly)/dx
        # `torch.einsum` throws a Runtime error without an explicit conversion to Double
        dEdx = torch.einsum('ij,ijk -> ik', dEdp.to(TORCH_FLOAT), dataset.dX.to(TORCH_FLOAT))

        # take into account normalization of model energy
        y_scale = torch.from_numpy(self.yscaler.scale_).to(self.device)
        dEdx = torch.mul(dEdx, y_scale)

        return y_pred, dEdx

    def compute_gradients_from_energy(self, X_subset, dX_subset, y_pred_subset):
        """
        Compute gradients for a subset given pre-computed energy predictions.
        This avoids a second forward pass through the model.

        Args:
            X_subset: Input polynomials for subset (must have requires_grad=True)
            dX_subset: Polynomial gradients for subset
            y_pred_subset: Energy predictions for subset (from same forward pass)
        """
        logging.debug("compute_gradients_from_energy: X_subset shape={}, dX_subset shape={}".format(
            X_subset.shape, dX_subset.shape))

        dEdp = torch.autograd.grad(
            outputs=y_pred_subset,
            inputs=X_subset,
            grad_outputs=torch.ones_like(y_pred_subset),
            retain_graph=True,
            create_graph=True
        )[0]

        # take into account normalization of polynomials
        x_scale = torch.from_numpy(self.xscaler.scale_).to(self.device)
        dEdp = torch.div(dEdp, x_scale)

        # gradient = dE/dx = \sigma(E) * dE/d(poly) * d(poly)/dx
        dEdx = torch.einsum('ij,ijk -> ik', dEdp.to(TORCH_FLOAT), dX_subset.to(TORCH_FLOAT))

        # take into account normalization of model energy
        y_scale = torch.from_numpy(self.yscaler.scale_).to(self.device)
        dEdx = torch.mul(dEdx, y_scale)

        return dEdx

    def compute_trust_mask(self, dataset):
        """
        Compute trust region active set based on energy prediction errors
        and optionally gradient errors from the previous epoch.

        Uses soft boundaries with sigmoid weighting:
            phi(e_i) = sigmoid((threshold - error) / soft_scale) in [0, 1]
        Active set = {i : phi(e_i) > soft_cutoff} (memory optimization).

        When GRADIENT_TRUST_THRESHOLD is set, configs with large gradient
        errors (from previous epoch) are down-weighted using a soft sigmoid.
        This is combined multiplicatively with energy-based weights.

        Returns:
          trust_indices : 1-D LongTensor of active-set config indices
          trust_mask    : 1-D BoolTensor of shape (N,) indicating membership
          energy_errors : 1-D float tensor of |E_pred - E_true| (cm^-1)
          gradient_weights : 1-D float tensor of combined weights for the active set
        """
        trust_threshold = getattr(self, 'current_trust_threshold', None)
        if trust_threshold is None:
            trust_threshold = self.cfg_loss.get('TRUST_THRESHOLD', 50.0)

        soft_scale = self.cfg_loss.get('TRUST_SOFT_SCALE', None)
        soft_cutoff = self.cfg_loss.get('TRUST_SOFT_CUTOFF', 0.01)

        # Gradient trust: filter by previous epoch's gradient errors
        grad_trust_threshold = self.cfg_loss.get('GRADIENT_TRUST_THRESHOLD', None)
        grad_trust_soft_scale = self.cfg_loss.get('GRADIENT_TRUST_SOFT_SCALE', None)

        with torch.no_grad():
            y_pred = self.model(dataset.X)

            # Descale energies
            en_mean = torch.from_numpy(self.yscaler.mean_).to(self.device)
            en_std = torch.from_numpy(self.yscaler.scale_).to(self.device)

            en_pred_descaled = y_pred * en_std + en_mean
            en_true_descaled = dataset.y * en_std + en_mean

            energy_errors = torch.abs(en_pred_descaled - en_true_descaled).view(-1)

            # Always use soft boundary with sigmoid weighting
            phi_energy = WMSELoss_TrustRegion_wgradients.soft_phi(
                energy_errors, trust_threshold, soft_scale=soft_scale
            )
            trust_mask = phi_energy > soft_cutoff
            trust_indices = torch.nonzero(trust_mask, as_tuple=False).view(-1)
            gradient_weights = phi_energy[trust_indices]

            # Apply gradient trust filtering (uses previous epoch's gradient errors)
            if (grad_trust_threshold is not None
                    and self._prev_train_gradient_errors is not None
                    and self._prev_train_gradient_errors.numel() == energy_errors.numel()):
                # Compute soft phi for gradient errors (same sigmoid as energy)
                phi_grad = WMSELoss_TrustRegion_wgradients.soft_phi(
                    self._prev_train_gradient_errors,
                    grad_trust_threshold,
                    soft_scale=grad_trust_soft_scale
                )
                # Multiply energy weights by gradient weights
                gradient_weights = gradient_weights * phi_grad[trust_indices]

        return trust_indices, trust_mask, energy_errors, gradient_weights

    def log_trust_region_diagnostics(self, epoch, trust_mask, energy_errors,
                                     gradient_weights):
        """Diagnose trust-region evolution: churn, eviction signal, flip counts.

        Compares the current trust mask against the previous epoch's mask
        and the per-config gradient errors recorded at the end of the previous
        validation pass. Writes a CSV row per epoch and logs a summary.

        The eviction signal is the key check for "evasion" behavior:
        if configs that just LEFT the trust set had systematically higher
        gradient errors than configs that STAYED, the optimizer is plausibly
        gaming the boundary by pushing hard configs out.
        """
        N_local = trust_mask.numel()
        cur = trust_mask.detach()
        n_in_local = int(cur.sum().item())

        # Initialize trackers lazily on the first call.
        if self._trust_flip_count is None:
            self._trust_flip_count = torch.zeros(N_local, dtype=torch.long, device=DEVICE)

        if self._prev_trust_mask is None:
            entered_local = n_in_local
            left_local = 0
            stable_in_local = n_in_local
            stable_out_local = N_local - n_in_local
            mean_err_left = float('nan')
            mean_err_stayed = float('nan')
            med_err_left = float('nan')
            med_err_stayed = float('nan')
        else:
            prev = self._prev_trust_mask
            entered_mask = cur & (~prev)
            left_mask    = (~cur) & prev
            stable_in_mask  = cur & prev
            stable_out_mask = (~cur) & (~prev)
            entered_local = int(entered_mask.sum().item())
            left_local = int(left_mask.sum().item())
            stable_in_local = int(stable_in_mask.sum().item())
            stable_out_local = int(stable_out_mask.sum().item())

            # Update cumulative flip count.
            flips = entered_mask | left_mask
            self._trust_flip_count[flips] += 1

            # Eviction signal: compare prev-epoch gradient errors of left vs stayed.
            if (self._prev_train_gradient_errors is not None
                    and self._prev_train_gradient_errors.numel() == N_local):
                pfe = self._prev_train_gradient_errors
                if left_local > 0:
                    mean_err_left = float(pfe[left_mask].mean().item())
                    med_err_left  = float(pfe[left_mask].median().item())
                else:
                    mean_err_left = float('nan')
                    med_err_left  = float('nan')
                if stable_in_local > 0:
                    mean_err_stayed = float(pfe[stable_in_mask].mean().item())
                    med_err_stayed  = float(pfe[stable_in_mask].median().item())
                else:
                    mean_err_stayed = float('nan')
                    med_err_stayed  = float('nan')
            else:
                mean_err_left = float('nan')
                mean_err_stayed = float('nan')
                med_err_left = float('nan')
                med_err_stayed = float('nan')

        if gradient_weights.numel() > 0:
            phi_sum = float(gradient_weights.sum().item())
            phi_mean = float(gradient_weights.mean().item())
            phi_min = float(gradient_weights.min().item())
        else:
            phi_sum = float('nan')
            phi_mean = float('nan')
            phi_min = float('nan')

        max_flips = int(self._trust_flip_count.max().item())
        ever_in = int((self._trust_flip_count > 0).sum().item()) + stable_in_local

        # Aggregate counts across ranks for logging
        if self.world_size > 1:
            counts = torch.tensor([N_local, n_in_local, entered_local, left_local, stable_in_local],
                                  dtype=torch.float32, device=DEVICE)
            counts = reduce_sum(counts)
            N, n_in, entered, left, stable_in = [int(c.item()) for c in counts]
            frac = n_in / max(N, 1)
        else:
            N, n_in, entered, left, stable_in = N_local, n_in_local, entered_local, left_local, stable_in_local
            frac = n_in / max(N, 1)

        if is_main_process():
            logging.info(
                "[trust-diag] epoch={} | n_in={}/{} ({:.1%}) | entered={} left={} "
                "stable_in={} | prev-epoch gradient-RMSE: left={:.2f} stayed={:.2f} "
                "(med {:.2f}/{:.2f}) | max_flips={}".format(
                    epoch, n_in, N, frac, entered, left, stable_in,
                    mean_err_left, mean_err_stayed,
                    med_err_left, med_err_stayed, max_flips
                )
            )

        # Append CSV row for post-hoc plotting (only on main process).
        stable_out = N - n_in  # Compute from aggregated values
        if is_main_process():
            if not self._trust_history_initialized:
                try:
                    with open(self._trust_history_path, "w") as f:
                        f.write("epoch,N,n_in,frac,entered,left,stable_in,stable_out,"
                                "mean_err_left,mean_err_stayed,med_err_left,med_err_stayed,"
                                "phi_sum,phi_mean,phi_min,max_flips\n")
                    self._trust_history_initialized = True
                except OSError as e:
                    logging.warning("Could not initialize trust history CSV: {}".format(e))
            try:
                with open(self._trust_history_path, "a") as f:
                    f.write("{},{},{},{:.6f},{},{},{},{},"
                            "{:.6f},{:.6f},{:.6f},{:.6f},"
                            "{:.6f},{:.6f},{:.6f},{}\n".format(
                        epoch, N, n_in, frac, entered, left, stable_in, stable_out,
                        mean_err_left, mean_err_stayed, med_err_left, med_err_stayed,
                        phi_sum, phi_mean, phi_min, max_flips
                    ))
            except OSError as e:
                logging.warning("Could not append to trust history CSV: {}".format(e))

        # Snapshot current mask for next-epoch comparison.
        self._prev_trust_mask = cur.clone()

    def log_gradient_loss_diagnostics(self, epoch, train_dy, train_dy_pred,
                                   train_e_d, train_e_pred,
                                   trust_indices, gradient_weights):
        """Per-config gradient-loss contribution + phi histogram on the active set.

        Contribution mirrors the loss term per config:
            c_i = phi_i * w_energy_i * w_focal_i * ||f_i - f_i_pred||^2 / (3 N_atoms)
        (un-normalized; we want raw share, not the loss value itself.)
        """
        if trust_indices is None or trust_indices.numel() == 0:
            return

        natoms = self.train.NATOMS
        n_active = int(trust_indices.numel())

        with torch.no_grad():
            # Per-config gradient squared error on the active set.
            dy_act      = train_dy[trust_indices]
            dy_pred_act = train_dy_pred[trust_indices]
            f_sq = (
                torch.sum((dy_act - dy_pred_act) ** 2, dim=1)
                / (3.0 * natoms)
            )  # (n_active,)

            # Re-derive w_energy * w_focal on the active set. _compute_weights
            # is safe to call here: error_scale was already updated inside the
            # closure, so the EMA guard prevents double-update.
            if hasattr(self.loss_fn, '_compute_weights'):
                w_full = self.loss_fn._compute_weights(train_e_d, train_e_pred)
                w_act = w_full.view(-1)[trust_indices]
            else:
                w_act = torch.ones(n_active, device=DEVICE)

            phi_act = gradient_weights.view(-1).to(f_sq.dtype)

            contrib = (phi_act * w_act * f_sq).detach().cpu()
            phi_cpu = phi_act.detach().cpu()

            qs = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
                              dtype=contrib.dtype)
            cq = torch.quantile(contrib, qs).tolist()
            contrib_sum = float(contrib.sum().item())
            contrib_max = float(contrib.max().item())

            # Top-k tail share (k = 1%, 5%, 10% of active set).
            sorted_c, _ = torch.sort(contrib, descending=True)
            def _tail_share(frac):
                k = max(1, int(round(frac * n_active)))
                return float(sorted_c[:k].sum().item()) / max(contrib_sum, 1e-30)
            top1  = _tail_share(0.01)
            top5  = _tail_share(0.05)
            top10 = _tail_share(0.10)

            pq = torch.quantile(phi_cpu, qs[:5].to(phi_cpu.dtype)).tolist()
            # Bin phi into membership categories.
            bins = torch.tensor([0.0, 0.25, 0.50, 0.75, 0.90, 1.0001])
            # counts per bin
            idx = torch.bucketize(phi_cpu, bins) - 1
            idx = idx.clamp(0, 4)
            bin_counts = [int((idx == b).sum().item()) for b in range(5)]

        if not self._gradient_diag_initialized:
            try:
                with open(self._gradient_diag_path, "w") as f:
                    f.write(
                        "epoch,n_active,contrib_sum,contrib_max,"
                        "contrib_q10,contrib_q25,contrib_q50,contrib_q75,"
                        "contrib_q90,contrib_q95,contrib_q99,"
                        "top1pct_share,top5pct_share,top10pct_share,"
                        "phi_q10,phi_q25,phi_q50,phi_q75,phi_q90,"
                        "phi_lt_25,phi_25_50,phi_50_75,phi_75_90,phi_ge_90\n"
                    )
                self._gradient_diag_initialized = True
            except OSError as e:
                logging.warning("Could not initialize gradient diag CSV: {}".format(e))
        try:
            with open(self._gradient_diag_path, "a") as f:
                f.write(
                    "{},{},{:.6e},{:.6e},"
                    "{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},"
                    "{:.6f},{:.6f},{:.6f},"
                    "{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},"
                    "{},{},{},{},{}\n".format(
                        epoch, n_active, contrib_sum, contrib_max,
                        cq[0], cq[1], cq[2], cq[3], cq[4], cq[5], cq[6],
                        top1, top5, top10,
                        pq[0], pq[1], pq[2], pq[3], pq[4],
                        bin_counts[0], bin_counts[1], bin_counts[2],
                        bin_counts[3], bin_counts[4],
                    )
                )
        except OSError as e:
            logging.warning("Could not append to gradient diag CSV: {}".format(e))

        if is_main_process():
            logging.info(
                "[grad-diag] epoch={} | top1%={:.1%} top5%={:.1%} top10%={:.1%} "
                "of gradient loss | contrib q50={:.3e} q95={:.3e} max={:.3e}".format(
                    epoch, top1, top5, top10, cq[2], cq[5], contrib_max
                )
            )

    def log_lbfgs_diagnostics(self, epoch, optimizer):
        """L-BFGS line-search telemetry, dumped per epoch.

        Pulls inner state from torch.optim.LBFGS or vendored FullBatchLBFGS:
          - this-step iteration / closure-call counts (deltas from cumulative)
          - last accepted step length t
          - initial Hessian diag scaling H_diag = (s . y) / (y . y)
          - curvature pair stats: <s_k, y_k> -- min/max/last/mean
            over the stored history (small or absent => degenerate curvature)
          - flat gradient norm at the last accepted iterate
        """
        if isinstance(optimizer, torch.optim.LBFGS):
            params = optimizer.param_groups[0]['params']
            if not params:
                return
            state = optimizer.state.get(params[0], {})
            if not state:
                return

            cum_n_iter     = int(state.get('n_iter', 0))
            cum_func_evals = int(state.get('func_evals', 0))
            iters_this_step = cum_n_iter - self._lbfgs_prev_n_iter
            evals_this_step = cum_func_evals - self._lbfgs_prev_func_evals
            self._lbfgs_prev_n_iter = cum_n_iter
            self._lbfgs_prev_func_evals = cum_func_evals

            t_val = state.get('t', None)
            try:
                t_val = float(t_val) if t_val is not None else float('nan')
            except (TypeError, ValueError):
                t_val = float('nan')

            H_diag = state.get('H_diag', None)
            try:
                H_diag = float(H_diag) if H_diag is not None else float('nan')
            except (TypeError, ValueError):
                H_diag = float('nan')

            ro = state.get('ro', []) or []
            n_pairs = len(ro)
            if n_pairs > 0:
                sy_vals = []
                for r in ro:
                    try:
                        rv = float(r)
                        if rv != 0.0:
                            sy_vals.append(1.0 / rv)
                    except (TypeError, ValueError, ZeroDivisionError):
                        pass
                if sy_vals:
                    sy_min  = min(sy_vals)
                    sy_max  = max(sy_vals)
                    sy_last = sy_vals[-1]
                    sy_mean = sum(sy_vals) / len(sy_vals)
                else:
                    sy_min = sy_max = sy_last = sy_mean = float('nan')
            else:
                sy_min = sy_max = sy_last = sy_mean = float('nan')

            prev_flat_grad = state.get('prev_flat_grad', None)
            if prev_flat_grad is not None:
                try:
                    grad_norm = float(prev_flat_grad.norm().item())
                except (RuntimeError, AttributeError):
                    grad_norm = float('nan')
            else:
                grad_norm = float('nan')

        elif isinstance(optimizer, HjmshiFullBatchLBFGS):
            state = optimizer.state['global_state']
            cum_n_iter = int(state.get('n_iter', 0))
            iters_this_step = cum_n_iter - self._lbfgs_prev_n_iter
            self._lbfgs_prev_n_iter = cum_n_iter
            # Closure evals are captured in train_epoch for vendored LBFGS
            evals_this_step = getattr(self, '_last_vendored_closure_eval', float('nan'))

            t_val = float(state.get('t', float('nan')))
            H_diag = float(state.get('H_diag', float('nan')))

            old_dirs = state.get('old_dirs', [])
            old_stps = state.get('old_stps', [])
            n_pairs = len(old_dirs)
            if n_pairs > 0:
                sy_vals = []
                for s, y in zip(old_stps, old_dirs):
                    try:
                        sy = float(s.dot(y).item())
                        if sy != 0.0:
                            sy_vals.append(sy)
                    except (TypeError, ValueError):
                        pass
                if sy_vals:
                    sy_min  = min(sy_vals)
                    sy_max  = max(sy_vals)
                    sy_last = sy_vals[-1]
                    sy_mean = sum(sy_vals) / len(sy_vals)
                else:
                    sy_min = sy_max = sy_last = sy_mean = float('nan')
            else:
                sy_min = sy_max = sy_last = sy_mean = float('nan')

            prev_flat_grad = state.get('prev_flat_grad', None)
            if prev_flat_grad is not None:
                try:
                    grad_norm = float(prev_flat_grad.norm().item())
                except (RuntimeError, AttributeError):
                    grad_norm = float('nan')
            else:
                grad_norm = float('nan')
        else:
            return

        if not self._lbfgs_diag_initialized:
            try:
                with open(self._lbfgs_diag_path, "w") as f:
                    f.write("epoch,iters_this_step,evals_this_step,t,H_diag,"
                            "n_pairs,grad_norm,sy_min,sy_mean,sy_max,sy_last\n")
                self._lbfgs_diag_initialized = True
            except OSError as e:
                logging.warning("Could not initialize lbfgs diag CSV: {}".format(e))
        try:
            with open(self._lbfgs_diag_path, "a") as f:
                f.write("{},{},{},{:.6e},{:.6e},{},{:.6e},"
                        "{:.6e},{:.6e},{:.6e},{:.6e}\n".format(
                    epoch, iters_this_step, evals_this_step, t_val, H_diag,
                    n_pairs, grad_norm, sy_min, sy_mean, sy_max, sy_last,
                ))
        except OSError as e:
            logging.warning("Could not append to lbfgs diag CSV: {}".format(e))

        if is_main_process():
            logging.info(
                "[lbfgs-diag] epoch={} | iters={} evals={} t={:.3e} H_diag={:.3e} "
                "pairs={} grad_norm={:.3e} sy(last/min/max)={:.3e}/{:.3e}/{:.3e}".format(
                    epoch, iters_this_step, evals_this_step, t_val, H_diag,
                    n_pairs, grad_norm, sy_last, sy_min, sy_max,
                )
            )

    def log_mgda_diagnostics(self, epoch, alpha, alpha_raw, cos_sim):
        """Log MGDA+GradNorm diagnostics.

        Args:
            epoch: current epoch
            alpha: EMA-smoothed weight for energy objective
            alpha_raw: raw (unsmoothed) weight from loss-ratio computation
            cos_sim: cosine similarity between normalized gradients
        """
        if not is_main_process():
            return

        # Get loss EMAs for logging
        e_loss_ema = self._mgda_energy_loss_ema if self._mgda_energy_loss_ema is not None else 0.0
        g_loss_ema = self._mgda_gradient_loss_ema if self._mgda_gradient_loss_ema is not None else 0.0

        if not self._mgda_diag_initialized:
            try:
                with open(self._mgda_diag_path, "w") as f:
                    f.write("epoch,alpha,alpha_raw,cos_sim,e_loss_ema,g_loss_ema\n")
                self._mgda_diag_initialized = True
            except OSError as e:
                logging.warning("Could not initialize MGDA diag CSV: {}".format(e))

        try:
            with open(self._mgda_diag_path, "a") as f:
                f.write("{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(
                    epoch, alpha, alpha_raw, cos_sim, e_loss_ema, g_loss_ema
                ))
        except OSError as e:
            logging.warning("Could not append to MGDA diag CSV: {}".format(e))

    def log_distributed_diagnostics(self, epoch, loss_local, e_rmse_local, n_trust_local, n_total_local):
        """Log verbose per-rank metrics for distributed training.

        Shows per-rank values + global aggregates to diagnose imbalanced shards,
        rank drift, or trust region distribution issues.

        Args:
            epoch: current epoch
            loss_local: local weighted MSE (before reduce_mean)
            e_rmse_local: local energy RMSE in cm-1 (before reduce)
            n_trust_local: number of configs in trust region on this rank
            n_total_local: total configs on this rank
        """
        if self.world_size <= 1:
            return

        # Gather values from all ranks
        losses = all_gather_scalar(float(loss_local), device=DEVICE)
        rmses = all_gather_scalar(float(e_rmse_local), device=DEVICE)
        trusts = all_gather_scalar(int(n_trust_local), device=DEVICE)
        totals = all_gather_scalar(int(n_total_local), device=DEVICE)

        # Log verbose multi-line format on rank 0
        if is_main_process():
            lines = [f"[dist-diag] epoch={epoch}"]
            for r in range(self.world_size):
                trust_pct = 100.0 * trusts[r] / max(totals[r], 1)
                lines.append(
                    f"  rank {r}: E-RMSE={rmses[r]:.2f} cm-1  "
                    f"trust={int(trusts[r])}/{int(totals[r])} ({trust_pct:.1f}%)  "
                    f"loss={losses[r]:.3f}"
                )
            # Global summary
            total_trust = sum(trusts)
            total_n = sum(totals)
            global_trust_pct = 100.0 * total_trust / max(total_n, 1)
            avg_loss = sum(losses) / len(losses)
            lines.append(
                f"  global: E-RMSE=<aggregated above>  "
                f"trust={int(total_trust)}/{int(total_n)} ({global_trust_pct:.1f}%)  "
                f"loss={avg_loss:.3f}"
            )
            logging.info("\n".join(lines))

        # Write CSV for post-hoc analysis (all ranks write their own row)
        if not self._dist_diag_initialized:
            if is_main_process():
                try:
                    with open(self._dist_diag_path, "w") as f:
                        f.write("epoch,rank,loss_local,e_rmse_local,n_trust,n_total\n")
                    self._dist_diag_initialized = True
                except OSError as e:
                    logging.warning("Could not initialize distributed diag CSV: {}".format(e))
            barrier()  # Ensure header is written before other ranks append
            self._dist_diag_initialized = True

        try:
            with open(self._dist_diag_path, "a") as f:
                f.write("{},{},{:.6e},{:.6e},{},{}\n".format(
                    epoch, self.rank, float(loss_local), float(e_rmse_local),
                    int(n_trust_local), int(n_total_local)
                ))
        except OSError as e:
            if is_main_process():
                logging.warning("Could not append to distributed diag CSV: {}".format(e))

    def compute_gradients_eval(self, dataset):
        """
        Compute gradients for evaluation (no create_graph needed).
        Much more memory efficient than compute_gradients() since we don't need
        to backpropagate through the gradient computation.
        """
        Xtr = dataset.X.clone().detach()
        Xtr.requires_grad = True

        with torch.enable_grad():
            y_pred = self.model(Xtr)
            dEdp = torch.autograd.grad(
                outputs=y_pred,
                inputs=Xtr,
                grad_outputs=torch.ones_like(y_pred),
                retain_graph=False,
                create_graph=False
            )[0]

        Xtr.requires_grad = False

        # take into account normalization of polynomials
        x_scale = torch.from_numpy(self.xscaler.scale_).to(self.device)
        dEdp = torch.div(dEdp, x_scale)

        # gradient = dE/dx
        dEdx = torch.einsum('ij,ijk -> ik', dEdp.to(TORCH_FLOAT), dataset.dX.to(TORCH_FLOAT))

        # take into account normalization of model energy
        y_scale = torch.from_numpy(self.yscaler.scale_).to(self.device)
        dEdx = torch.mul(dEdx, y_scale)

        return y_pred.detach(), dEdx.detach()

    def train_epoch(self, epoch, optimizer):
        CLOSURE_CALL_COUNT = 0

        # Precompute trust-region mask once per epoch so that the objective
        # stays fixed during the LBFGS step. Recomputing it inside the closure
        # breaks the line search because the loss landscape changes between
        # closure evaluations.
        use_trust_region = False
        trust_indices = None
        n_in_trust = 0
        X_subset = None
        dX_subset = None
        train_dy_subset = None
        gradient_weights = None
        energy_errors = None
        trust_mask = None

        if self.cfg_loss['USE_GRADIENTS']:
            trust_threshold = self.cfg_loss.get('TRUST_THRESHOLD', None)
            if trust_threshold is not None:
                use_trust_region = True
                trust_indices, trust_mask, energy_errors, gradient_weights = \
                    self.compute_trust_mask(self.train)
                n_in_trust = len(trust_indices)

                if n_in_trust > 0:
                    X_subset = self.train.X[trust_indices].clone()
                    X_subset.requires_grad = True
                    dX_subset = self.train.dX[trust_indices]
                    train_dy_subset = self.train.dy[trust_indices]

                # Aggregate trust region stats across ranks for correct logging
                n_in_trust_t = torch.tensor(n_in_trust, dtype=torch.float32, device=DEVICE)
                n_total_t = torch.tensor(len(self.train.X), dtype=torch.float32, device=DEVICE)
                err_min_t = energy_errors.min()
                err_max_t = energy_errors.max()
                if self.world_size > 1:
                    n_in_trust_global = int(reduce_sum(n_in_trust_t).item())
                    n_total_global = int(reduce_sum(n_total_t).item())
                    err_min_global = reduce_min(err_min_t).item()
                    err_max_global = torch.tensor(err_max_t.item(), device=DEVICE)
                    dist.all_reduce(err_max_global, op=dist.ReduceOp.MAX)
                    err_max_global = err_max_global.item()
                else:
                    n_in_trust_global = n_in_trust
                    n_total_global = len(self.train.X)
                    err_min_global = err_min_t.item()
                    err_max_global = err_max_t.item()
                frac_global = 100.0 * n_in_trust_global / max(n_total_global, 1)

                if n_in_trust > 0:
                    phi_sum_t = gradient_weights.sum()
                    phi_sum_global = reduce_sum(phi_sum_t).item() if self.world_size > 1 else phi_sum_t.item()
                    grad_trust_enabled = self.cfg_loss.get('GRADIENT_TRUST_THRESHOLD') is not None
                    label = "soft+grad" if grad_trust_enabled else "soft"
                    self._log(
                        "Trust region ({}): {}/{} configs ({:.1f}%) | "
                        "energy err: min={:.1f}, max={:.1f}, med={:.1f} | "
                        "weights: min={:.3f}, mean={:.3f}, sum={:.1f}".format(
                            label,
                            n_in_trust_global, n_total_global, frac_global,
                            err_min_global, err_max_global,
                            energy_errors.median().item(),
                            gradient_weights.min().item(), gradient_weights.mean().item(),
                            phi_sum_global))
                else:
                    self._log(
                        "Trust region: 0/{} configs (0.0%) | "
                        "energy err: min={:.1f}, max={:.1f}, med={:.1f}".format(
                            n_total_global,
                            err_min_global, err_max_global,
                            energy_errors.median().item()))

                # Run trust-region diagnostics (churn + eviction signal).
                self.log_trust_region_diagnostics(
                    epoch, trust_mask, energy_errors, gradient_weights
                )

        def _compute_loss(separate=False):
            """Compute training loss.

            Args:
                separate: If True and using gradients with trust region,
                         return (energy_loss, gradient_loss) tuple for MGDA.
                         Otherwise return combined loss.
            """
            if self.cfg_loss['USE_GRADIENTS']:
                if use_trust_region:
                    if n_in_trust > 0:
                        y_pred_subset = self.model(X_subset)
                        train_dy_pred_subset = self.compute_gradients_from_energy(
                            X_subset, dX_subset, y_pred_subset
                        )
                        train_y_pred = self.model(self.train.X)
                        if separate:
                            energy_loss, gradient_loss = self.loss_fn.forward_separate(
                                self.train.y, train_y_pred,
                                train_dy_subset, train_dy_pred_subset,
                                trust_indices, gradient_weights
                            )
                            # Add regularization to energy loss (it's model complexity, not gradient fitting)
                            if self.regularization is not None:
                                energy_loss = energy_loss + self.regularization(self.model)
                            return energy_loss, gradient_loss
                        else:
                            loss = self.loss_fn(
                                self.train.y, train_y_pred,
                                train_dy_subset, train_dy_pred_subset,
                                trust_indices, gradient_weights
                            )
                    else:
                        # No configs in trust region yet - energy only
                        train_y_pred = self.model(self.train.X)
                        loss = self.loss_fn.forward_energy_only(self.train.y, train_y_pred)
                        if separate:
                            if self.regularization is not None:
                                loss = loss + self.regularization(self.model)
                            return loss, torch.tensor(0.0, device=DEVICE)
                else:
                    # Original approach: compute gradients for ALL configs
                    train_y_pred, train_dy_pred = self.compute_gradients(self.train)
                    loss = self.loss_fn(self.train.y, train_y_pred, self.train.dy, train_dy_pred)

            elif self.cfg['TYPE'] == 'DIPOLE':
                y_pred = self.model(self.train.X)
                dip_pred = torch.einsum('ijk,ik->ij', self.train.grm, y_pred)
                loss = self.loss_fn(self.train.y, dip_pred)

            elif self.cfg['TYPE'] == 'DIPOLEQ':
                q_pred   = self.model(self.train.X)
                X_inf    = torch.zeros_like(self.train.X).cpu()
                X_inf_tr = torch.from_numpy(self.xscaler.transform(X_inf)).to(self.device)
                q_inf    = self.model(X_inf_tr)
                q_corr   = q_pred - q_inf
                dip_pred = torch.einsum('ijk,ij->ik', self.train.xyz_ordered.to(TORCH_FLOAT), q_corr)
                qsum     = torch.sum(q_corr, dim=1)
                qreg     = self.cfg_loss['LAMBDA_Q'] * torch.mean(qsum * qsum)
                loss     = self.loss_fn(self.train.y, dip_pred)
                loss = loss + qreg

            elif self.cfg['TYPE'] == 'DIPOLEC':
                dip_pred = self.model(self.train.X)
                loss = self.loss_fn(self.train.y, dip_pred)

            elif self.cfg['TYPE'] == 'ENERGY':
                y_pred = self.model(self.train.X)
                loss = self.loss_fn(self.train.y, y_pred)

            else:
                assert False, "unreachable"

            if self.regularization is not None:
                loss = loss + self.regularization(self.model)
            return loss

        def closure():
            nonlocal CLOSURE_CALL_COUNT
            CLOSURE_CALL_COUNT = CLOSURE_CALL_COUNT + 1
            optimizer.zero_grad()
            loss = _compute_loss()
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
            loss = _compute_loss()
            return loss

        # MGDA (Multi-objective Gradient Descent Algorithm) closure
        # Computes optimal combination of energy and gradient loss gradients
        use_mgda = (self.cfg_loss.get('USE_MGDA', False)
                    and self.cfg_loss['USE_GRADIENTS']
                    and use_trust_region
                    and n_in_trust > 0)
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
            energy_loss, gradient_loss = _compute_loss(separate=True)

            # Update loss EMAs for adaptive alpha computation
            energy_loss_val = energy_loss.detach().item()
            gradient_loss_val = gradient_loss.detach().item()

            if self._mgda_energy_loss_ema is None:
                self._mgda_energy_loss_ema = energy_loss_val
                self._mgda_gradient_loss_ema = gradient_loss_val
            else:
                self._mgda_energy_loss_ema = (mgda_ema_decay * self._mgda_energy_loss_ema +
                                              (1 - mgda_ema_decay) * energy_loss_val)
                self._mgda_gradient_loss_ema = (mgda_ema_decay * self._mgda_gradient_loss_ema +
                                                (1 - mgda_ema_decay) * gradient_loss_val)

            # Backward pass for energy gradient
            energy_loss.backward(retain_graph=True)
            g_energy = flatten_gradients(self.model)

            # Backward pass for gradient loss gradient
            optimizer.zero_grad()
            gradient_loss.backward()
            g_gradient = flatten_gradients(self.model)

            # Sync gradients across ranks before computing weights
            if self.world_size > 1:
                dist.all_reduce(g_energy, op=dist.ReduceOp.SUM)
                g_energy = g_energy / self.world_size
                dist.all_reduce(g_gradient, op=dist.ReduceOp.SUM)
                g_gradient = g_gradient / self.world_size

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
            """MGDA closure for line search (no backward needed)."""
            nonlocal CLOSURE_CALL_COUNT
            CLOSURE_CALL_COUNT = CLOSURE_CALL_COUNT + 1
            optimizer.zero_grad()
            energy_loss, gradient_loss = _compute_loss(separate=True)
            # Use current EMA alpha for consistent loss evaluation
            alpha = self._mgda_alpha_ema if self._mgda_alpha_ema is not None else 0.5
            combined_loss = alpha * energy_loss + (1 - alpha) * gradient_loss
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
                logging.debug(f"[rank {self.rank}] vendored LBFGS (MGDA): initial closure_mgda")
                loss = closure_mgda()  # This sets gradients via MGDA
                # Note: closure_mgda already syncs gradients and applies clipping
                # Build grad_sync closure that captures self.model
                def _grad_sync():
                    sync_gradients(self.model)
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
            obj, grad_new, t, ls_step, closure_eval, grad_eval, desc_dir, fail = optimizer.step(options=options)
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

        if self.cfg_loss['USE_GRADIENTS']:
            # Use memory-efficient gradient evaluation (no create_graph)
            train_y_pred, train_dy_pred = self.compute_gradients_eval(self.train)
            val_y_pred, val_dy_pred = self.compute_gradients_eval(self.val)

            # Compute energy metrics directly (works with any loss function)
            # Compute local values first, then aggregate via reduce_rmse/reduce_mae
            train_e_d    = self.loss_fn.descale_energies(self.train.y)
            train_e_pred = self.loss_fn.descale_energies(train_y_pred)
            train_e_errors = (train_e_d - train_e_pred).view(-1)
            train_e_rmse_local = torch.sqrt(torch.mean(train_e_errors ** 2)).item()
            train_e_mae  = reduce_mae(train_e_errors)
            train_e_rmse = reduce_rmse(train_e_errors)

            val_e_d    = self.loss_fn.descale_energies(self.val.y)
            val_e_pred = self.loss_fn.descale_energies(val_y_pred)
            val_e_errors = (val_e_d - val_e_pred).view(-1)
            val_e_rmse_local = torch.sqrt(torch.mean(val_e_errors ** 2)).item()
            val_e_mae  = reduce_mae(val_e_errors)
            val_e_rmse = reduce_rmse(val_e_errors)

            # Compute gradient metrics directly (per-component errors for RMSE/MAE)
            natoms   = self.train.NATOMS
            train_dy = self.train.dy.reshape(-1, 3 * natoms)
            val_dy   = self.val.dy.reshape(-1, 3 * natoms)
            train_g_errors = (train_dy - train_dy_pred).view(-1)
            val_g_errors   = (val_dy - val_dy_pred).view(-1)
            train_g_mae  = reduce_mae(train_g_errors)
            val_g_mae    = reduce_mae(val_g_errors)
            train_g_rmse = reduce_rmse(train_g_errors)
            val_g_rmse   = reduce_rmse(val_g_errors)

            # Snapshot per-config train gradient RMSE for next-epoch trust-region
            # diagnostics (eviction signal: do "left" configs have higher
            # gradient errors than "stayed" configs?).
            with torch.no_grad():
                per_config_f_rmse = torch.sqrt(
                    torch.sum((train_dy - train_dy_pred) ** 2, dim=1) / (3 * natoms)
                ).detach()
                self._prev_train_gradient_errors = per_config_f_rmse

            # Per-config gradient-loss contribution + phi histogram on the active set.
            if use_trust_region and trust_indices is not None and n_in_trust > 0:
                self.log_gradient_loss_diagnostics(
                    epoch, train_dy, train_dy_pred,
                    train_e_d, train_e_pred,
                    trust_indices, gradient_weights,
                )

            # Compute weighted loss values for logging (energy component only for scheduler)
            # Sync minimum across ranks for consistent weighting in distributed mode
            enmin_train = reduce_min(train_e_d.min())
            w_train = self.loss_fn.dwt / (self.loss_fn.dwt + train_e_d - enmin_train)
            loss_train_e = (w_train.view(-1) * (train_e_d - train_e_pred).view(-1)**2).mean()

            enmin_val = reduce_min(val_e_d.min())
            w_val = self.loss_fn.dwt / (self.loss_fn.dwt + val_e_d - enmin_val)
            loss_val_e = (w_val.view(-1) * (val_e_d - val_e_pred).view(-1)**2).mean()

            # Log verbose per-rank diagnostics before reducing
            if self.world_size > 1:
                self.log_distributed_diagnostics(
                    epoch,
                    loss_local=loss_val_e.item(),
                    e_rmse_local=val_e_rmse_local,
                    n_trust_local=n_in_trust,
                    n_total_local=len(self.train.X)
                )

            # Reduce weighted losses across ranks for scheduler / early stopping.
            # (MAE/RMSE already aggregated via reduce_mae/reduce_rmse above)
            if self.world_size > 1:
                loss_train_e = reduce_mean(loss_train_e)
                loss_val_e   = reduce_mean(loss_val_e)

            self._log("Epoch: {}; (energy) WMSE train: {:.3f}; (energy) WMSE val: {:.3f}\n \
                                           (energy) MAE train:  {:.3f} cm-1; (gradient) MAE train:  {:.3f} cm-1/bohr\n \
                                           (energy) MAE val:    {:.3f} cm-1; (gradient) MAE val:    {:.3f} cm-1/bohr\n \
                                           (energy) RMSE train: {:.3f} cm-1; (gradient) RMSE train: {:.3f} cm-1/bohr\n \
                                           (energy) RMSE val:   {:.3f} cm-1; (gradient) RMSE val:   {:.3f} cm-1/bohr".format(
                epoch, loss_train_e, loss_val_e, train_e_mae, train_g_mae, val_e_mae, val_g_mae, train_e_rmse, train_g_rmse, val_e_rmse, val_g_rmse
            ))

            # value to be passed to EarlyStopping/ReduceLR mechanisms
            self.loss_val = loss_val_e

            if self.writer is not None:
                self.writer.add_scalar("loss/train", loss_train_e, epoch)
                self.writer.add_scalar("loss/val", loss_val_e, epoch)

            # log metrics to WANDB to visualize model performance
            if is_main_process() and USE_WANDB:
                wandb.log({
                    "loss_train_e" : loss_train_e, "loss_val_e" : loss_val_e,
                    "train_e_mae" : train_e_mae, "train_e_rmse" : train_e_rmse, "val_e_mae" : val_e_mae, "val_e_rmse" : val_e_rmse,
                    "train_g_mae" : train_g_mae, "train_g_rmse" : train_g_rmse, "val_g_mae" : val_g_mae, "val_g_rmse" : val_g_rmse,
                    "lr" : current_lr})


        elif self.cfg['TYPE'] == 'DIPOLE':
            with torch.no_grad():
                train_y_pred   = self.model(self.train.X)
                dip_pred_train = torch.einsum('ijk,ik->ij', self.train.grm, train_y_pred)
                loss_train     = self.loss_fn(self.train.y, dip_pred_train)

                val_y_pred   = self.model(self.val.X)
                dip_pred_val = torch.einsum('ijk,ik->ij', self.val.grm, val_y_pred)
                loss_val     = self.loss_fn(self.val.y, dip_pred_val)

                if self.world_size > 1:
                    # DIPOLE mode: use loss as proxy for RMSE, no trust region
                    self.log_distributed_diagnostics(
                        epoch,
                        loss_local=loss_val.item(),
                        e_rmse_local=loss_val.item(),  # Use loss as proxy
                        n_trust_local=len(self.train.X),
                        n_total_local=len(self.train.X)
                    )
                    loss_train = reduce_mean(loss_train)
                    loss_val   = reduce_mean(loss_val)

                # value to be passed to EarlyStopping/ReduceLR mechanisms
                self.loss_val = loss_val

            # log metrics to WANDB to visualize model performance
            if is_main_process() and USE_WANDB:
                wandb.log({"loss_train": loss_train, "loss_val": loss_val})

            self._log("Epoch: {0}; loss train: {2:.{1}f}; loss val: {3:.{1}f}".format(epoch, PRINT_PRECISION, loss_train, loss_val))

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

                train_qsum = torch.sum(train_q_corr, dim=1)
                train_qreg = self.cfg_loss['LAMBDA_Q'] * torch.mean(train_qsum * train_qsum)
                val_qsum   = torch.sum(val_q_corr, dim=1)
                val_qreg   = self.cfg_loss['LAMBDA_Q'] * torch.mean(val_qsum * val_qsum)

                if self.world_size > 1:
                    loss_train = reduce_mean(loss_train)
                    loss_val   = reduce_mean(loss_val)
                    train_qreg = reduce_mean(train_qreg)
                    val_qreg   = reduce_mean(val_qreg)

                # value to be passed to EarlyStopping/ReduceLR mechanisms
                self.loss_val = loss_val

            # log metrics to WANDB to visualize model performance
            if is_main_process() and USE_WANDB:
                wandb.log({"loss_train": loss_train, "loss_val": loss_val, "train_qreg": train_qreg, "val_qreg": val_qreg, "lr" : current_lr})

            self._log("Epoch: {0}; loss train: {2:.{1}f}; qreg train: {3:{1}f}; loss val: {4:.{1}f}; qreg val: {5:.{1}f}".format(
                epoch, PRINT_PRECISION, loss_train, train_qreg, loss_val, val_qreg
            ))

        elif self.cfg['TYPE'] == 'DIPOLEC':
            with torch.no_grad():
                train_dip_pred = self.model(self.train.X)
                loss_train = self.loss_fn(self.train.y, train_dip_pred)

                val_dip_pred = self.model(self.val.X)
                loss_val = self.loss_fn(self.val.y, val_dip_pred)

                if self.world_size > 1:
                    loss_train = reduce_mean(loss_train)
                    loss_val   = reduce_mean(loss_val)

                self.loss_val = loss_val

            self._log("Epoch: {0}; loss train: {2:.{1}f}; loss val: {3:.{1}f}".format(epoch, PRINT_PRECISION, loss_train, loss_val))

        elif self.cfg['TYPE'] == 'ENERGY':
            # To disable the gradient calculation, set the .requires_grad attribute of all parameters to False 
            # or wrap the forward pass into with torch.no_grad().
            with torch.no_grad():
                train_y_pred = self.model(self.train.X)
                loss_train   = self.loss_fn(self.train.y, train_y_pred)

                val_y_pred = self.model(self.val.X)
                loss_val   = self.loss_fn(self.val.y, val_y_pred)

                if self.world_size > 1:
                    loss_train = reduce_mean(loss_train)
                    loss_val   = reduce_mean(loss_val)

                # value to be passed to EarlyStopping/ReduceLR mechanisms
                self.loss_val = loss_val

            # tensorboard writer
            if self.writer is not None:
                self.writer.add_scalar("loss/train", loss_train, epoch)
                self.writer.add_scalar("loss/val", loss_val, epoch)
                self.writer.add_scalar("lr", current_lr, epoch)

            # log metrics to WANDB to visualize model performance
            if is_main_process() and USE_WANDB:
                wandb.log({"loss_train" : loss_train, "loss_val" : loss_val, "lr" : current_lr})

            self._log("Epoch: {0}; loss train: {2:.{1}f} cm-1; loss val: {3:.{1}f} cm-1; lr: {4:.2e}".format(epoch, PRINT_PRECISION, loss_train, loss_val, current_lr))

        else:
            assert False, "unreachable"


    # ---- Multi-batch L-BFGS path (vendored hjmshi/PyTorch-LBFGS) --------------

    def _build_multibatch_optimizer(self):
        mode = self.cfg_batch['MODE']
        lr = float(self.cfg_batch['LR'])
        history_size = int(self.cfg_batch['HISTORY_SIZE'])

        if mode == 'multi_batch':
            line_search = 'None'   # fixed steplength; Powell damping handles curvature.
            opt_cls = HjmshiLBFGS
        else:
            line_search = self.cfg_batch['LINE_SEARCH']
            opt_cls = HjmshiFullBatchLBFGS

        opt = opt_cls(
            self.model.parameters(),
            lr=lr,
            history_size=history_size,
            line_search=line_search,
            debug=False,
        )
        self._log(
            "Built multi-batch LBFGS: mode={} lr={} history_size={} line_search={}".format(
                mode, lr, history_size, line_search
            )
        )
        return opt

    def _init_multibatch_sampler(self):
        n = self.train.X.shape[0]
        B = int(self.cfg_batch['BATCH_SIZE'])
        seed = int(self.cfg_batch['SEED'])
        mode = self.cfg_batch['MODE']
        if mode == 'multi_batch':
            self.sampler = MultiBatchSampler(
                n_samples=n,
                batch_size=B,
                overlap_fraction=float(self.cfg_batch['OVERLAP_FRACTION']),
                seed=seed,
            )
        elif self.world_size > 1:
            # Distributed mode: each rank gets a slice of the batch
            self.sampler = DistributedFullOverlapSampler(
                n_samples=n,
                batch_size=B,
                rank=self.rank,
                world_size=self.world_size,
                seed=seed,
            )
        else:
            # Single GPU mode
            self.sampler = FullOverlapSampler(
                n_samples=n,
                batch_size=B,
                seed=seed,
            )
        if is_main_process():
            dist_info = f" (distributed: {self.world_size} ranks)" if self.world_size > 1 else ""
            logging.info(
                "Initialized sampler: mode={} N={} batch_size={} steps/epoch={}{}".format(
                    mode, n, B, self.sampler.steps_per_epoch(), dist_info
                )
            )

    def _gather_batch(self, idx):
        """Move one batch of (X, y[, dX, dy]) to DEVICE. Returns a plain dict."""
        use_grad = self.cfg_loss['USE_GRADIENTS']
        non_blocking = torch.cuda.is_available()

        X_cpu = self.train.X[idx]
        y_cpu = self.train.y[idx]
        X = X_cpu.to(self.device, non_blocking=non_blocking)
        y = y_cpu.to(self.device, non_blocking=non_blocking)

        batch = {'X': X, 'y': y}
        if use_grad:
            dX_cpu = self.train.dX[idx]
            dy_cpu = self.train.dy[idx]
            batch['dX'] = dX_cpu.to(self.device, non_blocking=non_blocking)
            batch['dy'] = dy_cpu.to(self.device, non_blocking=non_blocking)
        return batch

    def _loss_and_flat_grad(self, batch):
        """Forward + backward on one batch; returns (loss_tensor, flat_grad)."""
        self.optimizer.zero_grad()

        if self.cfg_loss['USE_GRADIENTS']:
            X = batch['X'].clone()
            X.requires_grad = True
            y_pred = self.model(X)
            dy_pred = self.compute_gradients_from_energy(X, batch['dX'], y_pred)
            loss = self.loss_fn(batch['y'], y_pred, batch['dy'], dy_pred)
        else:
            y_pred = self.model(batch['X'])
            loss = self.loss_fn(batch['y'], y_pred)

        if self.regularization is not None:
            loss = loss + self.regularization(self.model)

        loss.backward()

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        flat_grad = self.optimizer._gather_flat_grad()
        return loss, flat_grad

    def _make_closure(self, batch):
        """Factory closure for Wolfe/Armijo line search.

        Returns a callable with no arguments that recomputes the objective on
        the *same* batch each call -- hjmshi's LBFGS expects the closure to
        return a scalar tensor (no backward inside).
        """
        def closure():
            self.optimizer.zero_grad()
            if self.cfg_loss['USE_GRADIENTS']:
                X = batch['X'].clone()
                X.requires_grad = True
                y_pred = self.model(X)
                dy_pred = self.compute_gradients_from_energy(X, batch['dX'], y_pred)
                loss = self.loss_fn(batch['y'], y_pred, batch['dy'], dy_pred)
            else:
                y_pred = self.model(batch['X'])
                loss = self.loss_fn(batch['y'], y_pred)
            if self.regularization is not None:
                loss = loss + self.regularization(self.model)
            return loss
        return closure

    def train_epoch_multibatch(self, epoch, optimizer):
        self.model.train()
        mode = self.cfg_batch['MODE']
        steps = self.sampler.steps_per_epoch()

        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)

        start_time = timeit.default_timer()

        if mode == 'multi_batch':
            alpha = float(self.cfg_batch['OVERLAP_FRACTION'])
            damping = bool(self.cfg_batch['DAMPING'])
            damping_eps = float(self.cfg_batch['DAMPING_EPS'])

            Ok_prev_idx = self.sampler.current_prev_overlap()
            batch_Ok_prev = self._gather_batch(Ok_prev_idx)
            _, g_Ok_prev = self._loss_and_flat_grad(batch_Ok_prev)

            last_loss = None
            for step in range(steps):
                Ok_idx, Nk_idx = self.sampler.next_step()

                batch_Ok = self._gather_batch(Ok_idx)
                loss_Ok, g_Ok = self._loss_and_flat_grad(batch_Ok)

                batch_Nk = self._gather_batch(Nk_idx)
                _, g_Nk = self._loss_and_flat_grad(batch_Nk)

                g_Sk = alpha * (g_Ok_prev + g_Ok) + (1.0 - 2.0 * alpha) * g_Nk

                p = optimizer.two_loop_recursion(-g_Sk)
                lr_used = optimizer.step(p, g_Ok, g_Sk=g_Sk)

                # Recompute Ok gradient at the new iterate for curvature pair.
                batch_Ok_new = self._gather_batch(Ok_idx)
                _, g_Ok_new = self._loss_and_flat_grad(batch_Ok_new)
                optimizer.curvature_update(g_Ok_new, eps=damping_eps, damping=damping)

                # Shift: this step's Ok becomes next step's "Ok_prev".
                self.sampler.advance(Ok_idx)
                g_Ok_prev = g_Ok_new
                last_loss = loss_Ok.detach()

            self._log(
                "Epoch {} multi_batch: {} steps, lr_last={}, loss_Ok_last={:.6e}".format(
                    epoch, steps, lr_used, float(last_loss) if last_loss is not None else float('nan')
                )
            )

        else:  # full_overlap
            last_loss = None
            max_iter = self.cfg_solver['OPTIMIZER'].get('MAX_ITER', 100)
            debug_timing = self.cfg_debug.get('TIMING', False)

            # Timing accumulators (only used if debug_timing)
            if debug_timing:
                t_gather = t_fwd_bwd = t_sync = t_optim = 0.0
                total_inner_iters = 0

            for step in range(steps):
                if debug_timing:
                    _t0 = timeit.default_timer()

                (Sk_idx,) = self.sampler.next_step()
                batch_Sk = self._gather_batch(Sk_idx)

                if debug_timing:
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    t_gather += timeit.default_timer() - _t0

                def closure():
                    optimizer.zero_grad()
                    if self.cfg_loss['USE_GRADIENTS']:
                        X = batch_Sk['X'].clone()
                        X.requires_grad = True
                        y_pred = self.model(X)
                        dy_pred = self.compute_gradients_from_energy(X, batch_Sk['dX'], y_pred)
                        loss = self.loss_fn(batch_Sk['y'], y_pred, batch_Sk['dy'], dy_pred)
                    else:
                        y_pred = self.model(batch_Sk['X'])
                        loss = self.loss_fn(batch_Sk['y'], y_pred)
                    if self.regularization is not None:
                        loss = loss + self.regularization(self.model)
                    return loss

                # Sync function for distributed: average loss across ranks after backward
                loss_sync_fn = reduce_mean if self.world_size > 1 else None

                # Pre-compute loss & gradient at the current iterate before the inner loop
                if debug_timing:
                    _t0 = timeit.default_timer()

                optimizer.zero_grad()
                loss = closure()
                loss.backward()

                if debug_timing:
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    t_fwd_bwd += timeit.default_timer() - _t0

                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

                if debug_timing:
                    _t0 = timeit.default_timer()

                if loss_sync_fn is not None:
                    loss = loss_sync_fn(loss.detach())

                if debug_timing:
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    t_sync += timeit.default_timer() - _t0

                options = {
                    'closure': closure,
                    'current_loss': loss,
                    'grad_clip_norm': self.grad_clip_norm,
                    'loss_sync_fn': loss_sync_fn,
                }

                for inner in range(max_iter):
                    if debug_timing:
                        _t0 = timeit.default_timer()

                    obj, grad_new, t, ls_step, closure_eval, grad_eval, desc_dir, fail = optimizer.step(options=options)

                    if debug_timing:
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        t_optim += timeit.default_timer() - _t0
                        total_inner_iters += 1

                    last_loss = obj.detach() if hasattr(obj, 'detach') else torch.as_tensor(obj)

                    # Stop early if line search failed or step size is zero
                    if fail or t == 0:
                        break

                    # Recompute gradient for next inner iteration
                    if debug_timing:
                        _t0 = timeit.default_timer()

                    optimizer.zero_grad()
                    loss = closure()
                    loss.backward()

                    if debug_timing:
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        t_fwd_bwd += timeit.default_timer() - _t0

                    if self.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

                    if debug_timing:
                        _t0 = timeit.default_timer()

                    if loss_sync_fn is not None:
                        loss = loss_sync_fn(loss.detach())

                    if debug_timing:
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        t_sync += timeit.default_timer() - _t0

                    options['current_loss'] = loss

            if debug_timing:
                self._log(
                    "Epoch {} full_overlap: {} steps, {} inner_iters, loss={:.6e}".format(
                        epoch, steps, total_inner_iters, float(last_loss) if last_loss is not None else float('nan')
                    )
                )
                self._log(
                    "  Timing: gather={:.2f}s fwd_bwd={:.2f}s sync={:.2f}s optim={:.2f}s".format(
                        t_gather, t_fwd_bwd, t_sync, t_optim
                    )
                )
            else:
                self._log(
                    "Epoch {} full_overlap: {} steps, loss_Sk_last={:.6e}".format(
                        epoch, steps, float(last_loss) if last_loss is not None else float('nan')
                    )
                )

        elapsed = timeit.default_timer() - start_time
        self._log("Epoch {} multibatch step time: {:.2f}s".format(epoch, elapsed))

        self.model.eval()
        with torch.no_grad():
            if self.cfg_loss['USE_GRADIENTS']:
                # Compute predictions for both train and val
                train_y_pred, train_dy_pred = self.compute_gradients_eval(self.train)
                val_y_pred, val_dy_pred = self.compute_gradients_eval(self.val)

                # Energy metrics - use reduce_rmse/reduce_mae for correct distributed aggregation
                train_e_d    = self.loss_fn.descale_energies(self.train.y)
                train_e_pred = self.loss_fn.descale_energies(train_y_pred)
                train_e_errors = (train_e_d - train_e_pred).view(-1)
                val_e_rmse_local = torch.sqrt(torch.mean(train_e_errors ** 2)).item()
                train_e_mae  = reduce_mae(train_e_errors)
                train_e_rmse = reduce_rmse(train_e_errors)

                val_e_d    = self.loss_fn.descale_energies(self.val.y)
                val_e_pred = self.loss_fn.descale_energies(val_y_pred)
                val_e_errors = (val_e_d - val_e_pred).view(-1)
                val_e_rmse_local = torch.sqrt(torch.mean(val_e_errors ** 2)).item()
                val_e_mae  = reduce_mae(val_e_errors)
                val_e_rmse = reduce_rmse(val_e_errors)

                # Gradient metrics (per-component errors)
                natoms = self.train.NATOMS
                train_dy = self.train.dy.reshape(-1, 3 * natoms)
                val_dy   = self.val.dy.reshape(-1, 3 * natoms)
                train_g_errors = (train_dy - train_dy_pred).view(-1)
                val_g_errors   = (val_dy - val_dy_pred).view(-1)
                train_g_mae  = reduce_mae(train_g_errors)
                val_g_mae    = reduce_mae(val_g_errors)
                train_g_rmse = reduce_rmse(train_g_errors)
                val_g_rmse   = reduce_rmse(val_g_errors)

                # Weighted MSE for scheduler
                # Sync minimum across ranks for consistent weighting in distributed mode
                enmin_train = reduce_min(train_e_d.min())
                w_train = self.loss_fn.dwt / (self.loss_fn.dwt + train_e_d - enmin_train)
                loss_train_e = (w_train.view(-1) * (train_e_d - train_e_pred).view(-1)**2).mean()

                enmin_val = reduce_min(val_e_d.min())
                w_val = self.loss_fn.dwt / (self.loss_fn.dwt + val_e_d - enmin_val)
                loss_val_e = (w_val.view(-1) * (val_e_d - val_e_pred).view(-1)**2).mean()

                if self.world_size > 1:
                    # Multi-batch mode doesn't use trust region, pass 0
                    self.log_distributed_diagnostics(
                        epoch,
                        loss_local=loss_val_e.item(),
                        e_rmse_local=val_e_rmse_local,
                        n_trust_local=len(self.train.X),  # All samples (no trust region)
                        n_total_local=len(self.train.X)
                    )
                    # MAE/RMSE already aggregated above via reduce_mae/reduce_rmse
                    loss_train_e = reduce_mean(loss_train_e)
                    loss_val_e   = reduce_mean(loss_val_e)

                self._log("Epoch: {}; (energy) WMSE train: {:.3f}; (energy) WMSE val: {:.3f}\n \
                                           (energy) MAE train:  {:.3f} cm-1; (gradient) MAE train:  {:.3f} cm-1/bohr\n \
                                           (energy) MAE val:    {:.3f} cm-1; (gradient) MAE val:    {:.3f} cm-1/bohr\n \
                                           (energy) RMSE train: {:.3f} cm-1; (gradient) RMSE train: {:.3f} cm-1/bohr\n \
                                           (energy) RMSE val:   {:.3f} cm-1; (gradient) RMSE val:   {:.3f} cm-1/bohr".format(
                    epoch, loss_train_e, loss_val_e, train_e_mae, train_g_mae, val_e_mae, val_g_mae, train_e_rmse, train_g_rmse, val_e_rmse, val_g_rmse
                ))

                loss_val = loss_val_e

                if self.writer is not None:
                    self.writer.add_scalar("loss/train", loss_train_e, epoch)
            else:
                val_y_pred = self.model(self.val.X)
                loss_val = self.loss_fn(self.val.y, val_y_pred)
                if self.world_size > 1:
                    loss_val = reduce_mean(loss_val)
                self._log("Epoch: {}; loss val: {:.3f} cm-1".format(epoch, loss_val))

        self.loss_val = loss_val
        current_lr = optimizer.param_groups[0]['lr']
        if self.writer is not None:
            self.writer.add_scalar("loss/val", loss_val, epoch)
            self.writer.add_scalar("lr", current_lr, epoch)


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

    if USE_WANDB:
        project_name = cfg_dataset['NAME'] + "-" + cfg['TYPE']
        wandb.init(project=project_name)

    rank, world_size, local_rank = setup_distributed()

    t = Training(MODEL_FOLDER, MODEL_NAME, chk_path, cfg, train, val, test,
                 rank=rank, world_size=world_size, local_rank=local_rank)

    try:
        t.train_model()
        t.model_eval()
    finally:
        cleanup()
