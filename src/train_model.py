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


def fit_scalers_to_train_dataset(train, cfg):
    if cfg['NORMALIZE'] == 'std':
        xscaler = StandardScaler()
        yscaler = StandardScaler()
    elif cfg['NORMALIZE'] == 'std-none':
        xscaler = StandardScaler()
        yscaler = IdentityScaler()
    else:
        raise ValueError("unreachable")

    xscaler.fit(train.X)
    yscaler.fit(train.y)

    return xscaler, yscaler


def load_from_checkpoint(chk_path):
    state = torch.load(chk_path, map_location=torch.device(DEVICE))
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

        ymin = yd.min()

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

        en_min = en.min()
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
        return wmse_en + wmse_gradients

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

        enmin   = _en.min()
        w       = self.dwt / (self.dwt + _en - enmin)
        w       = w.view(-1)
        wmse_en = (w * (_en - _en_pred)**2).mean()

        nconfigs = gradients.size()[0]

        gradients_pred = gradients_pred.reshape(nconfigs, self.natoms, 3)
        gradients    = gradients.reshape(nconfigs, self.natoms, 3)

        df = gradients - gradients_pred
        wdf = torch.einsum('ijk,i->ijk', df, w)
        wmse_gradients = self.g_lambda * torch.einsum('ijk,ijk->', wdf, df) / (3.0 * self.natoms) / nconfigs

        return wmse_en, wmse_gradients


class WMSELoss_TrustRegion_wgradients(torch.nn.Module):
    """
    Memory-efficient trust region loss for gradient training with optional focal
    weighting and an optional soft boundary.

    Gradients are pre-filtered before being passed to this loss (computed only
    for configs in the trust / active set). This avoids OOM by never computing
    gradients for configs outside the active set.

    Two modes (selected by `soft_boundary`):

      Hard mask (default; backward-compatible):
        Membership in the trust set is binary (energy error < trust_threshold).
        Gradient loss is normalized by n_in_trust.

      Soft boundary (soft_boundary=True):
        Each config carries a per-config weight
            phi(e_i) = sigmoid((trust_threshold - e_i) / soft_scale) in [0, 1]
        smoothly decaying with energy error. Gradient loss is normalized by the
        SUM of weights, so downweighting actually reduces a config's
        contribution rather than redistributing it.

        The "active set" passed in is the subset of configs whose phi exceeds
        soft_cutoff (a memory optimization only -- configs outside the
        active set contribute negligibly).

        Soft mode removes the discontinuous "evasion" incentive of the hard
        mask, where pushing a config across `tau` discretely zeroed its
        gradient loss.

    Focal weighting (when focal_gamma > 0) up-weights hard-to-fit configs.

    Expected inputs:
    - en, en_pred:        full energy tensors (all configs)
    - gradients_subset:      reference gradients for active-set configs
    - gradients_pred_subset: predicted gradients for active-set configs
    - trust_indices:      indices of active-set configs
    - gradient_weights:      optional per-config soft phi values for the active
                          set; when None we are in hard-mask mode.
    """
    def __init__(self, natoms, dwt=1.0, g_lambda=1.0, trust_threshold=100.0,
                 soft_boundary=False, soft_scale=None,
                 focal_gamma=0.0, focal_ema_decay=0.95,
                 gradient_clamp_quantile=None):
        super().__init__()
        self.natoms = natoms
        self.dwt = torch.tensor(dwt).to(DEVICE)
        self.g_lambda = torch.tensor(g_lambda).to(DEVICE)
        self.trust_threshold = trust_threshold
        # Soft boundary parameters (False = hard mask, backward-compatible)
        self.soft_boundary = soft_boundary
        self.soft_scale = soft_scale  # cm^-1; defaults to trust_threshold/4
        self.focal_gamma = focal_gamma
        self.focal_ema_decay = focal_ema_decay
        # Per-config gradient-loss clamping: clamp at this quantile (e.g. 0.95).
        # None = no clamping (backward-compatible).
        self.gradient_clamp_quantile = gradient_clamp_quantile

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
                "trust_threshold={}, soft_boundary={}, soft_scale={}, "
                "focal_gamma={}, focal_ema_decay={}, "
                "gradient_clamp_quantile={})").format(
            self.natoms, self.dwt, self.g_lambda, self.trust_threshold,
            self.soft_boundary, self.soft_scale,
            self.focal_gamma, self.focal_ema_decay,
            self.gradient_clamp_quantile)

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
        enmin = _en.min()
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
                trust_indices, gradient_weights=None):
        """
        Compute combined energy + gradient loss.
        Energy loss is computed on ALL configs, gradient loss only on the
        active (trust) subset.

        gradient_weights: optional 1-D tensor of soft phi values for the
        active subset (same length as trust_indices). If provided we are
        in soft-boundary mode and the gradient loss is normalized by the
        SUM of these weights; if None we are in hard-mask mode (count
        normalization, weights effectively 1.0).
        """
        wmse_en, wmse_gradients = self.forward_separate(
            en, en_pred, gradients_subset, gradients_pred_subset,
            trust_indices, gradient_weights
        )
        return wmse_en + wmse_gradients

    def forward_separate(self, en, en_pred, gradients_subset, gradients_pred_subset,
                         trust_indices, gradient_weights=None):
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
            #   hard mask: w_trusted = w_energy * w_focal
            #   soft mask: w_trusted = w_energy * w_focal * phi(e_i)
            w_trusted = w[trust_indices]
            if gradient_weights is not None:
                w_trusted = w_trusted * gradient_weights

            # Reshape gradients
            gradients_subset = gradients_subset.reshape(n_in_trust, self.natoms, 3)
            gradients_pred_subset = gradients_pred_subset.reshape(n_in_trust, self.natoms, 3)

            # Per-config weighted gradient squared error
            df = gradients_subset - gradients_pred_subset
            per_config_sq = torch.sum(df ** 2, dim=(1, 2))  # (n_in_trust,)
            contrib = w_trusted * per_config_sq               # (n_in_trust,)

            # Clamp outlier per-config contributions at a quantile threshold.
            # Gradient is zeroed for clamped configs, preventing them from
            # dominating the L-BFGS search direction.
            if self.gradient_clamp_quantile is not None:
                with torch.no_grad():
                    threshold = torch.quantile(
                        contrib, self.gradient_clamp_quantile
                    )
                contrib = torch.clamp(contrib, max=threshold)

            sq = contrib.sum()

            if gradient_weights is not None:
                # Soft mode: normalize by sum of soft weights so that
                # downweighting a config genuinely shrinks its contribution.
                denom = (gradient_weights.sum().clamp(min=1.0)
                         * 3.0 * self.natoms)
            else:
                # Hard mode: original count-based normalization.
                denom = 3.0 * self.natoms * n_in_trust

            wmse_gradients = self.g_lambda * sq / denom
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

        ymin = yd.min()

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
            logging.info("(Early Stopping) Best validation RMSE: {1:.{0}f}; current validation RMSE: {2:.{0}f}".format(PRINT_PRECISION, self.best_score, score))
            logging.info("(Early Stopping) counter: {}; patience: {}; tolerance: {}".format(self.counter, self.patience, self.tol))


def count_params(model):
    nparams = 0
    for name, param in model.named_parameters():
        params = torch.tensor(param.size())
        nparams += torch.prod(params, 0)

    return nparams



class Training:
    def __init__(self, model_folder, model_name, ckh_path, cfg, train, val, test):
        EVENTDIR = "runs"
        if not os.path.isdir(EVENTDIR):
            os.makedirs(EVENTDIR)

        log_dir = os.path.join("runs", model_name)
        self.writer = SummaryWriter(log_dir=log_dir)

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

            logging.info("Fitting scalers to training dataset...\n")
            self.xscaler, self.yscaler = fit_scalers_to_train_dataset(train, cfg['DATASET'])
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

    def build_optimizer(self, cfg_optimizer):
        if cfg_optimizer['NAME'] == 'LBFGS':
            lr               = cfg_optimizer.get('LR', 1.0)
            tolerance_grad   = cfg_optimizer.get('TOLERANCE_GRAD', 1e-14)
            tolerance_change = cfg_optimizer.get('TOLERANCE_CHANGE', 1e-14)
            max_iter         = cfg_optimizer.get('MAX_ITER', 100)

            optimizer        = torch.optim.LBFGS(self.model.parameters(), lr=lr, line_search_fn='strong_wolfe', tolerance_grad=tolerance_grad,
                                                 tolerance_change=tolerance_change, max_iter=max_iter)
        elif cfg_optimizer['NAME'] == 'Adam':
            lr           = cfg_optimizer.get('LR', 1e-3)
            optimizer    = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError("unreachable")

        logging.info("Build optimizer: {}".format(optimizer))

        return optimizer

    def build_loss(self):
        known_options = ('NAME', 'WEIGHT_TYPE', 'DWT', 'EREF', 'EMAX', 'USE_GRADIENTS', 'USE_GRADIENTS_AFTER_EPOCH', 'G_LAMBDA', 'G_LAMBDA_RAMP_EPOCHS', 'LAMBDA_Q', 'TRUST_THRESHOLD', 'TRUST_THRESHOLD_START', 'TRUST_THRESHOLD_RAMP_EPOCHS', 'TRUST_SOFT_BOUNDARY', 'TRUST_SOFT_SCALE', 'TRUST_SOFT_CUTOFF', 'FOCAL_GAMMA', 'FOCAL_EMA_DECAY', 'GRADIENT_CLAMP_QUANTILE')
        for option in self.cfg_loss.keys():
            assert option.upper() in known_options, "[build_loss] unknown option: {}".format(option)

        # have all defaults in the same place and set them to configuration if the value is omitted in the YAML file
        self.cfg_loss.setdefault('LAMBDA_Q', 1.0e3)
        self.cfg_loss.setdefault('USE_GRADIENTS_AFTER_EPOCH', None)
        self.cfg_loss.setdefault('USE_GRADIENTS', False)
        self.cfg_loss.setdefault('G_LAMBDA_RAMP_EPOCHS', 0)
        self.cfg_loss.setdefault('TRUST_THRESHOLD_RAMP_EPOCHS', 0)
        self.cfg_loss.setdefault('TRUST_SOFT_BOUNDARY', False)
        self.cfg_loss.setdefault('TRUST_SOFT_SCALE', None)
        self.cfg_loss.setdefault('TRUST_SOFT_CUTOFF', 0.01)
        self.cfg_loss.setdefault('FOCAL_GAMMA', 0.0)
        self.cfg_loss.setdefault('FOCAL_EMA_DECAY', 0.95)
        self.cfg_loss.setdefault('GRADIENT_CLAMP_QUANTILE', None)

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
                # Use memory-efficient trust region loss that expects pre-filtered gradients
                soft_boundary = self.cfg_loss.get('TRUST_SOFT_BOUNDARY', False)
                soft_scale = self.cfg_loss.get('TRUST_SOFT_SCALE', None)
                gradient_clamp_quantile = self.cfg_loss.get('GRADIENT_CLAMP_QUANTILE', None)
                loss_fn = WMSELoss_TrustRegion_wgradients(natoms=self.train.NATOMS, dwt=dwt, g_lambda=g_lambda,
                                                       trust_threshold=trust_threshold,
                                                       soft_boundary=soft_boundary,
                                                       soft_scale=soft_scale,
                                                       focal_gamma=focal_gamma,
                                                       focal_ema_decay=focal_ema_decay,
                                                       gradient_clamp_quantile=gradient_clamp_quantile)
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
        self.model = self.model.to(DEVICE)

        self.train.X = self.train.X.to(DEVICE)
        self.train.y = self.train.y.to(DEVICE)
        self.val.X = self.val.X.to(DEVICE)
        self.val.y = self.val.y.to(DEVICE)

        self.loss_fn = self.loss_fn.to(DEVICE)

        if self.cfg['TYPE'] == 'DIPOLE':
            self.train.grm = self.train.grm.to(DEVICE)
            self.val.grm   = self.val.grm.to(DEVICE)

        if self.cfg['TYPE'] == 'DIPOLEQ':
            self.train.xyz_ordered = self.train.xyz_ordered.to(DEVICE)
            self.val.xyz_ordered = self.val.xyz_ordered.to(DEVICE)
            self.test.xyz_ordered = self.test.xyz_ordered.to(DEVICE)

        if self.train.dX is not None:
            self.train.dX = self.train.dX.to(DEVICE)
            self.train.dy = self.train.dy.to(DEVICE)

            self.val.dX = self.val.dX.to(DEVICE)
            self.val.dy = self.val.dy.to(DEVICE)


        self.optimizer = self.build_optimizer(self.cfg_solver['OPTIMIZER'])
        self.scheduler = self.build_scheduler()

        start = time.time()

        MAX_EPOCHS = self.cfg_solver['MAX_EPOCHS']

        for epoch in range(MAX_EPOCHS):
            # switch into mixed loss function: E + F
            if self.cfg_loss['USE_GRADIENTS_AFTER_EPOCH'] is not None and epoch == self.cfg_loss['USE_GRADIENTS_AFTER_EPOCH']:
                self.cfg_loss['USE_GRADIENTS'] = True
                self.loss_fn = self.build_loss().to(DEVICE)
                self.loss_fn.set_scale(self.yscaler.mean_, self.yscaler.scale_)
                self.gradient_start_epoch = epoch

                self.es.reset()

                # Reset L-BFGS curvature history. The stored (s_k, y_k) pairs
                # describe the energy-only loss surface and produce degenerate
                # search directions on the new energy+gradient surface, causing
                # the Wolfe line search to return t=0 indefinitely.
                if isinstance(self.optimizer, torch.optim.LBFGS):
                    self.optimizer.state.clear()
                    self._lbfgs_prev_n_iter = 0
                    self._lbfgs_prev_func_evals = 0
                    logging.info("Reset L-BFGS state at gradient inclusion (epoch {})".format(epoch))

                # Reset LR to initial value so the optimizer has full step
                # budget to explore the new loss landscape.
                initial_lr = self.cfg_solver['OPTIMIZER'].get('LR', 0.1)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = initial_lr
                self.scheduler = self.build_scheduler()
                logging.info("Reset LR to {} and rebuilt scheduler at gradient inclusion".format(initial_lr))

            # Progressive G_LAMBDA ramp
            if self.cfg_loss['USE_GRADIENTS'] and self.cfg_loss.get('G_LAMBDA_RAMP_EPOCHS', 0) > 0:
                ramp_epochs = self.cfg_loss['G_LAMBDA_RAMP_EPOCHS']
                start_epoch = self.gradient_start_epoch if self.gradient_start_epoch is not None else 0
                progress = (epoch - start_epoch) / ramp_epochs
                progress = max(0.0, min(1.0, progress))
                target_g_lambda = self.cfg_loss.get('G_LAMBDA', 1.0)
                current_g_lambda = target_g_lambda * progress
                self.loss_fn.g_lambda = torch.tensor(current_g_lambda).to(DEVICE)
                if epoch % PRINT_TRAINING_STEPS == 0 or epoch == start_epoch or epoch == start_epoch + ramp_epochs:
                    logging.info("G_LAMBDA ramp: epoch {}, progress {:.1%}, g_lambda = {:.4f}".format(epoch, progress, current_g_lambda))

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
                    logging.info("Trust-threshold anneal: epoch {}, progress {:.1%}, threshold = {:.1f}".format(epoch, progress, self.current_trust_threshold))
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
                    and isinstance(self.optimizer, torch.optim.LBFGS)
                    and epoch > self.cfg_loss.get('USE_GRADIENTS_AFTER_EPOCH', 0)
                    and (epoch - self.cfg_loss.get('USE_GRADIENTS_AFTER_EPOCH', 0))
                        % lbfgs_reset_interval == 0):
                self.optimizer.state.clear()
                self._lbfgs_prev_n_iter = 0
                self._lbfgs_prev_func_evals = 0
                logging.info("Periodic L-BFGS state reset (epoch {})".format(epoch))

            print("loss function: {}".format(self.loss_fn))

            self.train_epoch(epoch, self.optimizer)

            # Step scheduler - ReduceLROnPlateau requires metric, CosineAnnealing does not
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(self.loss_val)
            else:
                self.scheduler.step()

            if epoch % PRINT_TRAINING_STEPS == 0:
                end = time.time()
                logging.info("Elapsed time: {:.0f}s\n".format(end - start))

            # writing all pending events to disk
            self.writer.flush()

            # pass loss values to EarlyStopping mechanism 
            self.es(epoch, self.loss_val, self.model, self.xscaler, self.yscaler, meta_info=self.meta_info)

            if self.es.status:
                logging.info("Invoking early stop.")
                break

        if self.loss_val < self.es.best_score:
            save_checkpoint(self.model, self.xscaler, self.yscaler, self.meta_info, self.chk_path)

        logging.info("\nReloading best model from the last checkpoint")

        self.reset_weights()
        checkpoint = torch.load(self.chk_path, map_location=torch.device(DEVICE))
        self.model.load_state_dict(checkpoint["model"])

        return self.model

    def compute_gradients(self, dataset):
        Xtr = dataset.X

        Xtr.requires_grad = True

        y_pred = self.model(Xtr)
        dEdp   = torch.autograd.grad(outputs=y_pred, inputs=Xtr, grad_outputs=torch.ones_like(y_pred), retain_graph=True, create_graph=True)[0]

        Xtr.requires_grad = False

        # take into account normalization of polynomials
        # now we have derivatives of energy w.r.t. to polynomials
        x_scale = torch.from_numpy(self.xscaler.scale_).to(DEVICE)
        dEdp = torch.div(dEdp, x_scale)

        # gradient = dE/dx = \sigma(E) * dE/d(poly) * d(poly)/dx
        # `torch.einsum` throws a Runtime error without an explicit conversion to Double
        dEdx = torch.einsum('ij,ijk -> ik', dEdp.double(), dataset.dX.double())

        # take into account normalization of model energy
        y_scale = torch.from_numpy(self.yscaler.scale_).to(DEVICE)
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
        x_scale = torch.from_numpy(self.xscaler.scale_).to(DEVICE)
        dEdp = torch.div(dEdp, x_scale)

        # gradient = dE/dx = \sigma(E) * dE/d(poly) * d(poly)/dx
        dEdx = torch.einsum('ij,ijk -> ik', dEdp.double(), dX_subset.double())

        # take into account normalization of model energy
        y_scale = torch.from_numpy(self.yscaler.scale_).to(DEVICE)
        dEdx = torch.mul(dEdx, y_scale)

        return dEdx

    def compute_trust_mask(self, dataset):
        """
        Compute trust region active set based on energy prediction errors.

        In hard-mask mode (default): active set = {i : e_i < trust_threshold}.
        In soft-boundary mode (TRUST_SOFT_BOUNDARY=True): active set =
            {i : phi(e_i) > soft_cutoff}, where phi is the sigmoid soft
            trust factor; the per-config phi values are returned as well.

        Returns:
          trust_indices : 1-D LongTensor of active-set config indices
          trust_mask    : 1-D BoolTensor of shape (N,) indicating membership
          energy_errors : 1-D float tensor of |E_pred - E_true| (cm^-1)
          gradient_weights : 1-D float tensor of phi(e_i) for the active set
                          in soft-boundary mode, or None in hard-mask mode
        """
        trust_threshold = getattr(self, 'current_trust_threshold', None)
        if trust_threshold is None:
            trust_threshold = self.cfg_loss.get('TRUST_THRESHOLD', 50.0)

        soft_boundary = self.cfg_loss.get('TRUST_SOFT_BOUNDARY', False)
        soft_scale = self.cfg_loss.get('TRUST_SOFT_SCALE', None)
        soft_cutoff = self.cfg_loss.get('TRUST_SOFT_CUTOFF', 0.01)

        with torch.no_grad():
            y_pred = self.model(dataset.X)

            # Descale energies
            en_mean = torch.from_numpy(self.yscaler.mean_).to(DEVICE)
            en_std = torch.from_numpy(self.yscaler.scale_).to(DEVICE)

            en_pred_descaled = y_pred * en_std + en_mean
            en_true_descaled = dataset.y * en_std + en_mean

            energy_errors = torch.abs(en_pred_descaled - en_true_descaled).view(-1)

            if soft_boundary:
                phi = WMSELoss_TrustRegion_wgradients.soft_phi(
                    energy_errors, trust_threshold, soft_scale=soft_scale
                )
                trust_mask = phi > soft_cutoff
                trust_indices = torch.nonzero(trust_mask, as_tuple=False).view(-1)
                gradient_weights = phi[trust_indices]
            else:
                trust_mask = energy_errors < trust_threshold
                trust_indices = torch.nonzero(trust_mask, as_tuple=False).view(-1)
                gradient_weights = None

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
        N = trust_mask.numel()
        cur = trust_mask.detach()
        n_in = int(cur.sum().item())
        frac = n_in / max(N, 1)

        # Initialize trackers lazily on the first call.
        if self._trust_flip_count is None:
            self._trust_flip_count = torch.zeros(N, dtype=torch.long, device=DEVICE)

        if self._prev_trust_mask is None:
            entered = n_in
            left = 0
            stable_in = n_in
            stable_out = N - n_in
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
            entered = int(entered_mask.sum().item())
            left = int(left_mask.sum().item())
            stable_in = int(stable_in_mask.sum().item())
            stable_out = int(stable_out_mask.sum().item())

            # Update cumulative flip count.
            flips = entered_mask | left_mask
            self._trust_flip_count[flips] += 1

            # Eviction signal: compare prev-epoch gradient errors of left vs stayed.
            if (self._prev_train_gradient_errors is not None
                    and self._prev_train_gradient_errors.numel() == N):
                pfe = self._prev_train_gradient_errors
                if left > 0:
                    mean_err_left = float(pfe[left_mask].mean().item())
                    med_err_left  = float(pfe[left_mask].median().item())
                else:
                    mean_err_left = float('nan')
                    med_err_left  = float('nan')
                if stable_in > 0:
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

        # Soft-mode phi statistics (None in hard mode).
        if gradient_weights is not None and gradient_weights.numel() > 0:
            phi_sum = float(gradient_weights.sum().item())
            phi_mean = float(gradient_weights.mean().item())
            phi_min = float(gradient_weights.min().item())
        else:
            phi_sum = float('nan')
            phi_mean = float('nan')
            phi_min = float('nan')

        max_flips = int(self._trust_flip_count.max().item())
        ever_in = int((self._trust_flip_count > 0).sum().item()) + stable_in
        # Configs that have never flipped AND are currently out: never-trusted.
        # (Approximate -- exact count requires another tracker; we don't bother.)

        logging.info(
            "[trust-diag] epoch={} | n_in={}/{} ({:.1%}) | entered={} left={} "
            "stable_in={} | prev-epoch gradient-RMSE: left={:.2f} stayed={:.2f} "
            "(med {:.2f}/{:.2f}) | max_flips={}".format(
                epoch, n_in, N, frac, entered, left, stable_in,
                mean_err_left, mean_err_stayed,
                med_err_left, med_err_stayed, max_flips
            )
        )

        # Append CSV row for post-hoc plotting.
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

        In hard-mask mode phi is treated as 1; phi columns are NaN.
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

            if gradient_weights is not None:
                phi_act = gradient_weights.view(-1).to(f_sq.dtype)
            else:
                phi_act = torch.ones(n_active, device=DEVICE, dtype=f_sq.dtype)

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

            if gradient_weights is not None:
                pq = torch.quantile(phi_cpu, qs[:5].to(phi_cpu.dtype)).tolist()
                # Bin phi into membership categories.
                bins = torch.tensor([0.0, 0.25, 0.50, 0.75, 0.90, 1.0001])
                # counts per bin
                idx = torch.bucketize(phi_cpu, bins) - 1
                idx = idx.clamp(0, 4)
                bin_counts = [int((idx == b).sum().item()) for b in range(5)]
            else:
                pq = [float('nan')] * 5
                bin_counts = [0] * 5  # all entries are phi=1, hard mode N/A

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

        logging.info(
            "[grad-diag] epoch={} | top1%={:.1%} top5%={:.1%} top10%={:.1%} "
            "of gradient loss | contrib q50={:.3e} q95={:.3e} max={:.3e}".format(
                epoch, top1, top5, top10, cq[2], cq[5], contrib_max
            )
        )

    def log_lbfgs_diagnostics(self, epoch, optimizer):
        """L-BFGS line-search telemetry, dumped per epoch.

        Pulls inner state from torch.optim.LBFGS:
          - this-step iteration / closure-call counts (deltas from cumulative)
          - last accepted step length t
          - initial Hessian diag scaling H_diag = (s . y) / (y . y)
          - curvature pair stats: <s_k, y_k> = 1 / ro_k -- min/max/last/mean
            over the stored history (small or absent => degenerate curvature)
          - flat gradient norm at the last accepted iterate
        """
        if not isinstance(optimizer, torch.optim.LBFGS):
            return

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

        logging.info(
            "[lbfgs-diag] epoch={} | iters={} evals={} t={:.3e} H_diag={:.3e} "
            "pairs={} grad_norm={:.3e} sy(last/min/max)={:.3e}/{:.3e}/{:.3e}".format(
                epoch, iters_this_step, evals_this_step, t_val, H_diag,
                n_pairs, grad_norm, sy_last, sy_min, sy_max,
            )
        )

    def compute_gradients_eval(self, dataset):
        """
        Compute gradients for evaluation (no create_graph needed).
        Much more memory efficient than compute_gradients() since we don't need
        to backpropagate through the gradient computation.
        """
        Xtr = dataset.X.clone().detach()
        Xtr.requires_grad = True

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
        x_scale = torch.from_numpy(self.xscaler.scale_).to(DEVICE)
        dEdp = torch.div(dEdp, x_scale)

        # gradient = dE/dx
        dEdx = torch.einsum('ij,ijk -> ik', dEdp.double(), dataset.dX.double())

        # take into account normalization of model energy
        y_scale = torch.from_numpy(self.yscaler.scale_).to(DEVICE)
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

                soft_boundary = self.cfg_loss.get('TRUST_SOFT_BOUNDARY', False)
                if soft_boundary and gradient_weights is not None and n_in_trust > 0:
                    logging.info(
                        "Trust region (soft): {}/{} configs ({:.1f}%) | "
                        "energy err: min={:.1f}, max={:.1f}, med={:.1f} | "
                        "phi: min={:.3f}, mean={:.3f}, sum={:.1f}".format(
                            n_in_trust, len(self.train.X),
                            100.0 * n_in_trust / len(self.train.X),
                            energy_errors.min().item(), energy_errors.max().item(),
                            energy_errors.median().item(),
                            gradient_weights.min().item(), gradient_weights.mean().item(),
                            gradient_weights.sum().item()))
                else:
                    logging.info(
                        "Trust region: {}/{} configs ({:.1f}%) | "
                        "energy err: min={:.1f}, max={:.1f}, med={:.1f}".format(
                            n_in_trust, len(self.train.X),
                            100.0 * n_in_trust / len(self.train.X),
                            energy_errors.min().item(), energy_errors.max().item(),
                            energy_errors.median().item()))

                # Run trust-region diagnostics (churn + eviction signal).
                self.log_trust_region_diagnostics(
                    epoch, trust_mask, energy_errors, gradient_weights
                )

        def closure():
            nonlocal CLOSURE_CALL_COUNT
            CLOSURE_CALL_COUNT = CLOSURE_CALL_COUNT + 1

            optimizer.zero_grad()

            if self.cfg_loss['USE_GRADIENTS']:
                if use_trust_region:
                    if n_in_trust > 0:
                        y_pred_subset = self.model(X_subset)
                        train_dy_pred_subset = self.compute_gradients_from_energy(
                            X_subset, dX_subset, y_pred_subset
                        )
                        train_y_pred = self.model(self.train.X)
                        loss = self.loss_fn(
                            self.train.y, train_y_pred,
                            train_dy_subset, train_dy_pred_subset,
                            trust_indices, gradient_weights
                        )
                    else:
                        # No configs in trust region yet - energy only
                        train_y_pred = self.model(self.train.X)
                        loss = self.loss_fn.forward_energy_only(self.train.y, train_y_pred)
                else:
                    # Original approach: compute gradients for ALL configs
                    train_y_pred, train_dy_pred = self.compute_gradients(self.train)
                    loss = self.loss_fn(self.train.y, train_y_pred, self.train.dy, train_dy_pred)

            elif self.cfg['TYPE'] == 'DIPOLE':
                # y_pred:    [(d, a1), (d, a2), (d, a3)] -- scalar products with anchor vectors 
                # dip_pred:  g @ y_pred                  -- Cartesian components of the predicted dipole

                y_pred = self.model(self.train.X)
                dip_pred = torch.einsum('ijk,ik->ij', self.train.grm, y_pred)

                loss = self.loss_fn(self.train.y, dip_pred)

            elif self.cfg['TYPE'] == 'DIPOLEQ':
                # y_pred: [q1, ... q7]      -- partial charges on atoms
                # dip_pred: sum(q_i * r_i)  -- Cartesian components of the predicted dipole [need to descale in the loss function]
                # additional term to `reqularize` the sum of partial charges

                q_pred   = self.model(self.train.X)
                X_inf    = torch.zeros_like(self.train.X).cpu()         # polynomials at infinite separation
                X_inf_tr = torch.from_numpy(xscaler.transform(X_inf)).to(DEVICE)
                q_inf    = self.model(X_inf_tr)                         # partial charges at infinite separation
                q_corr   = q_pred - q_inf                               # corrected partial charges
                dip_pred = torch.einsum('ijk,ij->ik', self.train.xyz_ordered.double(), q_corr)

                # charge regularization
                # NOTE: use `mean`
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

            loss.backward()

            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            return loss

        # Calling model.train() will change the behavior of some layers such as nn.Dropout and nn.BatchNormXd
        self.model.train()

        # Reset focal weighting flag to allow one error_scale update per optimizer step
        # (prevents non-deterministic loss during LBFGS line search)
        if hasattr(self.loss_fn, 'reset_error_scale_flag'):
            self.loss_fn.reset_error_scale_flag()

        start_time = timeit.default_timer()
        optimizer.step(closure)
        elapsed = timeit.default_timer() - start_time
        logging.info("Optimizer makes step in {:.2f}s".format(elapsed))
        logging.info("CLOSURE_CALL_COUNT = {}".format(CLOSURE_CALL_COUNT))

        current_lr = optimizer.param_groups[0]['lr']
        logging.info("(optimizer) current lr: {}".format(current_lr))

        # LBFGS line-search telemetry (no-op for non-LBFGS optimizers).
        if self.cfg_loss['USE_GRADIENTS']:
            self.log_lbfgs_diagnostics(epoch, optimizer)

        # Calling model.eval() will change the behavior of some layers, 
        # such as nn.Dropout, which will be disabled, and nn.BatchNormXd, which will use the running stats during evaluation.
        self.model.eval()

        if self.cfg_loss['USE_GRADIENTS']:
            # Use memory-efficient gradient evaluation (no create_graph)
            train_y_pred, train_dy_pred = self.compute_gradients_eval(self.train)
            val_y_pred, val_dy_pred = self.compute_gradients_eval(self.val)

            # Compute energy metrics directly (works with any loss function)
            train_e_d    = self.loss_fn.descale_energies(self.train.y)
            train_e_pred = self.loss_fn.descale_energies(train_y_pred)
            train_e_mae  = torch.mean(torch.abs(train_e_d - train_e_pred))
            train_e_rmse = torch.sqrt(torch.mean((train_e_d - train_e_pred)*(train_e_d - train_e_pred)))

            val_e_d    = self.loss_fn.descale_energies(self.val.y)
            val_e_pred = self.loss_fn.descale_energies(val_y_pred)
            val_e_mae  = torch.mean(torch.abs(val_e_d - val_e_pred))
            val_e_rmse = torch.sqrt(torch.mean((val_e_d - val_e_pred) * (val_e_d - val_e_pred)))

            # Compute gradient metrics directly
            natoms   = self.train.NATOMS
            train_dy = self.train.dy.reshape(-1, 3 * natoms)
            val_dy   = self.val.dy.reshape(-1, 3 * natoms)
            train_g_mae  = torch.mean(torch.sum(torch.abs(train_dy - train_dy_pred), dim=1) / (3 * natoms))
            val_g_mae    = torch.mean(torch.sum(torch.abs(val_dy - val_dy_pred), dim=1) / (3 * natoms))
            train_g_rmse = torch.sqrt(torch.mean(torch.sum((train_dy - train_dy_pred) * (train_dy - train_dy_pred), dim=1) / (3 * natoms)))
            val_g_rmse   = torch.sqrt(torch.mean(torch.sum((val_dy - val_dy_pred) * (val_dy - val_dy_pred), dim=1) / (3 * natoms)))

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
            enmin_train = train_e_d.min()
            w_train = self.loss_fn.dwt / (self.loss_fn.dwt + train_e_d - enmin_train)
            loss_train_e = (w_train.view(-1) * (train_e_d - train_e_pred).view(-1)**2).mean()

            enmin_val = val_e_d.min()
            w_val = self.loss_fn.dwt / (self.loss_fn.dwt + val_e_d - enmin_val)
            loss_val_e = (w_val.view(-1) * (val_e_d - val_e_pred).view(-1)**2).mean()

            logging.info("Epoch: {}; (energy) WMSE train: {:.3f}; (energy) WMSE val: {:.3f}\n \
                                           (energy) MAE train:  {:.3f} cm-1; (gradient) MAE train:  {:.3f} cm-1/bohr\n \
                                           (energy) MAE val:    {:.3f} cm-1; (gradient) MAE val:    {:.3f} cm-1/bohr\n \
                                           (energy) RMSE train: {:.3f} cm-1; (gradient) RMSE train: {:.3f} cm-1/bohr\n \
                                           (energy) RMSE val:   {:.3f} cm-1; (gradient) RMSE val:   {:.3f} cm-1/bohr".format(
                epoch, loss_train_e, loss_val_e, train_e_mae, train_g_mae, val_e_mae, val_g_mae, train_e_rmse, train_g_rmse, val_e_rmse, val_g_rmse
            ))

            # value to be passed to EarlyStopping/ReduceLR mechanisms
            self.loss_val = loss_val_e

            self.writer.add_scalar("loss/train", loss_train_e, epoch)
            self.writer.add_scalar("loss/val", loss_val_e, epoch)

            # log metrics to WANDB to visualize model performance
            if USE_WANDB:
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

                # value to be passed to EarlyStopping/ReduceLR mechanisms
                self.loss_val = loss_val

            # log metrics to WANDB to visualize model performance
            if USE_WANDB:
                wandb.log({"loss_train": loss_train, "loss_val": loss_val})

            logging.info("Epoch: {0}; loss train: {2:.{1}f}; loss val: {3:.{1}f}".format(epoch, PRINT_PRECISION, loss_train, loss_val))

        elif self.cfg['TYPE'] == 'DIPOLEQ':
            # To disable the gradient calculation, set the .requires_grad attribute of all parameters to False 
            # or wrap the forward pass into with torch.no_grad().
            with torch.no_grad():
                train_q_pred   = self.model(self.train.X)
                train_X_inf    = torch.zeros_like(self.train.X).cpu()
                train_X_inf_tr = torch.from_numpy(xscaler.transform(train_X_inf)).to(DEVICE)
                train_q_inf    = self.model(train_X_inf_tr)
                train_q_corr   = train_q_pred - train_q_inf
                dip_pred_train = torch.einsum('ijk,ij->ik', self.train.xyz_ordered.double(), train_q_corr)
                loss_train     = self.loss_fn(self.train.y, dip_pred_train)

                val_q_pred   = self.model(self.val.X)
                val_X_inf    = torch.zeros_like(self.val.X).cpu()
                val_X_inf_tr = torch.from_numpy(xscaler.transform(val_X_inf)).to(DEVICE)
                val_q_inf    = self.model(val_X_inf_tr)
                val_q_corr   = val_q_pred - val_q_inf
                dip_pred_val = torch.einsum('ijk,ij->ik', self.val.xyz_ordered.double(), val_q_corr)
                loss_val     = self.loss_fn(self.val.y, dip_pred_val)

                # value to be passed to EarlyStopping/ReduceLR mechanisms
                self.loss_val = loss_val

                train_qsum = torch.sum(train_q_corr, dim=1)
                train_qreg = self.cfg_loss['LAMBDA_Q'] * torch.mean(train_qsum * train_qsum)
                val_qsum   = torch.sum(val_q_corr, dim=1)
                val_qreg   = self.cfg_loss['LAMBDA_Q'] * torch.mean(val_qsum * val_qsum)

            # log metrics to WANDB to visualize model performance
            if USE_WANDB:
                wandb.log({"loss_train": loss_train, "loss_val": loss_val, "train_qreg": train_qreg, "val_qreg": val_qreg, "lr" : current_lr})

            logging.info("Epoch: {0}; loss train: {2:.{1}f}; qreg train: {3:{1}f}; loss val: {4:.{1}f}; qreg val: {5:.{1}f}".format(
                epoch, PRINT_PRECISION, loss_train, train_qreg, loss_val, val_qreg
            ))

        elif self.cfg['TYPE'] == 'DIPOLEC':
            with torch.no_grad():
                train_dip_pred = self.model(self.train.X)
                loss_train = self.loss_fn(self.train.y, train_dip_pred)

                val_dip_pred = self.model(self.val.X)
                loss_val = self.loss_fn(self.val.y, val_dip_pred)

                self.loss_val = loss_val

            logging.info("Epoch: {0}; loss train: {2:.{1}f}; loss val: {3:.{1}f}".format(epoch, PRINT_PRECISION, loss_train, loss_val))

        elif self.cfg['TYPE'] == 'ENERGY':
            # To disable the gradient calculation, set the .requires_grad attribute of all parameters to False 
            # or wrap the forward pass into with torch.no_grad().
            with torch.no_grad():
                train_y_pred = self.model(self.train.X)
                loss_train   = self.loss_fn(self.train.y, train_y_pred)

                val_y_pred = self.model(self.val.X)
                loss_val   = self.loss_fn(self.val.y, val_y_pred)

                # value to be passed to EarlyStopping/ReduceLR mechanisms
                self.loss_val = loss_val

            # tensorboard writer
            self.writer.add_scalar("loss/train", loss_train, epoch)
            self.writer.add_scalar("loss/val", loss_val, epoch)
            self.writer.add_scalar("lr", current_lr, epoch)

            # log metrics to WANDB to visualize model performance
            if USE_WANDB:
                wandb.log({"loss_train" : loss_train, "loss_val" : loss_val, "lr" : current_lr})

            logging.info("Epoch: {0}; loss train: {2:.{1}f} cm-1; loss val: {3:.{1}f} cm-1; lr: {4:.2e}".format(epoch, PRINT_PRECISION, loss_train, loss_val, current_lr))

        else:
            assert False, "unreachable"


    def model_eval(self):
        self.test.X = self.test.X.to(DEVICE)
        self.test.y = self.test.y.to(DEVICE)

        if self.test.dX is not None:
            self.test.dX = self.test.dX.to(DEVICE)
            self.test.dy = self.test.dy.to(DEVICE)

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
                train_indices = torch.arange(len(self.train.y), device=DEVICE)
                val_indices   = torch.arange(len(self.val.y), device=DEVICE)
                test_indices  = torch.arange(len(self.test.y), device=DEVICE)

                loss_train_e, loss_train_g = self.loss_fn.forward_separate(self.train.y, train_y_pred, self.train.dy, train_dy_pred, train_indices)
                loss_val_e, loss_val_g     = self.loss_fn.forward_separate(self.val.y, val_y_pred, self.val.dy, val_dy_pred, val_indices)
                loss_test_e, loss_test_g   = self.loss_fn.forward_separate(self.test.y, test_y_pred, self.test.dy, test_dy_pred, test_indices)
            else:
                loss_train_e, loss_train_g = self.loss_fn.forward_separate(self.train.y, train_y_pred, self.train.dy, train_dy_pred)
                loss_val_e, loss_val_g     = self.loss_fn.forward_separate(self.val.y, val_y_pred, self.val.dy, val_dy_pred)
                loss_test_e, loss_test_g   = self.loss_fn.forward_separate(self.test.y, test_y_pred, self.test.dy, test_dy_pred)

            logging.info("Model evaluation after training:")
            logging.info("Train      loss: {1:.{0}f} cm-1; gradient loss: {2:.{0}f} cm-1/bohr".format(PRINT_PRECISION, loss_train_e, loss_train_g))
            logging.info("Validation loss: {1:.{0}f} cm-1; gradient loss: {2:.{0}f} cm-1/bohr".format(PRINT_PRECISION, loss_val_e, loss_val_g))
            logging.info("Test       loss: {1:.{0}f} cm-1; gradient loss: {2:.{0}f} cm-1/bohr".format(PRINT_PRECISION, loss_test_e, loss_test_g))

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

            logging.info("Model evaluation after training:")
            logging.info("Train      loss: {1:.{0}f} cm-1".format(PRINT_PRECISION, loss_train))
            logging.info("Validation loss: {1:.{0}f} cm-1".format(PRINT_PRECISION, loss_val))
            logging.info("Test       loss: {1:.{0}f} cm-1".format(PRINT_PRECISION, loss_test))

        elif self.cfg['TYPE'] == 'DIPOLEQ':
            # To disable the gradient calculation, set the .requires_grad attribute of all parameters to False 
            # or wrap the forward pass into with torch.no_grad().
            with torch.no_grad():
                train_q_pred   = self.model(self.train.X)
                train_X_inf    = torch.zeros_like(self.train.X).cpu()
                train_X_inf_tr = torch.from_numpy(xscaler.transform(train_X_inf)).to(DEVICE)
                train_q_inf    = self.model(train_X_inf_tr)
                train_q_corr   = train_q_pred - train_q_inf
                dip_pred_train = torch.einsum('ijk,ij->ik', self.train.xyz_ordered.double(), train_q_corr)
                loss_train     = self.loss_fn(self.train.y, dip_pred_train)

                val_q_pred   = self.model(self.val.X)
                val_X_inf    = torch.zeros_like(self.val.X).cpu()
                val_X_inf_tr = torch.from_numpy(xscaler.transform(val_X_inf)).to(DEVICE)
                val_q_inf    = self.model(val_X_inf_tr)
                val_q_corr   = val_q_pred - val_q_inf
                dip_pred_val = torch.einsum('ijk,ij->ik', self.val.xyz_ordered.double(), val_q_corr)
                loss_val     = self.loss_fn(self.val.y, dip_pred_val)

                test_q_pred   = self.model(self.test.X)
                test_X_inf    = torch.zeros_like(self.test.X).cpu()
                test_X_inf_tr = torch.from_numpy(xscaler.transform(test_X_inf)).to(DEVICE)
                test_q_inf    = self.model(test_X_inf_tr)
                test_q_corr   = test_q_pred - test_q_inf
                dip_pred_test = torch.einsum('ijk,ij->ik', self.test.xyz_ordered.double(), test_q_corr)
                loss_test     = self.loss_fn(self.test.y, dip_pred_test)

            logging.info("Model evluation after training:")
            logging.info("Train      loss: {1:{0}f}".format(PRINT_PRECISION, loss_train))
            logging.info("Validation loss: {1:{0}f}".format(PRINT_PRECISION, loss_val))
            logging.info("Test       loss: {1:{0}f}".format(PRINT_PRECISION, loss_test))

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

    known_groups = ('TYPE', 'DATASET', 'MODEL', 'LOSS', 'TRAINING', 'PRINT_PRECISION', 'PRETRAINED_MODEL_SETTINGS', 'REGULARIZATION')
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

    t = Training(MODEL_FOLDER, MODEL_NAME, chk_path, cfg, train, val, test)

    t.train_model()
    t.model_eval()
