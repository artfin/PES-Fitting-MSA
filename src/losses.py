import logging

import torch
import numpy as np

from distributed import reduce_min, is_main_process
from data_io import save_checkpoint

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRINT_TRAINING_STEPS = 1
PRINT_PRECISION = 3

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
    def __init__(self, natoms, dwt=1.0, g_lambda=1.0, huber_delta=None):
        super().__init__()
        self.natoms = natoms
        self.dwt    = torch.tensor(dwt).to(DEVICE)
        self.g_lambda = torch.tensor(g_lambda).to(DEVICE)
        self.huber_delta = huber_delta

        self.en_mean = None
        self.en_std  = None

    def set_scale(self, en_mean, en_std):
        self.en_mean = torch.from_numpy(en_mean).to(DEVICE)
        self.en_std  = torch.from_numpy(en_std).to(DEVICE)

    def __repr__(self):
        return "WMSELoss_Ratio_wgradients(natoms={}, dwt={}, g_lambda={}, huber_delta={})".format(
            self.natoms, self.dwt, self.g_lambda, self.huber_delta)

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

        if self.huber_delta is not None:
            d = self.huber_delta
            per_config_loss = torch.sum(
                d * d * (torch.sqrt(1.0 + (df / d) ** 2) - 1.0),
                dim=(1, 2),
            )
            wmse_gradients = (w * per_config_loss).sum() / (3.0 * self.natoms) / nconfigs
        else:
            wdf = torch.einsum('ijk,i->ijk', df, w)
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
