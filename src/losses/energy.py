import torch

from config import DEVICE
from distributed import reduce_min


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
