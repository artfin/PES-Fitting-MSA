"""Loss functions and training utilities.

Split from the former monolithic ``losses.py``:
- ``energy``:   energy-only weighted (R)MSE losses
- ``gradient``: energy + gradient losses (ratio / trust-region)
- ``dipole``:   dipole-specific weighted loss
- ``early_stopping``: the EarlyStopping training utility

All names are re-exported here so ``from losses import X`` keeps working.
"""

from .energy import (
    WMSELoss_Boltzmann,
    WRMSELoss_Boltzmann,
    WMSELoss_Ratio,
    WRMSELoss_Ratio,
    WMSELoss_PS,
    WRMSELoss_PS,
)
from .gradient import (
    WMSELoss_Ratio_wgradients,
    WMSELoss_TrustRegion_wgradients,
)
from .dipole import WRMSELoss_Ratio_dipole
from .early_stopping import EarlyStopping

__all__ = [
    "WMSELoss_Boltzmann",
    "WRMSELoss_Boltzmann",
    "WMSELoss_Ratio",
    "WRMSELoss_Ratio",
    "WMSELoss_PS",
    "WRMSELoss_PS",
    "WMSELoss_Ratio_wgradients",
    "WMSELoss_TrustRegion_wgradients",
    "WRMSELoss_Ratio_dipole",
    "EarlyStopping",
]
