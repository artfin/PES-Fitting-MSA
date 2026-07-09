import logging

import torch

from build_model import build_network
from .base import BaseTrainer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DipoleGRMTrainer(BaseTrainer):
    """Dipole fitting via gradient response matrix (GRM)."""

    def build_model(self):
        cfg_model = self.cfg.get('MODEL', None)
        return build_network(cfg_model, hidden_dims=self.cfg['MODEL']['HIDDEN_DIMS'][0], input_features=self.train.NPOLY, output_features=3)
