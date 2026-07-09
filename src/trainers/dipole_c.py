import logging

import torch

from build_model import build_network
from .base import BaseTrainer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DipoleCTrainer(BaseTrainer):
    """Direct dipole component prediction."""

    def build_model(self):
        cfg_model = self.cfg.get('MODEL', None)
        return build_network(cfg_model, input_features=3 * self.train.NATOMS, output_features=1)
