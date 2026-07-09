import logging

import torch

from build_model import build_network
from .base import BaseTrainer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GradientTrainer(BaseTrainer):
    """Energy + gradient trainer with optional trust region, MGDA, Huber."""

    def build_model(self):
        cfg_model = self.cfg.get('MODEL', None)
        return build_network(cfg_model, hidden_dims=self.cfg['MODEL']['HIDDEN_DIMS'], input_features=self.train.NPOLY, output_features=1)
