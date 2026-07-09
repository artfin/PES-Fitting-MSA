import logging

import torch

from build_model import QModel
from .base import BaseTrainer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DipoleQTrainer(BaseTrainer):
    """Charge-based dipole fitting with neutrality constraint."""

    def build_model(self):
        cfg_model = self.cfg.get('MODEL', None)
        return QModel(cfg_model, input_features=self.train.NPOLY, output_features=[len(natoms) for natoms in self.train.symmetry.values()])
