from .base import BaseTrainer
from .energy import EnergyTrainer
from .gradient import GradientTrainer
from .dipole_grm import DipoleGRMTrainer
from .dipole_q import DipoleQTrainer
from .dipole_c import DipoleCTrainer


def get_trainer(model_folder, model_name, chk_path, cfg, train, val, test,
                rank=0, world_size=1, local_rank=0):
    """Dispatch to the appropriate trainer based on config."""
    fit_type = cfg['TYPE']
    use_gradients = cfg.get('LOSS', {}).get('USE_GRADIENTS', False)
    use_gradients_after = cfg.get('LOSS', {}).get('USE_GRADIENTS_AFTER_EPOCH') is not None

    kwargs = dict(
        model_folder=model_folder, model_name=model_name, chk_path=chk_path,
        cfg=cfg, train=train, val=val, test=test,
        rank=rank, world_size=world_size, local_rank=local_rank
    )

    if fit_type == 'ENERGY':
        if use_gradients or use_gradients_after:
            return GradientTrainer(**kwargs)
        else:
            return EnergyTrainer(**kwargs)
    elif fit_type == 'DIPOLE':
        return DipoleGRMTrainer(**kwargs)
    elif fit_type == 'DIPOLEQ':
        return DipoleQTrainer(**kwargs)
    elif fit_type == 'DIPOLEC':
        return DipoleCTrainer(**kwargs)
    else:
        raise ValueError("Unknown fit type: {}".format(fit_type))
