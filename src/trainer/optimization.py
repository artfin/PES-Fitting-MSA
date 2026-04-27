import logging

import torch

from regularization import L1Regularization, L2Regularization
from losses import (
    WMSELoss_Boltzmann, WRMSELoss_Boltzmann,
    WMSELoss_Ratio, WRMSELoss_Ratio, WRMSELoss_Ratio_dipole,
    WMSELoss_Ratio_wgradients, WMSELoss_TrustRegion_wgradients,
    WMSELoss_PS, WRMSELoss_PS,
    EarlyStopping,
)

import sys
import pathlib
BASEDIR = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(BASEDIR / "vendor"))
from pytorch_lbfgs import LBFGS as HjmshiLBFGS, FullBatchLBFGS as HjmshiFullBatchLBFGS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrainingOptimizationMixin:
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
            weight_decay = cfg_optimizer.get('WEIGHT_DECAY', 0.0)
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

        # Validate MGDA configuration
        if self.cfg_loss.get('USE_MGDA', False):
            gradients_enabled = (self.cfg_loss['USE_GRADIENTS'] or
                                 self.cfg_loss['USE_GRADIENTS_AFTER_EPOCH'] is not None)
            assert gradients_enabled, \
                "USE_MGDA requires USE_GRADIENTS or USE_GRADIENTS_AFTER_EPOCH to be enabled"

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
                use_huber = self.cfg_loss.get('USE_HUBER_GRADIENT', False)
                huber_delta = None
                if use_huber:
                    huber_delta = self.cfg_loss.get('HUBER_DELTA', None)
                    if huber_delta is not None:
                        logging.info("Huber delta (explicit) = {:.6e}".format(huber_delta))
                    else:
                        mad = getattr(self.train, 'mad_grad_components', None)
                        assert mad is not None and mad > 0, (
                            "USE_HUBER_GRADIENT requires train.mad_grad_components; "
                            "available only for gradient-loaded datasets.")
                        huber_delta = 2.0 * float(mad)
                        logging.info("Huber delta (auto) = 2 * MAD = {:.6e}".format(huber_delta))
                loss_fn = WMSELoss_Ratio_wgradients(natoms=self.train.NATOMS, dwt=dwt, g_lambda=g_lambda,
                                                   huber_delta=huber_delta)

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
