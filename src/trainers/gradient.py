import logging

import torch
import torch.distributed as dist

from config import TORCH_FLOAT
from build_model import build_network
from losses import (
    WMSELoss_Ratio, WRMSELoss_Ratio,
    WMSELoss_Boltzmann, WRMSELoss_Boltzmann,
    WMSELoss_PS, WRMSELoss_PS,
    WMSELoss_Ratio_wgradients, WMSELoss_TrustRegion_wgradients,
)
from distributed import (
    is_main_process, reduce_mean, reduce_mae, reduce_rmse, reduce_min, reduce_sum,
)
from .base import (
    BaseTrainer, DEVICE, PRINT_TRAINING_STEPS, PRINT_PRECISION, USE_WANDB,
    HjmshiLBFGS, HjmshiFullBatchLBFGS,
)

from .multibatch import MultibatchMixin

if USE_WANDB:
    import wandb


class GradientTrainer(MultibatchMixin, BaseTrainer):
    """Energy + gradient trainer with optional trust region, MGDA, Huber."""

    def build_model(self):
        cfg_model = self.cfg.get('MODEL', None)
        return build_network(cfg_model, hidden_dims=self.cfg['MODEL']['HIDDEN_DIMS'], input_features=self.train.NPOLY, output_features=1)

    def prepare_data_for_device(self):
        """Move energy + gradient dataset tensors (X/y, dX/dy) to self.device."""
        multibatch = bool(self.cfg_batch.get('MULTIBATCH_ENABLED', False))
        if multibatch:
            # Keep the training set on CPU; each step copies only its batch
            # to the GPU (pin_memory makes the per-batch copy faster when CUDA).
            if torch.cuda.is_available():
                self.train.X = self.train.X.pin_memory()
                self.train.y = self.train.y.pin_memory()
                if self.train.dX is not None:
                    self.train.dX = self.train.dX.pin_memory()
                    self.train.dy = self.train.dy.pin_memory()
            # Val stays on GPU for cheap eval.
            self.val.X = self.val.X.to(self.device)
            self.val.y = self.val.y.to(self.device)
        else:
            self.train.X = self.train.X.to(self.device)
            self.train.y = self.train.y.to(self.device)
            self.val.X = self.val.X.to(self.device)
            self.val.y = self.val.y.to(self.device)

        if self.train.dX is not None and not multibatch:
            self.train.dX = self.train.dX.to(self.device)
            self.train.dy = self.train.dy.to(self.device)

            self.val.dX = self.val.dX.to(self.device)
            self.val.dy = self.val.dy.to(self.device)
        elif self.train.dX is not None and multibatch:
            # Only move validation gradient tensors; train stays on pinned CPU.
            self.val.dX = self.val.dX.to(self.device)
            self.val.dy = self.val.dy.to(self.device)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
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

        if self.cfg_loss['NAME'] == 'WRMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'Boltzmann' and not self.cfg_loss['USE_GRADIENTS']:
            Eref = self.cfg_loss.get('EREF', 2000.0)
            loss_fn = WRMSELoss_Boltzmann(Eref=Eref)
        elif self.cfg_loss['NAME'] == 'WMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'Boltzmann' and not self.cfg_loss['USE_GRADIENTS']:
            Eref = self.cfg_loss.get('EREF', 2000.0)
            loss_fn = WMSELoss_Boltzmann(Eref=Eref)

        elif self.cfg_loss['NAME'] == 'WRMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'Ratio' and not self.cfg_loss['USE_GRADIENTS']:
            dwt = self.cfg_loss.get('DWT', 1.0)
            focal_gamma = self.cfg_loss.get('FOCAL_GAMMA', 0.0)
            focal_ema_decay = self.cfg_loss.get('FOCAL_EMA_DECAY', 0.95)
            loss_fn = WRMSELoss_Ratio(dwt=dwt, focal_gamma=focal_gamma, focal_ema_decay=focal_ema_decay)
        elif self.cfg_loss['NAME'] == 'WMSE' and self.cfg_loss['WEIGHT_TYPE'] == 'Ratio' and not self.cfg_loss['USE_GRADIENTS']:
            dwt = self.cfg_loss.get('DWT', 1.0)
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
            dwt = self.cfg_loss.get('DWT', 1.0)
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

    # ------------------------------------------------------------------
    # per-epoch preparation (gradient-inclusion switch, ramps, trust region)
    # ------------------------------------------------------------------
    def prepare_epoch(self, epoch):
        """Per-epoch precompute: gradient-inclusion switch, ramps, LBFGS resets,
        and the trust-region active-set mask. Sets self._trust_* attributes read
        by compute_loss / evaluate_and_log."""
        # switch into mixed loss function: E + F
        if self.cfg_loss['USE_GRADIENTS_AFTER_EPOCH'] is not None and epoch == self.cfg_loss['USE_GRADIENTS_AFTER_EPOCH']:
            self.cfg_loss['USE_GRADIENTS'] = True
            self.loss_fn = self.build_loss().to(self.device)
            self.loss_fn.set_scale(self.yscaler.mean_, self.yscaler.scale_)
            self.gradient_start_epoch = epoch

            self.es.reset()

            # Reset L-BFGS curvature history. The stored (s_k, y_k) pairs
            # describe the energy-only loss surface and produce degenerate
            # search directions on the new energy+gradient surface, causing
            # the Wolfe line search to return t=0 indefinitely.
            if isinstance(self.optimizer, (torch.optim.LBFGS, HjmshiLBFGS, HjmshiFullBatchLBFGS)):
                if isinstance(self.optimizer, torch.optim.LBFGS):
                    self.optimizer.state.clear()
                else:
                    # vendored LBFGS / FullBatchLBFGS
                    state = self.optimizer.state['global_state']
                    state['n_iter'] = 0
                    state['curv_skips'] = 0
                    state['fail_skips'] = 0
                    state['H_diag'] = 1
                    state['fail'] = True
                    state['old_dirs'] = []
                    state['old_stps'] = []
                    if 'rho' in state:
                        state['rho'] = [None] * self.optimizer.param_groups[0]['history_size']
                    if 'alpha' in state:
                        state['alpha'] = [None] * self.optimizer.param_groups[0]['history_size']
                self._lbfgs_prev_n_iter = 0
                self._lbfgs_prev_func_evals = 0
                self._log("Reset L-BFGS state at gradient inclusion (epoch {})".format(epoch))

            # Reset LR to initial value so the optimizer has full step
            # budget to explore the new loss landscape.
            initial_lr = self.cfg_solver['OPTIMIZER'].get('LR', 0.1)
            for pg in self.optimizer.param_groups:
                pg['lr'] = initial_lr
            self.scheduler = self.build_scheduler()
            self._log("Reset LR to {} and rebuilt scheduler at gradient inclusion".format(initial_lr))

        # Progressive G_LAMBDA ramp
        if self.cfg_loss['USE_GRADIENTS'] and self.cfg_loss.get('G_LAMBDA_RAMP_EPOCHS', 0) > 0:
            ramp_epochs = self.cfg_loss['G_LAMBDA_RAMP_EPOCHS']
            start_epoch = self.gradient_start_epoch if self.gradient_start_epoch is not None else 0
            progress = (epoch - start_epoch) / ramp_epochs
            progress = max(0.0, min(1.0, progress))
            target_g_lambda = self.cfg_loss.get('G_LAMBDA', 1.0)
            current_g_lambda = target_g_lambda * progress
            self.loss_fn.g_lambda = torch.tensor(current_g_lambda).to(self.device)
            if epoch % PRINT_TRAINING_STEPS == 0 or epoch == start_epoch or epoch == start_epoch + ramp_epochs:
                self._log("G_LAMBDA ramp: epoch {}, progress {:.1%}, g_lambda = {:.4f}".format(epoch, progress, current_g_lambda))

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
                self._log("Trust-threshold anneal: epoch {}, progress {:.1%}, threshold = {:.1f}".format(epoch, progress, self.current_trust_threshold))
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
                and isinstance(self.optimizer, (torch.optim.LBFGS, HjmshiLBFGS, HjmshiFullBatchLBFGS))
                and epoch > self.cfg_loss.get('USE_GRADIENTS_AFTER_EPOCH', 0)
                and (epoch - self.cfg_loss.get('USE_GRADIENTS_AFTER_EPOCH', 0))
                    % lbfgs_reset_interval == 0):
            if isinstance(self.optimizer, torch.optim.LBFGS):
                self.optimizer.state.clear()
            else:
                state = self.optimizer.state['global_state']
                state['n_iter'] = 0
                state['curv_skips'] = 0
                state['fail_skips'] = 0
                state['H_diag'] = 1
                state['fail'] = True
                state['old_dirs'] = []
                state['old_stps'] = []
                if 'rho' in state:
                    state['rho'] = [None] * self.optimizer.param_groups[0]['history_size']
                if 'alpha' in state:
                    state['alpha'] = [None] * self.optimizer.param_groups[0]['history_size']
            self._lbfgs_prev_n_iter = 0
            self._lbfgs_prev_func_evals = 0
            self._log("Periodic L-BFGS state reset (epoch {})".format(epoch))

            # Optionally reset LR to initial value on L-BFGS reset
            if self.cfg_solver['OPTIMIZER'].get('LR_RESET_ON_LBFGS_RESET', False):
                initial_lr = self.cfg_solver['OPTIMIZER'].get('LR', 0.1)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = initial_lr
                self.scheduler = self.build_scheduler()
                self._log("Reset LR to {} and rebuilt scheduler".format(initial_lr))

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

                # Aggregate trust region stats across ranks for correct logging
                n_in_trust_t = torch.tensor(n_in_trust, dtype=torch.float32, device=DEVICE)
                n_total_t = torch.tensor(len(self.train.X), dtype=torch.float32, device=DEVICE)
                err_min_t = energy_errors.min()
                err_max_t = energy_errors.max()
                if self.world_size > 1:
                    n_in_trust_global = int(reduce_sum(n_in_trust_t).item())
                    n_total_global = int(reduce_sum(n_total_t).item())
                    err_min_global = reduce_min(err_min_t).item()
                    err_max_global = torch.tensor(err_max_t.item(), device=DEVICE)
                    dist.all_reduce(err_max_global, op=dist.ReduceOp.MAX)
                    err_max_global = err_max_global.item()
                else:
                    n_in_trust_global = n_in_trust
                    n_total_global = len(self.train.X)
                    err_min_global = err_min_t.item()
                    err_max_global = err_max_t.item()
                frac_global = 100.0 * n_in_trust_global / max(n_total_global, 1)

                if n_in_trust > 0:
                    phi_sum_t = gradient_weights.sum()
                    phi_sum_global = reduce_sum(phi_sum_t).item() if self.world_size > 1 else phi_sum_t.item()
                    grad_trust_enabled = self.cfg_loss.get('GRADIENT_TRUST_THRESHOLD') is not None
                    label = "soft+grad" if grad_trust_enabled else "soft"
                    self._log(
                        "Trust region ({}): {}/{} configs ({:.1f}%) | "
                        "energy err: min={:.1f}, max={:.1f}, med={:.1f} | "
                        "weights: min={:.3f}, mean={:.3f}, sum={:.1f}".format(
                            label,
                            n_in_trust_global, n_total_global, frac_global,
                            err_min_global, err_max_global,
                            energy_errors.median().item(),
                            gradient_weights.min().item(), gradient_weights.mean().item(),
                            phi_sum_global))
                else:
                    self._log(
                        "Trust region: 0/{} configs (0.0%) | "
                        "energy err: min={:.1f}, max={:.1f}, med={:.1f}".format(
                            n_total_global,
                            err_min_global, err_max_global,
                            energy_errors.median().item()))

                # Run trust-region diagnostics (churn + eviction signal).
                self.log_trust_region_diagnostics(
                    epoch, trust_mask, energy_errors, gradient_weights
                )

        # publish per-epoch trust-region state for compute_loss / evaluate_and_log
        self._use_trust_region = use_trust_region
        self._trust_indices = trust_indices
        self._n_in_trust = n_in_trust
        self._trust_X_subset = X_subset
        self._trust_dX_subset = dX_subset
        self._trust_dy_subset = train_dy_subset
        self._trust_gradient_weights = gradient_weights
        self._trust_energy_errors = energy_errors
        self._trust_mask = trust_mask

    # ------------------------------------------------------------------
    # training loss
    # ------------------------------------------------------------------
    def compute_loss(self, separate=False):
        """Compute training loss.

        Args:
            separate: If True and using gradients with trust region,
                     return (energy_loss, gradient_loss) tuple for MGDA.
                     Otherwise return combined loss.
        """
        if self.cfg_loss['USE_GRADIENTS']:
            if self._use_trust_region:
                if self._n_in_trust > 0:
                    y_pred_subset = self.model(self._trust_X_subset)
                    train_dy_pred_subset = self.compute_gradients_from_energy(
                        self._trust_X_subset, self._trust_dX_subset, y_pred_subset
                    )
                    train_y_pred = self.model(self.train.X)
                    if separate:
                        energy_loss, gradient_loss = self.loss_fn.forward_separate(
                            self.train.y, train_y_pred,
                            self._trust_dy_subset, train_dy_pred_subset,
                            self._trust_indices, self._trust_gradient_weights
                        )
                        # Add regularization to energy loss (it's model complexity, not gradient fitting)
                        if self.regularization is not None:
                            energy_loss = energy_loss + self.regularization(self.model)
                        return energy_loss, gradient_loss
                    else:
                        loss = self.loss_fn(
                            self.train.y, train_y_pred,
                            self._trust_dy_subset, train_dy_pred_subset,
                            self._trust_indices, self._trust_gradient_weights
                        )
                else:
                    # No configs in trust region yet - energy only
                    train_y_pred = self.model(self.train.X)
                    loss = self.loss_fn.forward_energy_only(self.train.y, train_y_pred)
                    if separate:
                        if self.regularization is not None:
                            loss = loss + self.regularization(self.model)
                        return loss, torch.tensor(0.0, device=DEVICE)
            else:
                # Original approach: compute gradients for ALL configs
                train_y_pred, train_dy_pred = self.compute_gradients(self.train)
                if separate:
                    energy_loss, gradient_loss = self.loss_fn.forward_separate(
                        self.train.y, train_y_pred, self.train.dy, train_dy_pred
                    )
                    if self.regularization is not None:
                        energy_loss = energy_loss + self.regularization(self.model)
                    return energy_loss, gradient_loss
                loss = self.loss_fn(self.train.y, train_y_pred, self.train.dy, train_dy_pred)
        else:
            # Pre-switch (before USE_GRADIENTS_AFTER_EPOCH): energy-only loss.
            y_pred = self.model(self.train.X)
            loss = self.loss_fn(self.train.y, y_pred)

        if self.regularization is not None:
            loss = loss + self.regularization(self.model)
        return loss

    # ------------------------------------------------------------------
    # per-epoch evaluation + logging
    # ------------------------------------------------------------------
    def evaluate_and_log(self, epoch, current_lr):
        if self.cfg_loss['USE_GRADIENTS']:
            # Use memory-efficient gradient evaluation (no create_graph)
            train_y_pred, train_dy_pred = self.compute_gradients_eval(self.train)
            val_y_pred, val_dy_pred = self.compute_gradients_eval(self.val)

            # Compute energy metrics directly (works with any loss function)
            # Compute local values first, then aggregate via reduce_rmse/reduce_mae
            train_e_d    = self.loss_fn.descale_energies(self.train.y)
            train_e_pred = self.loss_fn.descale_energies(train_y_pred)
            train_e_errors = (train_e_d - train_e_pred).view(-1)
            train_e_rmse_local = torch.sqrt(torch.mean(train_e_errors ** 2)).item()
            train_e_mae  = reduce_mae(train_e_errors)
            train_e_rmse = reduce_rmse(train_e_errors)

            val_e_d    = self.loss_fn.descale_energies(self.val.y)
            val_e_pred = self.loss_fn.descale_energies(val_y_pred)
            val_e_errors = (val_e_d - val_e_pred).view(-1)
            val_e_rmse_local = torch.sqrt(torch.mean(val_e_errors ** 2)).item()
            val_e_mae  = reduce_mae(val_e_errors)
            val_e_rmse = reduce_rmse(val_e_errors)

            # Compute gradient metrics directly (per-component errors for RMSE/MAE)
            natoms   = self.train.NATOMS
            train_dy = self.train.dy.reshape(-1, 3 * natoms)
            val_dy   = self.val.dy.reshape(-1, 3 * natoms)
            train_g_errors = (train_dy - train_dy_pred).view(-1)
            val_g_errors   = (val_dy - val_dy_pred).view(-1)
            train_g_mae  = reduce_mae(train_g_errors)
            val_g_mae    = reduce_mae(val_g_errors)
            train_g_rmse = reduce_rmse(train_g_errors)
            val_g_rmse   = reduce_rmse(val_g_errors)

            # Snapshot per-config train gradient RMSE for next-epoch trust-region
            # diagnostics (eviction signal: do "left" configs have higher
            # gradient errors than "stayed" configs?).
            with torch.no_grad():
                per_config_f_rmse = torch.sqrt(
                    torch.sum((train_dy - train_dy_pred) ** 2, dim=1) / (3 * natoms)
                ).detach()
                self._prev_train_gradient_errors = per_config_f_rmse

            # Per-config gradient-loss contribution + phi histogram on the active set.
            if self._use_trust_region and self._trust_indices is not None and self._n_in_trust > 0:
                self.log_gradient_loss_diagnostics(
                    epoch, train_dy, train_dy_pred,
                    train_e_d, train_e_pred,
                    self._trust_indices, self._trust_gradient_weights,
                )

            # Compute weighted loss values for logging (energy component only for scheduler)
            # Sync minimum across ranks for consistent weighting in distributed mode
            enmin_train = reduce_min(train_e_d.min())
            w_train = self.loss_fn.dwt / (self.loss_fn.dwt + train_e_d - enmin_train)
            loss_train_e = (w_train.view(-1) * (train_e_d - train_e_pred).view(-1)**2).mean()

            enmin_val = reduce_min(val_e_d.min())
            w_val = self.loss_fn.dwt / (self.loss_fn.dwt + val_e_d - enmin_val)
            loss_val_e = (w_val.view(-1) * (val_e_d - val_e_pred).view(-1)**2).mean()

            # Log verbose per-rank diagnostics before reducing
            if self.world_size > 1:
                self.log_distributed_diagnostics(
                    epoch,
                    loss_local=loss_val_e.item(),
                    e_rmse_local=val_e_rmse_local,
                    n_trust_local=self._n_in_trust if self._use_trust_region else None,
                    n_total_local=len(self.train.X) if self._use_trust_region else None,
                    use_trust_region=self._use_trust_region
                )

            # Reduce weighted losses across ranks for scheduler / early stopping.
            # (MAE/RMSE already aggregated via reduce_mae/reduce_rmse above)
            if self.world_size > 1:
                loss_train_e = reduce_mean(loss_train_e)
                loss_val_e   = reduce_mean(loss_val_e)

            self._log("Epoch: {}; (energy) WMSE train: {:.3f}; (energy) WMSE val: {:.3f}\n \
                                           (energy) MAE train:  {:.3f} cm-1; (gradient) MAE train:  {:.3f} cm-1/bohr\n \
                                           (energy) MAE val:    {:.3f} cm-1; (gradient) MAE val:    {:.3f} cm-1/bohr\n \
                                           (energy) RMSE train: {:.3f} cm-1; (gradient) RMSE train: {:.3f} cm-1/bohr\n \
                                           (energy) RMSE val:   {:.3f} cm-1; (gradient) RMSE val:   {:.3f} cm-1/bohr".format(
                epoch, loss_train_e, loss_val_e, train_e_mae, train_g_mae, val_e_mae, val_g_mae, train_e_rmse, train_g_rmse, val_e_rmse, val_g_rmse
            ))

            # value to be passed to EarlyStopping/ReduceLR mechanisms
            self.loss_val = loss_val_e

            if self.writer is not None:
                self.writer.add_scalar("loss/train", loss_train_e, epoch)
                self.writer.add_scalar("loss/val", loss_val_e, epoch)

            # log metrics to WANDB to visualize model performance
            if is_main_process() and USE_WANDB:
                wandb.log({
                    "loss_train_e" : loss_train_e, "loss_val_e" : loss_val_e,
                    "train_e_mae" : train_e_mae, "train_e_rmse" : train_e_rmse, "val_e_mae" : val_e_mae, "val_e_rmse" : val_e_rmse,
                    "train_g_mae" : train_g_mae, "train_g_rmse" : train_g_rmse, "val_g_mae" : val_g_mae, "val_g_rmse" : val_g_rmse,
                    "lr" : current_lr})


        else:
            # To disable the gradient calculation, set the .requires_grad attribute of all parameters to False 
            # or wrap the forward pass into with torch.no_grad().
            with torch.no_grad():
                train_y_pred = self.model(self.train.X)
                loss_train   = self.loss_fn(self.train.y, train_y_pred)

                val_y_pred = self.model(self.val.X)
                loss_val   = self.loss_fn(self.val.y, val_y_pred)

                if self.world_size > 1:
                    loss_train = reduce_mean(loss_train)
                    loss_val   = reduce_mean(loss_val)

                # value to be passed to EarlyStopping/ReduceLR mechanisms
                self.loss_val = loss_val

            # tensorboard writer
            if self.writer is not None:
                self.writer.add_scalar("loss/train", loss_train, epoch)
                self.writer.add_scalar("loss/val", loss_val, epoch)
                self.writer.add_scalar("lr", current_lr, epoch)

            # log metrics to WANDB to visualize model performance
            if is_main_process() and USE_WANDB:
                wandb.log({"loss_train" : loss_train, "loss_val" : loss_val, "lr" : current_lr})

            self._log("Epoch: {0}; loss train: {2:.{1}f} cm-1; loss val: {3:.{1}f} cm-1; lr: {4:.2e}".format(epoch, PRINT_PRECISION, loss_train, loss_val, current_lr))

    # ------------------------------------------------------------------
    # final evaluation
    # ------------------------------------------------------------------
    def model_eval(self):
        self.test.X = self.test.X.to(self.device)
        self.test.y = self.test.y.to(self.device)

        if self.test.dX is not None:
            self.test.dX = self.test.dX.to(self.device)
            self.test.dy = self.test.dy.to(self.device)

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
                train_indices = torch.arange(len(self.train.y), device=self.device)
                val_indices   = torch.arange(len(self.val.y), device=self.device)
                test_indices  = torch.arange(len(self.test.y), device=self.device)

                # Full-dataset final eval: every config is "in trust" with uniform
                # soft weight 1.0 (forward_separate requires per-config gradient_weights).
                train_w = torch.ones(len(train_indices), device=self.device)
                val_w   = torch.ones(len(val_indices), device=self.device)
                test_w  = torch.ones(len(test_indices), device=self.device)

                loss_train_e, loss_train_g = self.loss_fn.forward_separate(self.train.y, train_y_pred, self.train.dy, train_dy_pred, train_indices, train_w)
                loss_val_e, loss_val_g     = self.loss_fn.forward_separate(self.val.y, val_y_pred, self.val.dy, val_dy_pred, val_indices, val_w)
                loss_test_e, loss_test_g   = self.loss_fn.forward_separate(self.test.y, test_y_pred, self.test.dy, test_dy_pred, test_indices, test_w)
            else:
                loss_train_e, loss_train_g = self.loss_fn.forward_separate(self.train.y, train_y_pred, self.train.dy, train_dy_pred)
                loss_val_e, loss_val_g     = self.loss_fn.forward_separate(self.val.y, val_y_pred, self.val.dy, val_dy_pred)
                loss_test_e, loss_test_g   = self.loss_fn.forward_separate(self.test.y, test_y_pred, self.test.dy, test_dy_pred)

            if self.world_size > 1:
                loss_train_e = reduce_mean(loss_train_e)
                loss_train_g = reduce_mean(loss_train_g)
                loss_val_e   = reduce_mean(loss_val_e)
                loss_val_g   = reduce_mean(loss_val_g)
                loss_test_e  = reduce_mean(loss_test_e)
                loss_test_g  = reduce_mean(loss_test_g)

            self._log("Model evaluation after training:")
            self._log("Train      loss: {1:.{0}f} cm-1; gradient loss: {2:.{0}f} cm-1/bohr".format(PRINT_PRECISION, loss_train_e, loss_train_g))
            self._log("Validation loss: {1:.{0}f} cm-1; gradient loss: {2:.{0}f} cm-1/bohr".format(PRINT_PRECISION, loss_val_e, loss_val_g))
            self._log("Test       loss: {1:.{0}f} cm-1; gradient loss: {2:.{0}f} cm-1/bohr".format(PRINT_PRECISION, loss_test_e, loss_test_g))

            self.eval_metrics = {
                "loss_train_e": float(loss_train_e), "loss_train_g": float(loss_train_g),
                "loss_val_e":   float(loss_val_e),   "loss_val_g":   float(loss_val_g),
                "loss_test_e":  float(loss_test_e),  "loss_test_g":  float(loss_test_g),
            }

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

            if self.world_size > 1:
                loss_train = reduce_mean(loss_train)
                loss_val   = reduce_mean(loss_val)
                loss_test  = reduce_mean(loss_test)

            self._log("Model evaluation after training:")
            self._log("Train      loss: {1:.{0}f} cm-1".format(PRINT_PRECISION, loss_train))
            self._log("Validation loss: {1:.{0}f} cm-1".format(PRINT_PRECISION, loss_val))
            self._log("Test       loss: {1:.{0}f} cm-1".format(PRINT_PRECISION, loss_test))

            self.eval_metrics = {
                "loss_train": float(loss_train),
                "loss_val":   float(loss_val),
                "loss_test":  float(loss_test),
            }
        else:
            assert False, "unreachable"

    # ------------------------------------------------------------------
    # MGDA support
    # ------------------------------------------------------------------
    def supports_mgda(self):
        return (self.cfg_loss.get('USE_MGDA', False)
                and self.cfg_loss['USE_GRADIENTS'])

    # ------------------------------------------------------------------
    # gradient + trust-region helpers and diagnostics
    # ------------------------------------------------------------------
    def compute_gradients(self, dataset):
        Xtr = dataset.X

        Xtr.requires_grad = True

        y_pred = self.model(Xtr)
        dEdp   = torch.autograd.grad(outputs=y_pred, inputs=Xtr, grad_outputs=torch.ones_like(y_pred), retain_graph=True, create_graph=True)[0]

        Xtr.requires_grad = False

        # take into account normalization of polynomials
        # now we have derivatives of energy w.r.t. to polynomials
        x_scale = torch.from_numpy(self.xscaler.scale_).to(self.device)
        dEdp = torch.div(dEdp, x_scale)

        # gradient = dE/dx = \sigma(E) * dE/d(poly) * d(poly)/dx
        # `torch.einsum` throws a Runtime error without an explicit conversion to Double
        dEdx = torch.einsum('ij,ijk -> ik', dEdp.to(TORCH_FLOAT), dataset.dX.to(TORCH_FLOAT))

        # take into account normalization of model energy
        y_scale = torch.from_numpy(self.yscaler.scale_).to(self.device)
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
        x_scale = torch.from_numpy(self.xscaler.scale_).to(self.device)
        dEdp = torch.div(dEdp, x_scale)

        # gradient = dE/dx = \sigma(E) * dE/d(poly) * d(poly)/dx
        dEdx = torch.einsum('ij,ijk -> ik', dEdp.to(TORCH_FLOAT), dX_subset.to(TORCH_FLOAT))

        # take into account normalization of model energy
        y_scale = torch.from_numpy(self.yscaler.scale_).to(self.device)
        dEdx = torch.mul(dEdx, y_scale)

        return dEdx
    def compute_trust_mask(self, dataset):
        """
        Compute trust region active set based on energy prediction errors
        and optionally gradient errors from the previous epoch.

        Uses soft boundaries with sigmoid weighting:
            phi(e_i) = sigmoid((threshold - error) / soft_scale) in [0, 1]
        Active set = {i : phi(e_i) > soft_cutoff} (memory optimization).

        When GRADIENT_TRUST_THRESHOLD is set, configs with large gradient
        errors (from previous epoch) are down-weighted using a soft sigmoid.
        This is combined multiplicatively with energy-based weights.

        Returns:
          trust_indices : 1-D LongTensor of active-set config indices
          trust_mask    : 1-D BoolTensor of shape (N,) indicating membership
          energy_errors : 1-D float tensor of |E_pred - E_true| (cm^-1)
          gradient_weights : 1-D float tensor of combined weights for the active set
        """
        trust_threshold = getattr(self, 'current_trust_threshold', None)
        if trust_threshold is None:
            trust_threshold = self.cfg_loss.get('TRUST_THRESHOLD', 50.0)

        soft_scale = self.cfg_loss.get('TRUST_SOFT_SCALE', None)
        soft_cutoff = self.cfg_loss.get('TRUST_SOFT_CUTOFF', 0.01)

        # Gradient trust: filter by previous epoch's gradient errors
        grad_trust_threshold = self.cfg_loss.get('GRADIENT_TRUST_THRESHOLD', None)
        grad_trust_soft_scale = self.cfg_loss.get('GRADIENT_TRUST_SOFT_SCALE', None)

        with torch.no_grad():
            y_pred = self.model(dataset.X)

            # Descale energies
            en_mean = torch.from_numpy(self.yscaler.mean_).to(self.device)
            en_std = torch.from_numpy(self.yscaler.scale_).to(self.device)

            en_pred_descaled = y_pred * en_std + en_mean
            en_true_descaled = dataset.y * en_std + en_mean

            energy_errors = torch.abs(en_pred_descaled - en_true_descaled).view(-1)

            # Always use soft boundary with sigmoid weighting
            phi_energy = WMSELoss_TrustRegion_wgradients.soft_phi(
                energy_errors, trust_threshold, soft_scale=soft_scale
            )
            trust_mask = phi_energy > soft_cutoff
            trust_indices = torch.nonzero(trust_mask, as_tuple=False).view(-1)
            gradient_weights = phi_energy[trust_indices]

            # Apply gradient trust filtering (uses previous epoch's gradient errors)
            if (grad_trust_threshold is not None
                    and self._prev_train_gradient_errors is not None
                    and self._prev_train_gradient_errors.numel() == energy_errors.numel()):
                # Compute soft phi for gradient errors (same sigmoid as energy)
                phi_grad = WMSELoss_TrustRegion_wgradients.soft_phi(
                    self._prev_train_gradient_errors,
                    grad_trust_threshold,
                    soft_scale=grad_trust_soft_scale
                )
                # Multiply energy weights by gradient weights
                gradient_weights = gradient_weights * phi_grad[trust_indices]

        return trust_indices, trust_mask, energy_errors, gradient_weights
    def compute_gradients_eval(self, dataset):
        """
        Compute gradients for evaluation (no create_graph needed).
        Much more memory efficient than compute_gradients() since we don't need
        to backpropagate through the gradient computation.
        """
        Xtr = dataset.X.clone().detach()
        Xtr.requires_grad = True

        with torch.enable_grad():
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
        x_scale = torch.from_numpy(self.xscaler.scale_).to(self.device)
        dEdp = torch.div(dEdp, x_scale)

        # gradient = dE/dx
        dEdx = torch.einsum('ij,ijk -> ik', dEdp.to(TORCH_FLOAT), dataset.dX.to(TORCH_FLOAT))

        # take into account normalization of model energy
        y_scale = torch.from_numpy(self.yscaler.scale_).to(self.device)
        dEdx = torch.mul(dEdx, y_scale)

        return y_pred.detach(), dEdx.detach()
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
        N_local = trust_mask.numel()
        cur = trust_mask.detach()
        n_in_local = int(cur.sum().item())

        # Initialize trackers lazily on the first call.
        if self._trust_flip_count is None:
            self._trust_flip_count = torch.zeros(N_local, dtype=torch.long, device=DEVICE)

        if self._prev_trust_mask is None:
            entered_local = n_in_local
            left_local = 0
            stable_in_local = n_in_local
            stable_out_local = N_local - n_in_local
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
            entered_local = int(entered_mask.sum().item())
            left_local = int(left_mask.sum().item())
            stable_in_local = int(stable_in_mask.sum().item())
            stable_out_local = int(stable_out_mask.sum().item())

            # Update cumulative flip count.
            flips = entered_mask | left_mask
            self._trust_flip_count[flips] += 1

            # Eviction signal: compare prev-epoch gradient errors of left vs stayed.
            if (self._prev_train_gradient_errors is not None
                    and self._prev_train_gradient_errors.numel() == N_local):
                pfe = self._prev_train_gradient_errors
                if left_local > 0:
                    mean_err_left = float(pfe[left_mask].mean().item())
                    med_err_left  = float(pfe[left_mask].median().item())
                else:
                    mean_err_left = float('nan')
                    med_err_left  = float('nan')
                if stable_in_local > 0:
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

        if gradient_weights.numel() > 0:
            phi_sum = float(gradient_weights.sum().item())
            phi_mean = float(gradient_weights.mean().item())
            phi_min = float(gradient_weights.min().item())
        else:
            phi_sum = float('nan')
            phi_mean = float('nan')
            phi_min = float('nan')

        max_flips = int(self._trust_flip_count.max().item())
        ever_in = int((self._trust_flip_count > 0).sum().item()) + stable_in_local

        # Aggregate counts across ranks for logging
        if self.world_size > 1:
            counts = torch.tensor([N_local, n_in_local, entered_local, left_local, stable_in_local],
                                  dtype=torch.float32, device=DEVICE)
            counts = reduce_sum(counts)
            N, n_in, entered, left, stable_in = [int(c.item()) for c in counts]
            frac = n_in / max(N, 1)
        else:
            N, n_in, entered, left, stable_in = N_local, n_in_local, entered_local, left_local, stable_in_local
            frac = n_in / max(N, 1)

        if is_main_process():
            logging.info(
                "[trust-diag] epoch={} | n_in={}/{} ({:.1%}) | entered={} left={} "
                "stable_in={} | prev-epoch gradient-RMSE: left={:.2f} stayed={:.2f} "
                "(med {:.2f}/{:.2f}) | max_flips={}".format(
                    epoch, n_in, N, frac, entered, left, stable_in,
                    mean_err_left, mean_err_stayed,
                    med_err_left, med_err_stayed, max_flips
                )
            )

        # Append CSV row for post-hoc plotting (only on main process).
        stable_out = N - n_in  # Compute from aggregated values
        if is_main_process():
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

            phi_act = gradient_weights.view(-1).to(f_sq.dtype)

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

            pq = torch.quantile(phi_cpu, qs[:5].to(phi_cpu.dtype)).tolist()
            # Bin phi into membership categories.
            bins = torch.tensor([0.0, 0.25, 0.50, 0.75, 0.90, 1.0001])
            # counts per bin
            idx = torch.bucketize(phi_cpu, bins) - 1
            idx = idx.clamp(0, 4)
            bin_counts = [int((idx == b).sum().item()) for b in range(5)]

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

        if is_main_process():
            logging.info(
                "[grad-diag] epoch={} | top1%={:.1%} top5%={:.1%} top10%={:.1%} "
                "of gradient loss | contrib q50={:.3e} q95={:.3e} max={:.3e}".format(
                    epoch, top1, top5, top10, cq[2], cq[5], contrib_max
                )
            )

    def log_mgda_diagnostics(self, epoch, alpha, alpha_raw, cos_sim):
        """Log MGDA+GradNorm diagnostics.

        Args:
            epoch: current epoch
            alpha: EMA-smoothed weight for energy objective
            alpha_raw: raw (unsmoothed) weight from loss-ratio computation
            cos_sim: cosine similarity between normalized gradients
        """
        if not is_main_process():
            return

        # Get loss EMAs for logging
        e_loss_ema = self._mgda_energy_loss_ema if self._mgda_energy_loss_ema is not None else 0.0
        g_loss_ema = self._mgda_gradient_loss_ema if self._mgda_gradient_loss_ema is not None else 0.0

        if not self._mgda_diag_initialized:
            try:
                with open(self._mgda_diag_path, "w") as f:
                    f.write("epoch,alpha,alpha_raw,cos_sim,e_loss_ema,g_loss_ema\n")
                self._mgda_diag_initialized = True
            except OSError as e:
                logging.warning("Could not initialize MGDA diag CSV: {}".format(e))

        try:
            with open(self._mgda_diag_path, "a") as f:
                f.write("{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(
                    epoch, alpha, alpha_raw, cos_sim, e_loss_ema, g_loss_ema
                ))
        except OSError as e:
            logging.warning("Could not append to MGDA diag CSV: {}".format(e))
