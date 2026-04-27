USE_WANDB = False
import os
import logging
import time
import timeit

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from distributed import is_main_process, reduce_mean, reduce_mae, reduce_rmse, reduce_min, reduce_sum, sync_gradients
from data_io import save_checkpoint
from losses import WMSELoss_TrustRegion_wgradients

import sys
import pathlib
BASEDIR = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(BASEDIR / "vendor"))
from pytorch_lbfgs import LBFGS as HjmshiLBFGS, FullBatchLBFGS as HjmshiFullBatchLBFGS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRINT_TRAINING_STEPS = 1
PRINT_PRECISION = 3


def flatten_gradients(model):
    """Flatten all parameter gradients into a single 1D tensor."""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
        else:
            grads.append(torch.zeros_like(p).view(-1))
    return torch.cat(grads)


def set_gradients(model, flat_grad):
    """Set model parameter gradients from a flattened 1D tensor."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.grad = flat_grad[offset:offset + numel].view_as(p)
        offset += numel


def compute_mgda_alpha(g_energy, g_gradient, alpha_min=0.0, alpha_max=1.0,
                       energy_loss=None, gradient_loss=None,
                       ema_energy_loss=None, ema_gradient_loss=None):
    """
    Compute optimal convex combination weight using GradNorm principles.
    """
    eps = 1e-12

    # Normalize gradients to unit vectors (GradNorm key idea #1)
    norm_e = torch.norm(g_energy) + eps
    norm_g = torch.norm(g_gradient) + eps
    g_energy_norm = g_energy / norm_e
    g_gradient_norm = g_gradient / norm_g

    # Cosine similarity between normalized gradients
    cos_sim = torch.dot(g_energy_norm, g_gradient_norm)

    # Adaptive alpha based on loss ratios (GradNorm key idea #2)
    if (energy_loss is not None and gradient_loss is not None and
        ema_energy_loss is not None and ema_gradient_loss is not None and
        ema_energy_loss > eps and ema_gradient_loss > eps):

        # Relative loss: current / EMA (>1 means task is falling behind)
        rel_energy = energy_loss / ema_energy_loss
        rel_gradient = gradient_loss / ema_gradient_loss

        # Convert to tensors if needed
        if not isinstance(rel_energy, torch.Tensor):
            rel_energy = torch.tensor(rel_energy, device=g_energy.device)
        if not isinstance(rel_gradient, torch.Tensor):
            rel_gradient = torch.tensor(rel_gradient, device=g_gradient.device)

        alpha = rel_gradient / (rel_energy + rel_gradient + eps)
    else:
        # Fallback: equal weighting when no loss history available
        alpha = torch.tensor(0.5, device=g_energy.device)

    # Clamp to bounds
    alpha = torch.clamp(alpha, alpha_min, alpha_max)

    # Combine NORMALIZED gradients (key difference from original MGDA)
    g_combined = alpha * g_energy_norm + (1 - alpha) * g_gradient_norm

    # CRITICAL: Rescale combined gradient to prevent near-cancellation
    combined_norm = torch.norm(g_combined) + eps
    target_norm = 0.5 * (norm_e + norm_g)  # Average of original gradient norms

    # Cap the rescaling factor to prevent Inf/NaN when gradients nearly cancel
    rescale_factor = torch.clamp(target_norm / combined_norm, max=10.0)
    g_combined = g_combined * rescale_factor

    return alpha, cos_sim, g_combined

class TrainingLoopMixin:
    def train_model(self):
        try:

            # Set device based on mode
            if self.world_size > 1:
                self.device = torch.device(f"cuda:{self.local_rank}")
            else:
                self.device = DEVICE

            self.model = self.model.to(self.device)

            if self.cfg_solver.get('TORCH_COMPILE', False):
                self.model = torch.compile(self.model, mode='reduce-overhead')
                if is_main_process():
                    logging.info("Model compiled with torch.compile(mode='reduce-overhead')")

            # Wrap with DDP for distributed training (except for LBFGS which uses
            # explicit gradient sync to avoid race conditions with line search)
            opt_name = self.cfg_solver['OPTIMIZER']['NAME']
            if self.world_size > 1 and opt_name != 'LBFGS':
                self.model = DDP(self.model, device_ids=[self.local_rank])
                if is_main_process():
                    logging.info(f"Distributed training enabled: {self.world_size} GPUs (DDP)")
            elif self.world_size > 1:
                if is_main_process():
                    logging.info(f"Distributed training enabled: {self.world_size} GPUs (explicit gradient sync, no DDP)")

            # Initialize TensorBoard only on rank 0 to avoid event-file corruption.
            if is_main_process():
                log_dir = os.path.join("runs", self.model_name)
                self.writer = SummaryWriter(log_dir=log_dir)
            else:
                self.writer = None

            # nn.Module.to() moves parameters and buffers, but our loss classes
            # store plain Tensor attributes (e.g. self.dwt). Move them explicitly.
            def _move_plain_tensors(mod, device):
                for k, v in mod.__dict__.items():
                    if isinstance(v, torch.Tensor) and not isinstance(v, torch.nn.Module):
                        setattr(mod, k, v.to(device))
            _move_plain_tensors(self.loss_fn, self.device)
            if self.regularization is not None:
                _move_plain_tensors(self.regularization, self.device)

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

            self.loss_fn = self.loss_fn.to(self.device)

            if self.cfg['TYPE'] == 'DIPOLE':
                self.train.grm = self.train.grm.to(self.device)
                self.val.grm   = self.val.grm.to(self.device)

            if self.cfg['TYPE'] == 'DIPOLEQ':
                self.train.xyz_ordered = self.train.xyz_ordered.to(self.device)
                self.val.xyz_ordered = self.val.xyz_ordered.to(self.device)
                self.test.xyz_ordered = self.test.xyz_ordered.to(self.device)

            if self.train.dX is not None and not multibatch:
                self.train.dX = self.train.dX.to(self.device)
                self.train.dy = self.train.dy.to(self.device)

                self.val.dX = self.val.dX.to(self.device)
                self.val.dy = self.val.dy.to(self.device)
            elif self.train.dX is not None and multibatch:
                # Only move validation gradient tensors; train stays on pinned CPU.
                self.val.dX = self.val.dX.to(self.device)
                self.val.dy = self.val.dy.to(self.device)


            if multibatch:
                self.optimizer = self._build_multibatch_optimizer()
                self._init_multibatch_sampler()
            else:
                self.optimizer = self.build_optimizer(self.cfg_solver['OPTIMIZER'])
            self.scheduler = self.build_scheduler()

            start = time.time()

            MAX_EPOCHS = self.cfg_solver['MAX_EPOCHS']

            for epoch in range(MAX_EPOCHS):
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

                self._log("loss function: {}".format(self.loss_fn))

                if bool(self.cfg_batch.get('MULTIBATCH_ENABLED', False)):
                    self.train_epoch_multibatch(epoch, self.optimizer)
                else:
                    self.train_epoch(epoch, self.optimizer)

                # Step scheduler - ReduceLROnPlateau requires metric, CosineAnnealing does not
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(self.loss_val)
                else:
                    self.scheduler.step()

                if epoch % PRINT_TRAINING_STEPS == 0:
                    end = time.time()
                    self._log("Elapsed time: {:.0f}s\n".format(end - start))

                # writing all pending events to disk
                if self.writer is not None:
                    self.writer.flush()

                # pass loss values to EarlyStopping mechanism 
                self.es(epoch, self.loss_val, self.model, self.xscaler, self.yscaler, meta_info=self.meta_info)

                if self.es.status:
                    self._log("Invoking early stop.")
                    break

            if self.loss_val < self.es.best_score:
                save_checkpoint(self.model, self.xscaler, self.yscaler, self.meta_info, self.chk_path)

            self._log("\nReloading best model from the last checkpoint")

            self.reset_weights()
            checkpoint = torch.load(self.chk_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])

            if is_main_process() and getattr(self, 'writer', None) is not None:
                self.writer.close()
            return self.model
        except Exception:
            if is_main_process() and getattr(self, 'writer', None) is not None:
                self.writer.close()
            raise
    def train_epoch(self, epoch, optimizer):
        CLOSURE_CALL_COUNT = 0

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

        def _compute_loss(separate=False):
            """Compute training loss.

            Args:
                separate: If True and using gradients with trust region,
                         return (energy_loss, gradient_loss) tuple for MGDA.
                         Otherwise return combined loss.
            """
            if self.cfg_loss['USE_GRADIENTS']:
                if use_trust_region:
                    if n_in_trust > 0:
                        y_pred_subset = self.model(X_subset)
                        train_dy_pred_subset = self.compute_gradients_from_energy(
                            X_subset, dX_subset, y_pred_subset
                        )
                        train_y_pred = self.model(self.train.X)
                        if separate:
                            energy_loss, gradient_loss = self.loss_fn.forward_separate(
                                self.train.y, train_y_pred,
                                train_dy_subset, train_dy_pred_subset,
                                trust_indices, gradient_weights
                            )
                            # Add regularization to energy loss (it's model complexity, not gradient fitting)
                            if self.regularization is not None:
                                energy_loss = energy_loss + self.regularization(self.model)
                            return energy_loss, gradient_loss
                        else:
                            loss = self.loss_fn(
                                self.train.y, train_y_pred,
                                train_dy_subset, train_dy_pred_subset,
                                trust_indices, gradient_weights
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

            elif self.cfg['TYPE'] == 'DIPOLE':
                y_pred = self.model(self.train.X)
                dip_pred = torch.einsum('ijk,ik->ij', self.train.grm, y_pred)
                loss = self.loss_fn(self.train.y, dip_pred)

            elif self.cfg['TYPE'] == 'DIPOLEQ':
                q_pred   = self.model(self.train.X)
                X_inf    = torch.zeros_like(self.train.X).cpu()
                X_inf_tr = torch.from_numpy(self.xscaler.transform(X_inf)).to(self.device)
                q_inf    = self.model(X_inf_tr)
                q_corr   = q_pred - q_inf
                dip_pred = torch.einsum('ijk,ij->ik', self.train.xyz_ordered.to(TORCH_FLOAT), q_corr)
                qsum     = torch.sum(q_corr, dim=1)
                qreg     = self.cfg_loss['LAMBDA_Q'] * torch.mean(qsum * qsum)
                loss     = self.loss_fn(self.train.y, dip_pred)
                loss = loss + qreg

            elif self.cfg['TYPE'] == 'DIPOLEC':
                dip_pred = self.model(self.train.X)
                loss = self.loss_fn(self.train.y, dip_pred)

            elif self.cfg['TYPE'] == 'ENERGY':
                y_pred = self.model(self.train.X)
                loss = self.loss_fn(self.train.y, y_pred)

            else:
                assert False, "unreachable"

            if self.regularization is not None:
                loss = loss + self.regularization(self.model)
            return loss

        def closure():
            nonlocal CLOSURE_CALL_COUNT
            CLOSURE_CALL_COUNT = CLOSURE_CALL_COUNT + 1
            optimizer.zero_grad()
            loss = _compute_loss()
            loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            # Synchronize loss across ranks so L-BFGS line search makes
            # identical decisions on every process.
            if self.world_size > 1:
                loss = reduce_mean(loss.detach())
            return loss

        def closure_no_backward():
            nonlocal CLOSURE_CALL_COUNT
            CLOSURE_CALL_COUNT = CLOSURE_CALL_COUNT + 1
            optimizer.zero_grad()
            loss = _compute_loss()
            return loss

        # MGDA (Multi-objective Gradient Descent Algorithm) closure
        # Computes optimal combination of energy and gradient loss gradients
        # MGDA is only active when gradients are currently being used
        use_mgda = (self.cfg_loss.get('USE_MGDA', False)
                    and self.cfg_loss['USE_GRADIENTS'])
        mgda_alpha_min = self.cfg_loss.get('MGDA_ALPHA_MIN', 0.1)
        mgda_alpha_max = self.cfg_loss.get('MGDA_ALPHA_MAX', 0.9)
        mgda_ema_decay = self.cfg_loss.get('MGDA_EMA_DECAY', 0.9)
        _mgda_alpha_raw = [None]  # Mutable container for closure
        _mgda_alpha = [None]
        _mgda_cos_sim = [None]

        def closure_mgda():
            """MGDA+GradNorm closure: normalized gradients + adaptive alpha from loss ratios."""
            nonlocal CLOSURE_CALL_COUNT
            CLOSURE_CALL_COUNT = CLOSURE_CALL_COUNT + 1
            optimizer.zero_grad()

            # Compute separate losses
            energy_loss, gradient_loss = _compute_loss(separate=True)

            # Update loss EMAs for adaptive alpha computation
            energy_loss_val = energy_loss.detach().item()
            gradient_loss_val = gradient_loss.detach().item()

            if self._mgda_energy_loss_ema is None:
                self._mgda_energy_loss_ema = energy_loss_val
                self._mgda_gradient_loss_ema = gradient_loss_val
            else:
                self._mgda_energy_loss_ema = (mgda_ema_decay * self._mgda_energy_loss_ema +
                                              (1 - mgda_ema_decay) * energy_loss_val)
                self._mgda_gradient_loss_ema = (mgda_ema_decay * self._mgda_gradient_loss_ema +
                                                (1 - mgda_ema_decay) * gradient_loss_val)

            # Backward pass for energy gradient
            energy_loss.backward(retain_graph=True)
            g_energy = flatten_gradients(self.model)

            # Backward pass for gradient loss gradient
            optimizer.zero_grad()
            gradient_loss.backward()
            g_gradient = flatten_gradients(self.model)

            # Sync gradients across ranks before computing weights
            if self.world_size > 1:
                dist.all_reduce(g_energy, op=dist.ReduceOp.SUM)
                g_energy = g_energy / self.world_size
                dist.all_reduce(g_gradient, op=dist.ReduceOp.SUM)
                g_gradient = g_gradient / self.world_size

            # Compute GradNorm weights: normalized gradients + adaptive alpha from loss ratios
            alpha_raw, cos_sim, combined_grad = compute_mgda_alpha(
                g_energy, g_gradient,
                mgda_alpha_min, mgda_alpha_max,
                energy_loss=energy_loss_val,
                gradient_loss=gradient_loss_val,
                ema_energy_loss=self._mgda_energy_loss_ema,
                ema_gradient_loss=self._mgda_gradient_loss_ema
            )

            # EMA smoothing of alpha to prevent oscillation
            if self._mgda_alpha_ema is None:
                alpha = alpha_raw
                self._mgda_alpha_ema = alpha.item()
            else:
                alpha = mgda_ema_decay * self._mgda_alpha_ema + (1 - mgda_ema_decay) * alpha_raw.item()
                self._mgda_alpha_ema = alpha
                alpha = torch.tensor(alpha, device=g_energy.device)

            # Store for diagnostics
            _mgda_alpha_raw[0] = alpha_raw.item()
            _mgda_alpha[0] = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
            _mgda_cos_sim[0] = cos_sim.item()

            # Set the combined normalized gradient
            set_gradients(self.model, combined_grad)

            # Gradient clipping on combined gradient
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            # Return combined loss for L-BFGS line search
            combined_loss = alpha * energy_loss.detach() + (1 - alpha) * gradient_loss.detach()
            if self.world_size > 1:
                combined_loss = reduce_mean(combined_loss)
            return combined_loss

        def closure_mgda_no_backward():
            """MGDA closure for line search (no backward needed)."""
            nonlocal CLOSURE_CALL_COUNT
            CLOSURE_CALL_COUNT = CLOSURE_CALL_COUNT + 1
            optimizer.zero_grad()
            energy_loss, gradient_loss = _compute_loss(separate=True)
            # Use current EMA alpha for consistent loss evaluation
            alpha = self._mgda_alpha_ema if self._mgda_alpha_ema is not None else 0.5
            combined_loss = alpha * energy_loss.detach() + (1 - alpha) * gradient_loss.detach()
            # Sync loss across ranks for consistent line search
            if self.world_size > 1:
                combined_loss = reduce_mean(combined_loss)
            return combined_loss

        # Calling model.train() will change the behavior of some layers such as nn.Dropout and nn.BatchNormXd
        self.model.train()

        # Reset focal weighting flag to allow one error_scale update per optimizer step
        # (prevents non-deterministic loss during LBFGS line search)
        if hasattr(self.loss_fn, 'reset_error_scale_flag'):
            self.loss_fn.reset_error_scale_flag()

        start_time = timeit.default_timer()
        if isinstance(optimizer, HjmshiFullBatchLBFGS):
            # Vendored FullBatchLBFGS for distributed training.
            if use_mgda:
                # MGDA mode: use MGDA closures that compute optimal gradient combination
                logging.debug(f"[rank {self.rank}] vendored LBFGS (MGDA): initial closure_mgda")
                loss = closure_mgda()  # This sets gradients via MGDA
                # Note: closure_mgda already syncs gradients and applies clipping
                # Build grad_sync closure that captures self.model
                def _grad_sync():
                    sync_gradients(self.model)
                options = {
                    'closure': closure_mgda_no_backward,
                    'current_loss': loss,
                    'grad_clip_norm': None,  # Already applied in closure_mgda
                    'loss_sync_fn': reduce_mean if self.world_size > 1 else None,
                    'grad_sync_fn': _grad_sync if self.world_size > 1 else None,
                }
            else:
                # Standard mode
                # Pre-compute loss & gradient at the current iterate.
                logging.debug(f"[rank {self.rank}] vendored LBFGS: zero_grad")
                optimizer.zero_grad()
                logging.debug(f"[rank {self.rank}] vendored LBFGS: closure_no_backward")
                loss = closure_no_backward()
                logging.debug(f"[rank {self.rank}] vendored LBFGS: backward (loss={loss.item():.4f})")
                loss.backward()
                # Explicit gradient sync - don't rely on DDP's implicit async sync
                if self.world_size > 1:
                    logging.debug(f"[rank {self.rank}] vendored LBFGS: sync_gradients START")
                    sync_gradients(self.model)
                    logging.debug(f"[rank {self.rank}] vendored LBFGS: sync_gradients DONE")
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                if self.world_size > 1:
                    logging.debug(f"[rank {self.rank}] vendored LBFGS: reduce_mean START")
                    loss = reduce_mean(loss.detach())
                    logging.debug(f"[rank {self.rank}] vendored LBFGS: reduce_mean DONE")
                # Build grad_sync closure that captures self.model
                def _grad_sync():
                    sync_gradients(self.model)
                options = {
                    'closure': closure_no_backward,
                    'current_loss': loss,
                    'grad_clip_norm': self.grad_clip_norm,
                    'loss_sync_fn': reduce_mean if self.world_size > 1 else None,
                    'grad_sync_fn': _grad_sync if self.world_size > 1 else None,
                }
            # Add line search options from config
            if hasattr(self, '_lbfgs_ls_options'):
                options.update(self._lbfgs_ls_options)
            obj, grad_new, t, ls_step, closure_eval, grad_eval, desc_dir, fail = optimizer.step(options=options)
            self._last_vendored_closure_eval = closure_eval + 1  # +1 for the initial evaluation above
            CLOSURE_CALL_COUNT = self._last_vendored_closure_eval
            elapsed = timeit.default_timer() - start_time
            self._log("Optimizer makes step in {:.2f}s".format(elapsed))
            self._log("CLOSURE_CALL_COUNT = {}".format(CLOSURE_CALL_COUNT))
        else:
            # Non-vendored optimizer (e.g., torch.optim.LBFGS)
            if use_mgda:
                optimizer.step(closure_mgda)
            else:
                optimizer.step(closure)
            elapsed = timeit.default_timer() - start_time
            self._log("Optimizer makes step in {:.2f}s".format(elapsed))
            self._log("CLOSURE_CALL_COUNT = {}".format(CLOSURE_CALL_COUNT))

        current_lr = optimizer.param_groups[0]['lr']
        self._log("(optimizer) current lr: {}".format(current_lr))

        # LBFGS line-search telemetry (no-op for non-LBFGS optimizers).
        self.log_lbfgs_diagnostics(epoch, optimizer)

        # MGDA diagnostics logging
        if use_mgda and _mgda_alpha[0] is not None:
            self._log("(MGDA) alpha={:.4f} (raw={:.4f}), cos_sim={:.4f}".format(
                _mgda_alpha[0], _mgda_alpha_raw[0], _mgda_cos_sim[0]))
            self.log_mgda_diagnostics(epoch, _mgda_alpha[0], _mgda_alpha_raw[0], _mgda_cos_sim[0])

        # Calling model.eval() will change the behavior of some layers, 
        # such as nn.Dropout, which will be disabled, and nn.BatchNormXd, which will use the running stats during evaluation.
        self.model.eval()

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
            if use_trust_region and trust_indices is not None and n_in_trust > 0:
                self.log_gradient_loss_diagnostics(
                    epoch, train_dy, train_dy_pred,
                    train_e_d, train_e_pred,
                    trust_indices, gradient_weights,
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
                    n_trust_local=n_in_trust if use_trust_region else None,
                    n_total_local=len(self.train.X) if use_trust_region else None,
                    use_trust_region=use_trust_region
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


        elif self.cfg['TYPE'] == 'DIPOLE':
            with torch.no_grad():
                train_y_pred   = self.model(self.train.X)
                dip_pred_train = torch.einsum('ijk,ik->ij', self.train.grm, train_y_pred)
                loss_train     = self.loss_fn(self.train.y, dip_pred_train)

                val_y_pred   = self.model(self.val.X)
                dip_pred_val = torch.einsum('ijk,ik->ij', self.val.grm, val_y_pred)
                loss_val     = self.loss_fn(self.val.y, dip_pred_val)

                if self.world_size > 1:
                    # DIPOLE mode: use loss as proxy for RMSE, no trust region
                    self.log_distributed_diagnostics(
                        epoch,
                        loss_local=loss_val.item(),
                        e_rmse_local=loss_val.item(),  # Use loss as proxy
                        use_trust_region=False
                    )
                    loss_train = reduce_mean(loss_train)
                    loss_val   = reduce_mean(loss_val)

                # value to be passed to EarlyStopping/ReduceLR mechanisms
                self.loss_val = loss_val

            # log metrics to WANDB to visualize model performance
            if is_main_process() and USE_WANDB:
                wandb.log({"loss_train": loss_train, "loss_val": loss_val})

            self._log("Epoch: {0}; loss train: {2:.{1}f}; loss val: {3:.{1}f}".format(epoch, PRINT_PRECISION, loss_train, loss_val))

        elif self.cfg['TYPE'] == 'DIPOLEQ':
            # To disable the gradient calculation, set the .requires_grad attribute of all parameters to False 
            # or wrap the forward pass into with torch.no_grad().
            with torch.no_grad():
                train_q_pred   = self.model(self.train.X)
                train_X_inf    = torch.zeros_like(self.train.X).cpu()
                train_X_inf_tr = torch.from_numpy(self.xscaler.transform(train_X_inf)).to(self.device)
                train_q_inf    = self.model(train_X_inf_tr)
                train_q_corr   = train_q_pred - train_q_inf
                dip_pred_train = torch.einsum('ijk,ij->ik', self.train.xyz_ordered.to(TORCH_FLOAT), train_q_corr)
                loss_train     = self.loss_fn(self.train.y, dip_pred_train)

                val_q_pred   = self.model(self.val.X)
                val_X_inf    = torch.zeros_like(self.val.X).cpu()
                val_X_inf_tr = torch.from_numpy(self.xscaler.transform(val_X_inf)).to(self.device)
                val_q_inf    = self.model(val_X_inf_tr)
                val_q_corr   = val_q_pred - val_q_inf
                dip_pred_val = torch.einsum('ijk,ij->ik', self.val.xyz_ordered.to(TORCH_FLOAT), val_q_corr)
                loss_val     = self.loss_fn(self.val.y, dip_pred_val)

                train_qsum = torch.sum(train_q_corr, dim=1)
                train_qreg = self.cfg_loss['LAMBDA_Q'] * torch.mean(train_qsum * train_qsum)
                val_qsum   = torch.sum(val_q_corr, dim=1)
                val_qreg   = self.cfg_loss['LAMBDA_Q'] * torch.mean(val_qsum * val_qsum)

                if self.world_size > 1:
                    loss_train = reduce_mean(loss_train)
                    loss_val   = reduce_mean(loss_val)
                    train_qreg = reduce_mean(train_qreg)
                    val_qreg   = reduce_mean(val_qreg)

                # value to be passed to EarlyStopping/ReduceLR mechanisms
                self.loss_val = loss_val

            # log metrics to WANDB to visualize model performance
            if is_main_process() and USE_WANDB:
                wandb.log({"loss_train": loss_train, "loss_val": loss_val, "train_qreg": train_qreg, "val_qreg": val_qreg, "lr" : current_lr})

            self._log("Epoch: {0}; loss train: {2:.{1}f}; qreg train: {3:{1}f}; loss val: {4:.{1}f}; qreg val: {5:.{1}f}".format(
                epoch, PRINT_PRECISION, loss_train, train_qreg, loss_val, val_qreg
            ))

        elif self.cfg['TYPE'] == 'DIPOLEC':
            with torch.no_grad():
                train_dip_pred = self.model(self.train.X)
                loss_train = self.loss_fn(self.train.y, train_dip_pred)

                val_dip_pred = self.model(self.val.X)
                loss_val = self.loss_fn(self.val.y, val_dip_pred)

                if self.world_size > 1:
                    loss_train = reduce_mean(loss_train)
                    loss_val   = reduce_mean(loss_val)

                self.loss_val = loss_val

            self._log("Epoch: {0}; loss train: {2:.{1}f}; loss val: {3:.{1}f}".format(epoch, PRINT_PRECISION, loss_train, loss_val))

        elif self.cfg['TYPE'] == 'ENERGY':
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

        else:
            assert False, "unreachable"
