import logging
import timeit

import torch

from batching import FullOverlapSampler, MultiBatchSampler, DistributedFullOverlapSampler
from distributed import is_main_process, reduce_mean, reduce_mae, reduce_rmse, reduce_min
from .base import DEVICE, HjmshiLBFGS, HjmshiFullBatchLBFGS


class MultibatchMixin:
    """Full-overlap / multi-batch L-BFGS training.

    Shared by EnergyTrainer and GradientTrainer (multi-batch is TYPE=ENERGY only;
    the USE_GRADIENTS branches inside these methods handle the energy+gradient case
    for GradientTrainer). Relies on the host trainer for compute_gradients_*,
    loss_fn, model, optimizer, and the cfg_* attributes.
    """
    def _build_multibatch_optimizer(self):
        mode = self.cfg_batch['MODE']
        lr = float(self.cfg_batch['LR'])
        history_size = int(self.cfg_batch['HISTORY_SIZE'])

        if mode == 'multi_batch':
            line_search = 'None'   # fixed steplength; Powell damping handles curvature.
            opt_cls = HjmshiLBFGS
        else:
            line_search = self.cfg_batch['LINE_SEARCH']
            opt_cls = HjmshiFullBatchLBFGS

        opt = opt_cls(
            self.model.parameters(),
            lr=lr,
            history_size=history_size,
            line_search=line_search,
            debug=False,
        )
        self._log(
            "Built multi-batch LBFGS: mode={} lr={} history_size={} line_search={}".format(
                mode, lr, history_size, line_search
            )
        )
        return opt
    def _init_multibatch_sampler(self):
        n = self.train.X.shape[0]
        B = int(self.cfg_batch['BATCH_SIZE'])
        seed = int(self.cfg_batch['SEED'])
        mode = self.cfg_batch['MODE']
        if mode == 'multi_batch':
            self.sampler = MultiBatchSampler(
                n_samples=n,
                batch_size=B,
                overlap_fraction=float(self.cfg_batch['OVERLAP_FRACTION']),
                seed=seed,
            )
        elif self.world_size > 1:
            # Distributed mode: each rank gets a slice of the batch
            self.sampler = DistributedFullOverlapSampler(
                n_samples=n,
                batch_size=B,
                rank=self.rank,
                world_size=self.world_size,
                seed=seed,
            )
        else:
            # Single GPU mode
            self.sampler = FullOverlapSampler(
                n_samples=n,
                batch_size=B,
                seed=seed,
            )
        if is_main_process():
            dist_info = f" (distributed: {self.world_size} ranks)" if self.world_size > 1 else ""
            logging.info(
                "Initialized sampler: mode={} N={} batch_size={} steps/epoch={}{}".format(
                    mode, n, B, self.sampler.steps_per_epoch(), dist_info
                )
            )
    def _gather_batch(self, idx):
        """Move one batch of (X, y[, dX, dy]) to DEVICE. Returns a plain dict."""
        use_grad = self.cfg_loss['USE_GRADIENTS']
        non_blocking = torch.cuda.is_available()

        X_cpu = self.train.X[idx]
        y_cpu = self.train.y[idx]
        X = X_cpu.to(self.device, non_blocking=non_blocking)
        y = y_cpu.to(self.device, non_blocking=non_blocking)

        batch = {'X': X, 'y': y}
        if use_grad:
            dX_cpu = self.train.dX[idx]
            dy_cpu = self.train.dy[idx]
            batch['dX'] = dX_cpu.to(self.device, non_blocking=non_blocking)
            batch['dy'] = dy_cpu.to(self.device, non_blocking=non_blocking)
        return batch
    def _loss_and_flat_grad(self, batch):
        """Forward + backward on one batch; returns (loss_tensor, flat_grad)."""
        self.optimizer.zero_grad()

        if self.cfg_loss['USE_GRADIENTS']:
            X = batch['X'].clone()
            X.requires_grad = True
            y_pred = self.model(X)
            dy_pred = self.compute_gradients_from_energy(X, batch['dX'], y_pred)
            loss = self.loss_fn(batch['y'], y_pred, batch['dy'], dy_pred)
        else:
            y_pred = self.model(batch['X'])
            loss = self.loss_fn(batch['y'], y_pred)

        if self.regularization is not None:
            loss = loss + self.regularization(self.model)

        loss.backward()

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        flat_grad = self.optimizer._gather_flat_grad()
        return loss, flat_grad
    def _make_closure(self, batch):
        """Factory closure for Wolfe/Armijo line search.

        Returns a callable with no arguments that recomputes the objective on
        the *same* batch each call -- hjmshi's LBFGS expects the closure to
        return a scalar tensor (no backward inside).
        """
        def closure():
            self.optimizer.zero_grad()
            if self.cfg_loss['USE_GRADIENTS']:
                X = batch['X'].clone()
                X.requires_grad = True
                y_pred = self.model(X)
                dy_pred = self.compute_gradients_from_energy(X, batch['dX'], y_pred)
                loss = self.loss_fn(batch['y'], y_pred, batch['dy'], dy_pred)
            else:
                y_pred = self.model(batch['X'])
                loss = self.loss_fn(batch['y'], y_pred)
            if self.regularization is not None:
                loss = loss + self.regularization(self.model)
            return loss
        return closure
    def train_epoch_multibatch(self, epoch, optimizer):
        self.model.train()
        mode = self.cfg_batch['MODE']
        steps = self.sampler.steps_per_epoch()

        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)

        start_time = timeit.default_timer()

        if mode == 'multi_batch':
            alpha = float(self.cfg_batch['OVERLAP_FRACTION'])
            damping = bool(self.cfg_batch['DAMPING'])
            damping_eps = float(self.cfg_batch['DAMPING_EPS'])

            Ok_prev_idx = self.sampler.current_prev_overlap()
            batch_Ok_prev = self._gather_batch(Ok_prev_idx)
            _, g_Ok_prev = self._loss_and_flat_grad(batch_Ok_prev)

            last_loss = None
            for step in range(steps):
                Ok_idx, Nk_idx = self.sampler.next_step()

                batch_Ok = self._gather_batch(Ok_idx)
                loss_Ok, g_Ok = self._loss_and_flat_grad(batch_Ok)

                batch_Nk = self._gather_batch(Nk_idx)
                _, g_Nk = self._loss_and_flat_grad(batch_Nk)

                g_Sk = alpha * (g_Ok_prev + g_Ok) + (1.0 - 2.0 * alpha) * g_Nk

                p = optimizer.two_loop_recursion(-g_Sk)
                lr_used = optimizer.step(p, g_Ok, g_Sk=g_Sk)

                # Recompute Ok gradient at the new iterate for curvature pair.
                batch_Ok_new = self._gather_batch(Ok_idx)
                _, g_Ok_new = self._loss_and_flat_grad(batch_Ok_new)
                optimizer.curvature_update(g_Ok_new, eps=damping_eps, damping=damping)

                # Shift: this step's Ok becomes next step's "Ok_prev".
                self.sampler.advance(Ok_idx)
                g_Ok_prev = g_Ok_new
                last_loss = loss_Ok.detach()

            self._log(
                "Epoch {} multi_batch: {} steps, lr_last={}, loss_Ok_last={:.6e}".format(
                    epoch, steps, lr_used, float(last_loss) if last_loss is not None else float('nan')
                )
            )

        else:  # full_overlap
            last_loss = None
            max_iter = self.cfg_solver['OPTIMIZER'].get('MAX_ITER', 100)
            debug_timing = self.cfg_debug.get('TIMING', False)

            # Timing accumulators (only used if debug_timing)
            if debug_timing:
                t_gather = t_fwd_bwd = t_sync = t_optim = 0.0
                total_inner_iters = 0

            for step in range(steps):
                if debug_timing:
                    _t0 = timeit.default_timer()

                (Sk_idx,) = self.sampler.next_step()
                batch_Sk = self._gather_batch(Sk_idx)

                if debug_timing:
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    t_gather += timeit.default_timer() - _t0

                def closure():
                    optimizer.zero_grad()
                    if self.cfg_loss['USE_GRADIENTS']:
                        X = batch_Sk['X'].clone()
                        X.requires_grad = True
                        y_pred = self.model(X)
                        dy_pred = self.compute_gradients_from_energy(X, batch_Sk['dX'], y_pred)
                        loss = self.loss_fn(batch_Sk['y'], y_pred, batch_Sk['dy'], dy_pred)
                    else:
                        y_pred = self.model(batch_Sk['X'])
                        loss = self.loss_fn(batch_Sk['y'], y_pred)
                    if self.regularization is not None:
                        loss = loss + self.regularization(self.model)
                    return loss

                # Sync function for distributed: average loss across ranks after backward
                loss_sync_fn = reduce_mean if self.world_size > 1 else None

                # Pre-compute loss & gradient at the current iterate before the inner loop
                if debug_timing:
                    _t0 = timeit.default_timer()

                optimizer.zero_grad()
                loss = closure()
                loss.backward()

                if debug_timing:
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    t_fwd_bwd += timeit.default_timer() - _t0

                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

                if debug_timing:
                    _t0 = timeit.default_timer()

                if loss_sync_fn is not None:
                    loss = loss_sync_fn(loss.detach())

                if debug_timing:
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    t_sync += timeit.default_timer() - _t0

                options = {
                    'closure': closure,
                    'current_loss': loss,
                    'grad_clip_norm': self.grad_clip_norm,
                    'loss_sync_fn': loss_sync_fn,
                }

                for inner in range(max_iter):
                    if debug_timing:
                        _t0 = timeit.default_timer()

                    obj, grad_new, t, ls_step, closure_eval, grad_eval, desc_dir, fail = optimizer.step(options=options)

                    if debug_timing:
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        t_optim += timeit.default_timer() - _t0
                        total_inner_iters += 1

                    last_loss = obj.detach() if hasattr(obj, 'detach') else torch.as_tensor(obj)

                    # Stop early if line search failed or step size is zero
                    if fail or t == 0:
                        break

                    # Recompute gradient for next inner iteration
                    if debug_timing:
                        _t0 = timeit.default_timer()

                    optimizer.zero_grad()
                    loss = closure()
                    loss.backward()

                    if debug_timing:
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        t_fwd_bwd += timeit.default_timer() - _t0

                    if self.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

                    if debug_timing:
                        _t0 = timeit.default_timer()

                    if loss_sync_fn is not None:
                        loss = loss_sync_fn(loss.detach())

                    if debug_timing:
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        t_sync += timeit.default_timer() - _t0

                    options['current_loss'] = loss

            if debug_timing:
                self._log(
                    "Epoch {} full_overlap: {} steps, {} inner_iters, loss={:.6e}".format(
                        epoch, steps, total_inner_iters, float(last_loss) if last_loss is not None else float('nan')
                    )
                )
                self._log(
                    "  Timing: gather={:.2f}s fwd_bwd={:.2f}s sync={:.2f}s optim={:.2f}s".format(
                        t_gather, t_fwd_bwd, t_sync, t_optim
                    )
                )
            else:
                self._log(
                    "Epoch {} full_overlap: {} steps, loss_Sk_last={:.6e}".format(
                        epoch, steps, float(last_loss) if last_loss is not None else float('nan')
                    )
                )

        elapsed = timeit.default_timer() - start_time
        self._log("Epoch {} multibatch step time: {:.2f}s".format(epoch, elapsed))

        self.model.eval()
        with torch.no_grad():
            if self.cfg_loss['USE_GRADIENTS']:
                # Compute predictions for both train and val
                train_y_pred, train_dy_pred = self.compute_gradients_eval(self.train)
                val_y_pred, val_dy_pred = self.compute_gradients_eval(self.val)

                # Energy metrics - use reduce_rmse/reduce_mae for correct distributed aggregation
                train_e_d    = self.loss_fn.descale_energies(self.train.y)
                train_e_pred = self.loss_fn.descale_energies(train_y_pred)
                train_e_errors = (train_e_d - train_e_pred).view(-1)
                val_e_rmse_local = torch.sqrt(torch.mean(train_e_errors ** 2)).item()
                train_e_mae  = reduce_mae(train_e_errors)
                train_e_rmse = reduce_rmse(train_e_errors)

                val_e_d    = self.loss_fn.descale_energies(self.val.y)
                val_e_pred = self.loss_fn.descale_energies(val_y_pred)
                val_e_errors = (val_e_d - val_e_pred).view(-1)
                val_e_rmse_local = torch.sqrt(torch.mean(val_e_errors ** 2)).item()
                val_e_mae  = reduce_mae(val_e_errors)
                val_e_rmse = reduce_rmse(val_e_errors)

                # Gradient metrics (per-component errors)
                natoms = self.train.NATOMS
                train_dy = self.train.dy.reshape(-1, 3 * natoms)
                val_dy   = self.val.dy.reshape(-1, 3 * natoms)
                train_g_errors = (train_dy - train_dy_pred).view(-1)
                val_g_errors   = (val_dy - val_dy_pred).view(-1)
                train_g_mae  = reduce_mae(train_g_errors)
                val_g_mae    = reduce_mae(val_g_errors)
                train_g_rmse = reduce_rmse(train_g_errors)
                val_g_rmse   = reduce_rmse(val_g_errors)

                # Weighted MSE for scheduler
                # Sync minimum across ranks for consistent weighting in distributed mode
                enmin_train = reduce_min(train_e_d.min())
                w_train = self.loss_fn.dwt / (self.loss_fn.dwt + train_e_d - enmin_train)
                loss_train_e = (w_train.view(-1) * (train_e_d - train_e_pred).view(-1)**2).mean()

                enmin_val = reduce_min(val_e_d.min())
                w_val = self.loss_fn.dwt / (self.loss_fn.dwt + val_e_d - enmin_val)
                loss_val_e = (w_val.view(-1) * (val_e_d - val_e_pred).view(-1)**2).mean()

                if self.world_size > 1:
                    # Multi-batch mode doesn't use trust region
                    self.log_distributed_diagnostics(
                        epoch,
                        loss_local=loss_val_e.item(),
                        e_rmse_local=val_e_rmse_local,
                        use_trust_region=False
                    )
                    # MAE/RMSE already aggregated above via reduce_mae/reduce_rmse
                    loss_train_e = reduce_mean(loss_train_e)
                    loss_val_e   = reduce_mean(loss_val_e)

                self._log("Epoch: {}; (energy) WMSE train: {:.3f}; (energy) WMSE val: {:.3f}\n \
                                           (energy) MAE train:  {:.3f} cm-1; (gradient) MAE train:  {:.3f} cm-1/bohr\n \
                                           (energy) MAE val:    {:.3f} cm-1; (gradient) MAE val:    {:.3f} cm-1/bohr\n \
                                           (energy) RMSE train: {:.3f} cm-1; (gradient) RMSE train: {:.3f} cm-1/bohr\n \
                                           (energy) RMSE val:   {:.3f} cm-1; (gradient) RMSE val:   {:.3f} cm-1/bohr".format(
                    epoch, loss_train_e, loss_val_e, train_e_mae, train_g_mae, val_e_mae, val_g_mae, train_e_rmse, train_g_rmse, val_e_rmse, val_g_rmse
                ))

                loss_val = loss_val_e

                if self.writer is not None:
                    self.writer.add_scalar("loss/train", loss_train_e, epoch)
            else:
                val_y_pred = self.model(self.val.X)
                loss_val = self.loss_fn(self.val.y, val_y_pred)
                if self.world_size > 1:
                    loss_val = reduce_mean(loss_val)
                self._log("Epoch: {}; loss val: {:.3f} cm-1".format(epoch, loss_val))

        self.loss_val = loss_val
        current_lr = optimizer.param_groups[0]['lr']
        if self.writer is not None:
            self.writer.add_scalar("loss/val", loss_val, epoch)
            self.writer.add_scalar("lr", current_lr, epoch)
