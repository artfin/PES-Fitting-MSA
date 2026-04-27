import logging

import torch
import torch.distributed as dist

from losses import WMSELoss_TrustRegion_wgradients
from distributed import reduce_sum, reduce_min, is_main_process, all_gather_scalar, barrier

import sys
import pathlib
BASEDIR = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(BASEDIR / "vendor"))
from pytorch_lbfgs import FullBatchLBFGS as HjmshiFullBatchLBFGS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrainingDiagnosticsMixin:
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
    def log_lbfgs_diagnostics(self, epoch, optimizer):
        """L-BFGS line-search telemetry, dumped per epoch.

        Pulls inner state from torch.optim.LBFGS or vendored FullBatchLBFGS:
          - this-step iteration / closure-call counts (deltas from cumulative)
          - last accepted step length t
          - initial Hessian diag scaling H_diag = (s . y) / (y . y)
          - curvature pair stats: <s_k, y_k> -- min/max/last/mean
            over the stored history (small or absent => degenerate curvature)
          - flat gradient norm at the last accepted iterate
        """
        if isinstance(optimizer, torch.optim.LBFGS):
            params = optimizer.param_groups[0]['params']
            if not params:
                return
            state = optimizer.state.get(params[0], {})
            if not state:
                return

            cum_n_iter     = int(state.get('n_iter', 0))
            cum_func_evals = int(state.get('func_evals', 0))
            iters_this_step = cum_n_iter - self._lbfgs_prev_n_iter
            evals_this_step = cum_func_evals - self._lbfgs_prev_func_evals
            self._lbfgs_prev_n_iter = cum_n_iter
            self._lbfgs_prev_func_evals = cum_func_evals

            t_val = state.get('t', None)
            try:
                t_val = float(t_val) if t_val is not None else float('nan')
            except (TypeError, ValueError):
                t_val = float('nan')

            H_diag = state.get('H_diag', None)
            try:
                H_diag = float(H_diag) if H_diag is not None else float('nan')
            except (TypeError, ValueError):
                H_diag = float('nan')

            ro = state.get('ro', []) or []
            n_pairs = len(ro)
            if n_pairs > 0:
                sy_vals = []
                for r in ro:
                    try:
                        rv = float(r)
                        if rv != 0.0:
                            sy_vals.append(1.0 / rv)
                    except (TypeError, ValueError, ZeroDivisionError):
                        pass
                if sy_vals:
                    sy_min  = min(sy_vals)
                    sy_max  = max(sy_vals)
                    sy_last = sy_vals[-1]
                    sy_mean = sum(sy_vals) / len(sy_vals)
                else:
                    sy_min = sy_max = sy_last = sy_mean = float('nan')
            else:
                sy_min = sy_max = sy_last = sy_mean = float('nan')

            prev_flat_grad = state.get('prev_flat_grad', None)
            if prev_flat_grad is not None:
                try:
                    grad_norm = float(prev_flat_grad.norm().item())
                except (RuntimeError, AttributeError):
                    grad_norm = float('nan')
            else:
                grad_norm = float('nan')

        elif isinstance(optimizer, HjmshiFullBatchLBFGS):
            state = optimizer.state['global_state']
            cum_n_iter = int(state.get('n_iter', 0))
            iters_this_step = cum_n_iter - self._lbfgs_prev_n_iter
            self._lbfgs_prev_n_iter = cum_n_iter
            # Closure evals are captured in train_epoch for vendored LBFGS
            evals_this_step = getattr(self, '_last_vendored_closure_eval', float('nan'))

            t_val = float(state.get('t', float('nan')))
            H_diag = float(state.get('H_diag', float('nan')))

            old_dirs = state.get('old_dirs', [])
            old_stps = state.get('old_stps', [])
            n_pairs = len(old_dirs)
            if n_pairs > 0:
                sy_vals = []
                for s, y in zip(old_stps, old_dirs):
                    try:
                        sy = float(s.dot(y).item())
                        if sy != 0.0:
                            sy_vals.append(sy)
                    except (TypeError, ValueError):
                        pass
                if sy_vals:
                    sy_min  = min(sy_vals)
                    sy_max  = max(sy_vals)
                    sy_last = sy_vals[-1]
                    sy_mean = sum(sy_vals) / len(sy_vals)
                else:
                    sy_min = sy_max = sy_last = sy_mean = float('nan')
            else:
                sy_min = sy_max = sy_last = sy_mean = float('nan')

            prev_flat_grad = state.get('prev_flat_grad', None)
            if prev_flat_grad is not None:
                try:
                    grad_norm = float(prev_flat_grad.norm().item())
                except (RuntimeError, AttributeError):
                    grad_norm = float('nan')
            else:
                grad_norm = float('nan')
        else:
            return

        if not self._lbfgs_diag_initialized:
            try:
                with open(self._lbfgs_diag_path, "w") as f:
                    f.write("epoch,iters_this_step,evals_this_step,t,H_diag,"
                            "n_pairs,grad_norm,sy_min,sy_mean,sy_max,sy_last\n")
                self._lbfgs_diag_initialized = True
            except OSError as e:
                logging.warning("Could not initialize lbfgs diag CSV: {}".format(e))
        try:
            with open(self._lbfgs_diag_path, "a") as f:
                f.write("{},{},{},{:.6e},{:.6e},{},{:.6e},"
                        "{:.6e},{:.6e},{:.6e},{:.6e}\n".format(
                    epoch, iters_this_step, evals_this_step, t_val, H_diag,
                    n_pairs, grad_norm, sy_min, sy_mean, sy_max, sy_last,
                ))
        except OSError as e:
            logging.warning("Could not append to lbfgs diag CSV: {}".format(e))

        if is_main_process():
            logging.info(
                "[lbfgs-diag] epoch={} | iters={} evals={} t={:.3e} H_diag={:.3e} "
                "pairs={} grad_norm={:.3e} sy(last/min/max)={:.3e}/{:.3e}/{:.3e}".format(
                    epoch, iters_this_step, evals_this_step, t_val, H_diag,
                    n_pairs, grad_norm, sy_last, sy_min, sy_max,
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
    def log_distributed_diagnostics(self, epoch, loss_local, e_rmse_local,
                                     n_trust_local=None, n_total_local=None,
                                     use_trust_region=False):
        """Log verbose per-rank metrics for distributed training.

        Shows per-rank values + global aggregates to diagnose imbalanced shards,
        rank drift, or trust region distribution issues.

        Args:
            epoch: current epoch
            loss_local: local weighted MSE (before reduce_mean)
            e_rmse_local: local energy RMSE in cm-1 (before reduce)
            n_trust_local: number of configs in trust region on this rank (optional)
            n_total_local: total configs on this rank (optional)
            use_trust_region: whether to log trust region statistics
        """
        if self.world_size <= 1:
            return

        # Gather values from all ranks
        losses = all_gather_scalar(float(loss_local), device=DEVICE)
        rmses = all_gather_scalar(float(e_rmse_local), device=DEVICE)
        if use_trust_region:
            trusts = all_gather_scalar(int(n_trust_local), device=DEVICE)
            totals = all_gather_scalar(int(n_total_local), device=DEVICE)

        # Log verbose multi-line format on rank 0
        if is_main_process():
            lines = [f"[dist-diag] epoch={epoch}"]
            for r in range(self.world_size):
                if use_trust_region:
                    trust_pct = 100.0 * trusts[r] / max(totals[r], 1)
                    lines.append(
                        f"  rank {r}: E-RMSE={rmses[r]:.2f} cm-1  "
                        f"trust={int(trusts[r])}/{int(totals[r])} ({trust_pct:.1f}%)  "
                        f"loss={losses[r]:.3f}"
                    )
                else:
                    lines.append(
                        f"  rank {r}: E-RMSE={rmses[r]:.2f} cm-1  "
                        f"loss={losses[r]:.3f}"
                    )
            # Global summary
            avg_loss = sum(losses) / len(losses)
            if use_trust_region:
                total_trust = sum(trusts)
                total_n = sum(totals)
                global_trust_pct = 100.0 * total_trust / max(total_n, 1)
                lines.append(
                    f"  global: E-RMSE=<aggregated above>  "
                    f"trust={int(total_trust)}/{int(total_n)} ({global_trust_pct:.1f}%)  "
                    f"loss={avg_loss:.3f}"
                )
            else:
                lines.append(
                    f"  global: E-RMSE=<aggregated above>  "
                    f"loss={avg_loss:.3f}"
                )
            logging.info("\n".join(lines))

        # Write CSV for post-hoc analysis (all ranks write their own row)
        if not self._dist_diag_initialized:
            if is_main_process():
                try:
                    with open(self._dist_diag_path, "w") as f:
                        if use_trust_region:
                            f.write("epoch,rank,loss_local,e_rmse_local,n_trust,n_total\n")
                        else:
                            f.write("epoch,rank,loss_local,e_rmse_local\n")
                    self._dist_diag_initialized = True
                except OSError as e:
                    logging.warning("Could not initialize distributed diag CSV: {}".format(e))
            barrier()  # Ensure header is written before other ranks append
            self._dist_diag_initialized = True

        try:
            with open(self._dist_diag_path, "a") as f:
                if use_trust_region:
                    f.write("{},{},{:.6e},{:.6e},{},{}\n".format(
                        epoch, self.rank, float(loss_local), float(e_rmse_local),
                        int(n_trust_local), int(n_total_local)
                    ))
                else:
                    f.write("{},{},{:.6e},{:.6e}\n".format(
                        epoch, self.rank, float(loss_local), float(e_rmse_local)
                    ))
        except OSError as e:
            if is_main_process():
                logging.warning("Could not append to distributed diag CSV: {}".format(e))
