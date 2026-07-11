import logging

from config import DEVICE
from distributed import is_main_process, all_gather_scalar, barrier


class DistributedDiagnosticsMixin:
    """Per-rank metric logging for distributed (DDP/sharded) training.

    Expects the host trainer to provide the following instance attributes:
      - self.world_size, self.rank
      - self._dist_diag_initialized, self._dist_diag_path
    """

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
