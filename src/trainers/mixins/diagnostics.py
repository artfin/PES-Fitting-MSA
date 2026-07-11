import logging

import torch

from config import DEVICE
from distributed import is_main_process

import sys
import pathlib
BASEDIR = pathlib.Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(BASEDIR / "vendor"))
from pytorch_lbfgs import FullBatchLBFGS as HjmshiFullBatchLBFGS


class DiagnosticsMixin:
    """L-BFGS line-search telemetry logging.

    Expects the host trainer to provide the following instance attributes:
      - self._lbfgs_prev_n_iter, self._lbfgs_prev_func_evals
      - self._last_vendored_closure_eval (optional)
      - self._lbfgs_diag_initialized, self._lbfgs_diag_path
    """

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
