"""Run provenance & metrics capture (Phase 1 of md-plans/run-tracking.md).

Records, per training run, enough to reproduce it (git SHA + working-tree diff,
resolved config, environment) and enough to compare it (final metrics), as
structured JSON written next to the run's log file.

Everything here is best-effort: a failure to capture provenance or metrics must
never abort a training run.
"""

import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime

import pathlib

# src/run_tracking/provenance.py -> repo root is three levels up.
BASEDIR = pathlib.Path(__file__).parent.parent.parent.resolve()


def _git(args, cwd):
    """Run a git command; return stdout on success, else None."""
    try:
        out = subprocess.run(
            ["git"] + args, cwd=cwd, capture_output=True, text=True, timeout=15
        )
        if out.returncode != 0:
            return None
        return out.stdout
    except Exception:
        return None


def _git_info(repo_dir):
    """Return ({sha, short_sha, branch, dirty, untracked}, dirty) or None.

    None means the directory is not a git repo or git is unavailable.
    """
    sha = _git(["rev-parse", "HEAD"], repo_dir)
    if sha is None:
        return None
    sha = sha.strip()

    branch = (_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir) or "").strip() or None
    status = _git(["status", "--porcelain"], repo_dir) or ""
    # "dirty" tracks modifications to tracked files (captured in the diff).
    # Untracked files ("?? ") are not part of the diff and, in this repo, are
    # dominated by data artifacts -- so record only their count, not the list.
    tracked_changes = [ln for ln in status.splitlines() if not ln.startswith("??")]
    untracked_count = sum(1 for ln in status.splitlines() if ln.startswith("??"))
    dirty = bool(tracked_changes)

    info = {
        "sha": sha,
        "short_sha": sha[:10],
        "branch": branch,
        "dirty": dirty,
        "untracked_count": untracked_count,
    }
    return info, dirty


def _env_info():
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "node": platform.node(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
        else:
            info["cuda_device"] = None
    except Exception:
        pass
    return info


def capture_provenance(cfg, out_dir, stem, repo_dir=None, extra=None):
    """Write ``<stem>.provenance.json`` (+ ``<stem>.provenance.diff`` if dirty).

    Args:
        cfg:      the resolved run configuration (dict).
        out_dir:  directory to write the manifest into (the model folder).
        stem:     run identifier / file stem (model name or log name).
        repo_dir: git repository root; defaults to the project root.
        extra:    optional dict of additional fields to record.

    Returns the provenance dict, or None on total failure. Never raises.
    """
    try:
        repo_dir = str(repo_dir or BASEDIR)

        git = _git_info(repo_dir)
        if git is None:
            git_info, dirty = None, False
        else:
            git_info, dirty = git

        prov = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "stem": stem,
            "git": git_info,
            "config": cfg,
            "env": _env_info(),
        }
        if extra:
            prov["extra"] = extra

        # Freeze the working-tree diff so a dirty run stays reconstructable:
        #   git checkout <sha> && git apply <stem>.provenance.diff
        if git_info is not None and dirty:
            diff = _git(["diff", "HEAD"], repo_dir)
            if diff:
                diff_path = os.path.join(out_dir, stem + ".provenance.diff")
                with open(diff_path, "w") as f:
                    f.write(diff)
                git_info["diff_file"] = os.path.basename(diff_path)

        prov_path = os.path.join(out_dir, stem + ".provenance.json")
        with open(prov_path, "w") as f:
            json.dump(prov, f, indent=2, default=str)

        logging.info("Wrote run provenance to {}".format(prov_path))
        if git_info is None:
            logging.warning("Provenance: not a git repository -- run is NOT reproducible.")
        elif dirty:
            logging.info("Provenance: working tree is dirty; diff saved alongside.")
        return prov
    except Exception as e:
        logging.warning("Failed to capture provenance: {}".format(e))
        return None


def collect_metrics(trainer, wall_time_s=None, extra=None):
    """Defensively gather final metrics from a trainer object.

    Reads only attributes that are known to exist (via ``getattr``) so it works
    across trainer types without coupling to any one of them.
    """
    m = {}

    if wall_time_s is not None:
        m["wall_time_s"] = round(float(wall_time_s), 1)

    es = getattr(trainer, "es", None)
    if es is not None:
        best = getattr(es, "best_score", None)
        if best is not None:
            m["best_val_rmse"] = float(best)
        best_epoch = getattr(es, "best_epoch", None)
        if best_epoch is not None:
            m["best_epoch"] = int(best_epoch)
        m["early_stopped"] = bool(getattr(es, "status", False))

    epochs_run = getattr(trainer, "epochs_run", None)
    if epochs_run is not None:
        m["epochs_run"] = int(epochs_run)

    # Final train/val/test evaluation losses, if the trainer stashed them.
    eval_metrics = getattr(trainer, "eval_metrics", None)
    if isinstance(eval_metrics, dict) and eval_metrics:
        m["final_eval"] = eval_metrics

    if extra:
        m.update(extra)

    return m


def write_metrics(out_dir, stem, metrics):
    """Write ``<stem>.metrics.json``. Best-effort; never raises."""
    try:
        path = os.path.join(out_dir, stem + ".metrics.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logging.info("Wrote run metrics to {}".format(path))
        return path
    except Exception as e:
        logging.warning("Failed to write metrics: {}".format(e))
        return None
