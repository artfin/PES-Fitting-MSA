#!/usr/bin/env python3
"""Backfill metrics for pre-tracking runs (Phase 4 of md-plans/run-tracking.md).

Old runs predate Phase-1 capture, so they have no metrics.json. Their logs,
however, still carry the numbers the leaderboard needs. This parses a run's log
and writes a ``<stem>.metrics.json`` (tagged ``backfilled: true``) plus a light
``<stem>.provenance.json`` with ``git: null`` -- honestly marking the run as
non-reproducible while getting its metrics into the report.

Only a chosen portion of runs is meant to be backfilled; pass them with --runs,
or --all for everything with a log.

Usage:
    python src/run_tracking/backfill.py --folder models/h2o-h2o --runs water-extended-72-sharded
    python src/run_tracking/backfill.py --folder models/h2o-h2o --all --dry-run
"""

import argparse
import datetime as _dt
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_tracking.registry import discover_runs  # noqa: E402

_EPOCH_RE = re.compile(r"Epoch:\s+(\d+);")
_BEST_RE = re.compile(r"Best validation RMSE:\s+([\d.]+)")
_ELAPSED_RE = re.compile(r"Elapsed time:\s+([\d.]+)s")
_EARLYSTOP_RE = re.compile(r"Invoking early stop")


def parse_log_metrics(log_path):
    """Extract final metrics from a training log. Returns {} if nothing usable.

    Reliable across the logs in this repo:
      - best_val_rmse : last (== best) 'Best validation RMSE' value
      - best_epoch    : the epoch at which that best was first reached
      - epochs_run    : highest 'Epoch: N' seen, + 1
      - early_stopped : whether 'Invoking early stop' appears
      - wall_time_s   : last 'Elapsed time' reading
    (The 'Model evaluation after training' block is absent from nearly all run
    logs, so final train/val/test losses are not backfilled.)
    """
    best = None
    best_epoch = None
    cur_epoch = None
    max_epoch = None
    wall = None
    early = False

    try:
        with open(log_path, errors="ignore") as f:
            for line in f:
                m = _EPOCH_RE.search(line)
                if m:
                    cur_epoch = int(m.group(1))
                    max_epoch = cur_epoch if max_epoch is None else max(max_epoch, cur_epoch)
                    continue

                m = _BEST_RE.search(line)
                if m:
                    val = float(m.group(1))
                    if best is None or val < best:
                        best = val
                        best_epoch = cur_epoch
                    continue

                m = _ELAPSED_RE.search(line)
                if m:
                    wall = float(m.group(1))
                    continue

                if _EARLYSTOP_RE.search(line):
                    early = True
    except Exception:
        return {}

    metrics = {}
    if best is not None:
        metrics["best_val_rmse"] = best
    if best_epoch is not None:
        metrics["best_epoch"] = best_epoch
    if max_epoch is not None:
        metrics["epochs_run"] = max_epoch + 1
    metrics["early_stopped"] = early
    if wall is not None:
        metrics["wall_time_s"] = wall

    # Nothing worth writing if we couldn't even find the headline metric.
    if "best_val_rmse" not in metrics:
        return {}
    return metrics


def backfill_run(run, overwrite=False, dry_run=False):
    """Backfill one run. Returns a short status string."""
    metrics_path = os.path.join(run.folder, run.stem + ".metrics.json")
    prov_path = os.path.join(run.folder, run.stem + ".provenance.json")

    if run.metrics and not overwrite:
        return "skip (metrics.json exists)"
    if not run.has_log:
        return "skip (no log)"

    metrics = parse_log_metrics(run.log_path)
    if not metrics:
        return "skip (no best-RMSE in log)"

    metrics["backfilled"] = True
    metrics["source"] = "log-backfill"

    log_ts = _dt.datetime.fromtimestamp(os.path.getmtime(run.log_path)).isoformat(timespec="seconds")
    provenance = {
        "timestamp": log_ts,
        "stem": run.stem,
        "git": None,
        "note": "pre-tracking; metrics backfilled from log, code state unknown",
        "config": run.config,
    }

    rmse = metrics["best_val_rmse"]
    if dry_run:
        return "would write (best_val_rmse={:.3f})".format(rmse)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    # Only write provenance if the run has none, so a real Phase-1 provenance is
    # never clobbered by a backfill.
    if not run.provenance:
        with open(prov_path, "w") as f:
            json.dump(provenance, f, indent=2, default=str)

    return "wrote (best_val_rmse={:.3f})".format(rmse)


def main():
    ap = argparse.ArgumentParser(description="Backfill metrics.json for pre-tracking runs.")
    ap.add_argument("--folder", required=True, help="model folder, e.g. models/h2o-h2o")
    ap.add_argument("--runs", nargs="+", default=None, help="run stems to backfill")
    ap.add_argument("--all", action="store_true", help="backfill every run with a log")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing metrics.json (backfilled ones only are safe)")
    ap.add_argument("--dry-run", action="store_true", help="report what would be written")
    args = ap.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        ap.error("folder not found: {}".format(folder))
    if not args.runs and not args.all:
        ap.error("specify --runs <stems...> or --all")

    runs = discover_runs(folder)
    by_stem = {r.stem: r for r in runs}

    if args.all:
        selected = runs
    else:
        selected = []
        for stem in args.runs:
            if stem not in by_stem:
                print("!! unknown run: {}".format(stem))
                continue
            selected.append(by_stem[stem])

    wrote = 0
    for run in selected:
        status = backfill_run(run, overwrite=args.overwrite, dry_run=args.dry_run)
        if status.startswith("wrote") or status.startswith("would"):
            wrote += 1
        print("  {:42s} {}".format(run.stem, status))

    verb = "would backfill" if args.dry_run else "backfilled"
    print("{} {}/{} run(s).".format(verb, wrote, len(selected)))


if __name__ == "__main__":
    main()
