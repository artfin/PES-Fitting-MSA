"""Run tracking: provenance capture, run discovery, and reporting.

See md-plans/run-tracking.md. Modules:
  - provenance : capture git SHA/diff + config + env, and final metrics, per run.
  - registry   : discover runs in a model folder and resolve their metrics.
  - build_report : generate a self-contained static HTML leaderboard/report.
"""

from run_tracking.provenance import (
    capture_provenance,
    collect_metrics,
    write_metrics,
)

__all__ = ["capture_provenance", "collect_metrics", "write_metrics"]
