"""Discover and describe training runs (Phase 2 of md-plans/run-tracking.md).

The data layer behind the run report. Scans a model folder, identifies runs, and
attaches their config / provenance / metrics manifests. Kept dependency-light
(stdlib + pyyaml) so it -- and the report built on it -- runs anywhere.

A "run" is identified by its config file ``<stem>.yaml`` (excluding ``*-test.yaml``
siblings). This works on today's flat layout and will keep working after the
Phase 3 migration to per-run directories.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Run:
    stem: str
    folder: str
    config: dict = field(default_factory=dict)
    intent: Optional[str] = None            # leading comment block from the YAML
    provenance: Optional[dict] = None       # <stem>.provenance.json, if present
    metrics: Optional[dict] = None          # <stem>.metrics.json, if present

    @property
    def has_log(self) -> bool:
        return os.path.isfile(self.log_path)

    @property
    def log_path(self) -> str:
        return os.path.join(self.folder, self.stem + ".log")

    @property
    def mtime(self) -> float:
        """Best available timestamp: log mtime, else config mtime."""
        for p in (self.log_path, os.path.join(self.folder, self.stem + ".yaml")):
            if os.path.isfile(p):
                return os.path.getmtime(p)
        return 0.0


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _load_yaml(path):
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _intent_comment(path, max_len=600):
    """Return the run's intent: full-line ``#`` comments from the YAML.

    Captures the ``# Run #NN: ... / # Strategy: ...`` notes configs carry, which
    are typically a few lines in rather than at the very top. Only whole-line
    comments are collected (not trailing inline comments), pure divider lines
    (``####``) are dropped, and the result is length-capped.
    """
    lines = []
    try:
        with open(path) as f:
            for raw in f:
                s = raw.strip()
                if not s.startswith("#"):
                    continue
                s = s.lstrip("#").strip()
                if not s or set(s) <= {"#", "=", "-", "*"}:
                    continue  # divider / empty comment
                lines.append(s)
    except Exception:
        return None
    text = " ".join(lines).strip()
    if len(text) > max_len:
        text = text[:max_len].rstrip() + "…"
    return text or None


def discover_runs(folder):
    """Return a list of Run objects found in ``folder`` (flat layout)."""
    runs = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".yaml"):
            continue
        stem = name[:-len(".yaml")]
        if stem.endswith("-test"):
            continue  # evaluation sidecar, not a run of its own

        cfg_path = os.path.join(folder, name)
        runs.append(Run(
            stem=stem,
            folder=folder,
            config=_load_yaml(cfg_path),
            intent=_intent_comment(cfg_path),
            provenance=_load_json(os.path.join(folder, stem + ".provenance.json")),
            metrics=_load_json(os.path.join(folder, stem + ".metrics.json")),
        ))
    return runs


# ---------------------------------------------------------------------------
# Metric resolution
# ---------------------------------------------------------------------------

_BEST_RMSE_RE = re.compile(r"Best validation RMSE:\s+([\d.]+)")


def best_val_rmse_from_log(log_path):
    """Fallback: scrape the final 'Best validation RMSE' from a log.

    The early stopper's best score is monotonic non-increasing, so the last
    occurrence is the run's best. Returns None if the log has no such line.
    """
    best = None
    try:
        with open(log_path, errors="ignore") as f:
            for line in f:
                m = _BEST_RMSE_RE.search(line)
                if m:
                    best = float(m.group(1))
    except Exception:
        return None
    return best


def resolve_best_val_rmse(run):
    """Return (value, source) where source is 'manifest', 'log', or None.

    Prefers structured metrics.json (Phase 1 / backfilled). Falls back to parsing
    the log so pre-tracking runs still appear in the report.
    """
    if run.metrics and run.metrics.get("best_val_rmse") is not None:
        return float(run.metrics["best_val_rmse"]), "manifest"
    if run.has_log:
        v = best_val_rmse_from_log(run.log_path)
        if v is not None:
            return v, "log"
    return None, None


# ---------------------------------------------------------------------------
# Reproducibility status
# ---------------------------------------------------------------------------

def repro_status(run):
    """Return (level, label): level in {'reproducible','dirty','pre-tracking'}."""
    prov = run.provenance
    if not prov or not prov.get("git"):
        return "pre-tracking", "no provenance"
    git = prov["git"]
    sha = (git.get("short_sha") or git.get("sha") or "")[:10]
    if git.get("dirty"):
        return "dirty", "{} +diff".format(sha)
    return "reproducible", sha


# ---------------------------------------------------------------------------
# Config summarization
# ---------------------------------------------------------------------------

def flatten_config(cfg, prefix=""):
    """Flatten a nested config to dotted keys, e.g. 'MODEL.HIDDEN_DIMS'."""
    out = {}
    for k, v in (cfg or {}).items():
        key = "{}.{}".format(prefix, k) if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_config(v, key))
        else:
            out[key] = v
    return out


def config_summary(cfg):
    """Pull the handful of fields worth showing as leaderboard columns."""
    ds = cfg.get("DATASET", {}) or {}
    model = cfg.get("MODEL", {}) or {}
    loss = cfg.get("LOSS", {}) or {}
    pre = cfg.get("PRETRAINED_MODEL_SETTINGS", {}) or {}

    def _fmt(v):
        return v if v is not None else ""

    pretrained = pre.get("SOURCE")
    if isinstance(pretrained, str):
        pretrained = os.path.basename(pretrained).replace(".pt", "")

    return {
        "type": cfg.get("TYPE", ""),
        "dataset": ds.get("NAME", ""),
        "order": _fmt(ds.get("ORDER")),
        "hidden_dims": _fmt(model.get("HIDDEN_DIMS")),
        "activation": _fmt(model.get("ACTIVATION")),
        "loss": loss.get("NAME", ""),
        "focal_gamma": _fmt(loss.get("FOCAL_GAMMA")),
        "f_lambda": _fmt(loss.get("F_LAMBDA")),
        "pretrained": _fmt(pretrained),
    }
