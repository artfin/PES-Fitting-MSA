#!/usr/bin/env python3
"""Analyze a training run from its log and diagnostic CSVs.

The per-run diagnostic plotter of the run-tracking pipeline: it turns one run's
``.log`` plus its ``.trust_history / .gradient_diagnostics / .lbfgs_diagnostics /
.mgda_diagnostics`` CSVs into a single ``<stem>.analysis.png``. That figure lands
in the run's folder, where ``registry.Run.figures`` discovers it and
``build_report`` embeds it -- so ``build_report --analyze`` regenerates these
automatically (see :func:`render_analysis`).

Kept as its own module (not folded into ``registry`` / ``build_report``) so the
matplotlib + pandas dependency stays quarantined to the plotting path and the
stdlib-light data layer keeps running anywhere.

Usage:
    python -m run_tracking.analyze water-extended-34-grad          # show
    python -m run_tracking.analyze water-extended-34-grad --save   # write PNG
    python -m run_tracking.analyze water-extended-34-grad \\
        --dir models/h2o-h2o/runs/water-extended-34-grad --save
"""

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow running as a script (python src/run_tracking/analyze.py ...).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_tracking.registry import Run  # noqa: E402

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_log(log_path):
    """Extract per-epoch metrics from the training log.

    Handles two formats:
      - Energy-only: 'Epoch: N; loss train: ... cm-1; loss val: ... cm-1; lr: ...'
      - Gradient phase: multi-line block with WMSE, MAE, RMSE for energy and gradient.
    """
    rows = []
    lr_map = {}

    # Patterns
    energy_only_re = re.compile(
        r"Epoch:\s+(\d+);\s+loss train:\s+([\d.]+)\s+cm-1;\s+loss val:\s+([\d.]+)\s+cm-1;\s+lr:\s+([\d.eE+-]+)"
    )
    gradient_epoch_re = re.compile(
        r"Epoch:\s+(\d+);\s+\(energy\)\s+WMSE train:\s+([\d.]+);\s+\(energy\)\s+WMSE val:\s+([\d.]+)"
    )
    rmse_re = re.compile(
        r"\(energy\)\s+RMSE\s+(train|val):\s+([\d.]+)\s+cm-1;\s+\(gradient\)\s+RMSE\s+(train|val):\s+([\d.]+)\s+cm-1/bohr"
    )
    mae_re = re.compile(
        r"\(energy\)\s+MAE\s+(train|val):\s+([\d.]+)\s+cm-1;\s+\(gradient\)\s+MAE\s+(train|val):\s+([\d.]+)\s+cm-1/bohr"
    )
    lr_re = re.compile(r"\(optimizer\)\s+current lr:\s+([\d.eE+-]+)")

    current_epoch = None
    current_row = {}

    with open(log_path) as f:
        for line in f:
            # LR
            m = lr_re.search(line)
            if m:
                lr_val = float(m.group(1))
                if current_epoch is not None:
                    lr_map[current_epoch] = lr_val

            # Energy-only epoch line
            m = energy_only_re.search(line)
            if m:
                ep = int(m.group(1))
                rows.append({
                    "epoch": ep,
                    "wmse_train": float(m.group(2)),
                    "wmse_val": float(m.group(3)),
                    "lr": float(m.group(4)),
                })
                current_epoch = ep
                continue

            # Gradient-phase epoch header
            m = gradient_epoch_re.search(line)
            if m:
                current_epoch = int(m.group(1))
                current_row = {
                    "epoch": current_epoch,
                    "wmse_train": float(m.group(2)),
                    "wmse_val": float(m.group(3)),
                }
                continue

            # RMSE lines (appear after gradient epoch header)
            m = rmse_re.search(line)
            if m and current_row:
                which = m.group(1)  # train or val
                current_row[f"e_rmse_{which}"] = float(m.group(2))
                current_row[f"g_rmse_{which}"] = float(m.group(4))
                if "e_rmse_val" in current_row and "g_rmse_val" in current_row:
                    current_row["lr"] = lr_map.get(current_epoch, np.nan)
                    rows.append(current_row)
                    current_row = {}
                continue

            # MAE lines
            m = mae_re.search(line)
            if m and current_row:
                which = m.group(1)
                current_row[f"e_mae_{which}"] = float(m.group(2))
                current_row[f"g_mae_{which}"] = float(m.group(4))

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _gradient_start_line(ax, df_log):
    """Draw a vertical line at the first epoch with gradient metrics."""
    if "g_rmse_val" in df_log.columns:
        gradient_epochs = df_log.dropna(subset=["g_rmse_val"])
        if not gradient_epochs.empty:
            ep0 = gradient_epochs["epoch"].iloc[0]
            ax.axvline(ep0, color="grey", ls="--", lw=0.8, alpha=0.6)


def _plot_energy_rmse(ax, df):
    if "e_rmse_train" in df.columns:
        sub = df.dropna(subset=["e_rmse_train"])
        ax.plot(sub["epoch"], sub["e_rmse_train"], label="train", lw=1)
        ax.plot(sub["epoch"], sub["e_rmse_val"], label="val", lw=1)
    else:
        # energy-only: WMSE is the only metric
        ax.plot(df["epoch"], df["wmse_train"] ** 0.5, label="train (√WMSE)", lw=1)
        ax.plot(df["epoch"], df["wmse_val"] ** 0.5, label="val (√WMSE)", lw=1)
    ax.set_ylabel("Energy RMSE (cm⁻¹)")
    ax.set_title("Energy RMSE")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    _gradient_start_line(ax, df)

    # Annotate best energy RMSE val
    if "e_rmse_val" in df.columns:
        sub = df.dropna(subset=["e_rmse_val"])
        if not sub.empty:
            best_idx = sub["e_rmse_val"].idxmin()
            best_ep = sub.loc[best_idx, "epoch"]
            best_val = sub.loc[best_idx, "e_rmse_val"]
            ax.axvline(best_ep, color="green", ls=":", lw=0.8, alpha=0.5)
            ax.annotate(f"best={best_val:.1f} @{best_ep}",
                        xy=(best_ep, best_val), fontsize=8,
                        xytext=(10, 10), textcoords="offset points",
                        arrowprops=dict(arrowstyle="->", lw=0.5),
                        color="green")


def _plot_gradient_rmse(ax, df):
    sub = df.dropna(subset=["g_rmse_train"])
    ax.plot(sub["epoch"], sub["g_rmse_train"], label="train", lw=1)
    ax.plot(sub["epoch"], sub["g_rmse_val"], label="val", lw=1)
    ax.set_ylabel("Gradient RMSE (cm⁻¹/bohr)")
    ax.set_title("Gradient RMSE")
    ax.legend(fontsize=8)
    _gradient_start_line(ax, df)


def _plot_lr(ax, df):
    ax.plot(df["epoch"], df["lr"], color="tab:blue", lw=1, label="LR")
    ax.set_ylabel("Learning rate", color="tab:blue")
    ax.set_yscale("log")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(df["epoch"], df["wmse_val"], color="tab:red", lw=0.8, alpha=0.7, label="WMSE val")
    ax2.set_ylabel("WMSE val", color="tab:red")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax.set_title("LR + WMSE val")
    _gradient_start_line(ax, df)


def _plot_trust_fraction(ax, df, df_log):
    ax.plot(df["epoch"], df["frac"], lw=1, color="tab:blue")
    ax.set_ylabel("Trust fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Trust set fraction")
    _gradient_start_line(ax, df_log)

    # Phi mean on twin axis (if available)
    if "phi_mean" in df.columns and df["phi_mean"].notna().any():
        ax_phi = ax.twinx()
        ax_phi.plot(df["epoch"], df["phi_mean"], lw=0.8, color="tab:orange", alpha=0.7)
        ax_phi.set_ylabel("φ mean", color="tab:orange")
        ax_phi.tick_params(axis="y", labelcolor="tab:orange")


def _plot_trust_churn(ax, df, df_log):
    ax.bar(df["epoch"], df["entered"], width=1.0, color="tab:green", alpha=0.7, label="entered")
    ax.bar(df["epoch"], -df["left"], width=1.0, color="tab:red", alpha=0.7, label="left")
    ax.set_ylabel("Configs entered / left")
    ax.set_title("Trust set churn")
    ax.legend(fontsize=8)
    _gradient_start_line(ax, df_log)


def _plot_mgda(ax, df, df_log):
    # Alpha (EMA-smoothed)
    ax.plot(df["epoch"], df["alpha"], lw=1, color="tab:blue", label="alpha (EMA)")
    if "alpha_raw" in df.columns:
        ax.plot(df["epoch"], df["alpha_raw"], lw=0.5, color="tab:blue", alpha=0.3, label="alpha (raw)")
    ax.set_ylabel("alpha", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("MGDA alpha + cos_sim")
    ax.legend(loc="upper left", fontsize=8)

    # Cosine similarity on twin axis
    if "cos_sim" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(df["epoch"], df["cos_sim"], lw=1, color="tab:orange", alpha=0.8)
        ax2.set_ylabel("cos_sim", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax2.set_ylim(-1.1, 1.1)
        ax2.axhline(0, color="tab:orange", ls=":", lw=0.5, alpha=0.5)

    _gradient_start_line(ax, df_log)


def _plot_tail_concentration(ax, df, df_log):
    ax.plot(df["epoch"], df["top1pct_share"] * 100, label="top 1%", lw=1.2)
    ax.plot(df["epoch"], df["top5pct_share"] * 100, label="top 5%", lw=1)
    ax.plot(df["epoch"], df["top10pct_share"] * 100, label="top 10%", lw=0.8)
    ax.set_ylabel("% of total gradient loss")
    ax.set_ylim(0, 105)
    ax.set_title("Tail concentration")
    ax.legend(fontsize=8)
    _gradient_start_line(ax, df_log)


def _plot_lbfgs_step(ax, df, df_log):
    # Step length + H_diag (both log scale)
    ax.plot(df["epoch"], df["t"].replace(0, np.nan), lw=1, color="tab:blue", label="step t")
    # Mark zero-step epochs
    zero_mask = df["t"] == 0
    if zero_mask.any():
        ax.scatter(df.loc[zero_mask, "epoch"],
                   [1e-8] * int(zero_mask.sum()),
                   marker="x", s=8, color="tab:blue", alpha=0.5, zorder=5)
    ax.set_ylabel("Step length t")
    ax.set_yscale("log")
    ax.set_title("L-BFGS step length + H_diag")
    _gradient_start_line(ax, df_log)

    ax_h = ax.twinx()
    ax_h.plot(df["epoch"], df["H_diag"], lw=0.8, color="tab:orange", alpha=0.7, label="H_diag")
    ax_h.set_ylabel("H_diag", color="tab:orange")
    ax_h.set_yscale("log")
    ax_h.tick_params(axis="y", labelcolor="tab:orange")


def _plot_curvature(ax, df, df_log):
    ax.plot(df["epoch"], df["sy_last"], label="last", lw=1)
    ax.plot(df["epoch"], df["sy_min"], label="min", lw=0.8, ls="--")
    ax.plot(df["epoch"], df["sy_max"], label="max", lw=0.8, ls="--")
    ax.plot(df["epoch"], df["sy_mean"], label="mean", lw=0.8, ls=":")
    ax.set_yscale("symlog", linthresh=1e-10)
    ax.set_ylabel("⟨s, y⟩")
    ax.set_title("Curvature pairs")
    ax.legend(fontsize=8)
    _gradient_start_line(ax, df_log)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Quantities plotted, one figure (``<stem>.<suffix>.png``) each. Ordered as they
# should appear in the report.
ANALYSIS_SUFFIXES = (
    "energy_rmse", "gradient_rmse", "lr",
    "trust_fraction", "trust_churn", "mgda",
    "tail_concentration", "lbfgs_step", "curvature",
)


def analysis_figure_paths(run):
    """Every possible ``<stem>.<quantity>.png`` path for ``run`` (present or not)."""
    return [os.path.join(run.folder, f"{run.stem}.{s}.png") for s in ANALYSIS_SUFFIXES]


def existing_analysis_figures(run):
    """The per-quantity diagnostic figures already on disk for ``run``."""
    return [p for p in analysis_figure_paths(run) if os.path.exists(p)]


def needs_analysis(run):
    """True if ``run`` can be plotted but none of its per-quantity figures exist.

    Figures are generated **once** and cached in the run folder: ``build_report``
    plots only runs that have no figure yet, so the plotting cost is paid a
    single time and later reports reuse the saved PNGs. Delete a run's
    ``<stem>.*.png`` diagnostics to force a rebuild. Runs without a log can't be
    plotted, so they never need analysis.
    """
    return run.has_log and not existing_analysis_figures(run)


def render_analysis(run, save=False):
    """Render one diagnostic figure per quantity for a :class:`registry.Run`.

    Parses ``run``'s log and any diagnostic CSVs and emits a separate figure for
    each available quantity (see :data:`ANALYSIS_SUFFIXES`) -- energy/gradient
    RMSE, LR, trust fraction/churn, MGDA, tail concentration, L-BFGS step and
    curvature -- each only when its underlying data is present. With
    ``save=True`` writes ``<stem>.<quantity>.png`` into the run's folder and
    returns the list of written paths; otherwise shows them interactively.
    Returns ``[]`` if the log is absent or has no parseable epochs.
    """
    base = os.path.join(run.folder, run.stem)
    log_path = run.log_path
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}", file=sys.stderr)
        return []

    df_log = parse_log(log_path)
    if df_log.empty or "epoch" not in df_log.columns:
        print(f"No parseable epoch metrics in {log_path}; nothing to plot", file=sys.stderr)
        return []
    print(f"Parsed {len(df_log)} epochs from log")

    def _load(suffix):
        p = base + suffix
        return pd.read_csv(p) if os.path.exists(p) else None

    df_trust = _load(".trust_history.csv")
    df_gradient = _load(".gradient_diagnostics.csv")
    df_lbfgs = _load(".lbfgs_diagnostics.csv")
    df_mgda = _load(".mgda_diagnostics.csv")

    # (suffix, draw) for every quantity whose data is available, in report order.
    plots = [("energy_rmse", lambda ax: _plot_energy_rmse(ax, df_log))]
    if "g_rmse_train" in df_log.columns and df_log["g_rmse_train"].notna().any():
        plots.append(("gradient_rmse", lambda ax: _plot_gradient_rmse(ax, df_log)))
    plots.append(("lr", lambda ax: _plot_lr(ax, df_log)))
    if df_trust is not None:
        plots.append(("trust_fraction", lambda ax: _plot_trust_fraction(ax, df_trust, df_log)))
        plots.append(("trust_churn", lambda ax: _plot_trust_churn(ax, df_trust, df_log)))
    if df_mgda is not None and not df_mgda.empty:
        plots.append(("mgda", lambda ax: _plot_mgda(ax, df_mgda, df_log)))
    if df_gradient is not None:
        plots.append(("tail_concentration", lambda ax: _plot_tail_concentration(ax, df_gradient, df_log)))
    if df_lbfgs is not None:
        plots.append(("lbfgs_step", lambda ax: _plot_lbfgs_step(ax, df_lbfgs, df_log)))
        plots.append(("curvature", lambda ax: _plot_curvature(ax, df_lbfgs, df_log)))

    written = []
    for suffix, draw in plots:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        draw(ax)
        ax.set_xlabel("Epoch")
        fig.suptitle(run.stem, fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        if save:
            out = f"{base}.{suffix}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(out)

    if save:
        print(f"Saved {len(written)} figure(s) for {run.stem}")
        return written

    plt.show()
    return []


def main():
    parser = argparse.ArgumentParser(description="Analyze a PES training run")
    parser.add_argument("model", help="Model name prefix, e.g. water-extended-34-grad")
    parser.add_argument("--dir", default=".", help="Directory containing model files")
    parser.add_argument("--save", action="store_true", help="Save PNG instead of showing")
    args = parser.parse_args()

    run = Run(stem=args.model, folder=args.dir)
    if not run.has_log:
        print(f"Log file not found: {run.log_path}", file=sys.stderr)
        sys.exit(1)
    render_analysis(run, save=args.save)


if __name__ == "__main__":
    main()
