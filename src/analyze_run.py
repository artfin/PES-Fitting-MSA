#!/usr/bin/env python3
"""Analyze a training run from its log and diagnostic CSVs.

Usage:
    python analyze_run.py water-extended-34-grad
    python analyze_run.py water-extended-34-grad --save  # save PNG instead of showing
"""

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def plot_training_metrics(axes, df):
    """Row 1: energy RMSE, gradient RMSE, LR + WMSE val."""
    ax_e, ax_g, ax_lr = axes

    # Energy RMSE
    if "e_rmse_train" in df.columns:
        sub = df.dropna(subset=["e_rmse_train"])
        ax_e.plot(sub["epoch"], sub["e_rmse_train"], label="train", lw=1)
        ax_e.plot(sub["epoch"], sub["e_rmse_val"], label="val", lw=1)
    else:
        # energy-only: WMSE is the only metric
        ax_e.plot(df["epoch"], df["wmse_train"] ** 0.5, label="train (√WMSE)", lw=1)
        ax_e.plot(df["epoch"], df["wmse_val"] ** 0.5, label="val (√WMSE)", lw=1)
    ax_e.set_ylabel("Energy RMSE (cm⁻¹)")
    ax_e.set_title("Energy RMSE")
    ax_e.legend(fontsize=7)
    ax_e.set_yscale("log")
    _gradient_start_line(ax_e, df)

    # Gradient RMSE
    if "g_rmse_train" in df.columns:
        sub = df.dropna(subset=["g_rmse_train"])
        if not sub.empty:
            ax_g.plot(sub["epoch"], sub["g_rmse_train"], label="train", lw=1)
            ax_g.plot(sub["epoch"], sub["g_rmse_val"], label="val", lw=1)
            ax_g.legend(fontsize=7)
    ax_g.set_ylabel("Gradient RMSE (cm⁻¹/bohr)")
    ax_g.set_title("Gradient RMSE")
    _gradient_start_line(ax_g, df)

    # LR + WMSE val
    ax_lr.plot(df["epoch"], df["lr"], color="tab:blue", lw=1, label="LR")
    ax_lr.set_ylabel("Learning rate", color="tab:blue")
    ax_lr.set_yscale("log")
    ax_lr.tick_params(axis="y", labelcolor="tab:blue", labelsize=7)
    ax2 = ax_lr.twinx()
    ax2.plot(df["epoch"], df["wmse_val"], color="tab:red", lw=0.8, alpha=0.7, label="WMSE val")
    ax2.set_ylabel("WMSE val", color="tab:red")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor="tab:red", labelsize=7)
    ax_lr.set_title("LR + WMSE val")
    _gradient_start_line(ax_lr, df)

    # Annotate best energy RMSE val
    if "e_rmse_val" in df.columns:
        sub = df.dropna(subset=["e_rmse_val"])
        if not sub.empty:
            best_idx = sub["e_rmse_val"].idxmin()
            best_ep = sub.loc[best_idx, "epoch"]
            best_val = sub.loc[best_idx, "e_rmse_val"]
            ax_e.axvline(best_ep, color="green", ls=":", lw=0.8, alpha=0.5)
            ax_e.annotate(f"best={best_val:.1f} @{best_ep}",
                          xy=(best_ep, best_val), fontsize=6,
                          xytext=(10, 10), textcoords="offset points",
                          arrowprops=dict(arrowstyle="->", lw=0.5),
                          color="green")


def plot_trust_region(axes, df, df_log):
    """Row 2: trust frac, churn, eviction signal."""
    ax_frac, ax_churn, ax_evict = axes

    # Fraction
    ax_frac.plot(df["epoch"], df["frac"], lw=1, color="tab:blue")
    ax_frac.set_ylabel("Trust fraction")
    ax_frac.set_ylim(-0.05, 1.05)
    ax_frac.set_title("Trust set fraction")
    _gradient_start_line(ax_frac, df_log)

    # Phi mean on twin axis (if available)
    if df["phi_mean"].notna().any():
        ax_phi = ax_frac.twinx()
        ax_phi.plot(df["epoch"], df["phi_mean"], lw=0.8, color="tab:orange", alpha=0.7)
        ax_phi.set_ylabel("φ mean", color="tab:orange")
        ax_phi.tick_params(axis="y", labelcolor="tab:orange")

    # Churn
    ax_churn.bar(df["epoch"], df["entered"], width=1.0, color="tab:green", alpha=0.7, label="entered")
    ax_churn.bar(df["epoch"], -df["left"], width=1.0, color="tab:red", alpha=0.7, label="left")
    ax_churn.set_ylabel("Configs entered / left")
    ax_churn.set_title("Trust set churn")
    ax_churn.legend(fontsize=7)
    _gradient_start_line(ax_churn, df_log)

    # Eviction signal
    valid = df.dropna(subset=["mean_err_left", "mean_err_stayed"])
    if not valid.empty:
        ax_evict.plot(valid["epoch"], valid["mean_err_left"], label="left (mean)", lw=1, color="tab:red")
        ax_evict.plot(valid["epoch"], valid["mean_err_stayed"], label="stayed (mean)", lw=1, color="tab:green")
        ax_evict.plot(valid["epoch"], valid["med_err_left"], label="left (med)", lw=0.7, ls="--", color="tab:red", alpha=0.6)
        ax_evict.plot(valid["epoch"], valid["med_err_stayed"], label="stayed (med)", lw=0.7, ls="--", color="tab:green", alpha=0.6)
        ax_evict.legend(fontsize=6)
    ax_evict.set_ylabel("Gradient RMSE (cm⁻¹/bohr)")
    ax_evict.set_title("Eviction signal")
    ax_evict.set_yscale("symlog", linthresh=1.0)
    _gradient_start_line(ax_evict, df_log)


def plot_gradient_diagnostics(axes, df, df_log):
    """Row 3: contribution quantiles, top-k share, phi histogram."""
    ax_cq, ax_topk, ax_phi = axes

    # Contribution quantiles (log scale)
    for col, label, alpha in [
        ("contrib_q50", "q50", 1.0),
        ("contrib_q90", "q90", 0.8),
        ("contrib_q95", "q95", 0.7),
        ("contrib_q99", "q99", 0.6),
        ("contrib_max", "max", 0.4),
    ]:
        ax_cq.plot(df["epoch"], df[col], label=label, lw=1, alpha=alpha)
    ax_cq.set_yscale("log")
    ax_cq.set_ylabel("Contribution")
    ax_cq.set_title("Per-config gradient-loss contribution")
    ax_cq.legend(fontsize=6, ncol=2)
    _gradient_start_line(ax_cq, df_log)

    # Top-k tail shares
    ax_topk.plot(df["epoch"], df["top1pct_share"] * 100, label="top 1%", lw=1.2)
    ax_topk.plot(df["epoch"], df["top5pct_share"] * 100, label="top 5%", lw=1)
    ax_topk.plot(df["epoch"], df["top10pct_share"] * 100, label="top 10%", lw=0.8)
    ax_topk.set_ylabel("% of total gradient loss")
    ax_topk.set_ylim(0, 105)
    ax_topk.set_title("Tail concentration")
    ax_topk.legend(fontsize=7)
    _gradient_start_line(ax_topk, df_log)

    # Phi distribution (stacked area)
    phi_cols = ["phi_lt_25", "phi_25_50", "phi_50_75", "phi_75_90", "phi_ge_90"]
    if all(c in df.columns for c in phi_cols):
        totals = df[phi_cols].sum(axis=1).replace(0, 1)
        fracs = df[phi_cols].div(totals, axis=0) * 100
        labels = ["φ<0.25", "0.25-0.50", "0.50-0.75", "0.75-0.90", "φ≥0.90"]
        colors = ["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1f77b4"]
        ax_phi.stackplot(df["epoch"], *[fracs[c] for c in phi_cols],
                         labels=labels, colors=colors, alpha=0.8)
        ax_phi.set_ylabel("% of active set")
        ax_phi.set_ylim(0, 100)
        ax_phi.set_title("φ distribution")
        ax_phi.legend(fontsize=6, loc="center left")
    _gradient_start_line(ax_phi, df_log)


def plot_lbfgs_diagnostics(axes, df, df_log):
    """Row 4: step length, curvature, grad norm + iters."""
    ax_t, ax_sy, ax_grad = axes

    # Step length + H_diag (both log scale)
    ax_t.plot(df["epoch"], df["t"].replace(0, np.nan), lw=1, color="tab:blue", label="step t")
    # Mark zero-step epochs
    zero_mask = df["t"] == 0
    if zero_mask.any():
        ax_t.scatter(df.loc[zero_mask, "epoch"],
                     [1e-8] * int(zero_mask.sum()),
                     marker="x", s=8, color="tab:blue", alpha=0.5, zorder=5)
    ax_t.set_ylabel("Step length t")
    ax_t.set_yscale("log")
    ax_t.set_title("L-BFGS step length + H_diag")
    _gradient_start_line(ax_t, df_log)

    ax_h = ax_t.twinx()
    ax_h.plot(df["epoch"], df["H_diag"], lw=0.8, color="tab:orange", alpha=0.7, label="H_diag")
    ax_h.set_ylabel("H_diag", color="tab:orange")
    ax_h.set_yscale("log")
    ax_h.tick_params(axis="y", labelcolor="tab:orange")

    # Curvature pairs <s,y>
    ax_sy.plot(df["epoch"], df["sy_last"], label="last", lw=1)
    ax_sy.plot(df["epoch"], df["sy_min"], label="min", lw=0.8, ls="--")
    ax_sy.plot(df["epoch"], df["sy_max"], label="max", lw=0.8, ls="--")
    ax_sy.plot(df["epoch"], df["sy_mean"], label="mean", lw=0.8, ls=":")
    ax_sy.set_yscale("symlog", linthresh=1e-10)
    ax_sy.set_ylabel("⟨s, y⟩")
    ax_sy.set_title("Curvature pairs")
    ax_sy.legend(fontsize=6)
    _gradient_start_line(ax_sy, df_log)

    # Grad norm + iters
    ax_grad.plot(df["epoch"], df["grad_norm"], lw=1, color="tab:red", label="grad norm")
    ax_grad.set_ylabel("Gradient norm", color="tab:red")
    ax_grad.set_yscale("log")
    ax_grad.tick_params(axis="y", labelcolor="tab:red")
    ax_grad.set_title("Grad norm + iters/step")
    _gradient_start_line(ax_grad, df_log)

    # Reference line for grad clip if constant
    gn = df["grad_norm"].dropna()
    if not gn.empty:
        clip_val = gn.mode().iloc[0] if not gn.mode().empty else None
        if clip_val is not None and (gn == clip_val).mean() > 0.5:
            ax_grad.axhline(clip_val, color="tab:red", ls=":", lw=0.7, alpha=0.5)
            ax_grad.annotate(f"clip={clip_val:.0f}",
                             xy=(df["epoch"].iloc[0], clip_val), fontsize=6,
                             color="tab:red", alpha=0.7)

    ax_it = ax_grad.twinx()
    ax_it.plot(df["epoch"], df["iters_this_step"], lw=0.8, color="tab:purple", alpha=0.7)
    ax_it.set_ylabel("Iters / step", color="tab:purple")
    ax_it.tick_params(axis="y", labelcolor="tab:purple")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze a PES training run")
    parser.add_argument("model", help="Model name prefix, e.g. water-extended-34-grad")
    parser.add_argument("--dir", default=".", help="Directory containing model files")
    parser.add_argument("--save", action="store_true", help="Save PNG instead of showing")
    args = parser.parse_args()

    base = os.path.join(args.dir, args.model)

    # Load files
    log_path = base + ".log"
    trust_path = base + ".trust_history.csv"
    gradient_path = base + ".gradient_diagnostics.csv"
    lbfgs_path = base + ".lbfgs_diagnostics.csv"

    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    df_log = parse_log(log_path)
    print(f"Parsed {len(df_log)} epochs from log")

    has_trust = os.path.exists(trust_path)
    has_gradient = os.path.exists(gradient_path)
    has_lbfgs = os.path.exists(lbfgs_path)

    df_trust = pd.read_csv(trust_path) if has_trust else None
    df_gradient = pd.read_csv(gradient_path) if has_gradient else None
    df_lbfgs = pd.read_csv(lbfgs_path) if has_lbfgs else None

    if df_trust is not None:
        print(f"Trust history: {len(df_trust)} rows, epochs {df_trust['epoch'].iloc[0]}-{df_trust['epoch'].iloc[-1]}")
    if df_gradient is not None:
        print(f"Gradient diagnostics: {len(df_gradient)} rows")
    if df_lbfgs is not None:
        print(f"LBFGS diagnostics: {len(df_lbfgs)} rows")

    # Determine layout
    n_rows = 1  # always have training metrics
    if has_trust:
        n_rows += 1
    if has_gradient:
        n_rows += 1
    if has_lbfgs:
        n_rows += 1

    fig, all_axes = plt.subplots(n_rows, 3, figsize=(16, 3.8 * n_rows))
    if n_rows == 1:
        all_axes = all_axes[np.newaxis, :]

    row = 0
    plot_training_metrics(all_axes[row], df_log)
    row += 1

    if has_trust:
        plot_trust_region(all_axes[row], df_trust, df_log)
        row += 1

    if has_gradient:
        plot_gradient_diagnostics(all_axes[row], df_gradient, df_log)
        row += 1

    if has_lbfgs:
        plot_lbfgs_diagnostics(all_axes[row], df_lbfgs, df_log)
        row += 1

    for ax_row in all_axes:
        for ax in ax_row:
            ax.set_xlabel("Epoch", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(args.model, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if args.save:
        out = base + ".analysis.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
