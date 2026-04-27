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
    """Row 2: trust frac, churn, MGDA (placeholder for 3rd panel)."""
    ax_frac, ax_churn, ax_mgda = axes

    # Fraction
    ax_frac.plot(df["epoch"], df["frac"], lw=1, color="tab:blue")
    ax_frac.set_ylabel("Trust fraction")
    ax_frac.set_ylim(-0.05, 1.05)
    ax_frac.set_title("Trust set fraction")
    _gradient_start_line(ax_frac, df_log)

    # Phi mean on twin axis (if available)
    if "phi_mean" in df.columns and df["phi_mean"].notna().any():
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

    # MGDA panel is filled by plot_mgda_diagnostics if data exists
    ax_mgda.set_title("MGDA alpha + cos_sim")
    ax_mgda.set_ylabel("alpha")
    _gradient_start_line(ax_mgda, df_log)


def plot_mgda_diagnostics(ax, df, df_log):
    """Plot MGDA alpha and cosine similarity on the given axis."""
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No MGDA data", ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='gray')
        return

    # Alpha (EMA-smoothed)
    ax.plot(df["epoch"], df["alpha"], lw=1, color="tab:blue", label="alpha (EMA)")
    if "alpha_raw" in df.columns:
        ax.plot(df["epoch"], df["alpha_raw"], lw=0.5, color="tab:blue", alpha=0.3, label="alpha (raw)")
    ax.set_ylabel("alpha", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", fontsize=6)

    # Cosine similarity on twin axis
    if "cos_sim" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(df["epoch"], df["cos_sim"], lw=1, color="tab:orange", alpha=0.8)
        ax2.set_ylabel("cos_sim", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax2.set_ylim(-1.1, 1.1)
        ax2.axhline(0, color="tab:orange", ls=":", lw=0.5, alpha=0.5)

    _gradient_start_line(ax, df_log)


def plot_gradient_diagnostics(ax, df, df_log):
    """Plot top-k tail share on the given axis."""
    # Top-k tail shares
    ax.plot(df["epoch"], df["top1pct_share"] * 100, label="top 1%", lw=1.2)
    ax.plot(df["epoch"], df["top5pct_share"] * 100, label="top 5%", lw=1)
    ax.plot(df["epoch"], df["top10pct_share"] * 100, label="top 10%", lw=0.8)
    ax.set_ylabel("% of total gradient loss")
    ax.set_ylim(0, 105)
    ax.set_title("Tail concentration")
    ax.legend(fontsize=7)
    _gradient_start_line(ax, df_log)


def plot_lbfgs_diagnostics(axes, df, df_log):
    """Row 3 panels 2-3: step length + H_diag, curvature pairs."""
    ax_t, ax_sy = axes

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
    mgda_path = base + ".mgda_diagnostics.csv"

    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    df_log = parse_log(log_path)
    print(f"Parsed {len(df_log)} epochs from log")

    has_trust = os.path.exists(trust_path)
    has_gradient = os.path.exists(gradient_path)
    has_lbfgs = os.path.exists(lbfgs_path)
    has_mgda = os.path.exists(mgda_path)

    df_trust = pd.read_csv(trust_path) if has_trust else None
    df_gradient = pd.read_csv(gradient_path) if has_gradient else None
    df_lbfgs = pd.read_csv(lbfgs_path) if has_lbfgs else None
    df_mgda = pd.read_csv(mgda_path) if has_mgda else None

    if df_trust is not None:
        print(f"Trust history: {len(df_trust)} rows, epochs {df_trust['epoch'].iloc[0]}-{df_trust['epoch'].iloc[-1]}")
    if df_gradient is not None:
        print(f"Gradient diagnostics: {len(df_gradient)} rows")
    if df_lbfgs is not None:
        print(f"LBFGS diagnostics: {len(df_lbfgs)} rows")
    if df_mgda is not None:
        print(f"MGDA diagnostics: {len(df_mgda)} rows")

    # Layout: 3 rows x 3 columns
    # Row 1: Energy RMSE, Gradient RMSE, LR + WMSE val
    # Row 2: Trust fraction, Trust churn, MGDA alpha + cos_sim
    # Row 3: Top-k tail share, L-BFGS step + H_diag, Curvature pairs
    n_rows = 1  # always have training metrics
    if has_trust or has_mgda:
        n_rows += 1
    if has_gradient or has_lbfgs:
        n_rows += 1

    fig, all_axes = plt.subplots(n_rows, 3, figsize=(16, 3.8 * n_rows))
    if n_rows == 1:
        all_axes = all_axes[np.newaxis, :]

    row = 0

    # Row 1: Training metrics
    plot_training_metrics(all_axes[row], df_log)
    row += 1

    # Row 2: Trust region + MGDA
    if has_trust or has_mgda:
        if has_trust:
            plot_trust_region(all_axes[row], df_trust, df_log)
        else:
            # No trust data - leave first two panels empty
            all_axes[row, 0].set_title("Trust set fraction")
            all_axes[row, 0].text(0.5, 0.5, "No trust data", ha='center', va='center',
                                   transform=all_axes[row, 0].transAxes, fontsize=10, color='gray')
            all_axes[row, 1].set_title("Trust set churn")
            all_axes[row, 1].text(0.5, 0.5, "No trust data", ha='center', va='center',
                                   transform=all_axes[row, 1].transAxes, fontsize=10, color='gray')

        # MGDA panel (3rd in row 2)
        plot_mgda_diagnostics(all_axes[row, 2], df_mgda, df_log)
        row += 1

    # Row 3: Gradient diagnostics + L-BFGS
    if has_gradient or has_lbfgs:
        if has_gradient:
            plot_gradient_diagnostics(all_axes[row, 0], df_gradient, df_log)
        else:
            all_axes[row, 0].set_title("Tail concentration")
            all_axes[row, 0].text(0.5, 0.5, "No gradient diagnostics", ha='center', va='center',
                                   transform=all_axes[row, 0].transAxes, fontsize=10, color='gray')

        if has_lbfgs:
            plot_lbfgs_diagnostics(all_axes[row, 1:], df_lbfgs, df_log)
        else:
            all_axes[row, 1].set_title("L-BFGS step length + H_diag")
            all_axes[row, 1].text(0.5, 0.5, "No L-BFGS data", ha='center', va='center',
                                   transform=all_axes[row, 1].transAxes, fontsize=10, color='gray')
            all_axes[row, 2].set_title("Curvature pairs")
            all_axes[row, 2].text(0.5, 0.5, "No L-BFGS data", ha='center', va='center',
                                   transform=all_axes[row, 2].transAxes, fontsize=10, color='gray')
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
