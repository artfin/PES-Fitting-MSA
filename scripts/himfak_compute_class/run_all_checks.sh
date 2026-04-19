#!/bin/bash
#
# run_all_checks.sh
# Runs the full cluster verification suite and produces a consolidated report.
# This directly answers all [TBD] questions from Section 1. Overview of
# DISTRIBUTED_TRAINING_PLAN.md.
#
# Usage: ./run_all_checks.sh [nodes.conf]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/nodes.conf}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Configuration file not found: $CONFIG"
    exit 1
fi

source "$CONFIG"

SUMMARY_FILE="cluster_summary_$(date +%Y%m%d_%H%M%S).txt"

echo "============================================"
echo "  PES-Fitting-MSA Cluster Verification"
echo "  himfak_compute_class"
echo "  $(date)"
echo "============================================"
echo ""

# Run each check and append to summary
{
    echo "============================================"
    echo "  PES-Fitting-MSA Cluster Verification"
    echo "  himfak_compute_class"
    echo "  $(date)"
    echo "============================================"
    echo ""
} > "$SUMMARY_FILE"

echo "[1/5] Checking GPUs across ${#NODES[@]} nodes..."
echo "--- GPU CHECK ---" >> "$SUMMARY_FILE"
"${SCRIPT_DIR}/check_gpus.sh" "$CONFIG" 2>&1 | tee -a "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

echo ""
echo "[2/5] Checking network connectivity and bandwidth..."
echo "--- NETWORK CHECK ---" >> "$SUMMARY_FILE"
"${SCRIPT_DIR}/check_network.sh" "$CONFIG" 2>&1 | tee -a "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

echo ""
echo "[3/5] Checking shared storage / NFS..."
echo "--- STORAGE CHECK ---" >> "$SUMMARY_FILE"
"${SCRIPT_DIR}/check_storage.sh" "$CONFIG" 2>&1 | tee -a "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

echo ""
echo "[4/5] Checking node homogeneity..."
echo "--- HOMOGENEITY CHECK ---" >> "$SUMMARY_FILE"
"${SCRIPT_DIR}/check_homogeneity.sh" "$CONFIG" 2>&1 | tee -a "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

echo ""
echo "[5/5] Checking job scheduler..."
echo "--- SCHEDULER CHECK ---" >> "$SUMMARY_FILE"
"${SCRIPT_DIR}/check_scheduler.sh" "$CONFIG" 2>&1 | tee -a "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Final summary block
{
    echo "============================================"
    echo "  SUMMARY OF FINDINGS"
    echo "============================================"
    echo ""
    echo "Please review the output above and update"
    echo "DISTRIBUTED_TRAINING_PLAN.md Section 1:"
    echo ""
    echo "  GPUs:               [update after reviewing gpu_report_*]"
    echo "  Network:            [update after reviewing network_report_*]"
    echo "  Shared storage:     [update after reviewing storage_report_*]"
    echo "  Node homogeneity:   [update after reviewing homogeneity_report_*]"
    echo "  Job scheduler:      [update after reviewing scheduler output]"
    echo ""
    echo "Individual reports:"
    ls -1 gpu_report_* network_report_* storage_report_* homogeneity_report_* 2>/dev/null || true
} | tee -a "$SUMMARY_FILE"

echo ""
echo "All checks complete. Consolidated summary: $SUMMARY_FILE"
