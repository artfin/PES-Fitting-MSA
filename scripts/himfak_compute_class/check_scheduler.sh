#!/bin/bash
#
# check_scheduler.sh
# Answers: Is SLURM available or are we running bare SSH?
#
# Usage: ./check_scheduler.sh [nodes.conf]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/nodes.conf}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Configuration file not found: $CONFIG"
    exit 1
fi

source "$CONFIG"

echo "=== Job Scheduler Check ==="
echo ""

# Check locally first
echo "--- Local environment ---"
if command -v sbatch &> /dev/null; then
    echo "SLURM found locally:"
    sbatch --version 2>/dev/null || true
    sinfo -V 2>/dev/null || true
else
    echo "No SLURM commands found locally (sbatch, sinfo, squeue)."
fi

if command -v qsub &> /dev/null; then
    echo "PBS/Torque found locally."
fi

echo ""
echo "--- Remote nodes ---"

for ip in "${NODES[@]}"; do
    has_slurm=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "${SSH_USER}@${ip}" \
        "command -v sbatch && echo YES || echo NO" 2>/dev/null || echo "SSH_FAIL")

    has_pbs=$(ssh -o ConnectTimeout=5 "${SSH_USER}@${ip}" \
        "command -v qsub && echo YES || echo NO" 2>/dev/null || echo "SSH_FAIL")

    if [[ "$has_slurm" == "YES" ]]; then
        sched="SLURM"
    elif [[ "$has_pbs" == "YES" ]]; then
        sched="PBS/Torque"
    elif [[ "$has_slurm" == "SSH_FAIL" ]]; then
        sched="UNREACHABLE"
    else
        sched="Bare SSH (none)"
    fi

    printf "%-16s %-20s\n" "$ip" "$sched"
done

echo ""
echo "Recommendation:"
if command -v sbatch &> /dev/null; then
    echo "  SLURM is available. Consider using sbatch + srun for launching instead of raw SSH."
    echo "  Example: sbatch --nodes=4 --gres=gpu:1 launch_slurm.sh"
else
    echo "  No job scheduler detected. You will use bare SSH + torchrun for launching."
    echo "  See launch_distributed.sh for the manual launch approach."
fi
