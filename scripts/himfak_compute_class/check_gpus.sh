#!/bin/bash
#
# check_gpus.sh
# Answers: Do all nodes have the same GPU? (e.g., RTX 4070 with 12GB)
#
# Usage: ./check_gpus.sh [nodes.conf]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/nodes.conf}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Configuration file not found: $CONFIG"
    echo "Usage: $0 [nodes.conf]"
    exit 1
fi

source "$CONFIG"

REPORT_FILE="gpu_report_$(date +%Y%m%d_%H%M%S).txt"
echo "=== GPU Verification Report ===" > "$REPORT_FILE"
echo "Date: $(date)" >> "$REPORT_FILE"
echo "Nodes checked: ${#NODES[@]}" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

printf "%-16s %-28s %-15s %-12s\n" "IP" "GPU Name" "Memory (MiB)" "Driver"
printf "%-16s %-28s %-15s %-12s\n" "---" "--------" "------------" "------"

for ip in "${NODES[@]}"; do
    result=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "${SSH_USER}@${ip}" \
        "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo 'N/A,N/A,N/A'" 2>/dev/null || echo "N/A,N/A,N/A")

    if [[ -z "$result" ]] || [[ "$result" == "N/A,N/A,N/A" ]]; then
        gpu="ERROR"
        mem="ERROR"
        driver="ERROR"
    else
        gpu=$(echo "$result" | cut -d',' -f1 | xargs)
        mem=$(echo "$result" | cut -d',' -f2 | xargs)
        driver=$(echo "$result" | cut -d',' -f3 | xargs)
    fi

    printf "%-16s %-28s %-15s %-12s\n" "$ip" "$gpu" "$mem" "$driver"
    printf "%-16s %-28s %-15s %-12s\n" "$ip" "$gpu" "$mem" "$driver" >> "$REPORT_FILE"
done

echo ""
echo "Report saved to: $REPORT_FILE"

# Quick homogeneity check
echo ""
echo "=== Homogeneity Summary ==="
unique_gpus=$(cut -c33-60 "$REPORT_FILE" | tail -n +5 | sort | uniq -c | sort -rn)
echo "$unique_gpus"
