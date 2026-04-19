#!/bin/bash
#
# check_homogeneity.sh
# Answers: Are all nodes identical? (CPU, RAM, OS, CUDA, PyTorch)
#
# Usage: ./check_homogeneity.sh [nodes.conf]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/nodes.conf}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Configuration file not found: $CONFIG"
    exit 1
fi

source "$CONFIG"

REPORT_FILE="homogeneity_report_$(date +%Y%m%d_%H%M%S).txt"
echo "=== Node Homogeneity Report ===" > "$REPORT_FILE"
echo "Date: $(date)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Helper function to run remote command safely
remote_cmd() {
    local ip="$1"
    local cmd="$2"
    ssh -o ConnectTimeout=5 -o BatchMode=yes "${SSH_USER}@${ip}" "$cmd" 2>/dev/null || echo "ERROR"
}

echo "--- 1. CPU Info ---"
printf "%-16s %-24s %-8s %-12s\n" "IP" "CPU Model" "Cores" "Threads"
printf "%-16s %-24s %-8s %-12s\n" "---" "---------" "-----" "-------" >> "$REPORT_FILE"

for ip in "${NODES[@]}"; do
    cpu=$(remote_cmd "$ip" "grep 'model name' /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs | cut -c1-24")
    cores=$(remote_cmd "$ip" "nproc --all")
    threads=$(remote_cmd "$ip" "grep -c '^processor' /proc/cpuinfo")
    printf "%-16s %-24s %-8s %-12s\n" "$ip" "$cpu" "$cores" "$threads"
    printf "%-16s %-24s %-8s %-12s\n" "$ip" "$cpu" "$cores" "$threads" >> "$REPORT_FILE"
done

echo ""
echo "--- 2. RAM Info ---"
echo "" >> "$REPORT_FILE"
echo "--- 2. RAM Info ---" >> "$REPORT_FILE"
printf "%-16s %-12s %-12s %-12s\n" "IP" "Total" "Available" "Free"
printf "%-16s %-12s %-12s %-12s\n" "---" "-----" "---------" "----" >> "$REPORT_FILE"

for ip in "${NODES[@]}"; do
    mem=$(remote_cmd "$ip" "free -h | awk 'NR==2{print \$2, \$7, \$4}'")
    read -r total avail free <<< "$mem"
    printf "%-16s %-12s %-12s %-12s\n" "$ip" "$total" "$avail" "$free"
    printf "%-16s %-12s %-12s %-12s\n" "$ip" "$total" "$avail" "$free" >> "$REPORT_FILE"
done

echo ""
echo "--- 3. OS / Kernel ---"
echo "" >> "$REPORT_FILE"
echo "--- 3. OS / Kernel ---" >> "$REPORT_FILE"
printf "%-16s %-20s %-20s\n" "IP" "OS" "Kernel"
printf "%-16s %-20s %-20s\n" "---" "--" "------" >> "$REPORT_FILE"

for ip in "${NODES[@]}"; do
    os=$(remote_cmd "$ip" "source /etc/os-release 2>/dev/null && echo \$NAME \$VERSION_ID || uname -o")
    kernel=$(remote_cmd "$ip" "uname -r")
    printf "%-16s %-20s %-20s\n" "$ip" "$os" "$kernel"
    printf "%-16s %-20s %-20s\n" "$ip" "$os" "$kernel" >> "$REPORT_FILE"
done

echo ""
echo "--- 4. Python / PyTorch / CUDA Versions ---"
echo "" >> "$REPORT_FILE"
echo "--- 4. Software Versions ---" >> "$REPORT_FILE"
printf "%-16s %-12s %-20s %-15s\n" "IP" "Python" "PyTorch" "CUDA"
printf "%-16s %-12s %-20s %-15s\n" "---" "------" "-------" "----" >> "$REPORT_FILE"

for ip in "${NODES[@]}"; do
    py=$(remote_cmd "$ip" "python3 --version 2>&1 | awk '{print \$2}' || echo 'N/A'")
    torch=$(remote_cmd "$ip" "python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A'")
    cuda=$(remote_cmd "$ip" "python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A'")
    printf "%-16s %-12s %-20s %-15s\n" "$ip" "$py" "$torch" "$cuda"
    printf "%-16s %-12s %-20s %-15s\n" "$ip" "$py" "$torch" "$cuda" >> "$REPORT_FILE"
done

echo ""
echo "--- 5. NCCL Version ---"
echo "" >> "$REPORT_FILE"
echo "--- 5. NCCL Version ---" >> "$REPORT_FILE"

for ip in "${NODES[@]}"; do
    nccl=$(remote_cmd "$ip" "python3 -c 'import torch; print(torch.cuda.nccl.version())' 2>/dev/null || echo 'N/A')
    printf "%-16s %-20s\n" "$ip" "$nccl"
    printf "%-16s %-20s\n" "$ip" "$nccl" >> "$REPORT_FILE"
done

echo ""
echo "Report saved to: $REPORT_FILE"
