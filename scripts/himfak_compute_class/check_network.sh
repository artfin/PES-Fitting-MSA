#!/bin/bash
#
# check_network.sh
# Answers: What is the network speed between nodes? (1GbE / 10GbE / InfiniBand)
# Also tests basic connectivity on the DDP rendezvous port.
#
# Usage: ./check_network.sh [nodes.conf]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/nodes.conf}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Configuration file not found: $CONFIG"
    exit 1
fi

source "$CONFIG"

REPORT_FILE="network_report_$(date +%Y%m%d_%H%M%S).txt"
echo "=== Network Verification Report ===" > "$REPORT_FILE"
echo "Date: $(date)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# 1. Basic ping latency
echo "--- 1. Ping Latency (master -> nodes) ---"
echo "--- 1. Ping Latency ---" >> "$REPORT_FILE"
printf "%-16s %-12s %-20s\n" "IP" "Latency" "Packet Loss"
printf "%-16s %-12s %-20s\n" "---" "-------" "-----------" >> "$REPORT_FILE"

for ip in "${NODES[@]}"; do
    if [[ "$ip" == "$MASTER_IP" ]]; then
        printf "%-16s %-12s %-20s\n" "$ip" "0.0 ms" "0% (localhost)"
        continue
    fi

    ping_out=$(ping -c 3 -W 2 "$ip" 2>/dev/null | tail -2)
    latency=$(echo "$ping_out" | grep 'avg' | sed 's/.*= \([0-9.]*\)\/\([0-9.]*\)\/\([0-9.]*\)\/\([0-9.]*\) ms/\2 ms/' || echo "TIMEOUT")
    loss=$(echo "$ping_out" | grep -oP '\d+(?=% packet loss)' || echo "100")

    printf "%-16s %-12s %-20s\n" "$ip" "$latency" "${loss}%"
    printf "%-16s %-12s %-20s\n" "$ip" "$latency" "${loss}%" >> "$REPORT_FILE"
done

# 2. TCP port connectivity for PyTorch DDP
echo ""
echo "--- 2. TCP Port ${MASTER_PORT} Connectivity ---"
echo "" >> "$REPORT_FILE"
echo "--- 2. TCP Port ${MASTER_PORT} Connectivity ---" >> "$REPORT_FILE"
printf "%-16s %-10s\n" "IP" "Port Open"
printf "%-16s %-10s\n" "---" "---------" >> "$REPORT_FILE"

for ip in "${NODES[@]}"; do
    if [[ "$ip" == "$MASTER_IP" ]]; then
        printf "%-16s %-10s\n" "$ip" "localhost"
        continue
    fi
    if timeout 5 bash -c "cat < /dev/null > /dev/tcp/${ip}/${MASTER_PORT}" 2>/dev/null; then
        status="YES"
    else
        status="NO/Firewall"
    fi
    printf "%-16s %-10s\n" "$ip" "$status"
    printf "%-16s %-10s\n" "$ip" "$status" >> "$REPORT_FILE"
done

# 3. Bandwidth test using iperf3 (if available)
echo ""
echo "--- 3. Bandwidth Test (iperf3) ---"
echo "" >> "$REPORT_FILE"
echo "--- 3. Bandwidth Test ---" >> "$REPORT_FILE"

if ! command -v iperf3 &> /dev/null; then
    echo "iperf3 not found. Install it for bandwidth testing:"
    echo "  sudo apt-get install iperf3"
    echo "Then run on master:  iperf3 -s"
    echo "And on workers:      iperf3 -c MASTER_IP"
    echo "iperf3 not installed" >> "$REPORT_FILE"
else
    printf "%-16s %-15s %-10s\n" "IP" "Bandwidth" "Direction"
    printf "%-16s %-15s %-10s\n" "---" "---------" "---------" >> "$REPORT_FILE"

    # Check if iperf3 server is running on master
    if ! nc -z "$MASTER_IP" 5201 2>/dev/null; then
        echo "Warning: iperf3 server not running on ${MASTER_IP}:5201"
        echo "Start it with: ssh ${SSH_USER}@${MASTER_IP} 'iperf3 -s -D'"
    else
        for ip in "${NODES[@]}"; do
            if [[ "$ip" == "$MASTER_IP" ]]; then
                printf "%-16s %-15s %-10s\n" "$ip" "N/A" "localhost"
                continue
            fi
            bw=$(ssh -o ConnectTimeout=5 "${SSH_USER}@${ip}" \
                "iperf3 -c ${MASTER_IP} -t 3 --format m 2>/dev/null | grep 'sender' | awk '{print \$7, \$8}'" 2>/dev/null || echo "FAILED")
            printf "%-16s %-15s %-10s\n" "$ip" "$bw" "-> master"
            printf "%-16s %-15s %-10s\n" "$ip" "$bw" "-> master" >> "$REPORT_FILE"
        done
    fi
fi

# 4. Network interface speed
echo ""
echo "--- 4. Interface Speed (master only) ---"
echo "" >> "$REPORT_FILE"
echo "--- 4. Interface Speed ---" >> "$REPORT_FILE"
interfaces=$(ip -o link show | awk -F': ' '{print $2}' | grep -v lo)
for iface in $interfaces; do
    speed=$(cat /sys/class/net/$iface/speed 2>/dev/null || echo "unknown")
    echo "Interface: $iface, Speed: ${speed} Mbps"
    echo "Interface: $iface, Speed: ${speed} Mbps" >> "$REPORT_FILE"
done

echo ""
echo "Report saved to: $REPORT_FILE"
