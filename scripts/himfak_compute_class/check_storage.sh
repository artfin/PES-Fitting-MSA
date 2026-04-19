#!/bin/bash
#
# check_storage.sh
# Answers: Is there a shared filesystem (NFS)? What path? Or is data local?
#
# Usage: ./check_storage.sh [nodes.conf]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/nodes.conf}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Configuration file not found: $CONFIG"
    exit 1
fi

source "$CONFIG"

REPORT_FILE="storage_report_$(date +%Y%m%d_%H%M%S).txt"
echo "=== Storage Verification Report ===" > "$REPORT_FILE"
echo "Date: $(date)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# 1. Check mount points on each node
echo "--- 1. NFS / Remote Mount Points ---"
echo "--- 1. Mount Points ---" >> "$REPORT_FILE"

for ip in "${NODES[@]}"; do
    echo "Node: $ip"
    echo "Node: $ip" >> "$REPORT_FILE"

    mounts=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "${SSH_USER}@${ip}" \
        "findmnt -t nfs,nfs4,ceph,fuse.sshfs -o TARGET,SOURCE,FSTYPE,AVAIL -n 2>/dev/null || true" 2>/dev/null)

    if [[ -z "$mounts" ]]; then
        echo "  No remote filesystems mounted."
        echo "  No remote filesystems mounted." >> "$REPORT_FILE"
    else
        echo "$mounts" | while read -r line; do
            echo "  $line"
            echo "  $line" >> "$REPORT_FILE"
        done
    fi
    echo "" >> "$REPORT_FILE"
done

# 2. Check if common shared paths exist and are actually shared
echo ""
echo "--- 2. Shared Path Consistency Check ---"
echo "" >> "$REPORT_FILE"
echo "--- 2. Shared Path Consistency ---" >> "$REPORT_FILE"

# Test with the current project directory
PROJECT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
echo "Testing path: $PROJECT_DIR"
echo "Testing path: $PROJECT_DIR" >> "$REPORT_FILE"

printf "%-16s %-10s %-20s\n" "IP" "Exists" "Writable"
printf "%-16s %-10s %-20s\n" "---" "------" "--------" >> "$REPORT_FILE"

for ip in "${NODES[@]}"; do
    if [[ "$ip" == "$MASTER_IP" ]]; then
        printf "%-16s %-10s %-20s\n" "$ip" "N/A" "N/A (localhost)"
        continue
    fi

    exists=$(ssh -o ConnectTimeout=5 "${SSH_USER}@${ip}" "test -d '${PROJECT_DIR}' && echo YES || echo NO" 2>/dev/null || echo "SSH_FAIL")
    if [[ "$exists" == "YES" ]]; then
        writable=$(ssh -o ConnectTimeout=5 "${SSH_USER}@${ip}" "test -w '${PROJECT_DIR}' && echo YES || echo NO" 2>/dev/null || echo "SSH_FAIL")
    else
        writable="N/A"
    fi

    printf "%-16s %-10s %-20s\n" "$ip" "$exists" "$writable"
    printf "%-16s %-10s %-20s\n" "$ip" "$exists" "$writable" >> "$REPORT_FILE"
done

# 3. Quick write speed test on shared path (from master)
echo ""
echo "--- 3. Shared Path Write Speed (master) ---"
echo "" >> "$REPORT_FILE"
echo "--- 3. Write Speed Test ---" >> "$REPORT_FILE"

TEST_FILE="${PROJECT_DIR}/.tmp_write_test_$$"
write_speed=$(dd if=/dev/zero of="$TEST_FILE" bs=1M count=256 conv=fsync 2>&1 | grep -oP '\d+\.?\d* [KMGT]B/s' || echo "unknown")
rm -f "$TEST_FILE"
echo "Sequential write: $write_speed"
echo "Sequential write: $write_speed" >> "$REPORT_FILE"

# 4. Check local disk space on all nodes
echo ""
echo "--- 4. Local Disk Space (/tmp and project dir) ---"
echo "" >> "$REPORT_FILE"
echo "--- 4. Disk Space ---" >> "$REPORT_FILE"
printf "%-16s %-12s %-12s %-12s\n" "IP" "/tmp Avail" "Project Avail" "Home Avail"
printf "%-16s %-12s %-12s %-12s\n" "---" "----------" "-------------" "----------" >> "$REPORT_FILE"

for ip in "${NODES[@]}"; do
    disk=$(ssh -o ConnectTimeout=5 "${SSH_USER}@${ip}" \
        "df -h /tmp $(dirname '${PROJECT_DIR}') /home 2>/dev/null | awk 'NR>1 {print \$4}' | tr '\n' ' '" 2>/dev/null || echo "ERROR ERROR ERROR")
    read -r tmp_avail proj_avail home_avail <<< "$disk"
    printf "%-16s %-12s %-12s %-12s\n" "$ip" "${tmp_avail:-N/A}" "${proj_avail:-N/A}" "${home_avail:-N/A}"
    printf "%-16s %-12s %-12s %-12s\n" "$ip" "${tmp_avail:-N/A}" "${proj_avail:-N/A}" "${home_avail:-N/A}" >> "$REPORT_FILE"
done

echo ""
echo "Report saved to: $REPORT_FILE"
