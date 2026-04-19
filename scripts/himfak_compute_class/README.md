# himfak_compute_class — Distributed Training Cluster Scripts

This directory contains cluster-specific diagnostic and launch scripts for the
**himfak_compute_class** GPU cluster (192.168.49.101–125).  
These scripts answer the open questions in
`models/h2o-h2o/DISTRIBUTED_TRAINING_PLAN.md` §1 and §10.

> **Note:** `nodes.conf` is git-ignored.  Copy the example and fill in your
> local values before running any script.

---

## Quick start

```bash
cd scripts/himfak_compute_class/
cp nodes.conf.example nodes.conf
# Edit nodes.conf with actual IPs, SSH user, etc.
vim nodes.conf

# Run the full verification suite
./run_all_checks.sh
```

Individual scripts can also be run stand-alone:

```bash
./check_gpus.sh         # GPU model & memory across nodes
./check_network.sh      # Ping, port 29500, iperf3 bandwidth
./check_storage.sh      # NFS mounts, shared-path consistency
./check_homogeneity.sh  # CPU, RAM, OS, PyTorch, CUDA, NCCL
./check_scheduler.sh    # SLURM vs bare SSH
```

Each script writes a timestamped `*_report_*.txt` file and prints to stdout.

---

## TODO — Answer these before starting distributed training

### 1. Overview (Section 1 TBDs)

Run `./run_all_checks.sh`, inspect the generated reports, and update
`DISTRIBUTED_TRAINING_PLAN.md` with the concrete values:

- [ ] **GPUs:** Confirm all nodes have the same GPU model and memory.
  - *Expected:* NVIDIA GeForce RTX 4070, 12288 MiB
  - *Run:* `./check_gpus.sh`
  - *Record finding in plan:* `models/h2o-h2o/DISTRIBUTED_TRAINING_PLAN.md` line 9

- [ ] **Network:** Measure inter-node bandwidth.
  - *Question:* 1 GbE / 10 GbE / InfiniBand?
  - *Run:* `./check_network.sh` (requires `iperf3` server on master)
  - *Record finding in plan:* `DISTRIBUTED_TRAINING_PLAN.md` line 10

- [ ] **Shared storage:** Identify if NFS (or similar) is mounted uniformly.
  - *Question:* NFS / local copy?
  - *Run:* `./check_storage.sh`
  - *Record finding in plan:* `DISTRIBUTED_TRAINING_PLAN.md` line 11

### 2. Open Questions (Section 10)

- [ ] **Shared filesystem path:**
  - Is NFS available? What is the mount point?
  - Record exact path (e.g. `/mnt/shared`, `/nfs`, `/home`).

- [ ] **Node homogeneity:**
  - Are all 24 nodes identical in CPU, RAM, OS, driver, and PyTorch version?
  - *Run:* `./check_homogeneity.sh`
  - Flag any outliers; they may need separate treatment or exclusion.

- [ ] **Job scheduler:**
  - SLURM available or bare SSH?
  - *Run:* `./check_scheduler.sh`
  - If SLURM is present, prefer `sbatch` / `srun` launch scripts over raw SSH.

- [ ] **Checkpoint strategy:**
  - Save on rank 0 to NFS, or local disk + aggregate at job end?
  - Decision depends on the bandwidth numbers from `check_network.sh` and
    `check_storage.sh`.

- [ ] **How many nodes to use:**
  - Start with 4–8 nodes for the first DDP smoke test.
  - The plan predicts diminishing returns after 8–16 nodes due to comms overhead.

### 3. Environment Setup Checklist (Phase 1, Day 1)

- [ ] Verify matching GPU, CUDA, PyTorch versions on all nodes.
- [ ] Set up passwordless SSH from master to all workers.
  ```bash
  # On master node
  for ip in 192.168.49.{101..125}; do
      ssh-copy-id admin@${ip}
  done
  ```
- [ ] Verify network connectivity on port 29500 (DDP rendezvous).
  - `check_network.sh` tests this automatically.
- [ ] Decide shared filesystem vs local copy.
  - If NFS is fast enough (>100 MB/s write), use it for code + data.
  - Write checkpoints to local `/tmp` or NVMe and `rsync` back at job end.
- [ ] Create a simple multi-node test script.
  - A minimal `test_distributed.py` is provided in the plan (§6, Phase 1).
  - Run it on 2 nodes before attempting full training.

---

## Files

| File | Purpose |
|------|---------|
| `nodes.conf.example` | Template for cluster-specific IPs and paths. Copy to `nodes.conf`. |
| `nodes.conf` | **Git-ignored.** Live configuration for this cluster. |
| `check_gpus.sh` | GPU verification across the node list. |
| `check_network.sh` | Latency, TCP port reachability, iperf3 bandwidth. |
| `check_storage.sh` | NFS mount detection, shared-path write test, disk space. |
| `check_homogeneity.sh` | CPU, RAM, OS, Python, PyTorch, CUDA, NCCL versions. |
| `check_scheduler.sh` | Detect SLURM / PBS / bare-SSH environment. |
| `run_all_checks.sh` | Orchestrates all checks and builds a consolidated summary. |

---

## Notes

- All diagnostic scripts are **read-only**; they do not modify code or data.
- `check_network.sh` needs `iperf3` installed for bandwidth numbers:
  ```bash
  sudo apt-get install iperf3
  # On master: iperf3 -s -D
  ```
- If a node is unreachable, scripts print `ERROR` or `SSH_FAIL` but continue
  with the remaining nodes.
