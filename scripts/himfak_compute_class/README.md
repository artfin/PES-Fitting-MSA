# himfak_compute_class — Distributed Training Cluster Scripts

`nodes.conf` is git-ignored.  Copy the example and fill in your local values before running any script.

## Quick start

```bash
cd scripts/himfak_compute_class/
cp nodes.conf.example nodes.conf
vim nodes.conf

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
