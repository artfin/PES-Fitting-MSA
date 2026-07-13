#!/bin/bash
#SBATCH --job-name=h2oar-grad-sharded
#SBATCH --nodes=4                     # world_size = NODES * GPUS_PER_NODE
#SBATCH --ntasks-per-node=1           # one torchrun launcher per node (it spawns the GPU workers)
#SBATCH --gres=gpu:1                  # GPUs per node -- keep in sync with GPUS_PER_NODE below
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
### ---- site-specific: uncomment / edit for your cluster ----
### #SBATCH --partition=gpu
### #SBATCH --account=YOUR_ACCOUNT
### #SBATCH --mem=32G
# ---------------------------------------------------------------------------
# Distributed (sharded) energy+gradient training for models/h2o-ar.
#   world_size = SLURM_NNODES * GPUS_PER_NODE  (NCCL backend => 1 GPU per rank)
#   Optimizer: vendored FullBatchLBFGS (auto-selected when world_size > 1)
#
# Launch:  sbatch scripts/slurm/gradient-sharded.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail

# --- edit these ---
REPO_DIR=/path/to/PES-Fitting-MSA        # repo root on the SHARED filesystem
GPUS_PER_NODE=1                          # must match --gres=gpu:N above
MASTER_PORT=29500
MODEL_FOLDER=models/h2o-ar
MODEL_NAME=gradient-sharded              # -> models/h2o-ar/gradient-sharded.yaml
# ------------------

cd "$REPO_DIR"

# Optionally activate your environment (edit or remove):
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate pes
# module load cuda

# Rendezvous host = first node in the allocation.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR MASTER_PORT

# NCCL: pin the fabric interface if auto-detection picks the wrong one.
# export NCCL_SOCKET_IFNAME=ib0        # or eth0 / enp5s0 ...
# export NCCL_DEBUG=INFO

echo "=== job $SLURM_JOB_ID | nodes=$SLURM_NNODES x ${GPUS_PER_NODE} gpu | master=$MASTER_ADDR:$MASTER_PORT ==="

# ---------------------------------------------------------------------------
# Preflight: build the interim .pk dataset ONCE on a single task.
# SHARDED makes every rank call load_dataset(); if the pickles are missing they
# would all race to build the same files (data_io.py:229). Build first, serially.
# No-op ("Dataset found.") when the pickles already exist on shared storage.
# ---------------------------------------------------------------------------
srun --nodes=1 --ntasks=1 python3 -c "
import sys; sys.path.insert(0, 'src')
from data_io import load_cfg, load_dataset
cfg = load_cfg('${MODEL_FOLDER}/${MODEL_NAME}.yaml')
load_dataset(cfg['DATASET'], cfg['TYPE'])
print('[preflight] interim dataset ready')
"

# ---------------------------------------------------------------------------
# Distributed training: one torchrun per node, all rendezvous at MASTER_ADDR.
# c10d rendezvous assigns ranks automatically (no --node_rank needed).
# ---------------------------------------------------------------------------
srun torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    src/train_model.py \
        --model_folder="$MODEL_FOLDER" \
        --model_name="$MODEL_NAME"

echo "=== done ==="
