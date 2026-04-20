"""
Distributed training utilities for PES-Fitting-MSA.

All functions gracefully handle single-GPU mode:
- is_distributed() returns False
- get_rank() returns 0
- get_world_size() returns 1
- reduce_mean() returns tensor unchanged
- is_main_process() returns True

No code changes needed in callers when switching modes.

Usage:
    # At start of training:
    rank, world_size, local_rank = setup_distributed()

    # Check if main process for logging:
    if is_main_process():
        logging.info(...)

    # Average metrics across ranks:
    val_loss = reduce_mean(val_loss_local)

    # At end of training:
    cleanup()
"""
import os
import logging
import torch
import torch.distributed as dist


def is_distributed():
    """Check if running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def setup_distributed():
    """Initialize distributed training if launched with torchrun.

    Automatically detects if running under torchrun by checking
    for RANK environment variable.

    Returns:
        tuple: (rank, world_size, local_rank)
        - Single GPU: (0, 1, 0)
        - Distributed: (actual rank, world_size, local_rank)
    """
    # Check if torchrun set the environment variables
    if os.environ.get("RANK") is None:
        # Single GPU mode - return defaults
        return 0, 1, 0

    # Distributed mode - initialize process group
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set CUDA device for this process
    torch.cuda.set_device(local_rank)

    if rank == 0:
        logging.info(f"Distributed training initialized: {world_size} processes")

    return rank, world_size, local_rank


def cleanup():
    """Clean up distributed process group (no-op if single GPU)."""
    if is_distributed():
        dist.destroy_process_group()


def get_rank():
    """Get current process rank (0 if not distributed)."""
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    """Get world size (1 if not distributed)."""
    return dist.get_world_size() if is_distributed() else 1


def is_main_process():
    """True if rank 0 or single GPU mode. Use for logging/checkpointing."""
    return get_rank() == 0


def reduce_mean(tensor):
    """Average tensor across all processes (no-op if single GPU).

    Args:
        tensor: PyTorch tensor to reduce

    Returns:
        Averaged tensor (clone, original unchanged)
    """
    if not is_distributed():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt


def reduce_sum(tensor):
    """Sum tensor across all processes (no-op if single GPU)."""
    if not is_distributed():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def reduce_min(tensor):
    """Min tensor across all processes (no-op if single GPU)."""
    if not is_distributed():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.MIN)
    return rt


def broadcast(tensor, src=0):
    """Broadcast tensor from src rank to all others (no-op if single GPU)."""
    if not is_distributed():
        return tensor
    dist.broadcast(tensor, src=src)
    return tensor


def barrier():
    """Synchronize all processes (no-op if single GPU)."""
    if is_distributed():
        dist.barrier()


def shard_dataset(dataset, rank, world_size):
    """
    Extract this rank's portion of a dataset, dropping remainder to ensure
    equal-sized shards across all ranks.

    Args:
        dataset: object with .X, .y, .dX, .dy, .xyz_ordered, .grm attributes
        rank: this process's rank
        world_size: total number of processes

    Returns:
        tuple: (dataset, dropped_count)
            - dataset is modified in-place
            - dropped_count is number of samples dropped from the end
    """
    N = len(dataset.y)
    samples_per_rank = N // world_size
    usable = samples_per_rank * world_size

    start = rank * samples_per_rank
    end = start + samples_per_rank

    dataset.X = dataset.X[start:end]
    dataset.y = dataset.y[start:end]
    if dataset.dX is not None:
        dataset.dX = dataset.dX[start:end]
        dataset.dy = dataset.dy[start:end]
    if getattr(dataset, 'xyz_ordered', None) is not None:
        dataset.xyz_ordered = dataset.xyz_ordered[start:end]
    if getattr(dataset, 'grm', None) is not None:
        dataset.grm = dataset.grm[start:end]

    return dataset, N - usable
