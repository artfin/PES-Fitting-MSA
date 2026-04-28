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


def reduce_rmse(errors):
    """Compute global RMSE from local error tensors across all processes.

    Unlike reduce_mean(local_rmse), this correctly computes:
        sqrt(sum_all_squared_errors / total_count)

    Args:
        errors: 1-D tensor of (pred - true) errors on this rank's shard

    Returns:
        Scalar tensor with the global RMSE
    """
    local_sse = torch.sum(errors ** 2)
    local_count = torch.tensor(errors.numel(), dtype=torch.float32, device=errors.device)

    if not is_distributed():
        return torch.sqrt(local_sse / local_count)

    total_sse = reduce_sum(local_sse)
    total_count = reduce_sum(local_count)
    return torch.sqrt(total_sse / total_count)


def reduce_mae(errors):
    """Compute global MAE from local error tensors across all processes.

    Args:
        errors: 1-D tensor of (pred - true) errors on this rank's shard

    Returns:
        Scalar tensor with the global MAE
    """
    local_sum = torch.sum(torch.abs(errors))
    local_count = torch.tensor(errors.numel(), dtype=torch.float32, device=errors.device)

    if not is_distributed():
        return local_sum / local_count

    total_sum = reduce_sum(local_sum)
    total_count = reduce_sum(local_count)
    return total_sum / total_count


def reduce_min(tensor):
    """Min tensor across all processes (no-op if single GPU)."""
    if not is_distributed():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.MIN)
    return rt


def reduce_max(tensor):
    """Max tensor across all processes (no-op if single GPU)."""
    if not is_distributed():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.MAX)
    return rt


def broadcast(tensor, src=0):
    """Broadcast tensor from src rank to all others (no-op if single GPU)."""
    if not is_distributed():
        return tensor
    dist.broadcast(tensor, src=src)
    return tensor


def all_gather_scalar(value, device=None):
    """Gather a scalar value from all ranks into a list.

    Args:
        value: Python scalar or 0-d tensor
        device: torch device (defaults to cuda if available)

    Returns:
        List of values from all ranks (on rank 0), or [value] if single GPU
    """
    if not is_distributed():
        return [value]

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    world_size = get_world_size()
    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return [t.item() for t in gathered]


def barrier():
    """Synchronize all processes (no-op if single GPU)."""
    if is_distributed():
        dist.barrier()


def sync_gradients(model):
    """Average gradients across all processes (no-op if single GPU).

    Call this after backward() to ensure all ranks have identical gradients.
    Unlike DDP's implicit sync (which can race with gradient reads), this is
    explicit and synchronous.

    Args:
        model: nn.Module whose .grad tensors to synchronize
    """
    if not is_distributed():
        return
    import logging
    rank = get_rank()
    world_size = get_world_size()
    logging.debug(f"[rank {rank}] sync_gradients: starting, world_size={world_size}")
    for i, param in enumerate(model.parameters()):
        if param.grad is not None:
            logging.debug(f"[rank {rank}] sync_gradients: all_reduce param {i} shape={param.grad.shape}")
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size
    logging.debug(f"[rank {rank}] sync_gradients: done")


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
