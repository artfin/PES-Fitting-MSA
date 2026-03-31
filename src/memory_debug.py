"""
Memory debugging utilities for PES-Fitting-MSA.
"""

import logging
import torch

# Global flag to enable/disable memory debugging
DEBUG_MEMORY = True

def get_device():
    """Get the current device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log_memory_usage(tag=""):
    """Log current GPU memory usage."""
    device = get_device()
    if not DEBUG_MEMORY or device.type != 'cuda':
        return
    
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3    # GB
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    logging.info(f"[MEMORY] {tag}: Allocated={allocated:.3f}GB, Reserved={reserved:.3f}GB, Max={max_allocated:.3f}GB")

def log_tensor_sizes(tag, **tensors):
    """Log sizes of named tensors."""
    if not DEBUG_MEMORY:
        return
    
    total_bytes = 0
    for name, tensor in tensors.items():
        if tensor is not None and isinstance(tensor, torch.Tensor):
            size_bytes = tensor.element_size() * tensor.nelement()
            total_bytes += size_bytes
            logging.info(f"[MEMORY] {tag} - {name}: shape={list(tensor.shape)}, dtype={tensor.dtype}, size={size_bytes/1024**2:.2f}MB")
    
    logging.info(f"[MEMORY] {tag} - Total tensors: {total_bytes/1024**3:.3f}GB")

def reset_peak_memory_stats():
    """Reset peak memory statistics."""
    device = get_device()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

def memory_summary(tag=""):
    """Get full memory summary."""
    device = get_device()
    if device.type != 'cuda':
        return "CUDA not available"
    
    torch.cuda.synchronize()
    summary = torch.cuda.memory_summary(device=device, abbreviated=False)
    return f"[MEMORY SUMMARY - {tag}]\n{summary}"
