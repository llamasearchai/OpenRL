import os
from typing import List, Any

import torch
import torch.distributed as dist

from rlhf.core.logging import get_logger

logger = get_logger("Distributed")


def is_distributed() -> bool:
    """Check if distributed training is enabled
    
    Returns:
        True if distributed training is enabled
    """
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Get the number of processes in the distributed training
    
    Returns:
        Number of processes (1 if not distributed)
    """
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get the rank of the current process
    
    Returns:
        Rank of the current process (0 if not distributed)
    """
    if is_distributed():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)
    
    Returns:
        True if this is the main process
    """
    return get_rank() == 0


def init_distributed(backend: str = "nccl") -> None:
    """Initialize distributed training
    
    Args:
        backend: PyTorch distributed backend
    """
    # Check if already initialized
    if is_distributed():
        logger.warning("Distributed training already initialized")
        return
    
    # Check if PyTorch distributed is available
    if not dist.is_available():
        logger.warning("PyTorch distributed not available")
        return
    
    # Get environment variables
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Initialize distributed
    if world_size > 1:
        logger.info(f"Initializing distributed training with {world_size} processes")
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )
        
        # Set device
        torch.cuda.set_device(local_rank)
        logger.info(f"Process {rank}/{world_size} using GPU: {torch.cuda.get_device_name(local_rank)}")
    else:
        logger.info("Running in single-process mode")


def all_gather(tensor: torch.Tensor) -> List[torch.Tensor]:
    """Gather tensors from all processes
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        List of gathered tensors
    """
    if not is_distributed():
        return [tensor]
    
    world_size = get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    
    return gathered_tensors