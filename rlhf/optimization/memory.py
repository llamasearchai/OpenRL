import gc
from typing import Optional

import torch

from rlhf.core.logging import get_logger

logger = get_logger("Memory")


def get_gpu_memory_usage() -> float:
    """Get GPU memory usage in GB
    
    Returns:
        Memory usage in GB
    """
    if not torch.cuda.is_available():
        return 0.0
    
    return torch.cuda.memory_allocated() / (1024 ** 3)


def clear_gpu_memory() -> None:
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"Cleared GPU memory cache. Current usage: {get_gpu_memory_usage():.2f} GB")


def memory_efficient_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    chunk_size: int = 4,
    **kwargs
):
    """Run forward pass with memory efficiency by chunking
    
    Args:
        model: Model to run forward pass on
        input_ids: Input ids tensor
        attention_mask: Attention mask tensor
        chunk_size: Number of sequences to process at once
        **kwargs: Additional arguments for forward pass
        
    Returns:
        Model outputs
    """
    batch_size = input_ids.shape[0]
    
    # If batch is small enough, just run directly
    if batch_size <= chunk_size:
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
    
    # Otherwise, chunk the batch
    outputs_list = []
    for i in range(0, batch_size, chunk_size):
        # Get chunk
        chunk_input_ids = input_ids[i:i+chunk_size]
        chunk_attention_mask = attention_mask[i:i+chunk_size] if attention_mask is not None else None
        
        # Run forward pass
        chunk_outputs = model(
            input_ids=chunk_input_ids,
            attention_mask=chunk_attention_mask,
            **kwargs
        )
        
        # Collect outputs
        outputs_list.append(chunk_outputs)
    
    # Combine outputs
    combined_outputs = {}
    for key in outputs_list[0].keys():
        if isinstance(outputs_list[0][key], torch.Tensor):
            combined_outputs[key] = torch.cat([o[key] for o in outputs_list], dim=0)
        else:
            combined_outputs[key] = outputs_list[0][key]
    
    return type(outputs_list[0])(**combined_outputs)