from typing import Optional, Dict, Any

import torch
from torch.cuda.amp import autocast, GradScaler

from rlhf.core.logging import get_logger

logger = get_logger("MixedPrecision")


class MixedPrecisionManager:
    """Manager for mixed precision training"""
    
    def __init__(
        self,
        enabled: bool = True,
        dtype: Optional[torch.dtype] = None,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        """Initialize mixed precision manager
        
        Args:
            enabled: Whether to enable mixed precision
            dtype: Data type to use for mixed precision (defaults to float16 or bfloat16 if available)
            init_scale: Initial scale for gradient scaler
            growth_factor: Growth factor for gradient scaler
            backoff_factor: Backoff factor for gradient scaler
            growth_interval: Growth interval for gradient scaler
        """
        self.enabled = enabled
        
        if dtype is None:
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
                    logger.info("Using bfloat16 for mixed precision training")
                else:
                    dtype = torch.float16
                    logger.info("Using float16 for mixed precision training")
        
        self.dtype = dtype
        self.scaler = GradScaler(
            enabled=enabled,
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
        )
        
        if enabled:
            logger.info(f"Mixed precision training enabled with {dtype}")
        else:
            logger.info("Mixed precision training disabled")
    
    def __call__(self, enabled: Optional[bool] = None):
        """Create autocast context manager
        
        Args:
            enabled: Override enabled setting
            
        Returns:
            Autocast context manager
        """
        return autocast(
            enabled=self.enabled if enabled is None else enabled,
            dtype=self.dtype,
        )
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training
        
        Args:
            loss: Loss to scale
            
        Returns:
            Scaled loss
        """
        return self.scaler.scale(loss)
    
    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Perform optimizer step with gradient scaling
        
        Args:
            optimizer: Optimizer to step
        """
        self.scaler.step(optimizer)
        self.scaler.update()
    
    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients for gradient clipping
        
        Args:
            optimizer: Optimizer containing gradients to unscale
        """
        self.scaler.unscale_(optimizer)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary of scaler
        
        Returns:
            State dictionary
        """
        return self.scaler.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary into scaler
        
        Args:
            state_dict: State dictionary to load
        """
        self.scaler.load_state_dict(state_dict)