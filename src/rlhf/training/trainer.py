import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from rlhf.algorithms.base import RLHFAlgorithm, RLHFBatch
from rlhf.core.logging import get_logger
from rlhf.data.dataset import RLHFDataset
from rlhf.tracking.experiment import ExperimentTracker
from rlhf.utils.distributed import is_distributed, get_world_size, get_rank


class RLHFTrainer:
    """Trainer for RLHF algorithms"""
    
    def __init__(
        self,
        algorithm: RLHFAlgorithm,
        train_dataset: RLHFDataset,
        eval_dataset: Optional[RLHFDataset] = None,
        experiment_tracker: Optional[ExperimentTracker] = None,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        eval_steps: int = 500,
        logging_steps: int = 10,
        save_steps: int = 1000,
        max_steps: int = 10000,
        num_workers: int = 4,
        output_dir: str = "outputs",
        device: Optional[torch.device] = None,
        deepspeed_config: Optional[str] = None,
        use_flash_attention: bool = False,
    ):
        """Initialize the RLHF trainer
        
        Args:
            algorithm: RLHF algorithm to train
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            experiment_tracker: Experiment tracker (e.g., W&B, TensorBoard)
            batch_size: Batch size per device
            gradient_accumulation_steps: Number of steps to accumulate gradients
            eval_steps: Frequency of evaluation
            logging_steps: Frequency of logging
            save_steps: Frequency of saving checkpoints
            max_steps: Maximum number of training steps
            num_workers: Number of data loading workers
            output_dir: Directory to save outputs
            device: Device to use for training
            deepspeed_config: Path to DeepSpeed configuration file
            use_flash_attention: Whether to use Flash Attention
        """
        self.algorithm = algorithm
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.experiment_tracker = experiment_tracker
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.max_steps = max_steps
        self.num_workers = num_workers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set up distributed training if needed
        self.is_distributed = is_distributed()
        self.world_size = get_world_size()
        self.rank = get_rank()
        
        # Set up logger
        self.logger = get_logger("RLHFTrainer")
        
        # Set up DeepSpeed if enabled
        self.deepspeed_config = deepspeed_config
        if self.deepspeed_config is not None:
            try:
                import deepspeed
                self.logger.info(f"Initializing DeepSpeed with config: {self.deepspeed_config}")
                # DeepSpeed initialization is deferred until train() is called
            except ImportError:
                self.logger.warning("DeepSpeed not available. Falling back to standard training.")
                self.deepspeed_config = None
        
        # Set up Flash Attention if enabled
        self.use_flash_attention = use_flash_attention
        if self.use_flash_attention:
            try:
                from flash_attn import flash_attn_func
                self.logger.info("Flash Attention is enabled")
                # Flash Attention is enabled at the model level, so we'll replace attention modules later
            except ImportError:
                self.logger.warning("Flash Attention not available. Falling back to standard attention.")
                self.use_flash_attention = False
        
        self.data_loaders = self._setup_data_loaders()
        
        self.logger.info(f"RLHF Trainer initialized with {self.world_size} processes")
        if self.is_distributed:
            self.logger.info(f"Process rank: {self.rank}")
    
    def _setup_data_loaders(self) -> Dict[str, DataLoader]:
        """Set up data loaders for training and evaluation"""
        train_batch_size = self.batch_size * self.gradient_accumulation_steps
        
        # For distributed training, adjust batch size and samplersmkdir -p src/rlhf/utils
cat > src/rlhf/utils/distributed.py << 'EOF'
import os
import torch
import torch.distributed as dist

def is_distributed() -> bool:
    """Check if distributed training is enabled
    
    Returns:
        True if distributed training is enabled
    """
    return dist.is_available() and dist.is_initialized()

def get_world_size() -> int:
    """Get the number of processes in the distributed group
    
    Returns:
        World size (number of processes)
    """
    if is_distributed():
        return dist.get_world_size()
    return 1

def get_rank() -> int:
    """Get the rank of the current process in the distributed group
    
    Returns:
        Rank of the current process
    """
    if is_distributed():
        return dist.get_rank()
    return 0

def is_main_process() -> bool:
    """Check if the current process is the main process (rank 0)
    
    Returns:
        True if current process is the main process
    """
    return get_rank() == 0

def init_distributed() -> None:
    """Initialize distributed training if environment variables are set"""
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        # Initialize with NCCL backend
        dist.init_process_group(backend="nccl")
        
        # Set device based on local rank
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        
        # Log initialization
        from rlhf.core.logging import get_logger
        logger = get_logger("distributed")
        logger.info(f"Initialized distributed training: rank={get_rank()}, world_size={get_world_size()}")

def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """Gather tensors from all processes and concatenate them
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        Concatenated tensor from all processes
    """
    if not is_distributed():
        return tensor
    
    world_size = get_world_size()
    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    
    dist.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0)
