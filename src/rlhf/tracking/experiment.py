from typing import Any, Dict, Optional, Union

from rlhf.core.logging import get_logger

logger = get_logger("ExperimentTracker")


class ExperimentTracker:
    """Base class for experiment tracking"""
    
    def __init__(self, run_name: str, output_dir: str):
        """Initialize experiment tracker
        
        Args:
            run_name: Name of the experiment run
            output_dir: Directory to save outputs
        """
        self.run_name = run_name
        self.output_dir = output_dir
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current training step
        """
        pass
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration
        
        Args:
            config: Dictionary of configuration parameters
        """
        pass
    
    def finish(self) -> None:
        """Finish tracking"""
        pass


class TensorBoardTracker(ExperimentTracker):
    """TensorBoard experiment tracker"""
    
    def __init__(self, run_name: str, output_dir: str):
        """Initialize TensorBoard tracker
        
        Args:
            run_name: Name of the experiment run
            output_dir: Directory to save outputs
        """
        super().__init__(run_name, output_dir)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=f"{output_dir}/tensorboard/{run_cat >> src/rlhf/tracking/experiment.py << 'EOF'
            run_name: Name of the experiment run
            output_dir: Directory to save outputs
        """
        super().__init__(run_name, output_dir)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=f"{output_dir}/tensorboard/{run_name}")
            logger.info(f"TensorBoard tracking initialized at {output_dir}/tensorboard/{run_name}")
        except ImportError:
            logger.warning("TensorBoard not available. Please install with 'pip install tensorboard'")
            self.writer = None
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to TensorBoard
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current training step
        """
        if self.writer is None:
            return
        
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration to TensorBoard
        
        Args:
            config: Dictionary of configuration parameters
        """
        if self.writer is None:
            return
        
        # Convert config to text (hparams are better but require extra work)
        config_str = "\n".join([f"{k}: {v}" for k, v in config.items()])
        self.writer.add_text("config", config_str)
    
    def finish(self) -> None:
        """Close TensorBoard writer"""
        if self.writer is not None:
            self.writer.close()


class WandbTracker(ExperimentTracker):
    """Weights & Biases experiment tracker"""
    
    def __init__(
        self,
        run_name: str,
        output_dir: str,
        project: str = "rlhf-engineering",
        entity: Optional[str] = None,
    ):
        """Initialize W&B tracker
        
        Args:
            run_name: Name of the experiment run
            output_dir: Directory to save outputs
            project: W&B project name
            entity: W&B entity name
        """
        super().__init__(run_name, output_dir)
        try:
            import wandb
            self.wandb = wandb
            
            # Initialize wandb run
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                dir=output_dir,
                resume="allow",
            )
            
            logger.info(f"W&B tracking initialized for project {project}, run {run_name}")
        except ImportError:
            logger.warning("Weights & Biases not available. Please install with 'pip install wandb'")
            self.wandb = None
            self.run = None
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to W&B
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current training step
        """
        if self.run is None:
            return
        
        if step is not None:
            metrics["step"] = step
        
        self.run.log(metrics)
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration to W&B
        
        Args:
            config: Dictionary of configuration parameters
        """
        if self.run is None:
            return
        
        self.run.config.update(config)
    
    def finish(self) -> None:
        """Finish W&B run"""
        if self.run is not None:
            self.run.finish()


def create_experiment_tracker(
    tracker_type: str,
    run_name: str,
    output_dir: str,
    **kwargs
) -> Optional[ExperimentTracker]:
    """Create experiment tracker based on type
    
    Args:
        tracker_type: Type of tracker ('tensorboard', 'wandb', 'none')
        run_name: Name of the experiment run
        output_dir: Directory to save outputs
        **kwargs: Additional arguments for specific trackers
        
    Returns:
        ExperimentTracker instance or None
    """
    tracker_type = tracker_type.lower()
    
    if tracker_type == "none" or not tracker_type:
        logger.info("Experiment tracking disabled")
        return None
    
    if tracker_type == "tensorboard":
        return TensorBoardTracker(run_name=run_name, output_dir=output_dir)
    
    if tracker_type == "wandb":
        return WandbTracker(
            run_name=run_name,
            output_dir=output_dir,
            project=kwargs.get("project", "rlhf-engineering"),
            entity=kwargs.get("entity", None),
        )
    
    # Invalid tracker type
    logger.warning(f"Unknown tracker type: {tracker_type}. Using TensorBoard as fallback.")
    return TensorBoardTracker(run_name=run_name, output_dir=output_dir)
