import abc
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from pydantic import BaseModel

from rlhf.core.logging import get_logger
from rlhf.models.policy import PolicyModelOutput, PolicyModelWrapper
from rlhf.models.reference import ReferenceModelWrapper
from rlhf.models.reward import RewardModelOutput, RewardModelWrapper


class RLHFAlgorithmType(str, Enum):
    """Types of RLHF algorithms supported by the system"""
    PPO = "ppo"
    DPO = "dpo"
    KTO = "kto"
    CUSTOM = "custom"


@dataclass
class RLHFBatch:
    """Batch of data for RLHF training"""
    # Tokenized prompt inputs
    prompt_input_ids: torch.Tensor  # [batch_size, prompt_seq_len]
    prompt_attention_mask: torch.Tensor  # [batch_size, prompt_seq_len]
    
    # Tokenized completions
    completion_input_ids: Optional[torch.Tensor] = None  # [batch_size, completion_seq_len]
    completion_attention_mask: Optional[torch.Tensor] = None  # [batch_size, completion_seq_len]
    
    # For pair-based methods like DPO, KTO
    chosen_input_ids: Optional[torch.Tensor] = None  # [batch_size, chosen_seq_len]
    chosen_attention_mask: Optional[torch.Tensor] = None  # [batch_size, chosen_seq_len]
    rejected_input_ids: Optional[torch.Tensor] = None  # [batch_size, rejected_seq_len]
    rejected_attention_mask: Optional[torch.Tensor] = None  # [batch_size, rejected_seq_len]
    
    # Optional reward labels (for supervised reward modeling)
    reward_scores: Optional[torch.Tensor] = None  # [batch_size]
    
    # Optional metadata
    metadata: Optional[Dict[str, Any]] = None
    
    def to(self, device: torch.device) -> "RLHFBatch":
        """Move batch to specified device"""
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
        return self
    
    def pin_memory(self) -> "RLHFBatch":
        """Pin memory for faster host-to-device transfer"""
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.pin_memory())
        return self


class RLHFAlgorithmConfig(BaseModel):
    """Base configuration for RLHF algorithms"""
    algorithm_type: RLHFAlgorithmType
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    use_adam8bit: bool = False
    kl_coef: float = 0.1
    discount_factor: float = 1.0
    gae_lambda: float = 0.95
    advantage_normalization: bool = True
    seed: int = 42
    mixed_precision: bool = True
    

class RLHFAlgorithm(abc.ABC):
    """Base class for RLHF algorithms"""
    
    def __init__(
        self,
        config: RLHFAlgorithmConfig,
        policy_model: PolicyModelWrapper,
        reference_model: Optional[ReferenceModelWrapper] = None,
        reward_model: Optional[RewardModelWrapper] = None,
    ):
        """Initialize RLHF algorithm
        
        Args:
            config: Algorithm configuration
            policy_model: Policy model to be fine-tuned
            reference_model: Reference model for KL divergence (often frozen)
            reward_model: Reward model for computing rewards
        """
        self.config = config
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.reward_model = reward_model
        self.logger = get_logger(f"{self.__class__.__name__}")
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize gradient scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)
        
        # Set random seed for reproducibility
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        
        # Initialize optimizer with AdamW (with optional 8-bit)
        if config.use_adam8bit:
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(
                    self.policy_model.trainable_parameters,
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
            except ImportError:
                self.logger.warning("bitsandbytes not installed, falling back to regular AdamW")
                self.optimizer = torch.optim.AdamW(
                    self.policy_model.trainable_parameters,
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
        else:
            self.optimizer = torch.optim.AdamW(
                self.policy_model.trainable_parameters,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        
        # Check models are compatible
        self._validate_models()
        
        self.logger.info(f"Initialized {config.algorithm_type} algorithm")
    
    def _validate_models(self) -> None:
        """Validate that the provided models are compatible with the algorithm"""
        if self.reference_model is not None:
            assert self.policy_model.config.vocab_size == self.reference_model.config.vocab_size, \
                "Policy and reference models must have the same vocabulary size"
        
        if self.reward_model is not None:
            assert self.policy_model.config.vocab_size == self.reward_model.config.vocab_size, \
                "Policy and reward models must have the same vocabulary size"
    
    @abc.abstractmethod
    def compute_loss(self, batch: RLHFBatch) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the loss for a batch of data
        
        Args:
            batch: Batch of data
            
        Returns:
            Tuple of (loss, metrics dictionary)
        """
        pass
    
    @torch.no_grad()
    def compute_rewards(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        completion_input_ids: torch.Tensor,
        completion_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute rewards for generated sequences
        
        Args:
            prompt_input_ids: Input IDs for prompts [batch_size, prompt_
cat >> src/rlhf/algorithms/base.py << 'EOF'
            prompt_input_ids: Input IDs for prompts [batch_size, prompt_seq_len]
            prompt_attention_mask: Attention mask for prompts [batch_size, prompt_seq_len]
            completion_input_ids: Input IDs for completions [batch_size, completion_seq_len]
            completion_attention_mask: Attention mask for completions [batch_size, completion_seq_len]
            
        Returns:
            Tensor of rewards [batch_size]
        """
        if self.reward_model is None:
            raise ValueError("Reward model is required to compute rewards")
        
        # Combine prompts and completions for reward model
        input_ids = torch.cat([prompt_input_ids, completion_input_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, completion_attention_mask], dim=1)
        
        # Compute rewards
        reward_outputs: RewardModelOutput = self.reward_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        return reward_outputs.rewards
    
    @torch.no_grad()
    def compute_kl_divergence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence between policy and reference models
        
        Args:
            input_ids: Input IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tensor of KL divergence values [batch_size]
        """
        if self.reference_model is None:
            raise ValueError("Reference model is required to compute KL divergence")
        
        # Get log probs from policy model
        policy_outputs = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        policy_logits = policy_outputs.logits[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
        policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
        
        # Get log probs from reference model
        reference_outputs = self.reference_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        reference_logits = reference_outputs.logits[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
        reference_log_probs = torch.log_softmax(reference_logits, dim=-1)
        
        # Compute KL divergence: E_p[log(p/q)] = E_p[log p - log q]
        # Get token probabilities
        target_ids = input_ids[:, 1:]  # Shift right for targets
        target_mask = attention_mask[:, 1:]  # Shift right for mask
        
        # Gather log probs at target positions
        target_policy_log_probs = torch.gather(
            policy_log_probs,
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)  # [batch_size, seq_len-1]
        
        target_reference_log_probs = torch.gather(
            reference_log_probs,
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)  # [batch_size, seq_len-1]
        
        # Compute KL divergence
        kl_div = (target_policy_log_probs - target_reference_log_probs) * target_mask
        kl_div = kl_div.sum(dim=1) / target_mask.sum(dim=1)  # Average over sequence length
        
        return kl_div
    
    def train_step(self, batch: RLHFBatch) -> Dict[str, float]:
        """Perform a single training step
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary of metrics
        """
        batch = batch.to(self.device)
        
        # Compute loss
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
            loss, metrics = self.compute_loss(batch)
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        # Optimization step if gradient accumulation is complete
        if self.optimizer.state["step"] % self.config.gradient_accumulation_steps == 0:
            # Clip gradients
            if self.config.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.policy_model.trainable_parameters,
                    self.config.max_grad_norm,
                )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        return {**metrics, "loss": loss.item() * self.config.gradient_accumulation_steps}
    
    def save_checkpoint(self, path: str) -> None:
        """Save algorithm checkpoint
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "policy_model": self.policy_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": self.config.dict(),
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load algorithm checkpoint
        
        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_model.load_state_dict(checkpoint["policy_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.logger.info(f"Loaded checkpoint from {path}")
