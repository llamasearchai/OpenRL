from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from pydantic import Field

from rlhf.algorithms.base import RLHFAlgorithm, RLHFAlgorithmConfig, RLHFAlgorithmType, RLHFBatch
from rlhf.models.policy import PolicyModelWrapper
from rlhf.models.reference import ReferenceModelWrapper


class KTOConfig(RLHFAlgorithmConfig):
    """Configuration for KL-constrained Preference Optimization (KTO) algorithm"""
    algorithm_type: RLHFAlgorithmType = RLHFAlgorithmType.KTO
    
    # KTO specific hyperparameters
    alpha: float = 5.0
    margin: float = 0.0
    use_ipo_loss: bool = False


class KTOAlgorithm(RLHFAlgorithm):
    """KL-constrained Preference Optimization (KTO) algorithm for RLHF"""
    
    def __init__(
        self,
        config: KTOConfig,
        policy_model: PolicyModelWrapper,
        reference_model: ReferenceModelWrapper,
    ):
        """Initialize KTO algorithm
        
        Args:
            config: KTO configuration
            policy_model: Policy model to be fine-tuned
            reference_model: Reference model (frozen)
        """
        super().__init__(
            config=config,
            policy_model=policy_model,
            reference_model=reference_model,
        )
        self.config = config  # For type hinting
        
        # KTO requires a reference model
        assert self.reference_model is not None, "Reference model is required for KTO"
        
        # Freeze reference model
        for param in self.reference_model.model.parameters():
            param.requires_grad = False
    
    def _get_logps(
        self,
        model: Union[PolicyModelWrapper, ReferenceModelWrapper],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get log probabilities from a model for the given sequence
        
        Args:
            model: Model to get log probabilities from
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Log probabilities [batch_size]
        """
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        logits = outputs.logits[:, :-1]  # [batch_size, seq_len-1, vocab_size]
        labels = input_ids[:, 1:]  # [batch_size, seq_len-1]
        
        # Create a mask for relevant tokens (exclude padding)
        label_mask = (labels != self.policy_model.tokenizer.pad_token_id).float()
        
        # Get log probs for chosen tokens
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, -1, labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Apply mask and sum log probs
        masked_log_probs = token_log_probs * label_mask
        seq_log_probs = masked_log_probs.sum(dim=-1) / (label_mask.sum(dim=-1) + 1e-6)
        
        return seq_log_probs
    
    def compute_loss(self, batch: RLHFBatch) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute KTO loss for a batch of data
        
        Args:
            batch: RLHFBatch containing chosen and rejected completions
            
        Returns:
            Tuple of (loss, metrics dictionary)
        """
        # Check that batch has both chosen and rejected inputs
        if batch.chosen_input_ids is None or batch.rejected_input_ids is None:
            raise ValueError("KTO requires both chosen and rejected completions")
        
        # Get log probs from policy model
        policy_chosen_logps = self._get_logps(
            model=self.policy_model,
            input_ids=batch.chosen_input_ids,
            attention_mask=batch.chosen_attention_mask,
        )
        
        policy_rejected_logps = self._get_logps(
            model=self.policy_model,
            input_ids=batch.rejected_input_ids,
            attention_mask=batch.rejected_attention_mask,
        )
        
        # Get log probs from reference model
        with torch.no_grad():
            reference_chosen_logps = self._get_logps(
                model=self.reference_model,
                input_ids=batch.chosen_input_ids,
                attention_mask=batch.chosen_attention_mask,
            )
            
            reference_rejected_logps = self._get_logps(
                model=self.reference_model,
                input_ids=batch.rejected_input_ids,
                attention_mask=batch.rejected_attention_mask,
            )
        
        # Compute KL divergences
        kl_chosen = reference_chosen_logps - policy_chosen_logps
        kl_rejected = reference_rejected_logps - policy_rejected_logps
        
        # Standard KTO loss
        if not self.config.use_ipo_loss:
            # Compute KTO loss: E[α * log(π(y_w)/π(y_l)) - KL(π_ref(y_w) || π(y_w)) + KL(π_ref(y_l) || π(y_l))]
            logits = self.config.alpha * (policy_chosen_logps - policy_rejected_logps) - (kl_chosen - kl_rejected)
            
            # Add margin if specified
            if self.config.margin > 0:
                logits = logits - self.config.margin
                
            # Use sigmoid loss
            loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        else:
            # Compute IPO (Implicit Preference Optimization) variant
            # IPO: E[max(0, KL(π_ref(y_w) || π(y_w)) - KL(π_ref(y_l) || π(y_l)) + α * log(π(y_l)/π(y_w)) + margin)]
            loss = torch.relu(kl_chosen - kl_rejected + self.config.alpha * (policy_rejected_logps - policy_chosen_logps) + self.config.margin).mean()
        
        # Compute accuracy
        with torch.no_grad():
            if not self.config.use_ipo_loss:
                accuracy = (logits > 0).float().mean()
            else:
                policy_logits = policy_chosen_logps - policy_rejected_logps
                accuracy = (policy_logits > 0).float().mean()
        
        # Compute metrics
        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "policy_chosen_logps_mean": policy_chosen_logps.mean().item(),
            "policy_rejected_logps_mean": policy_rejected_logps.mean().item(),
            "reference_chosen_logps_mean": reference_chosen_logps.mean().item(),
            "reference_rejected_logps_mean": reference_rejected_logps.mean().item(),
            "kl_chosen_mean": kl_chosen.mean().item(),
            "kl_rejected_mean": kl_rejected.mean().item(),
        }
        
        return loss, metrics