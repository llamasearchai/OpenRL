from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from rlhf.core.logging import get_logger


@dataclass
class ValueModelOutput:
    """Output for value model"""
    values: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None


class ValueHead(nn.Module):
    """Value head for critic model"""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        """Initialize value head
        
        Args:
            hidden_size: Size of hidden states
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.value_head = nn.Linear(hidden_size, 1)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass
        
        Args:
            hidden_states: Hidden states from the model [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Value estimates [batch_size, seq_len]
        """
        # Apply dropout
        x = self.dropout(hidden_states)
        
        # Apply value head
        values = self.value_head(x).squeeze(-1)  # [batch_size, seq_len]
        
        return values


class ValueModelWrapper:
    """Wrapper for value model (critic) for RLHF"""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        value_head: Optional[ValueHead] = None,
    ):
        """Initialize value model wrapper
        
        Args:
            model: HuggingFace model
            tokenizer: Tokenizer for the model
            value_head: Optional value head (critic head)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = model.config
        self.logger = get_logger("ValueModel")
        
        # Initialize value head if not provided
        self.value_head = value_head or ValueHead(
            hidden_size=self.config.hidden_size,
            dropout=0.1,
        )
        
        # Get trainable parameters
        self.trainable_parameters = list(self.get_trainable_parameters())
        
        self.logger.info(f"Value model initialized with {sum(p.numel() for p in self.trainable_parameters):,} trainable parameters")
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get trainable parameters
        
        Returns:
            List of trainable parameters
        """
        # For the value model, we only train the value head
        # Freeze base model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Train value head parameters
        for param in self.value_head.parameters():
            param.requires_grad = True
        
        return self.value_head.parameters()
    
    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Union[ValueModelOutput, Tuple]:
        """Forward pass
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_dict: Whether to return a dictionary or tuple
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            
        Returns:
            Value model outputs
        """
        # Forward pass through base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Need hidden states for value head
            return_dict=True,
        )
        
        # Get hidden states
        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
        
        # Forward pass through value head
        values = self.value_head(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        
        if return_dict:
            return ValueModelOutput(
                values=values,
                hidden_states=outputs.hidden_states if output_hidden_states else None,
                attentions=outputs.attentions if output_attentions else None,
            )
        else:
            return (values,)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary
        
        Returns:
            State dictionary
        """
        return {
            "value_head": self.value_head.state_dict(),
            "model": self.model.state_dict(),
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary
        
        Args:
            state_dict: State dictionary to load
        """
        if "value_head" in state_dict:
            self.value_head.load_state_dict(state_dict["value_head"])
        if "model" in state_dict:
            self.model.load_state_dict(state_dict["model"])
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        device_map: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> "ValueModelWrapper":
        """Load a value model from a pre-trained model
        
        Args:
            model_name_or_path: Model name or path
            tokenizer_name_or_path: Tokenizer name or path (defaults to model_name_or_path)
            device_map: Device map for loading large models
            **kwargs: Additional arguments for loading the model
            
        Returns:
            ValueModelWrapper instance
        """
        logger = get_logger("ValueModel")
        
        # Load tokenizer first
        tokenizer_path = tokenizer_name_or_path or model_name_or_path
        
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                padding_side="left",
                trust_remote_code=kwargs.get("trust_remote_code", False),
            )
            
            # Ensure tokenizer has padding token
            if tokenizer.pad_token is None:
                logger.info("Setting pad_token to eos_token")
                tokenizer.pad_token = tokenizer.eos_token
            
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
        
        # Load model
        try:
            # Set device map for efficient loading
            model_kwargs = kwargs.copy()
            if device_map is not None:
                model_kwargs["device_map"] = device_map
            
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                model_name_or_path,
                **model_kwargs
            )
            
            # Ensure model and tokenizer are compatible
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(len(tokenizer))
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        return cls(model=model, tokenizer=tokenizer)