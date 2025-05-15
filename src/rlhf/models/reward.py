from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from rlhf.core.logging import get_logger


@dataclass
class RewardModelOutput:
    """Output for reward model"""
    rewards: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None


class RewardModelWrapper:
    """Wrapper for reward model for RLHF"""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        reward_head: Optional[nn.Module] = None,
    ):
        """Initialize reward model wrapper
        
        Args:
            model: HuggingFace model
            tokenizer: Tokenizer for the model
            reward_head: Optional reward head for the model
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = model.config
        self.logger = get_logger("RewardModel")
        
        # Initialize reward head if not provided
        self.reward_head = reward_head or RewardHead(
            hidden_size=self.config.hidden_size,
            dropout=0.1,
        )
        
        # Freeze the base model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Enable training for reward head
        for param in self.reward_head.parameters():
            param.requires_grad = True
        
        self.logger.info(f"Reward model initialized with {sum(p.numel() for p in self.reward_head.parameters()):,} trainable parameters")
    
    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Union[RewardModelOutput, Tuple]:
        """Forward pass of the reward model
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            return_dict: Whether to return a dictionary or tuple
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            
        Returns:
            Output containing rewards and optionally other tensors
        """
        # Forward pass through base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Need hidden states for reward head
            return_dict=True,
        )
        
        # Get sequence output
        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
        
        # Get reward scores
        rewards = self.reward_head(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        
        if return_dict:
            return RewardModelOutput(
                rewards=rewards,
                hidden_states=outputs.hidden_states if output_hidden_states else None,
                attentions=outputs.attentions if output_attentions else None,
            )
        else:
            return (rewards,)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary of the reward head
        
        Returns:
            State dictionary
        """
        return {
            "reward_head": self.reward_head.state_dict(),
            "model": self.model.state_dict(),
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary into the model
        
        Args:
            state_dict: State dictionary to load
        """
        if "reward_head" in state_dict:
            self.reward_head.load_state_dict(state_dict["reward_head"])
        if "model" in state_dict:
            self.model.load_state_dict(state_dict["model"])
    
    def save_pretrained(self, path: str) -> None:
        """Save the model, reward head, and tokenizer
        
        Args:
            path: Path to save to
        """
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save base model and tokenizer
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save reward head
        torch.save(self.reward_head.state_dict(), os.path.join(path, "reward_head.pt"))
        
        self.logger.info(f"Reward model saved to {path}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        reward_head_path: Optional[str] = None,
        device_map: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> "RewardModelWrapper":
        """Load a reward model from a pre-trained model
        
        Args:
            model_name_or_path: Model name or path
            tokenizer_name_or_path: Tokenizer name or path (defaults to model_name_or_path)
            reward_head_path: Path to reward head state dictionary
            device_map: Device map for loading large models
            **kwargs: Additional arguments for loading the model
            
        Returns:
            RewardModelWrapper instance
        """
        logger = get_logger("RewardModel")
        
        # Load tokenizer first
        tokenizer_path = tokenizer_name_or_path or model_name_or_path
        
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
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
        
        # Create reward model
        reward_model = cls(model=model, tokenizer=tokenizer)
        
        # Load reward head if provided
        if reward_head_path is not None:
            try:
                reward_head_state = torch.load(reward_head_path, map_location="cpu")
                reward_model.reward_head.load_state_dict(reward_head_state)
                logger.info(f"Loaded reward head from {reward_head_path}")
            except Exception as e:
                logger.error(f"Error loading reward head: {e}")
                raise
        
        return reward_model


class RewardHead(nn.Module):
    """Reward head for reward model"""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        """Initialize reward head
        
        Args:
            hidden_size: Size of hidden states
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.reward = nn.Linear(hidden_size, 1)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the reward head
        
        Args:
            hidden_states: Hidden states from model [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Reward scores [batch_size]
        """
        # Use the last non-masked token for reward prediction
        if attention_mask is not None:
            # Find last non-masked token indices
            last_token_indices = attention_mask.sum(dim=1) - 1  # [batch_size]
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            
            # Select last hidden states
            last_states = hidden_states[batch_indices, last_token_indices]  # [batch_size, hidden_size]
        else:
            # Use the last token if no mask is provided
            last_states = hidden_states[:, -1]  # [batch_size, hidden_size]
        
        # Apply reward head
        x = self.dropout(last_states)
        x = self.dense(x)
        x = torch.tanh(x)  # Use tanh for stable reward values
        x = self.dropout(x)
        rewards = self.reward(x).squeeze(-1)  # [batch_size]
        
        return rewards
EOFcat > src/rlhf/models/critic.py << 'EOF'
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from rlhf.core.logging import get_logger


@dataclass
class ValueModelOutput:
    """Output for value model (critic)"""
    values: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.cat >> src/rlhf/models/critic.py << 'EOF'
    """Output for value model (critic)"""
    values: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None


class ValueModelWrapper:
    """Wrapper for value model (critic) for RLHF"""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        value_head: Optional[nn.Module] = None,
    ):
        """Initialize value model wrapper
        
        Args:
            model: HuggingFace model
            tokenizer: Tokenizer for the model
            value_head: Optional value head for the model
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
        
        # Freeze the base model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Enable training for value head
        for param in self.value_head.parameters():
            param.requires_grad = True
        
        # Get trainable parameters
        self.trainable_parameters = list(self.value_head.parameters())
        
        self.logger.info(f"Value model initialized with {sum(p.numel() for p in self.trainable_parameters):,} trainable parameters")
    
    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Union[ValueModelOutput, Tuple]:
        """Forward pass of the value model
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            return_dict: Whether to return a dictionary or tuple
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            
        Returns:
            Output containing values and optionally other tensors
        """
        # Forward pass through base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Need hidden states for value head
            return_dict=True,
        )
        
        # Get sequence output
        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
        
        # Get value predictions
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
        """Get state dictionary of the value head
        
        Returns:
            State dictionary
        """
        return {
            "value_head": self.value_head.state_dict(),
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary into the model
        
        Args:
            state_dict: State dictionary to load
        """
        if "value_head" in state_dict:
            self.value_head.load_state_dict(state_dict["value_head"])
    
    def save_pretrained(self, path: str) -> None:
        """Save the model and value head
        
        Args:
            path: Path to save to
        """
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save value head
        torch.save(self.value_head.state_dict(), os.path.join(path, "value_head.pt"))
        self.logger.info(f"Value model saved to {path}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        value_head_path: Optional[str] = None,
        device_map: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> "ValueModelWrapper":
        """Load a value model from a pre-trained model
        
        Args:
            model_name_or_path: Model name or path
            tokenizer_name_or_path: Tokenizer name or path (defaults to model_name_or_path)
            value_head_path: Path to value head state dictionary
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
        
        # Create value model
        value_model = cls(model=model, tokenizer=tokenizer)
        
        # Load value head if provided
        if value_head_path is not None:
            try:
                value_head_state = torch.load(value_head_path, map_location="cpu")
                value_model.value_head.load_state_dict(value_head_state)
                logger.info(f"Loaded value head from {value_head_path}")
            except Exception as e:
                logger.error(f"Error loading value head: {e}")
                raise
        
        return value_model


class ValueHead(nn.Module):
    """Value head for value model (critic)"""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        """Initialize value head
        
        Args:
            hidden_size: Size of hidden states
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the value head
        
        Args:
            hidden_states: Hidden states from model [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Value predictions [batch_size, seq_len]
        """
        # Apply value head at every token position
        x = self.dropout(hidden_states)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        values = self.value(x).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            values = values * attention_mask
        
        return values
