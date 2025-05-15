from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from rlhf.core.logging import get_logger
from rlhf.models.policy import PolicyModelOutput


class ReferenceModelWrapper:
    """Wrapper for reference model for RLHF"""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        """Initialize reference model wrapper
        
        Args:
            model: HuggingFace model
            tokenizer: Tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = model.config
        self.logger = get_logger("ReferenceModel")
        
        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.logger.info(f"Reference model initialized (frozen)")
    
    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Union[PolicyModelOutput, Tuple]:
        """Forward pass of the reference model
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            past_key_values: Past key values for efficient decoding
            position_ids: Position IDs
            return_dict: Whether to return a dictionary or tuple
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            
        Returns:
            Output containing logits and optionally other tensors
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=past_key_values is not None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        if return_dict:
            return PolicyModelOutput(
                logits=outputs.logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                past_key_values=outputs.past_key_values,
            )
        else:
            return outputs
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        device_map: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> "ReferenceModelWrapper":
        """Load a reference model from a pre-trained model
        
        Args:
            model_name_or_path: Model name or path
            tokenizer_name_or_path: Tokenizer name or path (defaults to model_name_or_path)
            device_map: Device map for loading large models
            **kwargs: Additional arguments for loading the model
            
        Returns:
            ReferenceModelWrapper instance
        """
        logger = get_logger("ReferenceModel")
        
        # Load tokenizer first
        tokenizer_path = tokenizer_name_or_path or model_name_or_path
        
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                padding_side="left",  # For efficient generation
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
            
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **model_kwargs
            )
            
            # Ensure model and tokenizer are compatible
            model.resize_token_embeddings(len(tokenizer))
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        return cls(model=model, tokenizer=tokenizer)