from typing import Optional, Union, Dict, Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from rlhf.core.logging import get_logger

logger = get_logger("FlashAttention")


def replace_with_flash_attention(model: nn.Module) -> None:
    """Replace standard attention with Flash Attention in the model
    
    Args:
        model: Model to modify
    """
    try:
        from flash_attn.flash_attention import FlashAttention
        from transformers.models.llama.modeling_llama import LlamaAttention
        from transformers.models.mistral.modeling_mistral import MistralAttention
    except ImportError:
        logger.error("Flash Attention not available. Install with 'pip install flash-attn'")
        return
    
    attention_counter = 0
    
    # Define Flash Attention wrapper for LlamaAttention
    class FlashLlamaAttention(LlamaAttention):
        def __init__(self, original_module):
            super().__init__(original_module.config)
            # Copy attributes from original module
            for attr_name, attr_value in original_module.__dict__.items():
                if not attr_name.startswith('_'):
                    setattr(self, attr_name, attr_value)
            
            # Initialize Flash Attention
            self.flash_attn = FlashAttention(
                softmax_scale=1.0 / (self.head_dim ** 0.5),
                attention_dropout=0.0
            )
        
        def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        ):
            # Normal query, key, value projections
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            
            # Reshape for Flash Attention
            batch_size, seq_length, _ = hidden_states.shape
            query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Handle cache for generation
            if past_key_value is not None:
                # Combine with cached keys and values
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            
            past_key_value = (key_states, value_states) if use_cache else None
            
            # No attention mask for Flash Attention (handled internally)
            # Convert query, key, value to correct shape for Flash Attention
            query_states = query_states.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
            key_states = key_states.transpose(1, 2)      # [batch, seq_len, num_heads, head_dim]
            value_states = value_states.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
            
                    # Apply Flash Attention
                    attn_output, _ = self.flash_attn(
                        q=query_states,
                        k=key_states,
                        v=value_states,
                        causal=True
                    )
            
                    # Reshape output
                    attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
            
                    # Output projection
                    attn_output = self.o_proj(attn_output)
            
                    return attn_output, None, past_key_value
    
            # Define Flash Attention wrapper for MistralAttention
            class FlashMistralAttention(MistralAttention):
                def __init__(self, original_module):
                    super().__init__(original_module.config)
                    # Copy attributes from original module
                    for attr_name, attr_value in original_module.__dict__.items():
                        if not attr_name.startswith('_'):
                            setattr(self, attr_name, attr_value)
            
                    # Initialize Flash Attention
                    self.flash_attn = FlashAttention(
                        softmax_scale=1.0 / (self.head_dim ** 0.5),
                        attention_dropout=0.0
                    )
        
                def forward(
                    self,
                    hidden_states,
                    attention_mask=None,
                    position_ids=None,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                ):
                    # Normal query, key, value projections
                    query_states = self.q_proj(hidden_states)
                    key_states = self.k_proj(hidden_states)
                    value_states = self.v_proj(hidden_states)
            
                    # Reshape for Flash Attention
                    batch_size, seq_length, _ = hidden_states.shape
                    query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
                    key_states = key_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
                    value_states = value_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            
                    # Handle cache for generation
                    if past_key_value is not None:
                        # Combine with cached keys and values
                        key_states = torch.cat([past_key_value[0], key_states], dim=2)
                        value_states = torch.cat([past_key_value[1], value_states], dim=2)
            
                    past_key_value = (key_states, value_states) if use_cache else None
            
                    # Convert query, key, value to correct shape for Flash Attention
                    query_states = query_states.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
                    key_states = key_states.transpose(1, 2)      # [batch, seq_len, num_heads, head_dim]
                    value_states = value_states.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
            
                    # Apply Flash Attention
                    attn_output, _ = self.flash_attn(
                        q=query_states,
                        k=key_states,
                        v=value_states,
                        causal=True
                    )
            
                    # Reshape output
                    attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
            
                    # Output projection
                    attn_output = self.o_proj(attn_output)
            
                    return attn_output, None, past_key_value
    
            # Recursively replace attention modules
            for name, module in model.named_children():
                if isinstance(module, LlamaAttention):
                    setattr(model, name, FlashLlamaAttention(module))
                    attention_counter += 1
                elif isinstance(module, MistralAttention):
                    setattr(model, name, FlashMistralAttention(module))
                    attention_counter += 1
                else:
                    # Recursively apply to child modules
                    replace_with_flash_attention(module)
    
            logger.info(f"Replaced {attention_counter} attention modules with Flash Attention")            attn_output = self.flash_attn(