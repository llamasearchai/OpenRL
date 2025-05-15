import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from rlhf.algorithms.base import RLHFBatch
from rlhf.core.logging import get_logger

logger = get_logger("RLHFDataset")


class RLHFDataset(Dataset):
    """Base dataset for RLHF training"""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
        algorithm_type: str = "dpo",
    ):
        """Initialize RLHF dataset
        
        Args:
            data: List of data samples
            tokenizer: Tokenizer for encoding texts
            max_length: Maximum sequence length
            algorithm_type: Type of algorithm (e.g., "dpo", "ppo", "kto")
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.algorithm_type = algorithm_type.lower()
        
        logger.info(f"Initialized {self.algorithm_type} dataset with {len(data)} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary of tokenized data
        """
        item = self.data[idx]
        
        # Process based on algorithm type
        if self.algorithm_type in ["dpo", "kto"]:
            return self._process_preference_item(item)
        elif self.algorithm_type == "ppo":
            return self._process_ppo_item(item)
        else:
            raise ValueError(f"Unsupported algorithm type: {self.algorithm_type}")
    
    def _process_preference_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process item for preference-based algorithms (DPO, KTO)
        
        Args:
            item: Data item with prompt, chosen, and rejected completions
            
        Returns:
            Dictionary of tokenized data
        """
        prompt = item["prompt"]
        chosen = item.get("chosen", item.get("chosen_response", ""))
        rejected = item.get("rejected", item.get("rejected_response", ""))
        
        if not chosen or not rejected:
            logger.warning(f"Missing chosen or rejected response in item: {item}")
        
        # Tokenize prompt separately
        prompt_tokens = self.tokenizer(
            prompt,
            max_length=self.max_length // 2,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Tokenize chosen completion (with prompt)
        chosen_tokens = self.tokenizer(
            prompt + chosen,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Tokenize rejected completion (with prompt)
        rejected_tokens = self.tokenizer(
            prompt + rejected,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "prompt_input_ids": prompt_tokens.input_ids.squeeze(0),
            "prompt_attention_mask": prompt_tokens.attention_mask.squeeze(0),
            "chosen_input_ids": chosen_tokens.input_ids.squeeze(0),
            "chosen_attention_mask": chosen_tokens.attention_mask.squeeze(0),
            "rejected_input_ids": rejected_tokens.input_ids.squeeze(0),
            "rejected_attention_mask": rejected_tokens.attention_mask.squeeze(0),
            "metadata": {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            },
        }
    
    def _process_ppo_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process item for PPO algorithm
        
        Args:
            item: Data item with prompt
            
        Returns:
            Dictionary of tokenized data
        """
        prompt = item["prompt"]
        
        # Tokenize prompt
        prompt_tokens = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "prompt_input_ids": prompt_tokens.input_ids.squeeze(0),
            "prompt_attention_mask": prompt_tokens.attention_mask.squeeze(0),
            "metadata": {
                "prompt": prompt,
            },
        }
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> RLHFBatch:
        """Collate function for dataloader
        
        Args:
            batch: Batch of data items
            
        Returns:
            RLHFBatch object
        """
        # Extract tensors from batch
        prompt_input_ids = torch.stack([item["prompt_input_ids"] for item in batch])
        prompt_attention_mask = torch.stack([item["prompt_attention_mask"] for item in batch])
        
        # Initialize optional tensors
        chosen_input_ids = None
        chosen_attention_mask = None
        rejected_input_ids = None
        rejected_attention_mask = None
        
        # Process based on algorithm type
        if self.algorithm_type in ["dpo", "kto"]:
            chosen_input_ids = torch.stack([item["chosen_input_ids"] for item in batch])
            chosen_attention_mask = torch.stack([item["chosen_attention_mask"] for item in batch])
            rejected_input_ids = torch.stack([item["rejected_input_ids"] for item in batch])
            rejected_attention_mask = torch.stack([item["rejected_attention_mask"] for item in batch])
        
        # Collect metadata
        metadata = [item["metadata"] for item in batch]
        
        # Create batch
        return RLHFBatch(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            chosen_input_ids=chosen_input_ids,
            chosen_attention_mask=chosen_attention_mask,
            rejected_input_ids=rejected_input_ids,
            rejected_attention_mask=rejected_attention_mask,
            metadata=metadata,
        )


def create_dataset_from_jsonl(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    train_test_split: float = 0.9,
    max_length: int = 1024,
    algorithm: str = "dpo",
) -> Tuple[RLHFDataset, Optional[RLHFDataset]]:
    """Create datasets from a JSONL file
    
    Args:
        data_path: Path to JSONL file or Hugging Face dataset name
        tokenizer: Tokenizer for encoding texts
        train_test_split: Fraction of data to use for training
        max_length: Maximum sequence length
        algorithm: Algorithm type (e.g., "dpo", "ppo", "kto")
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    # Check if path exists or if it's a Hugging Face dataset
    if os.path.exists(data_path):
        # Load from local file
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse line: {line}")
        
        logger.info(f"Loaded {len(data)} samples from {data_path}")
    else:
        # Attempt to load from Hugging Face datasets
        try:
            from datasets import load_dataset
            dataset = load_dataset(data_path)
            
            # Convert to list of dictionaries
            if "train" in dataset:
                data = list(dataset["train"])
            else:
                data = list(dataset)
            
            logger.info(f"Loaded {len(data)} samples from Hugging Face dataset {data_path}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    # Split data into train and eval
    split_idx = int(len(data) * train_test_split)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]
    
    # Create datasets
    train_dataset = RLHFDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=max_length,
        algorithm_type=algorithm,
    )
    
    eval_dataset = RLHFDataset(
        data=eval_data,
        tokenizer=tokenizer,
        max_length=max_length,
        algorithm_type=algorithm,
    ) if eval_data else None
    
    return train_dataset, eval_dataset
