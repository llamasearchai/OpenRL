from typing import Dict, List, Optional, Tuple, Union, Callable

import os
import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from rlhf.algorithms.base import RLHFBatch
from rlhf.core.logging import get_logger

logger = get_logger("Dataset")


class RLHFDataset(Dataset):
    """Dataset for RLHF training"""
    
    def __init__(
        self,
        items: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
        algorithm_type: str = "ppo",
    ):
        """Initialize RLHF dataset
        
        Args:
            items: List of data items
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            algorithm_type: Type of RLHF algorithm (ppo, dpo, kto)
        """
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.algorithm_type = algorithm_type.lower()
        
        logger.info(f"Initialized {self.algorithm_type} dataset with {len(items)} items")
        
        # Validate algorithm type
        if self.algorithm_type not in ["ppo", "dpo", "kto"]:
            raise ValueError(f"Unsupported algorithm type: {self.algorithm_type}")
    
    def __len__(self) -> int:
        """Get number of items in dataset
        
        Returns:
            Number of items
        """
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get item by index
        
        Args:
            idx: Item index
            
        Returns:
            Item dictionary
        """
        return self.items[idx]
    
    def _tokenize(self, text: str, max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Tokenize text
        
        Args:
            text: Text to tokenize
            max_length: Maximum sequence length (overrides self.max_length)
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        max_length = max_length or self.max_length
        
        # Tokenize
        tokenized = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
        }
    
    def collate_fn(self, batch: List[Dict]) -> RLHFBatch:
        """Collate batch of items
        
        Args:
            batch: Batch of items
            
        Returns:
            Collated batch
        """
        if self.algorithm_type == "ppo":
            return self._collate_ppo(batch)
        elif self.algorithm_type in ["dpo", "kto"]:
            return self._collate_preference(batch)
        else:
            raise ValueError(f"Unsupported algorithm type: {self.algorithm_type}")
    
    def _collate_ppo(self, batch: List[Dict]) -> RLHFBatch:
        """Collate batch for PPO
        
        Args:
            batch: Batch of items
            
        Returns:
            Collated batch
        """
        # Extract prompts from batch
        prompts = [item["prompt"] for item in batch]
        
        # Tokenize prompts
        tokenized_prompts = [self._tokenize(prompt) for prompt in prompts]
        
        # Stack tensors
        prompt_input_ids = torch.stack([item["input_ids"] for item in tokenized_prompts])
        prompt_attention_mask = torch.stack([item["attention_mask"] for item in tokenized_prompts])
        
        # Create batch
        return RLHFBatch(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            metadata=[{k: v for k, v in item.items() if k != "prompt"} for item in batch],
        )
    
    def _collate_preference(self, batch: List[Dict]) -> RLHFBatch:
        """Collate batch for preference-based methods (DPO, KTO)
        
        Args:
            batch: Batch of items
            
        Returns:
            Collated batch
        """
        # Extract data from batch
        chosen_texts = [item["chosen"] for item in batch]
        rejected_texts = [item["rejected"] for item in batch]
        prompts = [item.get("prompt", "") for item in batch]
        
        # Tokenize pairs
        tokenized_chosen = [self._tokenize(prompt + chosen) for prompt, chosen in zip(prompts, chosen_texts)]
        tokenized_rejected = [self._tokenize(prompt + rejected) for prompt, rejected in zip(prompts, rejected_texts)]
        tokenized_prompts = [self._tokenize(prompt, max_length=min(len(prompt.split()), self.max_length // 2)) for prompt in prompts]
        
        # Stack tensors
        chosen_input_ids = torch.stack([item["input_ids"] for item in tokenized_chosen])
        chosen_attention_mask = torch.stack([item["attention_mask"] for item in tokenized_chosen])
        rejected_input_ids = torch.stack([item["input_ids"] for item in tokenized_rejected])
        rejected_attention_mask = torch.stack([item["attention_mask"] for item in tokenized_rejected])
        prompt_input_ids = torch.stack([item["input_ids"] for item in tokenized_prompts])
        prompt_attention_mask = torch.stack([item["attention_mask"] for item in tokenized_prompts])
        
        # Create batch
        return RLHFBatch(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            chosen_input_ids=chosen_input_ids,
            chosen_attention_mask=chosen_attention_mask,
            rejected_input_ids=rejected_input_ids,
            rejected_attention_mask=rejected_attention_mask,
            metadata=[{k: v for k, v in item.items() if k not in ["chosen", "rejected", "prompt"]} for item in batch],
        )


def create_dataset_from_jsonl(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    train_test_split: float = 0.9,
    max_length: int = 1024,
    algorithm: str = "ppo",
    filter_fn: Optional[Callable[[Dict], bool]] = None,
) -> Tuple[RLHFDataset, Optional[RLHFDataset]]:
    """Create dataset from JSONL file
    
    Args:
        data_path: Path to JSONL file
        tokenizer: Tokenizer for encoding text
        train_test_split: Train/test split ratio
        max_length: Maximum sequence length
        algorithm: Type of RLHF algorithm (ppo, dpo, kto)
        filter_fn: Optional function to filter items
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    if data_path.startswith("hf://"):
        # Load from Hugging Face dataset
        try:
            from datasets import load_dataset
            dataset_name = data_path[5:]
            dataset = load_dataset(dataset_name)
            data_items = list(dataset["train"])
            logger.info(f"Loaded {len(data_items)} items from Hugging Face dataset {dataset_name}")
        except Exception as e:
            logger.error(f"Error loading Hugging Face dataset: {e}")
            raise
    else:
        # Load from local file
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        data_items = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data_items.append(item)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse line: {line}")
                    continue
        
        logger.info(f"Loaded {len(data_items)} items from {data_path}")
    
    # Apply filter if provided
    if filter_fn is not None:
        original_count = len(data_items)
        data_items = [item for item in data_items if filter_fn(item)]
        logger.info(f"Filtered {original_count - len(data_items)} items")
    
    # Split into train and eval
    if train_test_split < 1.0:
        split_idx = int(len(data_items) * train_test_split)
        train_items = data_items[:split_idx]
        eval_items = data_items[split_idx:]
        
        train_dataset = RLHFDataset(
            items=train_items,
            tokenizer=tokenizer,
            max_length=max_length,
            algorithm_type=algorithm,
        )
        
        eval_dataset = RLHFDataset(
            items=eval_items,
            tokenizer=tokenizer,
            max_length=max_length,
            algorithm_type=algorithm,
        )
        
        logger.info(f"Created train dataset with {len(train_items)} items")
        logger.info(f"Created eval dataset with {len(eval_items)} items")
        
        return train_dataset, eval_dataset
    else:
        # No split, return all data as train dataset
        train_dataset = RLHFDataset(
            items=data_items,
            tokenizer=tokenizer,
            max_length=max_length,
            algorithm_type=algorithm,
        )
        
        logger.info(f"Created train dataset with {len(data_items)} items (no eval split)")
        
        return train_dataset, None