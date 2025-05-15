#!/usr/bin/env python
"""
Simple DPO Fine-tuning Example

This script demonstrates how to use the RLHF engineering system to fine-tune
a language model using Direct Preference Optimization (DPO).
"""

import os
import argparse
from pathlib import Path

import torch
from transformers import set_seed

from rlhf.algorithms.dpo import DPOAlgorithm, DPOConfig
from rlhf.models.policy import PolicyModelWrapper
from rlhf.models.reference import ReferenceModelWrapper
from rlhf.training.trainer import RLHFTrainer
from rlhf.data.dataset import create_dataset_from_jsonl
from rlhf.tracking.experiment import create_experiment_tracker


def parse_args():
    parser = argparse.ArgumentParser(description="Simple DPO fine-tuning example")
    
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path (JSONL)")
    parser.add_argument("--output-dir", type=str, default="./outputs/dpo", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tracker", type=str, default="tensorboard", help="Experiment tracker (tensorboard, wandb, none)")
    parser.add_argument("--run-name", type=str, default=None, help="Experiment run name")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup LoRA if enabled
    lora_config = None
    if args.use_lora:
        lora_config = {
            "rank": args.lora_rank,
            "alpha": args.lora_rank * 2,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        }
    
    # Load models
    print(f"Loading model from {args.model}")
    policy_model = PolicyModelWrapper.from_pretrained(
        model_name_or_path=args.model,
        lora_config=lora_config,
        device_map="auto",
    )
    
    reference_model = ReferenceModelWrapper.from_pretrained(
        model_name_or_path=args.model,
        device_map="auto",
    )
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}")
    train_dataset, eval_dataset = create_dataset_from_jsonl(
        data_path=args.dataset,
        tokenizer=policy_model.tokenizer,

            train_test_split=0.9,
            max_length=args.max_length,
            algorithm="dpo",
        )
    
        # Create experiment tracker
        run_name = args.run_name or f"dpo_{Path(args.model).name}"
        tracker = create_experiment_tracker(
            tracker_type=args.tracker,
            run_name=run_name,
            output_dir=str(output_dir),
        )
    
        # Create DPO config
        dpo_config = DPOConfig(
            learning_rate=args.lr,
            beta=args.beta,
            weight_decay=0.01,
            seed=args.seed,
            mixed_precision=True,
        )
    
        # Create DPO algorithm
        print("Initializing DPO algorithm")
        algorithm = DPOAlgorithm(
            config=dpo_config,
            policy_model=policy_model,
            reference_model=reference_model,
        )
    
        # Create trainer
        print("Creating trainer")
        trainer = RLHFTrainer(
            algorithm=algorithm,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            experiment_tracker=tracker,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            eval_steps=200,
            logging_steps=10,
            save_steps=200,
            max_steps=args.max_steps,
            output_dir=str(output_dir),
            use_flash_attention=False,  # Enable if you have flash-attention installed
        )
    
        # Start training
        print("Starting training")
        trainer.train()
    
        # Save final model
        final_model_path = output_dir / "final_model"
        policy_model.save_pretrained(str(final_model_path))
        print(f"Model saved to {final_model_path}")


if __name__ == "__main__":
        main()            max_length=args.max_length,
            algorithm="dpo",
        )
    
        # Create experiment tracker
        run_name = args.run_name or f"dpo_{Path(args.model).name}"
        tracker = create_experiment_tracker(
            tracker_type=args.tracker,
            run_name=run_name,
            output_dir=str(output_dir),
        )
    
        # Create DPO config
        dpo_config = DPOConfig(
            learning_rate=args.lr,
            beta=args.beta,
            weight_decay=0.01,
            seed=args.seed,
            mixed_precision=True,
        )
    
        # Create DPO algorithm
        print("Initializing DPO algorithm")
        algorithm = DPOAlgorithm(
            config=dpo_config,
            policy_model=policy_model,
            reference_model=reference_model,
        )
    
        # Create trainer
        print("Creating trainer")
        trainer = RLHFTrainer(
            algorithm=algorithm,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            experiment_tracker=tracker,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            eval_steps=200,
            logging_steps=10,
            save_steps=200,
            max_steps=args.max_steps,
            output_dir=str(output_dir),
            use_flash_attention=False,  # Enable if you have flash-attention installed
        )
    
        # Start training
        print("Starting training")
        trainer.train()
    
        # Save final model
        final_model_path = output_dir / "final_model"
        policy_model.save_pretrained(str(final_model_path))
        print(f"Model saved to {final_model_path}")


if __name__ == "__main__":
        main()        max_length=args.max_length,