#!/usr/bin/env python3
"""
GPT-2 Pretraining Script using HuggingFace Trainer
"""

import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional
import multiprocessing

# Set environment variables for parallelism BEFORE importing tokenizers/torch
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OMP_NUM_THREADS"] = str(min(multiprocessing.cpu_count(), 64))
os.environ["MKL_NUM_THREADS"] = str(min(multiprocessing.cpu_count(), 64))

import torch
from datasets import load_from_disk
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    config_path: str = field(
        default="./configs/gpt2/config.json",
        metadata={"help": "Path to the model configuration file"}
    )
    tokenizer_path: str = field(
        default="./model_gpt2_tokenizer",
        metadata={"help": "Path to the pretrained tokenizer"}
    )


@dataclass
class DataArguments:
    """Arguments for data loading."""
    data_path: str = field(
        default=None,
        metadata={"help": "Path to the dataset (arrow format)"}
    )
    text_file: str = field(
        default="datasets/mapped_bnc_sentences.txt",
        metadata={"help": "Path to a text file (one example per line)"}
    )
    max_seq_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length for training"}
    )
    validation_split_percentage: int = field(
        default=5,
        metadata={"help": "Percentage of data to use for validation"}
    )


def load_config(config_path: str) -> GPT2Config:
    """Load GPT-2 configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return GPT2Config(**config_dict)


def prepare_dataset(dataset, tokenizer, max_seq_length: int):
    """Tokenize and prepare dataset for training."""
    
    # Use all available CPU cores
    num_proc = min(multiprocessing.cpu_count(), 64)  # Cap at 64 to avoid overhead
    print(f"Using {num_proc} CPU cores for dataset processing")
    print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    print(f"TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM', 'not set')}")
    
    def tokenize_function(examples):
        # Tokenize the texts
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=True,
            return_attention_mask=False,
        )
    
    # Tokenize the train dataset
    print("Tokenizing train dataset...")
    tokenized_train = dataset["train"].map(
        tokenize_function,
        batched=True,
        batch_size=100,  # Even smaller batches for more parallelism
        num_proc=num_proc,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing train dataset",
        writer_batch_size=100,
        load_from_cache_file=False,  # Force reprocessing to use all cores
    )
    
    # Tokenize the validation dataset
    print("Tokenizing validation dataset...")
    tokenized_val = dataset["validation"].map(
        tokenize_function,
        batched=True,
        batch_size=100,
        num_proc=num_proc,
        remove_columns=dataset["validation"].column_names,
        desc="Tokenizing validation dataset",
        writer_batch_size=100,
        load_from_cache_file=False,
    )
    
    # Group texts into chunks of max_seq_length
    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # Drop the small remainder
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        
        # Split by chunks of max_seq_length
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result
    
    print("Grouping train texts into chunks...")
    processed_train = tokenized_train.map(
        group_texts,
        batched=True,
        batch_size=100,
        num_proc=num_proc,
        desc="Grouping train texts",
        writer_batch_size=100,
        load_from_cache_file=False,
    )
    
    print("Grouping validation texts into chunks...")
    processed_val = tokenized_val.map(
        group_texts,
        batched=True,
        batch_size=100,
        num_proc=num_proc,
        desc="Grouping validation texts",
        writer_batch_size=100,
        load_from_cache_file=False,
    )
    
    from datasets import DatasetDict
    return DatasetDict({
        "train": processed_train,
        "validation": processed_val
    })

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="GPT-2 Pretraining")
    parser.add_argument("--config-path", type=str, default="./configs/gpt2/config.json",
                        help="Path to model config JSON")
    parser.add_argument("--tokenizer-path", type=str, default="./model_gpt2_tokenizer",
                        help="Path to tokenizer")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to dataset (arrow format)")
    parser.add_argument("--text-file", type=str, default=None,
                        help="Path to text file (one example per line)")
    parser.add_argument("--max-seq-length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--validation-split", type=float, default=5,
                        help="Validation split percentage")
    parser.add_argument("--output-dir", type=str, default="./output_gpt2_pretrained",
                        help="Output directory for checkpoints")
    parser.add_argument("--num-epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size per device")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
                        help="Learning rate")
    
    args = parser.parse_args()
    
    # Create arguments objects from parsed args
    model_args = ModelArguments(
        config_path=args.config_path,
        tokenizer_path=args.tokenizer_path
    )
    data_args = DataArguments(
        data_path=args.data_path,
        text_file=args.text_file,
        max_seq_length=args.max_seq_length,
        validation_split_percentage=args.validation_split
    )
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Set PyTorch threads
    torch.set_num_threads(min(multiprocessing.cpu_count(), 64))
    
    print("=" * 80)
    print("GPT-2 Pretraining")
    print("=" * 80)
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print(f"CPU cores available: {multiprocessing.cpu_count()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Load configuration
    print(f"\nLoading config from: {model_args.config_path}")
    config = load_config(model_args.config_path)
    print(f"Model config: {config.n_layer} layers, {config.n_embd} hidden size, {config.n_head} heads")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from: {model_args.tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_args.tokenizer_path)
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Update config vocab size to match tokenizer
    config.vocab_size = len(tokenizer)
    
    # Initialize model from scratch
    print("\nInitializing model from scratch...")
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.max_len = config.n_ctx
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {num_params:,} parameters")
    
    # Validate input arguments
    if not data_args.data_path and not data_args.text_file:
        raise ValueError("Must provide either --data-path or --text-file")
    if data_args.data_path and data_args.text_file:
        raise ValueError("Cannot use both --data-path and --text-file")
    
    # Load dataset
    if data_args.text_file:
        print(f"\nLoading text file: {data_args.text_file}")
        from datasets import Dataset
        
        # Read text file
        with open(data_args.text_file, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f if line.strip()]
        
        print(f"Loaded {len(lines):,} lines from text file")
        
        # Create dataset
        train_dataset = Dataset.from_dict({"text": lines})
        print(f"Created dataset with {len(train_dataset)} examples")
    else:
        print(f"\nLoading dataset from: {data_args.data_path}")
        dataset = load_from_disk(data_args.data_path)
        
        # Handle DatasetDict or Dataset
        if hasattr(dataset, 'keys') and 'train' in dataset:
            # Already a DatasetDict with train split
            print(f"Dataset loaded with train split: {len(dataset['train'])} examples")
            train_dataset = dataset['train']
        else:
            # Single Dataset
            print(f"Dataset loaded: {len(dataset)} examples")
            train_dataset = dataset
    
    # Split into train and validation
    print(f"\nSplitting dataset (train: {100 - data_args.validation_split_percentage}%, val: {data_args.validation_split_percentage}%)")
    from datasets import DatasetDict
    
    dataset_split = train_dataset.train_test_split(
        test_size=data_args.validation_split_percentage / 100,
        seed=42
    )
    
    dataset = DatasetDict({
        "train": dataset_split["train"],
        "validation": dataset_split["test"]
    })
    
    print(f"Train examples: {len(dataset['train'])}")
    print(f"Validation examples: {len(dataset['validation'])}")
    
    # Prepare dataset (tokenize and group into chunks)
    processed_dataset = prepare_dataset(dataset, tokenizer, data_args.max_seq_length)
    
    print(f"\nProcessed dataset:")
    print(f"Train examples: {len(processed_dataset['train'])}")
    print(f"Validation examples: {len(processed_dataset['validation'])}")

    # Data collator for language modeling (handles masking for causal LM)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # False for causal language modeling (GPT-2 style)
    )
    
    # Verify the dataset is already tokenized
    print(f"\nDataset columns: {processed_dataset['train'].column_names}")
    print(f"First example keys: {processed_dataset['train'][0].keys() if len(processed_dataset['train']) > 0 else 'empty'}")
    
    # Set format for faster data loading
    processed_dataset.set_format(type='torch', columns=['input_ids'])
    
    # Training arguments
    output_dir = args.output_dir
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Training hyperparameters
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        weight_decay=0.1,
        warmup_steps=1000,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,  # Gradient clipping to prevent explosion
        
        # Multi-GPU settings
        ddp_find_unused_parameters=False,
        
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=5000,
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=3,
        
        # Logging
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        logging_first_step=True,
        report_to=["tensorboard"],
        
        # Optimization
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,  # Use many workers to saturate CPUs
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=8,  # Prefetch even more batches per worker
        
        # Misc
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    print("\n" + "=" * 80)
    print("Training Arguments:")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Number of epochs: {training_args.num_train_epochs}")
    print(f"Batch size per device: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * torch.cuda.device_count() if torch.cuda.is_available() else training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"FP16 training: {training_args.fp16}")
    print(f"Dataloader workers: {training_args.dataloader_num_workers}")
    print(f"Prefetch factor: {training_args.dataloader_prefetch_factor}")
    print(f"Total workers (all GPUs): {training_args.dataloader_num_workers * torch.cuda.device_count() if torch.cuda.is_available() else training_args.dataloader_num_workers}")
    print("=" * 80 + "\n")
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        data_collator=data_collator,
    )

    
    # Train
    print("\nStarting training...")
    train_result = trainer.train()
    
    # Save model
    print("\nSaving final model...")
    trainer.save_model()
    
    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Final evaluation
    print("\nRunning final evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Model saved to: {output_dir}")
    print(f"Final train loss: {metrics['train_loss']:.4f}")
    print(f"Final eval loss: {eval_metrics['eval_loss']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
