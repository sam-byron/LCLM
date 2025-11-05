from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast
import glob
import os
from datasets import load_from_disk

def train_tokenizer(dataset, output_dir, vocab_size=8000):
    """Train a custom GPT-2 style tokenizer with reduced vocabulary."""
    
    # Initialize a BPE tokenizer (same as GPT-2)
    tokenizer = Tokenizer(models.BPE())
    
    # Use GPT-2's preprocessing (ByteLevel)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # Special tokens
    special_tokens = ["<|endoftext|>"]
    
    # Trainer configuration
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=25,
        special_tokens=special_tokens,
        show_progress=True,
    )

    texts = [line.strip() for line in open(dataset, "r", encoding="utf-8")]
    # Train the tokenizer
    tokenizer.train_from_iterator(iter(texts), trainer)

    # Add post-processor (for special tokens)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # Wrap in HuggingFace tokenizer
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        unk_token="<|endoftext|>",
    )
    
    # Save
    wrapped_tokenizer.save_pretrained(output_dir)
    print(f"Saved 16k tokenizer to {output_dir}")
    print(f"Final vocab size: {len(wrapped_tokenizer)}")
    
    return wrapped_tokenizer

if __name__ == "__main__":
    train_tokenizer(
        dataset="./datasets/packed_bnc_blocks.txt",
        output_dir="./model_gpt2_tokenizer",
        vocab_size=8000
    )