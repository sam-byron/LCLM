import json
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast, AddedToken
import glob
import os
from datasets import load_from_disk

def train_tokenizer(dataset, vocab_size=8000):
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
    if wrapped_tokenizer.pad_token is None:
        wrapped_tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

   
    
    return wrapped_tokenizer

def add_special_tokens(tok: PreTrainedTokenizerFast):
    with open("data/synset2ids.json", "r") as f:
        synset2ids = json.load(f)

    cluster_added = [
        AddedToken(v, lstrip=False, rstrip=False, single_word=False, normalized=False)
        for k, v in synset2ids.items()
    ]
    added_codes = tok.add_tokens(cluster_added, special_tokens=False)
    print(f"Added {added_codes} cluster tokens as AddedToken entries.")

if __name__ == "__main__":
    wrapped_tokenizer = train_tokenizer(
        dataset="./datasets/embed_synset_bnc_sentences.txt",
        vocab_size=12000
    )

    output_dir="./model_gpt2_tokenizer"
     # Save
    wrapped_tokenizer.save_pretrained(output_dir)
    print(f"Saved tokenizer to {output_dir}")
    print(f"Final vocab size: {len(wrapped_tokenizer)}")
