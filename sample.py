#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample a percentage of lines from a text dataset.

Usage:
    python sample.py --input data.txt --percent 10
    # Creates: data_sampled_10.txt

    python sample.py --input corpus.txt --percent 25 --prefix my_subset
    # Creates: my_subset_corpus_sampled_25.txt
"""

import argparse
import random
import sys
from pathlib import Path


def sample_lines(input_path: Path, percent: float, prefix: str = None, seed: int = 42) -> Path:
    """
    Sample a percentage of lines from input_path and write to a new file.
    
    Args:
        input_path: Path to input .txt file
        percent: Percentage of lines to sample (0-100)
        prefix: Optional prefix for output filename
        seed: Random seed for reproducibility
        
    Returns:
        Path to the output file
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if not 0 < percent <= 100:
        raise ValueError(f"Percent must be between 0 and 100, got {percent}")
    
    # Build output filename
    stem = input_path.stem
    suffix = input_path.suffix
    if prefix:
        output_name = f"{prefix}_{stem}_sampled_{percent:.1f}{suffix}"
    else:
        output_name = f"{stem}_sampled_{percent:.1f}{suffix}"
    output_path = input_path.parent / output_name
    
    # Read all lines
    print(f"Reading lines from {input_path}...", file=sys.stderr)
    with input_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    if total_lines == 0:
        print("Warning: Input file is empty", file=sys.stderr)
        output_path.write_text("", encoding="utf-8")
        return output_path
    
    # Sample lines
    sample_size = max(1, int(total_lines * percent / 100))
    print(f"Sampling {sample_size} / {total_lines} lines ({percent}%)...", file=sys.stderr)
    
    random.seed(seed)
    sampled_lines = random.sample(lines, sample_size)
    
    # Write output
    with output_path.open("w", encoding="utf-8") as f:
        f.writelines(sampled_lines)
    
    print(f"Wrote {len(sampled_lines)} lines to {output_path}", file=sys.stderr)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Sample a percentage of lines from a text dataset."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input .txt file"
    )
    parser.add_argument(
        "--percent", "-p",
        type=float,
        required=True,
        help="Percentage of lines to sample (0-100)"
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional prefix for output filename"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    try:
        output_path = sample_lines(
            input_path=input_path,
            percent=args.percent,
            prefix=args.prefix,
            seed=args.seed
        )
        print(f"{output_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
