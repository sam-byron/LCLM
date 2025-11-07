#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract synset ID to words mapping from clusters.json.

Usage:
    python clusters2words.py --input data/clusters.json --output synset2words.json

Output format:
    {
        "0": ["entity", "something"],
        "1": ["physical_entity", "physical entity"],
        ...
    }
    
Where keys are integer positions (order of appearance) in clusters.json.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def extract_synset_to_words(clusters_path: Path) -> Dict[int, List[str]]:
    """
    Extract mapping from synset position indices to their word lists.
    
    Args:
        clusters_path: Path to clusters.json file
        
    Returns:
        Dictionary mapping synset index (position in clusters.json) -> list of words
    """
    with clusters_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    synset2words: Dict[int, List[str]] = {}
    
    # Track global position across all POS categories
    global_index = 0
    
    # Iterate through all POS categories in order
    clusters_by_pos = data.get("clusters_by_pos", {})
    for pos_tag in ["n", "v", "a", "r"]:  # Preserve order
        entries = clusters_by_pos.get(pos_tag, [])
        for entry in entries:
            words = entry.get("words", [])
            synset2words[global_index] = words
            global_index += 1
    
    return synset2words


def main():
    parser = argparse.ArgumentParser(
        description="Extract synset ID to words mapping from clusters.json"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input clusters.json file"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print output JSON with indentation"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Reading clusters from {input_path}...", file=sys.stderr)
    synset2words = extract_synset_to_words(input_path)
    
    print(f"Extracted {len(synset2words)} synset mappings", file=sys.stderr)
    
    # Write output
    with output_path.open("w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(synset2words, f, indent=2, ensure_ascii=False)
        else:
            json.dump(synset2words, f, ensure_ascii=False)
    
    print(f"Wrote synset-to-words mapping to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
