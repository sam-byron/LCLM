#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Map synset names to their position indices in clusters.json.

Usage:
    python synset2IDs.py --input data/clusters.json --output synset2ids.json

Output format:
    {
        "entity.n.01": "C<0>",
        "physical_entity.n.01": "C<1>",
        "kick.v.03": "C<4567>",
        ...
    }
    
Where values are formatted as C<position> with position being the order of appearance in clusters.json.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict


def extract_synset_to_id(clusters_path: Path) -> Dict[str, str]:
    """
    Extract mapping from synset names to their position identifiers.
    
    Args:
        clusters_path: Path to clusters.json file
        
    Returns:
        Dictionary mapping synset name -> "C<position>" string
    """
    with clusters_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    synset2id: Dict[str, str] = {}
    
    # Track global position across all POS categories
    global_index = 0
    
    # Iterate through all POS categories in order
    clusters_by_pos = data.get("clusters_by_pos", {})
    for pos_tag in ["n", "v", "a", "r"]:  # Preserve order
        entries = clusters_by_pos.get(pos_tag, [])
        for entry in entries:
            synset_name = entry.get("synset")
            if synset_name:
                synset2id[synset_name] = f"C<{global_index}>"
            global_index += 1
    
    return synset2id


def main():
    parser = argparse.ArgumentParser(
        description="Map synset names to position indices in clusters.json"
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
    synset2id = extract_synset_to_id(input_path)
    
    print(f"Extracted {len(synset2id)} synset mappings", file=sys.stderr)
    
    # Write output
    with output_path.open("w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(synset2id, f, indent=2, ensure_ascii=False)
        else:
            json.dump(synset2id, f, ensure_ascii=False)
    
    print(f"Wrote synset-to-ID mapping to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
