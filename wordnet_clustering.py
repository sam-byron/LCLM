#!/usr/bin/env python3
"""
Paraphrase-Based Clustering with WordNet (NLTK)

Given a list of words (one per line) in words.txt, group items that are
synonymous *for the same sense* (synset). Clusters are separated by POS:
- nouns (n), verbs (v), adjectives (a), adverbs (r)

Each cluster is labeled by its synset id and definition for interpretability.
This implements the "paraphrase-based clustering" approach using WordNet.

Example:
    pip install nltk
    python cluster_words.py --input words.txt --out clusters.json --min-size 2
"""

from __future__ import annotations

import argparse
import json
import sys
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set, Tuple

# NLTK / WordNet
try:
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.stem import WordNetLemmatizer
except Exception as e:
    print("This script requires NLTK. Install with: pip install nltk", file=sys.stderr)
    raise

POS_LABELS = {
    "n": "Nouns",
    "v": "Verbs",
    "a": "Adjectives",
    "r": "Adverbs",
}

# Map CLI POS to NLTK lemmatizer POS
LEMMATIZER_POS = {
    "n": "n",
    "v": "v",
    "a": "a",  # includes adjective satellites
    "r": "r",
}

def ensure_wordnet():
    """
    Ensure the WordNet data is available; download if needed.
    """
    try:
        wn.ensure_loaded()
    except LookupError:
        nltk.download("wordnet", quiet=True)
        # omw-1.4 adds many synonyms and lemmas for non‑English mappings but also enriches English
        nltk.download("omw-1.4", quiet=True)
        wn.ensure_loaded()

def normalize(token: str) -> str:
    """
    Normalize input tokens to WordNet-friendly lemma keys:
    - lowercase
    - strip surrounding whitespace
    - convert internal whitespace to underscores (WordNet uses underscores)
    """
    token = token.strip().lower()
    token = re.sub(r"\s+", "_", token)
    return token

def load_words(path: str) -> List[str]:
    """
    Read non-empty, non-comment (# ... ) lines as tokens.
    Each line is treated as a single entry (multi-word OK).
    """
    words: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            words.append(line)
    return words

def build_canonical_maps(
    words: List[str],
    pos_list: List[str],
    lemmatizer: WordNetLemmatizer,
) -> Tuple[Set[str], Dict[str, Set[str]], Dict[str, Dict[str, Set[str]]]]:
    """
    Prepare:
      - input_norm_set: all normalized input forms
      - orig_map: normalized form -> set of original surface forms (for pretty output)
      - canon_map_by_pos: for each POS, map canonical lemma -> set of original forms that reduce to it
        (this lets 'cars' join a 'car' synset, etc.)
    """
    input_norm_set: Set[str] = set()
    orig_map: Dict[str, Set[str]] = defaultdict(set)
    canon_map_by_pos: Dict[str, Dict[str, Set[str]]] = {p: defaultdict(set) for p in pos_list}

    for w in words:
        norm = normalize(w)
        input_norm_set.add(norm)
        orig_map[norm].add(w)

    for p in pos_list:
        for w in words:
            norm = normalize(w)
            lemma = lemmatizer.lemmatize(norm, pos=LEMMATIZER_POS[p])
            # Keep normalized style (underscores) for WordNet alignment
            canon = normalize(lemma)
            canon_map_by_pos[p][canon].add(w)

            # Also allow the raw normalized form as a fallback canonical
            # (covers cases where lemmatization doesn't change anything)
            canon_map_by_pos[p][norm].add(w)

    return input_norm_set, orig_map, canon_map_by_pos

def synset_clusters_for_pos(
    pos: str,
    input_norm_set: Set[str],
    orig_map: Dict[str, Set[str]],
    canon_map: Dict[str, Set[str]],
    min_size: int = 2,
) -> Tuple[List[dict], List[str]]:
    """
    Build clusters for a given POS using WordNet synsets.
    A cluster is created for each synset that contains at least 'min_size' words from the input list.

    Returns:
      clusters: List of dicts {synset, pos, definition, examples, words}
      singletons: List of input words (original surface forms) that didn't appear in any >=min_size cluster
    """
    # Map synset -> set of normalized canonical keys intersecting our input
    syn_to_keys: Dict[object, Set[str]] = defaultdict(set)

    # For efficiency, iterate over all synsets only where we have overlap.
    # Strategy: derive candidate lookup keys from our canonical map;
    # then for each candidate key, pull synsets and intersect.
    candidate_keys = set(canon_map.keys())

    for key in candidate_keys:
        # For each key, look up synsets that contain it
        ssets = wn.synsets(key, pos=pos)
        if not ssets:
            continue
        for s in ssets:
            # All lemma names in this synset
            lemma_keys = {normalize(l.name()) for l in s.lemmas()}
            # If any synset lemma connects to our canonical input keys, record it
            overlap_keys = candidate_keys & lemma_keys
            if overlap_keys:
                syn_to_keys[s].update(overlap_keys)

    clusters: List[dict] = []
    words_in_any_cluster: Set[str] = set()

    for s, keys in syn_to_keys.items():
        # Collect *original* surface forms that reduce to those canonical keys
        word_variants: Set[str] = set()
        for k in keys:
            word_variants.update(canon_map.get(k, set()))
        # Only keep cluster if it covers at least 'min_size' items from the input
        if len(word_variants) >= min_size:
            clusters.append({
                "synset": s.name(),
                "pos": pos,
                "definition": s.definition(),
                "examples": s.examples(),
                "words": sorted(word_variants, key=lambda x: (x.lower(), x)),
            })
            words_in_any_cluster.update(word_variants)

    # Sort clusters by synset name for stable output
    clusters.sort(key=lambda c: (c["synset"], c["words"]))

    # Identify singletons (original surface forms) that didn't join any cluster
    # We consider "singleton" at the POS level: if a word didn’t appear in any >=min_size cluster for this POS
    # we list its original forms (as entered).
    all_input_originals: Set[str] = set()
    for _, originals in canon_map.items():
        all_input_originals.update(originals)

    singletons = sorted([w for w in all_input_originals if w not in words_in_any_cluster],
                        key=lambda x: (x.lower(), x))
    return clusters, singletons

def build_clusters(
    words: List[str],
    pos_choice: str = "all",
    min_size: int = 2,
    show_singletons: bool = False,
) -> dict:
    """
    Orchestrates clustering across POS choices.
    """
    ensure_wordnet()
    lemmatizer = WordNetLemmatizer()

    if pos_choice == "all":
        pos_list = ["n", "v", "a", "r"]
    else:
        pos_list = [pos_choice]

    input_norm_set, orig_map, canon_map_by_pos = build_canonical_maps(words, pos_list, lemmatizer)

    result = {
        "metadata": {
            "method": "WordNet synset-based paraphrase clustering",
            "min_cluster_size": min_size,
            "pos": pos_list,
            "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
        "clusters_by_pos": {},
    }
    if show_singletons:
        result["singletons_by_pos"] = {}

    for p in pos_list:
        clusters, singletons = synset_clusters_for_pos(
            p, input_norm_set, orig_map, canon_map_by_pos[p], min_size=min_size
        )
        result["clusters_by_pos"][p] = clusters
        if show_singletons:
            result["singletons_by_pos"][p] = singletons

    return result

def print_human_readable(clusters_result: dict, show_singletons: bool):
    """
    Pretty-print clusters by POS.
    """
    print("\n=== Paraphrase-Based Clusters (WordNet Synsets) ===")
    meta = clusters_result["metadata"]
    pos_list = meta["pos"]
    print(f"Method: {meta['method']}")
    print(f"Min cluster size: {meta['min_cluster_size']}")
    print(f"Generated at (UTC): {meta['generated_at_utc']}\n")

    for p in pos_list:
        label = POS_LABELS.get(p, p)
        clusters = clusters_result["clusters_by_pos"].get(p, [])
        print(f"--- {label} ({p}) ---")
        if not clusters:
            print("No multi-word clusters found.")
        else:
            for c in clusters:
                syn = c["synset"]
                definition = c["definition"]
                words = ", ".join(c["words"])
                print(f"[{syn}] {definition}")
                print(f"  -> {words}")
        if show_singletons:
            singletons = clusters_result.get("singletons_by_pos", {}).get(p, [])
            if singletons:
                print("Singletons (not clustered at this POS): " + ", ".join(singletons))
        print()

def main():
    parser = argparse.ArgumentParser(description="Paraphrase-based clustering of words using WordNet synsets.")
    parser.add_argument("--input", "-i", type=str, default="words.txt",
                        help="Path to input file (one word or multi-word expression per line). Default: words.txt")
    parser.add_argument("--out", "-o", type=str, default="",
                        help="Optional path to write JSON results (e.g., clusters.json).")
    parser.add_argument("--pos", type=str, default="all", choices=["all", "n", "v", "a", "r"],
                        help="Which POS to cluster: all, n (nouns), v (verbs), a (adjectives), r (adverbs). Default: all")
    parser.add_argument("--min-size", type=int, default=2,
                        help="Minimum cluster size to include. Default: 2")
    parser.add_argument("--show-singletons", action="store_true",
                        help="Also list words that didn't fall into any multi-word cluster for each POS.")

    args = parser.parse_args()

    try:
        words = load_words(args.input)
    except FileNotFoundError:
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not words:
        print("No words found in input file (non-empty, non-comment lines).", file=sys.stderr)
        sys.exit(1)

    clusters_result = build_clusters(
        words=words,
        pos_choice=args.pos,
        min_size=args.min_size,
        show_singletons=args.show_singletons,
    )

    print_human_readable(clusters_result, show_singletons=args.show_singletons)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(clusters_result, f, ensure_ascii=False, indent=2)
        print(f"JSON written to: {args.out}")

if __name__ == "__main__":
    main()
