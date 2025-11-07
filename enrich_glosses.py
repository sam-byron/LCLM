#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enrich clusters.json glosses with WordNet definitions, examples, and neighbor glosses.

Usage:
  python enrich_glosses.py \
    --candidates data/clusters.json \
    --out data/clusters_enriched.json \
    [--limit 0]

Notes:
- This augments each entry's 'definition' to include an expanded gloss so
  downstream consumers like lexicon_compress.py pick it up without code changes.
- Also stores the full expansion in 'expanded_gloss' for inspection.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import nltk
from nltk.corpus import wordnet as wn


def ensure_wordnet() -> None:
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)


def expand_gloss(synset_id: str, max_hyper: int = 3, max_hypo: int = 3) -> str:
    syn = wn.synset(synset_id)
    parts: List[str] = []
    # base gloss + examples
    parts.append(syn.definition())
    ex = syn.examples()
    if ex:
        parts.append(" ".join(ex))
    # neighbors: a few hyper/hypo definitions
    hypers = syn.hypernyms()[:max_hyper]
    if hypers:
        parts.append(" ".join(h.definition() for h in hypers))
    hypos = syn.hyponyms()[:max_hypo]
    if hypos:
        parts.append(" ".join(h.definition() for h in hypos))
    # synonyms/lemmas
    lemmas = syn.lemma_names()
    if lemmas:
        parts.append(" ".join(lemmas))
    # Join and normalize whitespace
    return " ".join(p.strip() for p in parts if p).strip()


def enrich_entry(entry: Dict[str, Any]) -> bool:
    syn_id = entry.get("synset")
    if not syn_id:
        return False
    try:
        expanded = expand_gloss(syn_id)
    except Exception as e:
        print(f"[warn] Failed to expand {syn_id}: {e}", file=sys.stderr)
        return False

    # Merge into 'definition' so downstream code uses it
    base_def = (entry.get("definition") or "").strip()
    if base_def:
        merged = f"{base_def} {expanded}".strip()
    else:
        merged = expanded
    entry["definition"] = merged
    entry["expanded_gloss"] = expanded
    return True


def main():
    ap = argparse.ArgumentParser(description="Enrich clusters.json entries with expanded gloss text from WordNet.")
    ap.add_argument("--candidates", required=True, help="Path to input clusters.json")
    ap.add_argument("--out", required=True, help="Path to write enriched JSON")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on number of entries to enrich (0 = all)")
    args = ap.parse_args()

    ensure_wordnet()

    inp = Path(args.candidates)
    data = json.loads(inp.read_text(encoding="utf-8"))
    clusters_by_pos = data.get("clusters_by_pos", {})

    total = 0
    touched = 0
    for pos, entries in clusters_by_pos.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            total += 1
            if args.limit and touched >= args.limit:
                continue
            ok = enrich_entry(entry)
            if ok:
                touched += 1

    outp = Path(args.out)
    outp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Enriched {touched} entries out of {total} scanned. Wrote: {outp}")


if __name__ == "__main__":
    main()