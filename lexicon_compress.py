#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lexicon Compression via Unsupervised Gloss Ranking (Dual-Encoder + Lesk)
-------------------------------------------------------------------------

Usage:
    python lexicon_compress.py \
        --candidates candidates.json \
        --in input.txt \
        --out output.txt \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --abstain_margin 0.05 \
        --min_score 0.25

What it does:
- Loads candidate synset clusters (precomputed) from a JSON file.
- Precomputes an "expanded gloss" string per synset (definition + examples).
- Encodes all expanded glosses once (batched) with a sentence embedding model.
- Processes the input text with spaCy (sentences, POS, lemmas, morphology).
- For each content token (NOUN/VERB/ADJ/ADV) that has candidate synsets in the file,
  computes a score = cosine(context_emb, sense_emb) + 0.1 * lesk_overlap.
- Picks the best sense if confident (margin & min_score thresholds), otherwise keeps original.
- Replaces token with a lemma from the chosen synset's word list, preserving inflection via lemminflect.

Notes:
- "Context" embedding is the sentence embedding (no per-token marker for speed).
- Lesk overlap uses lemmatized content words in the sentence vs. sense text bag.
- Multiword replacements are inserted as-is (no head inflection) to keep things simple and robust.
- Proper nouns and function words are not replaced.

Dependencies:
  pip install spacy lemminflect sentence-transformers numpy
  python -m spacy download en_core_web_sm
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Iterator, Optional

import numpy as np
import spacy
from lemminflect import getInflection
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# -------------------------
# Utility: Simple cosine sim
# -------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (n, d), b: (m, d) => returns (n, m)
    # Normalize rows to unit length
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T

# -------------------------
# Load candidate synsets
# -------------------------
def load_candidates(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Expect structure:
    # { "metadata": {...},
    #   "clusters_by_pos": { "n": [ {synset, pos, definition, examples[], words[]}, ...], ... } }
    return data

# -------------------------
# Build sense bank and indices
# -------------------------
def build_sense_bank(candidates: Dict[str, Any]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int], List[str]]:
    """
    Returns:
        senses_by_pos: {'NOUN': [entry, ...], 'VERB': [...], 'ADJ': [...], 'ADV': [...]}
        sense_index: maps synset string -> global index in sense_texts
        sense_texts: list of expanded gloss texts (aligned with global index)
    """
    pos_map = {'n': 'NOUN', 'v': 'VERB', 'a': 'ADJ', 's': 'ADJ', 'r': 'ADV'}
    senses_by_pos = {'NOUN': [], 'VERB': [], 'ADJ': [], 'ADV': []}
    sense_index: Dict[str, int] = {}
    sense_texts: List[str] = []
    # Flatten while keeping POS grouping
    for short_pos, entries in candidates.get("clusters_by_pos", {}).items():
        upos = pos_map.get(short_pos)
        if upos is None:
            continue
        for e in entries:
            # expanded gloss: definition + first example if any
            definition = (e.get("definition") or "").strip()
            examples = e.get("examples") or []
            ex = (" " + examples[0].strip()) if examples else ""
            expanded = (definition + ex).strip()
            if not expanded:
                expanded = definition or e.get("synset", "")
            # register
            global_idx = len(sense_texts)
            sense_index[e["synset"]] = global_idx
            sense_texts.append(expanded)
            senses_by_pos[upos].append(e)
    return senses_by_pos, sense_index, sense_texts

# -------------------------
# Lesk overlap (bag overlap)
# -------------------------
def lesk_overlap(context_doc, sense_text: str, nlp) -> float:
    # Simple, fast overlap: lemmatized content words only
    # Build sense bag once per call; tokenize via nlp.pipe at higher level would be overkill for short strings
    sense_doc = nlp.make_doc(sense_text)
    # Quick lemmatization via tagger? We'll just lower for sense text to keep it fast (no tagging overhead).
    # To keep speed, we do a lightweight token filter here.
    sense_bag = set([t.lower_ for t in sense_doc if t.is_alpha and len(t) > 2])
    if not sense_bag:
        return 0.0
    ctx_bag = set([t.lemma_.lower() for t in context_doc if t.is_alpha and len(t) > 2 and t.pos_ in {"NOUN","VERB","ADJ","ADV"}])
    if not ctx_bag:
        return 0.0
    return float(len(ctx_bag & sense_bag)) / (len(ctx_bag) + 1e-6)

# -------------------------
# Replacement choice & inflection
# -------------------------
SYNSET_PREFIX_RE = re.compile(r"^([^.]+)")


def synset_prefix(entry: Dict[str, Any], fallback: str) -> str:
    syn = entry.get("synset", "")
    match = SYNSET_PREFIX_RE.match(syn)
    if not match:
        return fallback
    prefix = match.group(1)
    # WordNet synsets use underscores for multiword lemmas; normalize to spaces.
    return prefix.replace("_", " ") or fallback

def inflect_like(repl: str, tok) -> str:
    """
    Try to inflect 'repl' (single token) like 'tok'. If 'repl' is multiword, return as-is.
    We keep this intentionally simple and robust.
    """
    if " " in repl:
        # keep multiword as-is for simplicity/robustness
        return repl

    # Handle by POS
    upos = tok.pos_
    orth = repl

    # Nouns: plural/singular
    if upos == "NOUN":
        num = tok.morph.get("Number")
        if num and num[0] == "Plur":
            forms = getInflection(orth, tag="NNS")
            if forms:
                return forms[0]
        # default singular
        forms = getInflection(orth, tag="NN")
        return forms[0] if forms else orth

    # Verbs: tense/mood/person (simplified)
    if upos == "VERB" or upos == "AUX":
        # Heuristics: look at Morph features
        tense = tok.morph.get("Tense")
        verbform = tok.morph.get("VerbForm")
        person = tok.morph.get("Person")
        number = tok.morph.get("Number")

        # Gerund/participle
        if verbform and "Ger" in verbform:
            forms = getInflection(orth, tag="VBG")
            return forms[0] if forms else orth
        if verbform and "Part" in verbform:
            # past participle
            forms = getInflection(orth, tag="VBN")
            return forms[0] if forms else orth

        # Past tense
        if tense and "Past" in tense:
            forms = getInflection(orth, tag="VBD")
            return forms[0] if forms else orth

        # Present tense; 3rd person singular vs base
        if tense and "Pres" in tense:
            if person and "3" in person and number and "Sing" in number:
                forms = getInflection(orth, tag="VBZ")
                return forms[0] if forms else orth
            else:
                forms = getInflection(orth, tag="VBP")
                return forms[0] if forms else orth

        # Default to base form
        forms = getInflection(orth, tag="VB")
        return forms[0] if forms else orth

    # Adjectives: comparative/superlative
    if upos == "ADJ":
        degree = tok.morph.get("Degree")
        if degree and degree[0] == "Cmp":
            forms = getInflection(orth, tag="JJR")
            return forms[0] if forms else orth
        if degree and degree[0] == "Sup":
            forms = getInflection(orth, tag="JJS")
            return forms[0] if forms else orth
        forms = getInflection(orth, tag="JJ")
        return forms[0] if forms else orth

    # Adverbs: comparative/superlative (rare)
    if upos == "ADV":
        degree = tok.morph.get("Degree")
        if degree and degree[0] == "Cmp":
            forms = getInflection(orth, tag="RBR")
            return forms[0] if forms else orth
        if degree and degree[0] == "Sup":
            forms = getInflection(orth, tag="RBS")
            return forms[0] if forms else orth
        return orth

    # Default: return as-is
    return orth

# -------------------------
# Token replacement policy
# -------------------------
CONTENT_POS = {"NOUN","VERB","ADJ","ADV"}
FUNCTION_POS = {"ADP","AUX","CCONJ","DET","INTJ","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","X","SPACE"}

def should_replace(tok) -> bool:
    if tok.is_space or tok.is_punct:
        return False
    if tok.pos_ in FUNCTION_POS:
        return False
    if tok.ent_type_:
        # named entity: leave as-is
        return False
    return tok.pos_ in CONTENT_POS

# -------------------------
# Build word -> candidate entries mapping for quick lookup
# -------------------------
def build_word_lookup(senses_by_pos: Dict[str, List[Dict[str, Any]]]) -> Dict[Tuple[str,str], List[Dict[str, Any]]]:
    """
    Key: (lemma_lower, UPOS) -> list of entries (each has fields incl. 'synset','words','definition',...)
    """
    lookup: Dict[Tuple[str,str], List[Dict[str, Any]]] = {}
    for upos, entries in senses_by_pos.items():
        for e in entries:
            for w in e.get("words", []):
                lemma = w.replace("_"," ").lower()
                lookup.setdefault((lemma, upos), []).append(e)
    return lookup

# -------------------------
# Vectorized sense encoding
# -------------------------
def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 256,
    *,
    pool=None,
    chunk_size: Optional[int] = None,
    normalize: bool = False,
) -> np.ndarray:
    """Encode *texts* with SentenceTransformer, optionally via multi-process pool."""
    if not texts:
        dim = model.get_sentence_embedding_dimension() or 0
        return np.zeros((0, dim), dtype=np.float32)

    if pool is not None:
        embs = model.encode_multi_process(
            texts,
            pool,
            batch_size=batch_size,
            chunk_size=chunk_size,
            normalize_embeddings=normalize,
        )
    else:
        embs = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )

    if not isinstance(embs, np.ndarray):
        embs = np.asarray(embs)
    if embs.dtype != np.float32:
        embs = embs.astype(np.float32, copy=False)
    if embs.ndim == 1:
        embs = embs.reshape(1, -1)
    return embs

# -------------------------
# Stream large text files safely
# -------------------------
def stream_text_chunks(path: Path, chunk_chars: int) -> Iterator[str]:
    """Yield text chunks capped near chunk_chars to avoid giant spaCy docs."""
    buffer = ""
    with path.open("r", encoding="utf-8") as src:
        for line in src:
            buffer += line
            while len(buffer) >= chunk_chars:
                split_idx = buffer.rfind("\n", 0, chunk_chars)
                if split_idx > 0:
                    split_idx += 1
                else:
                    split_idx = buffer.rfind(" ", 0, chunk_chars)
                    if split_idx <= 0:
                        split_idx = chunk_chars
                chunk = buffer[:split_idx]
                if chunk:
                    yield chunk
                buffer = buffer[split_idx:]
        if buffer:
            yield buffer

# -------------------------
# Main pipeline
# -------------------------
def process_file(args):
    # Load NLP
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError as e:
        print("Error: spaCy model 'en_core_web_sm' not installed. Run: python -m spacy download en_core_web_sm", file=sys.stderr)
        raise
    nlp.enable_pipe("lemmatizer")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    nlp.max_length = max(nlp.max_length, args.max_length, args.chunk_chars * 2)

    # Load candidates and build banks
    cand = load_candidates(Path(args.candidates))
    senses_by_pos, sense_index, sense_texts = build_sense_bank(cand)
    # Word lookup
    word_lookup = build_word_lookup(senses_by_pos)

    # Embedding model
    model = SentenceTransformer(args.model)

    devices_arg = (args.devices or "").strip().lower()
    target_devices: List[str] = []
    mp_pool = None
    if devices_arg and devices_arg != "none":
        if devices_arg == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    gpu_count = max(0, torch.cuda.device_count())
                    if gpu_count > 1:
                        target_devices = [f"cuda:{i}" for i in range(gpu_count)]
                else:
                    workers = args.cpu_workers if args.cpu_workers > 0 else max(1, min(8, os.cpu_count() or 1))
                    if workers > 1:
                        target_devices = ["cpu"] * workers
            except ImportError:
                workers = args.cpu_workers if args.cpu_workers > 0 else max(1, min(8, os.cpu_count() or 1))
                if workers > 1:
                    target_devices = ["cpu"] * workers
        else:
            target_devices = [dev.strip() for dev in devices_arg.split(",") if dev.strip()]

        if len(target_devices) > 1:
            print(
                f"Starting SentenceTransformer multi-process pool on devices: {target_devices}",
                file=sys.stderr,
            )
            mp_pool = model.start_multi_process_pool(target_devices=target_devices)

    encode_pool = mp_pool if mp_pool is not None else None

    try:
        # Precompute sense embeddings (vectorized)
        sense_embs = encode_texts(
            model,
            sense_texts,
            batch_size=args.batch_size,
            pool=encode_pool,
            chunk_size=args.mp_chunk_size,
        )
        sense_norm = sense_embs / (np.linalg.norm(sense_embs, axis=1, keepdims=True) + 1e-12)

        # Pre-tokenize sense bags for Lesk quickly (use spaCy make_doc to avoid heavy tagging)
        # We'll just lowercase alpha tokens length>2 for sense bag.
        def sense_bag(text: str) -> set:
            doc = nlp.make_doc(text)
            return set([t.lower_ for t in doc if t.is_alpha and len(t) > 2])

        sense_bags = [sense_bag(t) for t in sense_texts]

        input_path = Path(args.input)
        output_path = Path(args.output)
        print(
            f"Streaming input from {input_path} in ~{args.chunk_chars} character chunks",
            file=sys.stderr,
        )

        sentence_bar = tqdm(desc="Processing sentences", unit="sent")
        with output_path.open("w", encoding="utf-8") as out_f:
            for doc in nlp.pipe(
                stream_text_chunks(input_path, args.chunk_chars),
                batch_size=args.nlp_batch_size,
                n_process=max(1, args.nlp_processes),
            ):
                sent_items = []
                for sent in doc.sents:
                    sent_text = sent.text
                    if not sent_text:
                        continue
                    sent_items.append((sent, sent_text))

                if not sent_items:
                    continue

                sent_texts = [txt for _, txt in sent_items]
                ctx_embs = encode_texts(
                    model,
                    sent_texts,
                    batch_size=args.context_batch_size,
                    pool=encode_pool,
                    chunk_size=args.mp_chunk_size,
                )
                ctx_norms = ctx_embs / (np.linalg.norm(ctx_embs, axis=1, keepdims=True) + 1e-12)

                for (sent, _), ctx_norm in zip(sent_items, ctx_norms):
                    ctx_bag = {
                        t.lemma_.lower()
                        for t in sent
                        if t.is_alpha and len(t) > 2 and t.pos_ in CONTENT_POS
                    }

                    out_tokens: List[str] = []
                    for tok in sent:
                        if not should_replace(tok):
                            out_tokens.append(tok.text_with_ws)
                            continue

                        key = (tok.lemma_.lower(), tok.pos_)
                        entries = word_lookup.get(key)

                        if not entries:
                            key2 = (tok.text.lower(), tok.pos_)
                            entries = word_lookup.get(key2)

                        if not entries:
                            out_tokens.append(tok.text_with_ws)
                            continue

                        cand_idxs = []
                        for e in entries:
                            idx = sense_index.get(e["synset"])
                            if idx is not None:
                                cand_idxs.append(idx)
                        if not cand_idxs:
                            out_tokens.append(tok.text_with_ws)
                            continue
                        cand_idxs = list(dict.fromkeys(cand_idxs))

                        cand_norm = sense_norm[cand_idxs]
                        cos = cand_norm @ ctx_norm

                        if ctx_bag:
                            lesk_vals = np.array(
                                [
                                    float(len(ctx_bag & sense_bags[i]))
                                    / (len(ctx_bag) + 1e-6)
                                    for i in cand_idxs
                                ],
                                dtype=np.float32,
                            )
                        else:
                            lesk_vals = np.zeros(len(cand_idxs), dtype=np.float32)

                        scores = cos + args.lesk_weight * lesk_vals

                        if len(scores) == 1:
                            best_idx = 0
                            margin = scores[0]
                        else:
                            order = np.argsort(-scores)
                            best_idx = order[0]
                            second = scores[order[1]] if len(scores) > 1 else -1.0
                            margin = scores[order[0]] - second

                        best_score = float(scores[best_idx])
                        if best_score < args.min_score or margin < args.abstain_margin:
                            out_tokens.append(tok.text_with_ws)
                            continue

                        chosen_entry = entries[best_idx]
                        prefix = synset_prefix(chosen_entry, tok.lemma_)
                        repl_infl = inflect_like(prefix, tok)

                        ws = tok.whitespace_
                        out_tokens.append(repl_infl + ws)

                    out_f.write("".join(out_tokens))
                    sentence_bar.update(1)
        sentence_bar.close()
        print(f"Wrote: {output_path}")
    finally:
        if mp_pool is not None:
            SentenceTransformer.stop_multi_process_pool(mp_pool)

def main():
    p = argparse.ArgumentParser(description="Unsupervised lexicon compression using candidate synsets and sentence embeddings.")
    p.add_argument("--candidates", "-c", required=True, help="Path to candidate synsets JSON file.")
    p.add_argument("--in", dest="input", required=True, help="Path to input text file.")
    p.add_argument("--out", dest="output", required=True, help="Path to write transformed text.")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model name.")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size for encoding sense texts.")
    p.add_argument("--context_batch_size", type=int, default=256, help="Batch size for sentence/context encoding.")
    p.add_argument("--lesk_weight", type=float, default=0.1, help="Weight for Lesk overlap term.")
    p.add_argument("--abstain_margin", type=float, default=0.05, help="Min margin between top-2 scores to accept replacement.")
    p.add_argument("--min_score", type=float, default=0.25, help="Min absolute score to accept replacement.")
    p.add_argument("--chunk_chars", type=int, default=500_000, help="Approximate number of characters per spaCy chunk.")
    p.add_argument("--nlp_batch_size", type=int, default=4, help="spaCy pipe batch size for streaming chunks.")
    p.add_argument("--nlp_processes", type=int, default=1, help="Number of spaCy processes for nlp.pipe (>=1).")
    p.add_argument(
        "--devices",
        default="auto",
        help="Comma-separated devices for SentenceTransformer multi-process encoding (e.g. 'cuda:0,cuda:1'). Use 'auto' (default) to detect or 'none' to disable.",
    )
    p.add_argument("--mp_chunk_size", type=int, default=500, help="Chunk size per worker for multi-process encoding.")
    p.add_argument("--cpu_workers", type=int, default=0, help="CPU worker count when auto-detecting devices (0=auto).")
    p.add_argument("--max_length", type=int, default=2_000_000, help="Override spaCy max_length safeguard (in characters).")
    args = p.parse_args()
    process_file(args)

if __name__ == "__main__":
    main()
