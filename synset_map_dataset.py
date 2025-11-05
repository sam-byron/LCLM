import json
import multiprocessing
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Iterable, List, Tuple

import synset_mapping as sm
from spacy.tokens import Doc
from spacy.util import filter_spans
from tqdm import tqdm

CHUNK_LINES = 64
PIPE_BATCH_SIZE = 128
NUM_PROCESSES = max(1, os.cpu_count() or 1)
WORKER_READY = False


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def read_chunks(handle, chunk_size: int):
    while True:
        batch = list(islice(handle, chunk_size))
        if not batch:
            break
        yield [line.rstrip("\n") for line in batch]


def init_worker():
    global WORKER_READY
    sm.get_nlp()
    WORKER_READY = True


def _collect_candidates(span) -> List[str]:
    cands = set()
    txt = span.text.lower()
    cands.update(sm.word2synsets.get(txt, set()))
    for tok in span:
        cands.update(sm.word2synsets.get(tok.text.lower(), set()))
        cands.update(sm.word2synsets.get(tok.lemma_.lower(), set()))
    return list(cands)


def _transform_doc(doc) -> Tuple[str, List[Tuple[str, str]]]:
    covered = [False] * len(doc)
    token_map = {}
    span_starts = {}

    spans = filter_spans([doc[start:end] for _, start, end in sm.matcher(doc)])
    for sp in spans:
        if any(covered[i] for i in range(sp.start, sp.end)):
            continue
        cand = _collect_candidates(sp)
        best = sm.best_synset_for_item(doc, sp, cand)
        if not best:
            continue
        canon = best
        match = re.match(r"^([^.]+)", best)
        if match:
            canon = match.group(1)
        inflected = sm.inflect_like(sp.root, canon, sp.root.pos_, sp.root.tag_)
        span_starts[sp.start] = (sp.end, inflected)
        for i in range(sp.start, sp.end):
            covered[i] = True
            token_map[i] = inflected

    for i, tok in enumerate(doc):
        if covered[i] or not tok.is_alpha:
            continue
        cands = set()
        for form in (tok.text.lower(), tok.lemma_.lower()):
            cands.update(sm.word2synsets.get(form, set()))
        if not cands:
            continue
        best = sm.best_synset_for_item(doc, doc[i:i+1], list(cands))
        if not best:
            continue
        canon = best
        match = re.match(r"^([^.]+)", best)
        if match:
            canon = match.group(1)
        repl = sm.inflect_like(tok, canon, tok.pos_, tok.tag_)
        token_map[i] = repl
        covered[i] = True

    for i, tok in enumerate(doc):
        if i not in token_map:
            token_map[i] = tok.text

    words = []
    idx = 0
    while idx < len(doc):
        if idx in span_starts:
            end, value = span_starts[idx]
            words.append(value)
            idx = end
            continue
        words.append(token_map[idx])
        idx += 1

    text = Doc(sm.get_nlp().vocab, words=words).text

    mapping_pairs: List[Tuple[str, str]] = []
    for i, tok in enumerate(doc):
        if not tok.is_alpha:
            continue
        orig = tok.text.lower()
        repl = token_map[i].lower()
        mapping_pairs.append((orig, repl))

    return text, mapping_pairs


def process_chunk(lines: List[str]) -> tuple[str, int, List[Tuple[str, str]]]:
    if not WORKER_READY:
        init_worker()
    nlp = sm.get_nlp()
    docs = nlp.pipe(lines, batch_size=PIPE_BATCH_SIZE)
    mapped_texts = []
    mapping_pairs: List[Tuple[str, str]] = []
    for doc in docs:
        text, pairs = _transform_doc(doc)
        mapped_texts.append(text)
        mapping_pairs.extend(pairs)
    return "\n".join(mapped_texts), len(mapped_texts), mapping_pairs


def main() -> None:
    # dataset_path = Path("datasets/bnc_sentences.txt")
    dataset_path = Path("datasets/packed_bnc_blocks.txt")
    output_path = dataset_path.with_name(f"mapped_{dataset_path.name}")

    total_lines = count_lines(dataset_path)

    mappings = defaultdict(set)

    with dataset_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        chunks = read_chunks(src, CHUNK_LINES)
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=NUM_PROCESSES, initializer=init_worker, mp_context=mp_ctx) as executor:
            with tqdm(total=total_lines, desc="Mapping sentences", unit="line") as pbar:
                for mapped_text, count, pairs in executor.map(process_chunk, chunks, chunksize=1):
                    dst.write(mapped_text.lower() + "\n")
                    for orig, repl in pairs:
                        mappings[orig].add(repl)
                    pbar.update(count)

    mapping_output = output_path.with_name(f"{output_path.stem}_mappings.json")
    serializable = {word: sorted(repls) for word, repls in sorted(mappings.items())}
    with mapping_output.open("w", encoding="utf-8") as mf:
        json.dump(serializable, mf, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
    