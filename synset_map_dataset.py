import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path

from synset_mapping import get_nlp, replace_doc
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
    get_nlp()
    WORKER_READY = True


def process_chunk(lines: list[str]) -> tuple[str, int]:
    if not WORKER_READY:
        init_worker()
    nlp = get_nlp()
    docs = nlp.pipe(lines, batch_size=PIPE_BATCH_SIZE)
    mapped = [replace_doc(doc) for doc in docs]
    return "\n".join(mapped), len(mapped)


def main() -> None:
    # dataset_path = Path("datasets/bnc_sentences.txt")
    dataset_path = Path("datasets/packed_bnc_blocks.txt")
    output_path = dataset_path.with_name(f"mapped_{dataset_path.name}")

    total_lines = count_lines(dataset_path)

    with dataset_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        chunks = read_chunks(src, CHUNK_LINES)
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=NUM_PROCESSES, initializer=init_worker, mp_context=mp_ctx) as executor:
            with tqdm(total=total_lines, desc="Mapping sentences", unit="line") as pbar:
                for mapped_text, count in executor.map(process_chunk, chunks, chunksize=1):
                    dst.write(mapped_text.lower() + "\n")
                    pbar.update(count)


if __name__ == "__main__":
    main()
    