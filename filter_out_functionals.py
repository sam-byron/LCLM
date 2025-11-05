"""Utility to filter functional words out of a vocabulary list.

The script removes:
  * stopwords / function words (built-in list, optional NLTK stopwords)
  * tokens with length <= 2 characters
  * tokens that map to proper nouns in WordNet (if NLTK is available)

Example:
	python filter_out_functionals.py --input frequent_words.txt --output content_words.txt
"""

from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Set

# Optional NLTK imports (WordNet + stopwords). We degrade gracefully if missing.
try:
	import nltk
	from nltk.corpus import stopwords
	from nltk.corpus import wordnet as wn
except Exception:  # pragma: no cover - protects against missing NLTK at runtime
	nltk = None  # type: ignore
	stopwords = None  # type: ignore
	wn = None  # type: ignore


# Conservative fallback list of English function words if NLTK stopwords are unavailable.
FALLBACK_STOPWORDS: Set[str] = {
	"a",
	"about",
	"above",
	"after",
	"again",
	"against",
	"all",
	"am",
	"an",
	"and",
	"any",
	"are",
	"aren't",
	"as",
	"at",
	"be",
	"because",
	"been",
	"before",
	"being",
	"below",
	"between",
	"both",
	"but",
	"by",
	"can",
	"can't",
	"could",
	"couldn't",
	"did",
	"didn't",
	"do",
	"does",
	"doesn't",
	"doing",
	"don't",
	"down",
	"during",
	"each",
	"few",
	"for",
	"from",
	"further",
	"had",
	"hadn't",
	"has",
	"hasn't",
	"have",
	"haven't",
	"having",
	"he",
	"he'd",
	"he'll",
	"he's",
	"her",
	"here",
	"here's",
	"hers",
	"herself",
	"him",
	"himself",
	"his",
	"how",
	"how's",
	"i",
	"i'd",
	"i'll",
	"i'm",
	"i've",
	"if",
	"in",
	"into",
	"is",
	"isn't",
	"it",
	"it's",
	"its",
	"itself",
	"just",
	"let's",
	"me",
	"more",
	"most",
	"mustn't",
	"my",
	"myself",
	"no",
	"nor",
	"not",
	"of",
	"off",
	"on",
	"once",
	"only",
	"or",
	"other",
	"ought",
	"our",
	"ours",
	"ourselves",
	"out",
	"over",
	"own",
	"same",
	"shall",
	"shan't",
	"she",
	"she'd",
	"she'll",
	"she's",
	"should",
	"shouldn't",
	"so",
	"some",
	"such",
	"than",
	"that",
	"that's",
	"the",
	"their",
	"theirs",
	"them",
	"themselves",
	"then",
	"there",
	"there's",
	"these",
	"they",
	"they'd",
	"they'll",
	"they're",
	"they've",
	"this",
	"those",
	"through",
	"to",
	"too",
	"under",
	"until",
	"up",
	"very",
	"was",
	"wasn't",
	"we",
	"we'd",
	"we'll",
	"we're",
	"we've",
	"were",
	"weren't",
	"what",
	"what's",
	"when",
	"when's",
	"where",
	"where's",
	"which",
	"while",
	"who",
	"who's",
	"whom",
	"why",
	"why's",
	"with",
	"won't",
	"would",
	"wouldn't",
	"you",
	"you'd",
	"you'll",
	"you're",
	"you've",
	"your",
	"yours",
	"yourself",
	"yourselves",
}


def ensure_wordnet() -> None:
	"""Ensure WordNet is downloaded if NLTK is available."""

	if wn is None or nltk is None:
		raise RuntimeError("NLTK with WordNet is required for proper noun detection.")

	try:
		wn.ensure_loaded()
	except LookupError:  # Download WordNet data lazily when missing
		nltk.download("wordnet", quiet=True)
		nltk.download("omw-1.4", quiet=True)
		wn.ensure_loaded()


def load_words(path: Path) -> List[str]:
	"""Read newline-delimited words, ignoring empty lines and comments."""

	words: List[str] = []
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			token = line.strip()
			if not token or token.startswith("#"):
				continue
			words.append(token)
	return words


def load_stopword_inventory(extra_sources: Iterable[Path]) -> Set[str]:
	"""Aggregate stopwords from built-in sets, optional NLTK, and user-provided files."""

	inventory: Set[str] = {w.lower() for w in FALLBACK_STOPWORDS}

	if stopwords is not None and nltk is not None:
		try:
			stopwords.words("english")  # Trigger lookup before potential download.
		except LookupError:
			nltk.download("stopwords", quiet=True)
		try:
			inventory.update(w.lower() for w in stopwords.words("english"))
		except (LookupError, OSError):
			# If download fails we keep the fallback list.
			pass

	for path in extra_sources:
		text = path.read_text(encoding="utf-8")
		for raw in text.splitlines():
			cleaned = raw.strip().lower()
			if cleaned and not cleaned.startswith("#"):
				inventory.add(cleaned)

	return inventory


@lru_cache(maxsize=8192)
def is_proper_noun(token: str) -> bool:
	"""Return True if WordNet marks the token as a proper noun."""

	if wn is None:
		return False

	ensure_wordnet()
	candidates = {token, token.lower(), token.title(), token.upper()}
	for probe in candidates:
		for syn in wn.synsets(probe, pos=wn.NOUN):
			for lemma in syn.lemmas():  # type: ignore[union-attr]
				name = lemma.name()
				if not name:
					continue
				if name[0].isupper() or any(ch.isupper() for ch in name[1:]):
					return True
	return False


def filter_words(
	words: Iterable[str],
	stopword_inventory: Set[str],
	min_length: int,
	require_wordnet: bool,
	drop_titlecase: bool,
) -> List[str]:
	"""Filter out function words, short tokens, and proper nouns."""

	filtered: List[str] = []
	for word in words:
		surface = word.strip()
		if not surface:
			continue

		if drop_titlecase and surface != surface.lower():
			continue

		lowered = surface.lower()

		if len(lowered) < min_length:
			continue

		if lowered in stopword_inventory:
			continue

		if require_wordnet and is_proper_noun(lowered):
			continue

		if drop_titlecase or surface == surface.lower():
			filtered.append(lowered)
		else:
			filtered.append(surface)

	return filtered


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input", "-i", type=Path, default=Path("frequent_words.txt"), help="Path to source word list.")
	parser.add_argument(
		"--output",
		"-o",
		type=Path,
		default=Path("frequent_words_filtered.txt"),
		help="Where to write the filtered vocabulary.",
	)
	parser.add_argument(
		"--extra-stopwords",
		type=Path,
		nargs="*",
		default=(),
		help="Optional files containing additional stopwords (one per line).",
	)
	parser.add_argument(
		"--min-length",
		type=int,
		default=3,
		help="Minimum token length to keep; defaults to 3 (filters <= 2 characters).",
	)
	parser.add_argument(
		"--skip-proper-nouns",
		action="store_true",
		help="Skip WordNet-based proper-noun detection.",
	)
	parser.add_argument(
		"--keep-titlecase",
		action="store_true",
		help="Keep tokens that contain uppercase characters (defaults to dropping them).",
	)
	return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
	args = parse_args(sys.argv[1:] if argv is None else argv)

	if not args.input.exists():
		print(f"Input file not found: {args.input}", file=sys.stderr)
		return 1

	try:
		words = load_words(args.input)
	except OSError as exc:
		print(f"Failed to read {args.input}: {exc}", file=sys.stderr)
		return 1

	stopword_inventory = load_stopword_inventory(args.extra_stopwords)

	require_wordnet = not args.skip_proper_nouns
	if require_wordnet and wn is None:
		print("Warning: NLTK not available; proper noun filtering disabled.", file=sys.stderr)
		require_wordnet = False

	filtered_words = filter_words(
		words=words,
		stopword_inventory=stopword_inventory,
		min_length=max(1, args.min_length),
		require_wordnet=require_wordnet,
		drop_titlecase=not args.keep_titlecase,
	)

	try:
		args.output.parent.mkdir(parents=True, exist_ok=True)
		with args.output.open("w", encoding="utf-8") as handle:
			for token in filtered_words:
				handle.write(f"{token}\n")
	except OSError as exc:
		print(f"Failed to write {args.output}: {exc}", file=sys.stderr)
		return 1

	removed = len(words) - len(filtered_words)
	print(
		f"Filtered {removed} of {len(words)} tokens; wrote {len(filtered_words)} items to {args.output}",
		file=sys.stderr,
	)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
