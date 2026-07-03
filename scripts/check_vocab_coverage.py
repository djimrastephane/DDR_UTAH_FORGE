from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
from ddr_rag.vocab import ABBREVIATION_MAP, COMPOUND_TERMS, STOP_WORDS, UNIT_TOKENS

# Tokens already handled by stop_words or unit_tokens are not gaps
_ALREADY_COVERED = STOP_WORDS | UNIT_TOKENS

# Pre-build a set of all source phrases from COMPOUND_TERMS for O(1) lookup
_COMPOUND_SOURCES: set[str] = {src.lower() for src, _ in COMPOUND_TERMS}

_ABBR_RE = re.compile(r"\b([A-Z]{2,6})\b")
_WORD_RE = re.compile(r"[a-z]{2,}")


def _iter_texts(chunks_path: Path):
    df = pd.read_parquet(chunks_path)
    col = "chunk_text"
    if col not in df.columns:
        raise ValueError(f"Expected column '{col}' in {chunks_path}. Found: {df.columns.tolist()}")
    for text in df[col].dropna():
        yield str(text)


def _collect_unknown_abbrs(texts, top: int) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for text in texts:
        for match in _ABBR_RE.finditer(text):
            tok = match.group(1)
            if tok.lower() not in ABBREVIATION_MAP and tok.lower() not in _ALREADY_COVERED:
                counts[tok] += 1
    return counts.most_common(top)


def _collect_unknown_phrases(texts, min_freq: int, top: int) -> list[tuple[str, int]]:
    bigrams: Counter[str] = Counter()
    trigrams: Counter[str] = Counter()

    for text in texts:
        words = _WORD_RE.findall(text.lower())
        for i in range(len(words) - 1):
            bigrams[f"{words[i]} {words[i+1]}"] += 1
        for i in range(len(words) - 2):
            trigrams[f"{words[i]} {words[i+1]} {words[i+2]}"] += 1

    results: Counter[str] = Counter()
    for phrase, count in bigrams.items():
        if count >= min_freq and phrase not in _COMPOUND_SOURCES:
            results[phrase] = count
    for phrase, count in trigrams.items():
        if count >= min_freq and phrase not in _COMPOUND_SOURCES:
            results[phrase] = count

    return results.most_common(top)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit vocab coverage against processed chunks.")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=REPO_ROOT / "data" / "global_index" / "chunks.parquet",
        help="Path to chunks.parquet (default: data/global_index/chunks.parquet)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of candidates to show per section (default: 20)",
    )
    parser.add_argument(
        "--min-phrase-freq",
        type=int,
        default=8,
        help="Minimum occurrence count for phrase candidates (default: 8)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.chunks.exists():
        print(f"ERROR: chunks file not found at {args.chunks}", file=sys.stderr)
        print("Run scripts/build_global_index.py first, or pass --chunks <path>", file=sys.stderr)
        sys.exit(1)

    print(f"Loading chunks from {args.chunks} …")
    texts = list(_iter_texts(args.chunks))
    print(f"  {len(texts):,} chunks loaded\n")

    print(f"{'─' * 60}")
    print(f"Unknown abbreviation-like tokens  (top {args.top})")
    print(f"  → Add matching entries to abbreviation_map in ddr_vocab.yaml")
    print(f"{'─' * 60}")

    unknown_abbrs = _collect_unknown_abbrs(texts, top=args.top)
    if unknown_abbrs:
        width = max(len(tok) for tok, _ in unknown_abbrs)
        for tok, count in unknown_abbrs:
            print(f"  {tok:<{width}}  {count:>6,}x")
    else:
        print("  (none found)")

    print()
    print(f"{'─' * 60}")
    print(f"Frequent unmatched phrases  (≥{args.min_phrase_freq}x, top {args.top})")
    print(f"  → Add matching entries to compound_terms in ddr_vocab.yaml")
    print(f"{'─' * 60}")

    unknown_phrases = _collect_unknown_phrases(texts, min_freq=args.min_phrase_freq, top=args.top)
    if unknown_phrases:
        width = max(len(p) for p, _ in unknown_phrases)
        for phrase, count in unknown_phrases:
            print(f"  {phrase:<{width}}  {count:>6,}x")
    else:
        print("  (none found — try lowering --min-phrase-freq)")

    print()
    print("To add a term: edit configs/vocab/ddr_vocab.yaml and restart the service.")


if __name__ == "__main__":
    main()
