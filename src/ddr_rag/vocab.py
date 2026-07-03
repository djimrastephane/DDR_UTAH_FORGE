from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Tokens that match this pattern and are not in ABBREVIATION_MAP are logged
# at DEBUG level so new abbreviations surface without manual corpus scanning.
_ABBR_CANDIDATE_RE = re.compile(r"^[A-Z]{2,6}$")


def _load_vocab() -> dict:
    path_env = os.getenv("DDR_VOCAB_FILE")
    if path_env:
        path = Path(path_env)
    else:
        # src/ddr_rag/vocab.py → parents[2] = project root
        path = Path(__file__).resolve().parents[2] / "configs" / "vocab" / "ddr_vocab.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Vocab file not found at {path}. "
            "Create configs/vocab/ddr_vocab.yaml or set DDR_VOCAB_FILE."
        )
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


_vocab = _load_vocab()

MIN_TOKEN_LEN: int = int(_vocab.get("min_token_len", 3))

STOP_WORDS: frozenset[str] = frozenset(_vocab.get("stop_words", []))

UNIT_TOKENS: frozenset[str] = frozenset(_vocab.get("unit_tokens", []))

COMPOUND_TERMS: list[tuple[str, str]] = [
    (str(entry["from"]), str(entry["to"]))
    for entry in _vocab.get("compound_terms", [])
]

ABBREVIATION_MAP: dict[str, Optional[str]] = {
    str(entry["abbr"]): (str(entry["expands_to"]) if entry.get("expands_to") is not None else None)
    for entry in _vocab.get("abbreviation_map", [])
}

OP_CODE_LABELS: dict[str, str] = {
    str(k): str(v) for k, v in (_vocab.get("op_code_labels") or {}).items()
}

ACTIVITY_CODE_LABELS: dict[str, str] = {
    str(k): str(v) for k, v in (_vocab.get("activity_code_labels") or {}).items()
}

# Compile compound patterns once (order preserved from YAML)
_COMPOUND_RE: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(src) + r"\b"), tgt)
    for src, tgt in COMPOUND_TERMS
]


def _load_phase_labels() -> dict[str, str]:
    try:
        from ddr_rag.ddr_profile import load_profile
        profile = load_profile("operator_alpha")
        if profile.phase_labels:
            return dict(profile.phase_labels)
    except Exception:
        pass
    return {
        "MIRU":   "Move In / Rig Up",
        "COND1":  "Conductor Section",
        "INTRM1": "Intermediate Section 1",
        "INTRM2": "Intermediate Section 2",
        "PROD1":  "Production / Reservoir Section",
        "COMPZN": "Completion / Zonal",
    }


PHASE_LABELS: dict[str, str] = _load_phase_labels()


def _apply_compounds(text: str) -> str:
    for pattern, replacement in _COMPOUND_RE:
        text = pattern.sub(replacement, text)
    return text


def _expand_abbreviations(text: str) -> str:
    tokens = text.split()
    result = []
    for tok in tokens:
        clean = tok.strip(".,;:()[]\"'")
        expanded = ABBREVIATION_MAP.get(clean)
        if expanded is None and clean in ABBREVIATION_MAP:
            # Explicitly mapped to None → drop token
            continue
        if expanded is not None:
            if expanded:
                result.append(expanded)
        else:
            # Log uppercase tokens that look like abbreviations but aren't mapped.
            # Enable with: logging.getLogger("ddr_rag.vocab").setLevel(logging.DEBUG)
            if _ABBR_CANDIDATE_RE.match(clean):
                logger.debug("Unknown abbreviation-like token: %r", clean)
            result.append(tok)
    return " ".join(result)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    t = _apply_compounds(t)
    t = _expand_abbreviations(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


_TOKEN_RE = re.compile(r"[a-z][a-z0-9'-]{2,}")
_NUMERIC_RE = re.compile(r"^\d")
_MEASUREMENT_RE = re.compile(
    r"^\d[\d,./'-]*(?:ft|psi|ppg|gpm|bpm|rpm|klbs|lbs|bbls?|kft|"
    r"mins?|hrs?|ppge|btu|kips?|gal|kpa|bar|mpa)?$"
)


def tokenize_for_graph(text: str) -> list[str]:
    if not text:
        return []

    normalised = normalize_text(text)

    tokens: list[str] = []
    for tok in normalised.split():
        clean = tok.strip(".,;:()[]\"'/*+-")
        if not clean:
            continue
        if _MEASUREMENT_RE.match(clean):
            continue
        if _NUMERIC_RE.match(clean):
            continue
        if "-" in clean:
            if len(clean) >= MIN_TOKEN_LEN and clean not in STOP_WORDS:
                tokens.append(clean)
            continue
        if not _TOKEN_RE.match(clean):
            continue
        if clean in STOP_WORDS or clean in UNIT_TOKENS:
            continue
        tokens.append(clean)

    return tokens


def label_op_code(code: str) -> str:
    return OP_CODE_LABELS.get(str(code).strip(), str(code).strip())


def label_activity_code(code: str) -> str:
    return ACTIVITY_CODE_LABELS.get(str(code).strip(), str(code).strip())


def label_phase(code: str) -> str:
    return PHASE_LABELS.get(str(code).strip(), str(code).strip())
