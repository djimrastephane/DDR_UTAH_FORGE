from __future__ import annotations

import os
from pathlib import Path

# The production code path (rag_pdf.services.search_service) forces offline
# mode by default so it never reaches the hub at request time. This script is
# the one place allowed to actually fetch the model, to populate that cache.
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = REPO_ROOT / "models" / "all-MiniLM-L6-v2"
HUB_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> int:
    if (TARGET_DIR / "config.json").exists():
        print(f"Model already present at {TARGET_DIR}, skipping download.")
        return 0
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(HUB_NAME)
    model.save(str(TARGET_DIR))
    print(f"Saved {HUB_NAME} to {TARGET_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())