from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

OUT_DIR = repo_root / "data" / "processed" / "qc"
PROCESSED_DIR = repo_root / "data" / "processed"

COMPLETION_STRING_COLUMNS = [
    "component", "component_type", "depth_top_ft", "depth_ft",
    "od_in", "id_in", "drift_in", "weight_lbft", "grade", "connection",
    "vendor", "notes", "depth_source", "ddr_citation", "review_note",
    "is_id_restriction",
]


def build_completion_string() -> pd.DataFrame:
    """
    Completion-string extraction is not yet implemented for this asset.

    Earlier versions of this script hardcoded a specific well's completion
    component list and frac-sleeve numbering scheme (vendor, tally, and
    depths) as a stand-in dataset. That was specific to a different well's
    real design and not applicable here, so it has been removed rather than
    adapted — a wrong or fabricated completion string is worse than none.
    Implement extraction from processed DDR text for this asset's actual
    completion design before re-enabling this page.
    """
    return pd.DataFrame(columns=COMPLETION_STRING_COLUMNS)


def main() -> None:
    print("Building completion string...")
    df = build_completion_string()

    if df.empty:
        print(
            "Completion string extraction is not implemented for this asset yet — "
            "skipping (no output written)."
        )
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = OUT_DIR / "ddr_completion_string.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved {len(df)} components → {parquet_path}")


if __name__ == "__main__":
    main()
