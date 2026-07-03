from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

repo_root   = Path(__file__).resolve().parents[1]
FIELDS_DIR  = repo_root / "data" / "fields"
PROCESSED   = repo_root / "data" / "processed"
sys.path.insert(0, str(repo_root / "src"))


def _load_well_manifest(field_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    wells_dir = field_dir / "wells"
    if not wells_dir.exists():
        return mapping
    for well_dir in sorted(wells_dir.iterdir()):
        if not well_dir.is_dir():
            continue
        well_id  = well_dir.name
        ids_file = well_dir / "ddr_ids.txt"
        if ids_file.exists():
            for line in ids_file.read_text().splitlines():
                prefix = line.strip()
                if prefix:
                    mapping[prefix] = well_id
        meta_file = well_dir / "metadata.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                if "doc_id_prefix" in meta:
                    mapping[meta["doc_id_prefix"]] = well_id
            except Exception:
                pass
    return mapping


def _assign_well_id(doc_id: str, manifest: dict[str, str], fallback: str = "Unknown") -> str:
    for prefix, well_id in manifest.items():
        if str(doc_id).startswith(prefix):
            return well_id
    return fallback


def rebuild(field_name: str = "UtahForge", dry_run: bool = False) -> None:
    field_dir    = FIELDS_DIR / field_name
    analysis_dir = field_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_well_manifest(field_dir)
    if not manifest:
        print(
            f"[warn] No well manifest found in {field_dir / 'wells'}. "
            "All DDRs will be assigned to the first well or 'Unknown'.\n"
            "Create data/fields/<field>/wells/<well_id>/ddr_ids.txt with doc_id prefixes."
        )

    facts_frames: list[pd.DataFrame] = []
    hdr_frames:   list[pd.DataFrame] = []
    wells_seen: set[str] = set()

    for doc_dir in sorted(PROCESSED.iterdir()):
        if not doc_dir.is_dir():
            continue
        facts_path = doc_dir / "ddr_facts.parquet"
        hdr_path   = doc_dir / "ddr_header.parquet"

        doc_id  = doc_dir.name
        well_id = _assign_well_id(doc_id, manifest)
        wells_seen.add(well_id)

        if facts_path.exists():
            df = pd.read_parquet(facts_path)
            df["well_id"] = well_id
            facts_frames.append(df)

        if hdr_path.exists():
            df = pd.read_parquet(hdr_path)
            df["well_id"] = well_id
            hdr_frames.append(df)

    if not facts_frames:
        print("No ddr_facts.parquet files found — nothing to rebuild.")
        return

    combined_facts   = pd.concat(facts_frames,   ignore_index=True)
    combined_headers = pd.concat(hdr_frames,      ignore_index=True) if hdr_frames else pd.DataFrame()

    def _n(s):
        try: return float(re.sub(r"[^\d.]", "", str(s).split()[0]))
        except: return None
    if not combined_headers.empty:
        combined_headers["daily_cost_num"] = combined_headers["daily_cost"].apply(_n)
        combined_headers["cum_cost_num"]   = combined_headers["cumulative_cost"].apply(_n)
        combined_headers["end_depth_num"]  = combined_headers.get("end_depth_md_ft", pd.Series(dtype=float)).apply(_n)
        combined_headers["report_date_dt"] = pd.to_datetime(
            combined_headers["report_date"], dayfirst=True, errors="coerce"
        )

    print(f"Field: {field_name}")
    print(f"Wells: {sorted(wells_seen)}")
    print(f"Total ops rows: {len(combined_facts):,}")
    print(f"Total header rows: {len(combined_headers):,}")

    if dry_run:
        print("[dry-run] No files written.")
        return

    facts_out = analysis_dir / "combined_facts.parquet"
    hdrs_out  = analysis_dir / "combined_headers.parquet"
    combined_facts.to_parquet(facts_out, index=False)
    print(f"Wrote {facts_out}")
    if not combined_headers.empty:
        combined_headers.to_parquet(hdrs_out, index=False)
        print(f"Wrote {hdrs_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild field analysis combined parquets")
    parser.add_argument("--field",   default="UtahForge", help="Field name under data/fields/")
    parser.add_argument("--dry-run", action="store_true",   help="Show what would be written without writing")
    args = parser.parse_args()
    rebuild(args.field, args.dry_run)


if __name__ == "__main__":
    main()
