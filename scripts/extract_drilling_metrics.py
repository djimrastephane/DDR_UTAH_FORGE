from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterator

import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

OUT_DIR       = repo_root / "data" / "processed" / "qc"
PROCESSED_DIR = repo_root / "data" / "processed"

_ROP_INST = re.compile(
    r"(?:instantaneous|inst\.?)\s+rop[^\d]{0,12}(\d+\.?\d*)\s*ft/h",
    re.I,
)
_ROP_AVG  = re.compile(
    r"(?:average|avg\.?)\s+rop[^\d]{0,12}(\d+\.?\d*)\s*ft/h",
    re.I,
)
_ROP_GEN  = re.compile(
    r"\brop\b[^\d]{0,12}(\d+\.?\d*)\s*ft/h",
    re.I,
)

_TRIP_SPEED = re.compile(r"(\d+)\s*ft/min", re.I)

_OH_RE = re.compile(r"\bopen\s*hole\b|\boh\b", re.I)
_CH_RE = re.compile(
    r"\bcased\s*hole\b|\bcasing\b|\bliner\b|inside.*liner|through.*casing|"
    r"\binside\s+\d+[\"′]|in\s+liner",
    re.I,
)

_JTS_HR = re.compile(r"(\d+\.?\d*)\s*(?:joints?|jts?|stands?)\s*/\s*hr", re.I)

_DEPTH_INTERVAL = re.compile(
    r"(?:from|rih from)\s+([\d,]+)\s*ft[^.]{0,30}to\s+([\d,]+)\s*ft",
    re.I,
)

_WOB = re.compile(r"\bwob\b[^\d]{0,6}(\d+\.?\d*)\s*klbs?", re.I)

_FLOW = re.compile(r"(\d{3,4})\s*gpm", re.I)   # ≥100 gpm to avoid false positives

_MUT = re.compile(
    r"(?:made\s+up\s+to|mut|optimum\s+mut|maximum\s+mut)[^\d]{0,20}"
    r"([\d,]+)\s*ft\.?\s*lbs?",
    re.I,
)


def _snip(text: str, pat: re.Pattern, window: int = 60) -> str:
    m = pat.search(text)
    if not m:
        return text[:80]
    start = max(0, m.start() - window // 2)
    end   = min(len(text), m.end() + window // 2)
    return f"…{text[start:end]}…"


def _depth_interval(text: str) -> tuple[float | None, float | None]:
    m = _DEPTH_INTERVAL.search(text)
    if m:
        try:
            return (
                float(m.group(1).replace(",", "")),
                float(m.group(2).replace(",", "")),
            )
        except ValueError:
            pass
    return None, None


def _hole_type(text: str, phase: str = "") -> str:
    oh = _OH_RE.search(text)
    ch = _CH_RE.search(text)
    if oh and not ch:
        return "OH"
    if ch and not oh:
        return "CH"
    # Phase-based fallback
    if phase in ("COND1", "INTRM1", "INTRM2"):
        return "OH"
    if phase in ("COMPZN",):
        return "CH"
    if phase == "PROD1":
        # Liner RIH is CH, open-hole section is OH
        if re.search(r"\bliner\b|\bcasing\b", text, re.I):
            return "CH"
        return "OH"
    return "unknown"


def _extract_rop(text: str, meta: dict) -> list[dict]:
    records = []
    d_from, d_to = _depth_interval(text)

    for pat, mtype in [(_ROP_INST, "rop_inst"), (_ROP_AVG, "rop_avg")]:
        for m in pat.finditer(text):
            try:
                val = float(m.group(1))
            except ValueError:
                continue
            if val <= 0 or val > 2000:   # sanity range
                continue
            records.append({
                **meta,
                "metric_type": mtype,
                "value":       val,
                "unit":        "ft/hr",
                "hole_type":   _hole_type(text, meta.get("phase", "")),
                "depth_from_ft": d_from,
                "depth_to_ft":   d_to,
                "raw_snippet": _snip(text, pat),
            })

    # Generic \brop\b ft/hr — only if no specific match already found
    if not records:
        for m in _ROP_GEN.finditer(text):
            try:
                val = float(m.group(1))
            except ValueError:
                continue
            if val <= 0 or val > 2000:
                continue
            records.append({
                **meta,
                "metric_type": "rop_gen",
                "value":       val,
                "unit":        "ft/hr",
                "hole_type":   _hole_type(text, meta.get("phase", "")),
                "depth_from_ft": d_from,
                "depth_to_ft":   d_to,
                "raw_snippet": _snip(text, _ROP_GEN),
            })

    return records


def _extract_trip_speed(text: str, meta: dict) -> list[dict]:
    records = []
    d_from, d_to = _depth_interval(text)
    ht = _hole_type(text, meta.get("phase", ""))

    for m in _TRIP_SPEED.finditer(text):
        try:
            val = int(m.group(1))
        except ValueError:
            continue
        if val <= 0 or val > 500:   # unrealistic range
            continue
        # Local context: 40 chars either side of the number
        ctx      = text[max(0, m.start()-40): m.end()+40]
        local_oh = bool(_OH_RE.search(ctx))
        local_ch = bool(_CH_RE.search(ctx))
        local_ht = ("OH" if local_oh and not local_ch else
                    "CH" if local_ch and not local_oh else ht)

        records.append({
            **meta,
            "metric_type": "trip_speed",
            "value":       float(val),
            "unit":        "ft/min",
            "hole_type":   local_ht,
            "depth_from_ft": d_from,
            "depth_to_ft":   d_to,
            "raw_snippet": _snip(text, _TRIP_SPEED),
        })

    return records


def _extract_running_speed(text: str, meta: dict) -> list[dict]:
    records = []
    d_from, d_to = _depth_interval(text)
    is_tubing = bool(re.search(
        r"\btubing\b|\bwash.?pipe\b|\b3-1/2\"\b|\b4-1/2\"\b|\b5-1/2\"\s+dp\b|\b5-1/2\"\s+tubing",
        text, re.I,
    ))
    is_casing = bool(re.search(
        r"\bcasing\b|\bliner\b|\b20\"\b|\b13-3/8\"\b|\b9-7/8\"\b|\b7\"\b",
        text, re.I,
    ))
    mtype = ("tubing_speed" if is_tubing
             else "casing_speed" if is_casing
             else "running_speed")

    for m in _JTS_HR.finditer(text):
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        if val <= 0 or val > 200:
            continue
        records.append({
            **meta,
            "metric_type": mtype,
            "value":       val,
            "unit":        "joints/hr",
            "hole_type":   _hole_type(text, meta.get("phase", "")),
            "depth_from_ft": d_from,
            "depth_to_ft":   d_to,
            "raw_snippet": _snip(text, _JTS_HR),
        })

    return records


def _extract_wob(text: str, meta: dict) -> list[dict]:
    records = []
    d_from, d_to = _depth_interval(text)
    for m in _WOB.finditer(text):
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        if val <= 0 or val > 100:
            continue
        records.append({
            **meta,
            "metric_type": "wob",
            "value": val, "unit": "klbs",
            "hole_type": "OH",
            "depth_from_ft": d_from, "depth_to_ft": d_to,
            "raw_snippet": _snip(text, _WOB),
        })
    return records


def _extract_flow(text: str, meta: dict) -> list[dict]:
    records = []
    d_from, d_to = _depth_interval(text)
    for m in _FLOW.finditer(text):
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        if val < 100 or val > 2000:
            continue
        records.append({
            **meta,
            "metric_type": "flow_rate",
            "value": val, "unit": "gpm",
            "hole_type": "unknown",
            "depth_from_ft": d_from, "depth_to_ft": d_to,
            "raw_snippet": _snip(text, _FLOW),
        })
    return records


def _load_corpus() -> pd.DataFrame:
    frames, hdr_frames = [], []
    for doc_dir in sorted(PROCESSED_DIR.iterdir()):
        f = doc_dir / "ddr_facts.parquet"
        h = doc_dir / "ddr_header.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
        if h.exists():
            hdr_frames.append(pd.read_parquet(h, columns=["report_date", "report_no"]))

    if not frames:
        raise FileNotFoundError("No ddr_facts.parquet found")

    facts = pd.concat(frames, ignore_index=True)
    facts["report_date_dt"] = pd.to_datetime(
        facts["report_date"], dayfirst=True, errors="coerce"
    )

    if hdr_frames:
        hdr = pd.concat(hdr_frames, ignore_index=True)
        rno_map = dict(zip(hdr["report_date"].astype(str), hdr["report_no"]))
        facts["report_no"] = facts["report_date"].astype(str).map(rno_map)
    else:
        facts["report_no"] = None

    return facts


def run_extraction() -> pd.DataFrame:
    corpus = _load_corpus()
    print(f"Scanning {len(corpus):,} operation rows…")

    all_records: list[dict] = []

    for _, row in corpus.iterrows():
        text = str(row.get("operation_text") or "")
        if not text.strip():
            continue

        rno  = row.get("report_no", "?")
        pg   = row.get("page", "?")
        cite = f"DDR-{rno} · p.{pg} · {row.get('report_date','')}"

        meta = {
            "report_date":    row.get("report_date", ""),
            "report_date_dt": row.get("report_date_dt"),
            "phase":          row.get("phase", ""),
            "activity_code":  row.get("activity_code", ""),
            "op_code":        row.get("op_code", ""),
            "doc_id":         row.get("doc_id", ""),
            "page":           pg,
            "ddr_citation":   cite,
        }

        all_records.extend(_extract_rop(text, meta))
        all_records.extend(_extract_trip_speed(text, meta))
        all_records.extend(_extract_running_speed(text, meta))
        all_records.extend(_extract_wob(text, meta))
        all_records.extend(_extract_flow(text, meta))

    if not all_records:
        print("No metrics extracted.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["report_date_dt"] = pd.to_datetime(df["report_date_dt"])
    df = df.sort_values(["report_date_dt", "metric_type"]).reset_index(drop=True)
    return df


def print_summary(df: pd.DataFrame) -> None:
    for mtype in df["metric_type"].unique():
        sub = df[df["metric_type"] == mtype]
        print(
            f"\n{mtype} ({sub['unit'].iloc[0]})"
            f"  n={len(sub)}"
            f"  min={sub['value'].min():.1f}"
            f"  p50={sub['value'].median():.1f}"
            f"  p90={sub['value'].quantile(0.9):.1f}"
            f"  max={sub['value'].max():.1f}"
        )
        by_phase = sub.groupby("phase")["value"].agg(["count","median","max"])
        for ph, r in by_phase.iterrows():
            print(f"    {ph:8}: n={r['count']:4.0f}  p50={r['median']:6.1f}  max={r['max']:6.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract drilling performance metrics from DDRs")
    parser.add_argument("--summary", action="store_true", help="Print summary statistics")
    args = parser.parse_args()

    print("Extracting drilling performance metrics…")
    df = run_extraction()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "ddr_drilling_metrics.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df):,} metric records → {out_path}")

    if args.summary:
        print_summary(df)
    else:
        print("\nMetric counts by type:")
        for mtype, cnt in df["metric_type"].value_counts().items():
            print(f"  {mtype:<18}: {cnt:,}")


if __name__ == "__main__":
    main()
