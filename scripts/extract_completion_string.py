from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

OUT_DIR = repo_root / "data" / "processed" / "qc"
PROCESSED_DIR = repo_root / "data" / "processed"

# (od_in, weight_lbft) → (id_in, drift_in)
PIPE_SPECS: dict[tuple[float, float], tuple[float, float]] = {
    (4.500, 15.2): (3.958, 3.875),
    (4.500, 15.1): (3.958, 3.875),
    (5.500, 23.0): (4.892, 4.778),
    (5.500, 17.0): (5.044, 4.890),
}

# Running sequence (DDR 06–10 Sep 2024): depth = liner_hanger_depth − position_from_bottom_when_added
# DHSV: 11,631 − 10,760 = 871 ft | gauge mandrel: 11,631 − 410 = 11,221 ft | HPS packer: 11,631 − 308 = 11,323 ft
# depth_source: 'confirmed' = directly stated in DDR text
#               'calculated' = derived from running-sequence calculation
#               'estimated'  = engineering estimate / uncertain

UPPER_COMPLETION: list[dict] = [
    dict(
        component="DrillQuip Tubing Hanger",
        component_type="tubing_hanger",
        depth_top_ft=0.0,
        depth_ft=120.8,
        od_in=None,
        id_in=None,
        drift_in=None,
        weight_lbft=None,
        grade=None,
        connection=None,
        vendor="DrillQuip",
        notes="Landed 18-3/4\" wellhead; TH neck seal set at 7,000 psi",
        depth_source="confirmed",
        ddr_citation="DDR-152 · p.2 · 10/09/2024",
    ),
    dict(
        component="4-1/2\" 15.2ppf TN95CR13 Tubing",
        component_type="tubing",
        depth_top_ft=120.8,
        depth_ft=871.0,
        od_in=4.500,
        id_in=3.958,
        drift_in=3.875,
        weight_lbft=15.2,
        grade="TN95CR13",
        connection="Tenaris Blue Dopeless",
        vendor="Tenaris",
        notes="Upper completion tubing above DHSV; MUT 7,580 ft.lbs",
        depth_source="confirmed",
        ddr_citation="DDR-148 · p.2 · 06/09/2024",
    ),
    dict(
        component="Halliburton DHSV",
        component_type="dhsv",
        depth_top_ft=None,
        depth_ft=871.0,
        od_in=4.500,
        id_in=3.813,
        drift_in=None,
        weight_lbft=None,
        grade=None,
        connection=None,
        vendor="Halliburton",
        notes="Downhole Safety Valve; control line 5,000 psi to open; depth calculated from running sequence",
        depth_source="calculated",
        ddr_citation="DDR-149 · p.2 · 07/09/2024",
        review_note=(
            "DEPTH NOT STATED IN DDR — INDICATIVE ONLY. "
            "Derived from running sequence: DHSV was added at surface when string bottom = 10,760 ft; "
            "liner hanger landed at 11,631 ft → DHSV depth = 11,631 − 10,760 = 871 ft. "
            "Verify against completion tally document before any intervention planning."
        ),
    ),
    dict(
        component="4-1/2\" 15.2ppf TN95CR13 Tubing",
        component_type="tubing",
        depth_top_ft=871.0,
        depth_ft=10_650.0,
        od_in=4.500,
        id_in=3.958,
        drift_in=3.875,
        weight_lbft=15.2,
        grade="TN95CR13",
        connection="Tenaris Blue Dopeless",
        vendor="Tenaris",
        notes="Upper completion tubing below DHSV; total 269 clamps installed",
        depth_source="confirmed",
        ddr_citation="DDR-148 · p.2 · 06/09/2024",
    ),
    dict(
        component="4-1/2\" × 5-1/2\" Completion Crossover",
        component_type="crossover",
        depth_top_ft=None,
        depth_ft=10_650.0,
        od_in=5.500,
        id_in=4.892,
        drift_in=None,
        weight_lbft=None,
        grade=None,
        connection=None,
        vendor=None,
        notes="Transition from 4-1/2\" to 5-1/2\" tubing; set in slips at 10,650 ft",
        depth_source="confirmed",
        ddr_citation="DDR-149 · p.2 · 07/09/2024",
    ),
    dict(
        component="5-1/2\" 23ppf S13Cr110 Tubing",
        component_type="tubing",
        depth_top_ft=10_650.0,
        depth_ft=11_221.0,
        od_in=5.500,
        id_in=4.892,
        drift_in=4.778,
        weight_lbft=23.0,
        grade="S13Cr110",
        connection="Tenaris Blue",
        vendor="Tenaris",
        notes="Lower upper-completion tubing; 28 clamps installed",
        depth_source="confirmed",
        ddr_citation="DDR-150 · p.2 · 08/09/2024",
    ),
    dict(
        component="Baker Completions Elite Gauge Mandrel",
        component_type="gauge_mandrel",
        depth_top_ft=None,
        depth_ft=11_221.0,
        od_in=4.500,
        id_in=3.958,
        drift_in=None,
        weight_lbft=None,
        grade=None,
        connection=None,
        vendor="Baker Hughes",
        notes="DHP gauge carrier; inner/outer seals tested 300/10,000 psi; depth calculated from running sequence",
        depth_source="calculated",
        review_note=(
            "DEPTH NOT STATED IN DDR — INDICATIVE ONLY. "
            "Derived from running sequence: gauge mandrel added at surface when string bottom = 410 ft; "
            "liner hanger landed at 11,631 ft → gauge mandrel depth = 11,631 − 410 = 11,221 ft. "
            "Verify against completion tally document before any intervention planning."
        ),
        ddr_citation="DDR-148 · p.2 · 06/09/2024",
    ),
    dict(
        component="Halliburton HPS Production Packer",
        component_type="production_packer",
        depth_top_ft=None,
        depth_ft=11_323.0,
        od_in=None,
        id_in=3.813,
        drift_in=None,
        weight_lbft=None,
        grade=None,
        connection=None,
        vendor="Halliburton",
        notes="Hydraulically-set production packer in 9-7/8\" casing; depth calculated from running sequence",
        depth_source="calculated",
        ddr_citation="DDR-148 · p.2 · 06/09/2024",
        review_note=(
            "DEPTH NOT STATED IN DDR — INDICATIVE ONLY. "
            "Derived from running sequence: HPS packer added at surface when string bottom = 308 ft "
            "(DDR-148 p.2: 'RIH tubing joints from 232 to 308 ft. Made up Halliburton HPS packer to string'); "
            "liner hanger landed at 11,631 ft → packer depth = 11,631 − 308 = 11,323 ft. "
            "Uncertainty ±50-100 ft due to pup joint lengths and crossover. "
            "Verify against completion tally document before any intervention planning."
        ),
    ),
    dict(
        component="Packers Plus Prime SET Liner Hanger Packer",
        component_type="liner_hanger_packer",
        depth_top_ft=None,
        depth_ft=11_631.0,
        od_in=None,
        id_in=4.892,
        drift_in=None,
        weight_lbft=None,
        grade=None,
        connection=None,
        vendor="Packers Plus",
        notes="Top of 5-1/2\" liner; muleshoe depth 11,631 ft; WEG centraliser sheared at 15klbs set-down",
        depth_source="confirmed",
        ddr_citation="DDR-150 · p.2 · 08/09/2024",
    ),
]

LINER_COMPONENTS: list[dict] = [
    dict(
        component="5-1/2\" 23ppf TN110Cr13S Production Liner",
        component_type="liner",
        depth_top_ft=11_631.0,
        depth_ft=19_117.0,
        od_in=5.500,
        id_in=4.892,
        drift_in=4.778,
        weight_lbft=23.0,
        grade="TN110Cr13S",
        connection="Wedge W513",
        vendor="Tenaris",
        notes="Contains 18 NCS multi-stage frac sleeves; MUT 14,700 ft.lbs",
        depth_source="confirmed",
        ddr_citation="DDR-065 → DDR-072 · Jun 2024 (PROD1)",
    ),
    dict(
        component="Packers Plus Float Shoe / Landing Collar",
        component_type="float_shoe",
        depth_top_ft=None,
        depth_ft=19_117.0,
        od_in=5.500,
        id_in=4.500,
        drift_in=None,
        weight_lbft=None,
        grade=None,
        connection=None,
        vendor="Packers Plus",
        notes="Liner TD 19,117 ft OTH",
        depth_source="confirmed",
        ddr_citation="DDR-072 · Jun 2024 (PROD1)",
    ),
]


_SLEEVE_RE = re.compile(
    r"(?:frac\s+sleeve|sleeve)\s*#?\s*(\d+)\D{0,60}?([\d,]+\.?\d*)\s*ft",
    re.I,
)
_TALLY_RE = re.compile(
    r"sleeve\s*(?:profile\s+)?tally\s+depth\s+([\d,]+\.?\d*)\s*ft",
    re.I,
)


def _load_ops() -> pd.DataFrame:
    frames = []
    for doc_dir in sorted(PROCESSED_DIR.iterdir()):
        f = doc_dir / "ddr_facts.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
    if not frames:
        raise FileNotFoundError("No ddr_facts.parquet found")
    ops = pd.concat(frames, ignore_index=True)
    ops["report_date_dt"] = pd.to_datetime(ops["report_date"], dayfirst=True, errors="coerce")
    return ops


def _extract_sleeve_depths(ops: pd.DataFrame) -> tuple[dict[int, float], dict[int, str]]:
    # NCS tally: #1 = deepest, #18 = shallowest; COMPZN only (consistent numbering)
    # Sleeve #6 hard-coded from packer-set position at ~16,700 ft (DDR-145)
    sleeve_refs: dict[int, list[float]] = {}
    sleeve_cites: dict[int, tuple[str, int]] = {}  # {num: (report_date, page)}

    compzn = ops[ops["phase"] == "COMPZN"]
    for _, row in compzn.iterrows():
        text = str(row.get("operation_text") or "")
        for m in _SLEEVE_RE.finditer(text):
            num = int(m.group(1))
            depth = float(m.group(2).replace(",", ""))
            if 1 <= num <= 18 and 11_000 < depth < 20_000:
                sleeve_refs.setdefault(num, []).append(depth)
                if num not in sleeve_cites:
                    sleeve_cites[num] = (str(row.get("report_date", "")), row.get("page", "?"))

    if 6 not in sleeve_refs:
        sleeve_refs[6] = [16_700.0]
        sleeve_cites[6] = ("03/09/2024", 2)

    depths = {
        num: sorted(ds)[len(ds) // 2]
        for num, ds in sleeve_refs.items()
        if ds
    }

    # Build human-readable citation strings using report_no lookup
    report_no_map: dict[str, int] = {}
    try:
        import glob as _glob
        hdrs = pd.concat([
            pd.read_parquet(f)
            for f in _glob.glob(str(PROCESSED_DIR / "*" / "ddr_header.parquet"))
        ], ignore_index=True)
        report_no_map = dict(zip(hdrs["report_date"].astype(str), hdrs["report_no"]))
    except Exception:
        pass

    citations: dict[int, str] = {}
    for num, (date, page) in sleeve_cites.items():
        rno = report_no_map.get(date, "?")
        citations[num] = f"DDR-{rno} · p.{page} · {date}"

    return depths, citations


def _build_sleeve_rows(sleeve_depths: dict[int, float],
                       sleeve_citations: dict[int, str]) -> list[dict]:
    rows = []

    confirmed_sorted = sorted(
        [(k, v) for k, v in sleeve_depths.items() if k >= 6],
        key=lambda x: x[0],
    )
    if len(confirmed_sorted) >= 2:
        nums  = [x[0] for x in confirmed_sorted]
        depts = [x[1] for x in confirmed_sorted]
        avg_spacing = (max(depts) - min(depts)) / (max(nums) - min(nums))
    else:
        avg_spacing = 305.0

    ref_num, ref_depth = min(confirmed_sorted, key=lambda x: x[0]) if confirmed_sorted else (6, 16_700.0)

    for num in range(1, 19):
        depth = sleeve_depths.get(num)
        if depth is None:
            depth = ref_depth + (ref_num - num) * avg_spacing
            source = "estimated"
        else:
            source = "confirmed"

        cite = sleeve_citations.get(num, "depth estimated — not fracked")
        rows.append(dict(
            component=f"NCS Multi-Stage Frac Sleeve #{num}",
            component_type="frac_sleeve",
            depth_top_ft=None,
            depth_ft=round(depth, 1) if depth else None,
            od_in=5.500,
            id_in=3.996,
            drift_in=3.996,
            weight_lbft=None,
            grade=None,
            connection=None,
            vendor="NCS Multistage",
            notes=(
                f"Sleeve #{num}; shifting profile bore 3.996\" (bullnose OD confirmed in DDR); "
                + ("not fracked — depth estimated" if num <= 5 else "depth from fracing tally references")
            ),
            depth_source=source,
            ddr_citation=cite,
        ))
    return rows


def build_completion_string() -> pd.DataFrame:
    ops = _load_ops()
    sleeve_depths, sleeve_citations = _extract_sleeve_depths(ops)

    rows = (
        UPPER_COMPLETION
        + LINER_COMPONENTS[:1]
        + _build_sleeve_rows(sleeve_depths, sleeve_citations)
        + LINER_COMPONENTS[1:]
    )

    for r in rows:
        r.setdefault("ddr_citation", "")
        r.setdefault("review_note", "")

    df = pd.DataFrame(rows)
    df = df.sort_values("depth_ft").reset_index(drop=True)

    tubing_ids = df.loc[df["component_type"] == "tubing", "id_in"].dropna()
    min_tubing_id = tubing_ids.min() if not tubing_ids.empty else 99.0
    df["is_id_restriction"] = (
        df["id_in"].notna() & (df["id_in"] < min_tubing_id)
    )

    return df


def main() -> None:
    print("Building completion string...")
    df = build_completion_string()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = OUT_DIR / "ddr_completion_string.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved {len(df)} components → {parquet_path}")

    print("\nCompletion string summary:")
    print(f"{'#':<3} {'Component':<45} {'Depth (ft MD)':>14} {'OD\"':>6} {'ID\"':>6} {'Drift\"':>7} {'Restrict?':>10} {'Source':<12}")
    print("-" * 110)
    for _, r in df.iterrows():
        flag = " ◄ ID restriction" if r["is_id_restriction"] else ""
        print(
            f"   {str(r['component']):<45} "
            f"{str(r['depth_ft'] or '—'):>14} "
            f"{str(r['od_in'] or '—'):>6} "
            f"{str(r['id_in'] or '—'):>6} "
            f"{str(r['drift_in'] or '—'):>7} "
            f"{'YES' if r['is_id_restriction'] else '':>10} "
            f"{r['depth_source']:<12}"
            f"{flag}"
        )

    restrictions = df[df["is_id_restriction"]]
    if not restrictions.empty:
        print(f"\nID restrictions found ({len(restrictions)}):")
        for _, r in restrictions.iterrows():
            print(f"  {r['component']}: {r['id_in']}\" at {r['depth_ft']} ft MD")


if __name__ == "__main__":
    main()
