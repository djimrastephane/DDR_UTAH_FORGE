from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

RAW_DIR  = repo_root / "data" / "raw"
OUT_DIR  = repo_root / "data" / "processed" / "qc"

DDR_DATE_RE = re.compile(
    r"(\d{2})\.(\d{2})\.(\d{4})\.+pdf$", re.IGNORECASE  # .+ handles single & double-dot filenames
)

def _parse_grams(text: str) -> float | None:
    cleaned = text.replace(",", "")
    m = re.search(r"([\d]+(?:\.\d+)?)\s*g", cleaned)
    if not m:
        return None
    raw = m.group(1)
    try:
        value = float(raw)
        # Heuristic: if there is exactly one period followed by 3 digits,
        # treat it as a thousands separator (e.g. "14.945" → 14945).
        # Genuine decimals like "3.5" or "0.25" have fewer digits after the dot.
        if "." in raw:
            parts = raw.split(".")
            if len(parts) == 2 and len(parts[1]) == 3:
                value = float(parts[0] + parts[1])   # "14" + "945" → 14945
        return value
    except ValueError:
        return None


def _parse_qualifier(text: str) -> str:
    m_paren = re.search(r"\(([^)]+)\)", text)
    # Dash-separated note: "- Includes recovery from..."
    m_dash  = re.search(r"-\s*(.+)$", re.sub(r"\(.*?\)", "", text).rstrip("., "))
    parts = []
    if m_paren:
        parts.append(m_paren.group(1).strip())
    if m_dash and m_dash.group(1).strip():
        parts.append(m_dash.group(1).strip())
    return "; ".join(parts)


_DITCH_BLOCK_RE = re.compile(
    r"Ditch\s*Magnets?\s*[:\-]?\s*\n(.*?)(?=\n={4,}|\nDaily\s+Offline|\nDRILLING\s+ORIGINAL|\Z)",
    re.DOTALL | re.IGNORECASE,
)

_DAILY_RE = re.compile(
    r"Previous\s+(\d+)\s+hours?\s*[:\-]?\s*([\d,]+(?:\.\d+)?)\s*g",
    re.IGNORECASE,
)

_SECTION_RE = re.compile(
    # Capture: hole size descriptor + grams
    r"([\d\-/]+\"\s*(?:x\s*[\d\-/]+\")?\s*Section\s*Total)\s*[:\-]?\s*([\d,]+(?:\.\d+)?)\s*g",
    re.IGNORECASE,
)

_ASSEMBLY_RE = re.compile(
    r"Assembly\s*#(\d+)\s*=\s*([\d,]+(?:\.\d+)?)\s*g",
    re.IGNORECASE,
)


def extract_ditch_magnets(pdf_path: Path) -> dict:
    result = {
        "has_ditch_magnet":    False,
        "daily_grams":         None,
        "daily_hours":         None,
        "daily_qualifier":     "",
        "section_name":        "",
        "section_total_grams": None,
        "section_qualifier":   "",
        "assembly_breakdown":  {},
        "assembly_total_grams":None,
        "raw_block":           "",
    }

    try:
        import pymupdf
        doc = pymupdf.open(str(pdf_path))
        for page in doc.pages():
            txt = page.get_text()
            m_block = _DITCH_BLOCK_RE.search(txt)
            if not m_block:
                continue

            result["has_ditch_magnet"] = True
            block = m_block.group(1)
            result["raw_block"] = block.strip()[:400]

            for line in block.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue

                m_daily = _DAILY_RE.search(stripped)
                if m_daily and result["daily_grams"] is None:
                    result["daily_hours"]    = int(m_daily.group(1))
                    result["daily_grams"]    = _parse_grams(stripped)
                    result["daily_qualifier"] = _parse_qualifier(stripped)
                    continue

                m_sec = _SECTION_RE.search(stripped)
                if m_sec and result["section_total_grams"] is None:
                    result["section_name"]        = m_sec.group(1).strip()
                    result["section_total_grams"] = _parse_grams(stripped)
                    result["section_qualifier"]   = _parse_qualifier(stripped)
                    continue

                m_asm = _ASSEMBLY_RE.search(stripped)
                if m_asm:
                    asm_id = f"assembly_{m_asm.group(1)}"
                    asm_g  = _parse_grams(stripped)
                    if asm_g is not None:
                        result["assembly_breakdown"][asm_id] = asm_g

            # Sum assemblies if present
            if result["assembly_breakdown"]:
                result["assembly_total_grams"] = sum(
                    result["assembly_breakdown"].values()
                )

            break   # stop at first page with Ditch Magnet section
        doc.close()
    except Exception as exc:
        result["parse_error"] = str(exc)

    return result


def build_corpus(raw_dir: Path) -> pd.DataFrame:
    pdfs = sorted(raw_dir.glob("*.pdf"))
    rows = []

    print(f"Scanning {len(pdfs)} PDFs for Ditch Magnet data...")
    found = 0

    for i, pdf_path in enumerate(pdfs, 1):
        m_date = DDR_DATE_RE.search(pdf_path.name)
        if not m_date:
            continue
        report_date = f"{m_date.group(3)}-{m_date.group(2)}-{m_date.group(1)}"

        result = extract_ditch_magnets(pdf_path)
        row = {
            "source_filename": pdf_path.name,
            "report_date":     report_date,
            **{k: v for k, v in result.items()
               if k not in ("assembly_breakdown", "raw_block")},
            "assembly_detail": str(result["assembly_breakdown"]) if result["assembly_breakdown"] else "",
        }
        rows.append(row)
        if result["has_ditch_magnet"]:
            found += 1

        if i % 20 == 0:
            print(f"  {i}/{len(pdfs)}  found={found}")

    df = pd.DataFrame(rows)
    df["report_date"] = pd.to_datetime(df["report_date"])
    df = df.sort_values("report_date").reset_index(drop=True)
    print(f"\nTotal: {len(df)} DDRs, {found} with Ditch Magnet data "
          f"({found/len(df)*100:.0f}% coverage)")
    return df


def print_findings(df: pd.DataFrame) -> None:
    has = df[df["has_ditch_magnet"]]
    print()
    print("=== Ditch Magnet Findings ===")
    print(f"Coverage:  {len(has)}/{len(df)} DDRs  ({len(has)/len(df)*100:.0f}%)")

    daily = has[has["daily_grams"].notna()]
    print(f"Daily readings available: {len(daily)}")
    if len(daily):
        print(f"  Total debris (sum of daily): {daily['daily_grams'].sum():,.0f}g  "
              f"({daily['daily_grams'].sum()/1000:.2f}kg)")
        print(f"  Max single day: {daily['daily_grams'].max():,.0f}g  "
              f"({daily.loc[daily['daily_grams'].idxmax(), 'source_filename']})")
        print(f"  Days with 0g: {(daily['daily_grams']==0).sum()}")
        print(f"  Days with >100g: {(daily['daily_grams']>100).sum()}")
        print(f"  Days with >1000g: {(daily['daily_grams']>1000).sum()}")

    sections = has[has["section_name"].notna() & (has["section_name"] != "")]
    print(f"\nSection totals available: {len(sections)}")
    for sname, grp in sections.groupby("section_name"):
        max_total = grp["section_total_grams"].dropna().max()
        print(f"  {sname}: max cumulative = {max_total:,.0f}g ({max_total/1000:.2f}kg)")

    with_assembly = has[has["assembly_detail"] != ""]
    if len(with_assembly):
        print(f"\nAssembly breakdowns: {len(with_assembly)} DDRs")
        for _, r in with_assembly.head(3).iterrows():
            print(f"  {r['source_filename']}: {r['assembly_detail'][:80]}")

    qualifiers = has[has["daily_qualifier"] != ""]["daily_qualifier"].value_counts()
    if len(qualifiers):
        print(f"\nDaily qualifiers observed:")
        for q, c in qualifiers.items():
            print(f"  {c:3d}x  {q[:70]}")

    print()
    print("Top 10 highest daily readings:")
    top = daily.nlargest(10, "daily_grams")[
        ["report_date", "source_filename", "daily_grams", "daily_qualifier",
         "section_name", "section_total_grams"]
    ]
    print(top.to_string(index=False))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--out",     default=str(OUT_DIR / "ddr_ditch_magnets.parquet"))
    args = parser.parse_args()

    df = build_corpus(Path(args.raw_dir))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Written to: {out_path}")

    print_findings(df)

    csv_path = out_path.with_suffix(".csv")
    df[df["has_ditch_magnet"]].to_csv(csv_path, index=False)
    print(f"CSV (magnet-only rows): {csv_path}")


if __name__ == "__main__":
    main()
