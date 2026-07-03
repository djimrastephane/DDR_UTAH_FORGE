from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

_REPORT_NO_RE  = re.compile(r"Report\s+No\.?\s*:?\s*(\d+)", re.I)
_DATE_RE        = re.compile(r"\b(\d{2}[./]\d{2}[./]\d{4})\b")
_DDR_MARKER_RE  = re.compile(
    r"Daily\s+Well\s+Oper|DRILLING\s+ORIGINAL|DAILY\s+DRILLING\s+REPORT|"
    r"WELL\s+OPERATIONS\s+REPORT",
    re.I,
)


def detect_ddr_starts(pdf_path: Path) -> list[tuple[int, int, str]]:
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("pdfplumber is required: pip install pdfplumber")

    starts: list[tuple[int, int, str]] = []
    seen_reports: set[int] = set()

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = page.extract_text() or ""

            m_rno  = _REPORT_NO_RE.search(text)
            m_date = _DATE_RE.search(text)

            if not m_rno or not m_date:
                continue

            report_no = int(m_rno.group(1))

            # Only accept the first page where this report number appears
            if report_no in seen_reports:
                continue

            # Require at least one DDR structural marker nearby to avoid false positives
            if not _DDR_MARKER_RE.search(text[:800]):
                continue

            date_str = m_date.group(1).replace("/", ".").replace("-", ".")
            starts.append((page_idx, report_no, date_str))
            seen_reports.add(report_no)

    # Sort by page index (ascending)
    starts.sort(key=lambda x: x[0])
    return starts


def split_combined_ddr(
    pdf_path: Path,
    output_dir: Path,
    well_id: str = "Unknown-Well",
    rig_name: str = "Rig",
) -> list[tuple[int, str, Path]]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError("PyMuPDF is required: pip install pymupdf")

    starts = detect_ddr_starts(pdf_path)
    if not starts:
        raise ValueError(
            f"No DDR report headers detected in '{pdf_path.name}'. "
            "Check that the file is a valid combined DDR PDF."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    results: list[tuple[int, str, Path]] = []

    for i, (start_page, report_no, date_str) in enumerate(starts):
        end_page = starts[i + 1][0] if i + 1 < len(starts) else total_pages

        safe_well = well_id.replace("/", "-").replace(" ", "_")
        filename  = f"{rig_name} DDR {report_no:03d} {safe_well} {date_str}.pdf"
        out_path  = output_dir / filename

        sub = fitz.open()
        sub.insert_pdf(doc, from_page=start_page, to_page=end_page - 1)
        sub.save(str(out_path))
        sub.close()

        results.append((report_no, date_str, out_path))
        print(f"  DDR-{report_no:03d}  {date_str}  → {filename}")

    doc.close()
    print(f"\nSplit {len(results)} DDRs from '{pdf_path.name}'")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Split combined DDR PDF into individual files")
    parser.add_argument("--input",   required=True, help="Path to combined DDR PDF")
    parser.add_argument("--outdir",  default="data/raw", help="Output directory for split PDFs")
    parser.add_argument("--well-id", default="Unknown-Well", help="Well identifier (e.g. Block-A-W2)")
    parser.add_argument("--rig",     default="Rig",          help="Rig name for filename prefix")
    parser.add_argument("--dry-run", action="store_true",    help="Detect only, don't write files")
    args = parser.parse_args()

    pdf_path   = Path(args.input)
    output_dir = repo_root / args.outdir

    if not pdf_path.exists():
        print(f"Error: '{pdf_path}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning '{pdf_path.name}' for DDR boundaries...")
    starts = detect_ddr_starts(pdf_path)
    print(f"Detected {len(starts)} DDR(s):")
    for page_idx, report_no, date_str in starts:
        print(f"  Report No. {report_no:3d}  date={date_str}  first_page={page_idx + 1}")

    if args.dry_run or not starts:
        return

    print(f"\nSplitting into {output_dir}/...")
    split_combined_ddr(pdf_path, output_dir, args.well_id, args.rig)


if __name__ == "__main__":
    main()
