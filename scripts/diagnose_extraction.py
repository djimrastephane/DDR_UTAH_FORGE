from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore")

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

import pymupdf  # noqa: E402

from ddr_rag.filename_qc import parse_ddr_filename  # noqa: E402
from ddr_rag.ddr_extractor import (  # noqa: E402
    extract_header_fields,
    extract_op_summary,
)


@dataclass
class DocDiagnostic:
    source_filename: str
    safe_doc_id: str
    page_count: int
    header: dict = field(default_factory=dict)
    header_fields_found: int = 0
    op_summary_rows: list[dict] = field(default_factory=list)
    op_summary_hours_total: float = 0.0
    op_summary_hours_gap: float = 0.0
    op_summary_pt_x_dist: dict[str, int] = field(default_factory=dict)
    op_summary_phases: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def _validate_header(fields: dict) -> tuple[int, list[str]]:
    found = sum(1 for v in fields.values() if v)
    issues: list[str] = []
    for required in ("report_date", "report_no", "wellbore", "daily_cost", "rig_name"):
        if not fields.get(required):
            issues.append(f"{required} not found")
    return found, issues


def _validate_ops(rows: list[dict]) -> tuple[float, float, list[str]]:
    # Valid range is 20–31h (24h day + optional 6h night carry-over)
    total = sum(r["duration_hr"] for r in rows if r.get("duration_hr") is not None)
    gap = abs(24.0 - total)
    issues: list[str] = []

    if not rows:
        issues.append("no operation summary rows extracted")
        return 0.0, 24.0, issues

    if total < 20 or total > 31:
        issues.append(f"hours sum to {total:.2f} — outside expected 20–31h range")

    invalid_ptx = {r["pt_x"] for r in rows if r.get("pt_x")} - {"P", "T", "X", "N", ""}
    if invalid_ptx:
        issues.append(f"unexpected P-T-X values: {sorted(invalid_ptx)}")

    missing_phase = [r for r in rows if not r.get("phase")]
    if missing_phase:
        issues.append(f"{len(missing_phase)} rows missing phase")

    return round(total, 2), round(gap, 2), issues


def select_sample_pdfs(raw_dir: Path, n: int) -> list[Path]:
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        return []
    parsed = []
    for p in pdfs:
        record = parse_ddr_filename(p)
        sort_key = record.report_date_iso if record.parsed else p.name
        parsed.append((sort_key, p))
    parsed.sort(key=lambda x: x[0])
    if len(parsed) <= n:
        return [p for _, p in parsed]
    step = len(parsed) / n
    return [parsed[int(i * step)][1] for i in range(n)]


def run_diagnostic(pdf_path: Path) -> DocDiagnostic:
    doc = pymupdf.open(str(pdf_path))
    page_count = len(doc)
    doc.close()

    record = parse_ddr_filename(pdf_path)
    safe_doc_id = record.safe_doc_id if record.parsed else ""

    diag = DocDiagnostic(
        source_filename=pdf_path.name,
        safe_doc_id=safe_doc_id,
        page_count=page_count,
    )

    try:
        fields = extract_header_fields(pdf_path)
        diag.header = fields
        found, h_issues = _validate_header(fields)
        diag.header_fields_found = found
        diag.issues.extend(h_issues)
    except Exception as exc:
        diag.issues.append(f"header extraction failed: {exc}")

    try:
        rows = extract_op_summary(pdf_path)
        diag.op_summary_rows = rows
        total, gap, v_issues = _validate_ops(rows)
        diag.op_summary_hours_total = total
        diag.op_summary_hours_gap = gap
        diag.issues.extend(v_issues)
        pt_x_dist: dict[str, int] = {}
        phases: list[str] = []
        for r in rows:
            k = r.get("pt_x", "")
            pt_x_dist[k] = pt_x_dist.get(k, 0) + 1
            ph = r.get("phase", "")
            if ph and ph not in phases:
                phases.append(ph)
        diag.op_summary_pt_x_dist = pt_x_dist
        diag.op_summary_phases = phases
    except Exception as exc:
        diag.issues.append(f"op summary extraction failed: {exc}")

    return diag


def print_summary(diagnostics: list[DocDiagnostic]) -> None:
    col_w = [40, 5, 8, 4, 8, 6, 6, 30]
    headers = ["source_filename", "pages", "hdr_fld", "ops", "hrs_tot", "gap", "issues", "phases"]
    sep = "  ".join("-" * w for w in col_w)
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)

    print()
    print(fmt.format(*headers))
    print(sep)
    for d in diagnostics:
        print(fmt.format(
            d.source_filename[:col_w[0]],
            d.page_count,
            d.header_fields_found,
            len(d.op_summary_rows),
            f"{d.op_summary_hours_total:.2f}",
            f"{d.op_summary_hours_gap:.2f}",
            len(d.issues),
            ", ".join(d.op_summary_phases)[:col_w[7]],
        ))
    print()

    any_issues = [d for d in diagnostics if d.issues]
    if any_issues:
        print("=== Issues ===")
        for d in any_issues:
            print(f"  {d.source_filename}:")
            for iss in d.issues:
                print(f"    - {iss}")
        print()

    if diagnostics and diagnostics[0].op_summary_rows:
        d = diagnostics[0]
        print(f"=== Op Summary sample: {d.source_filename} ({len(d.op_summary_rows)} rows) ===")
        print(f"  {'start':5}  {'end':5}  {'dur':5}  {'phase':6}  {'op_code':6}  {'ptx':3}  operation_text")
        print(f"  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*3}  {'-'*50}")
        for r in d.op_summary_rows[:10]:
            print(
                f"  {r.get('start_time',''):5}  {r.get('end_time',''):5}  "
                f"{r.get('duration_hr') or '':5}  {r.get('phase',''):6}  "
                f"{r.get('op_code',''):6}  {r.get('pt_x',''):3}  "
                f"{r.get('operation_text','')[:80]}"
            )
        print()

    if diagnostics and diagnostics[0].header:
        d = diagnostics[0]
        h = d.header
        print(f"=== Header sample: {d.source_filename} ===")
        for fname in [
            "report_date", "report_no", "wellbore", "job_start", "spud_date",
            "rig_name", "field_name", "end_depth_md_ft", "daily_cost",
            "cumulative_cost", "water_depth_ft", "avg_rop", "last_casing_string",
            "morning_report_ops",
        ]:
            val = str(h.get(fname, ""))
            status = "OK" if val else "MISSING"
            print(f"  [{status:7}] {fname:<25} {val[:80]}")
        print()


def write_outputs(diagnostics: list[DocDiagnostic], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for d in diagnostics:
        out_path = out_dir / f"{d.safe_doc_id or d.source_filename}.json"
        out_path.write_text(json.dumps({
            "source_filename": d.source_filename,
            "safe_doc_id": d.safe_doc_id,
            "page_count": d.page_count,
            "header": d.header,
            "header_fields_found": d.header_fields_found,
            "op_summary_rows": d.op_summary_rows,
            "op_summary_hours_total": d.op_summary_hours_total,
            "op_summary_hours_gap": d.op_summary_hours_gap,
            "op_summary_pt_x_dist": d.op_summary_pt_x_dist,
            "op_summary_phases": d.op_summary_phases,
            "issues": d.issues,
        }, indent=2, default=str))

    csv_path = out_dir / "extraction_diagnostics_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_filename", "safe_doc_id", "page_count",
            "header_fields_found", "op_summary_rows", "op_hours_total",
            "op_hours_gap", "n_issues", "phases", "pt_x_dist", "issues",
        ])
        for d in diagnostics:
            writer.writerow([
                d.source_filename, d.safe_doc_id, d.page_count,
                d.header_fields_found, len(d.op_summary_rows),
                d.op_summary_hours_total, d.op_summary_hours_gap,
                len(d.issues),
                "|".join(d.op_summary_phases),
                json.dumps(d.op_summary_pt_x_dist),
                " | ".join(d.issues),
            ])

    ops_csv_path = out_dir / "op_summary_rows_sample.csv"
    with ops_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_filename", "safe_doc_id", "page", "shift_block",
            "start_time", "end_time", "duration_hr",
            "phase", "op_code", "activity_code", "pt_x", "is_npt",
            "operation_text", "parse_warning",
        ])
        for d in diagnostics:
            for r in d.op_summary_rows:
                writer.writerow([
                    d.source_filename, d.safe_doc_id,
                    r.get("page"), r.get("shift_block"),
                    r.get("start_time"), r.get("end_time"), r.get("duration_hr"),
                    r.get("phase"), r.get("op_code"), r.get("activity_code"),
                    r.get("pt_x"), r.get("is_npt"),
                    r.get("operation_text"), r.get("parse_warning"),
                ])

    print(f"Outputs written to: {out_dir}")
    print(f"  {csv_path.name}  ({len(diagnostics)} docs)")
    print(f"  {ops_csv_path.name}")
    print(f"  {len(diagnostics)} per-doc JSON files")


def main() -> None:
    parser = argparse.ArgumentParser(description="DDR extraction diagnostics")
    parser.add_argument("--raw-dir", default=str(repo_root / "data" / "raw"))
    parser.add_argument("--out-dir", default=str(repo_root / "data" / "processed" / "qc" / "extraction_diagnostics"))
    parser.add_argument("--sample", type=int, default=15, help="Number of DDRs to sample (default 15)")
    parser.add_argument("--pdf", default="", help="Diagnose a single specific PDF instead of sampling")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if args.pdf:
        pdfs = [Path(args.pdf)]
    else:
        pdfs = select_sample_pdfs(raw_dir, args.sample)
        if not pdfs:
            print(f"No PDFs found in {raw_dir}")
            sys.exit(1)
        print(f"Sampled {len(pdfs)} PDFs from {raw_dir}")

    diagnostics: list[DocDiagnostic] = []
    for i, pdf_path in enumerate(pdfs, 1):
        print(f"  [{i:2d}/{len(pdfs)}] {pdf_path.name} ...", end=" ", flush=True)
        try:
            diag = run_diagnostic(pdf_path)
            diagnostics.append(diag)
            status = "OK" if not diag.issues else f"{len(diag.issues)} issues"
            print(f"{status} | hdr={diag.header_fields_found} ops={len(diag.op_summary_rows)} hrs={diag.op_summary_hours_total:.1f}")
        except Exception as exc:
            print(f"FAILED: {exc}")

    print_summary(diagnostics)
    write_outputs(diagnostics, out_dir)


if __name__ == "__main__":
    main()
