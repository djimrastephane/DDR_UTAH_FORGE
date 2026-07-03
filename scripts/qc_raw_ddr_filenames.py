from __future__ import annotations

import argparse
import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from ddr_rag.filename_qc import audit_raw_pdfs, write_audit_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QA/QC raw DDR PDF filenames and generate safe document ids."
    )
    parser.add_argument(
        "--raw-dir",
        default=str(repo_root / "data" / "raw"),
        help="Directory containing source DDR PDFs.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(repo_root / "data" / "processed" / "qc"),
        help="Directory where QA/QC CSV and JSON outputs are written.",
    )
    parser.add_argument(
        "--block-id",
        default="FORGE",
        help="Optional project/block identifier used to derive block_wellbore values.",
    )
    parser.add_argument(
        "--skip-pdf-open-check",
        action="store_true",
        help="Skip PyMuPDF readability and page-count checks.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero for blocking errors. Missing DDRs remain warnings.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = audit_raw_pdfs(
        args.raw_dir,
        block_id=str(args.block_id or ""),
        check_pdf_readability=not bool(args.skip_pdf_open_check),
    )
    outputs = write_audit_outputs(audit, args.out_dir)
    summary = audit.summary()

    print("DDR raw filename QA/QC")
    print(f"  Raw directory: {summary['raw_dir']}")
    print(f"  PDFs found: {summary['pdf_count']}")
    print(f"  Parsed filenames: {summary['parsed_count']}")
    print(f"  Blocking issues: {summary['blocking_issue_count']}")
    print(f"  Warnings: {summary['warning_count']}")
    print(f"  Missing reports flagged: {summary['missing_report_count']}")
    print(f"  Date range: {summary['date_range_start']} to {summary['date_range_end']}")
    print(f"  DDR range: {summary['ddr_number_min']} to {summary['ddr_number_max']}")
    print("")
    print("Outputs")
    for label, path in outputs.items():
        print(f"  {label}: {path}")

    if args.strict and audit.blocking_issue_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
