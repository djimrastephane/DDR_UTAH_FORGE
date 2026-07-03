from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from ddr_rag.filename_qc import audit_raw_pdfs, write_audit_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch preprocess raw DDR PDFs using QA/QC-derived safe document ids."
    )
    parser.add_argument("--raw-dir", default=str(repo_root / "data" / "raw"))
    parser.add_argument("--out-root", default=str(repo_root / "data" / "processed"))
    parser.add_argument("--qc-out-dir", default=str(repo_root / "data" / "processed" / "qc"))
    parser.add_argument("--block-id", default="FORGE")
    parser.add_argument("--table-chunking", default="row_blocks")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N valid PDFs; 0 means all.")
    parser.add_argument("--dry-run", action="store_true", help="Write QA/QC outputs but do not preprocess PDFs.")
    parser.add_argument("--no-resume", action="store_true", help="Reprocess PDFs even when chunks.parquet already exists.")
    parser.add_argument(
        "--no-dedupe-report-dates",
        action="store_true",
        help="Process every parsed PDF, including duplicate files for the same report date.",
    )
    parser.add_argument("--stop-on-error", action="store_true", help="Stop at the first preprocessing failure.")
    parser.add_argument("--skip-pdf-open-check", action="store_true")
    parser.add_argument("--strict-qc", action="store_true", help="Stop before preprocessing if QA/QC has blocking errors.")
    parser.add_argument(
        "--no-table-page-backup-text-chunks",
        action="store_true",
        help="Do not pass --table-page-backup-text-chunks to preprocessing.",
    )
    parser.add_argument(
        "--no-table-extract-return-all-tables",
        action="store_true",
        help="Do not pass --table-extract-return-all-tables to preprocessing.",
    )
    parser.add_argument("--build-index", action="store_true", help="Run scripts/build_index.py after preprocessing.")
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _preprocess_command(args: argparse.Namespace, source_path: str, safe_doc_id: str) -> list[str]:
    cmd = [
        sys.executable,
        "preprocess_hybrid.py",
        "--pdf-path",
        source_path,
        "--doc-id",
        safe_doc_id,
        "--out-root",
        str(args.out_root),
        "--table-chunking",
        str(args.table_chunking),
    ]
    if not args.no_table_page_backup_text_chunks:
        cmd.append("--table-page-backup-text-chunks")
    if not args.no_table_extract_return_all_tables:
        cmd.append("--table-extract-return-all-tables")
    return cmd


def _canonical_record_sort_key(record: Any) -> tuple[int, int, int, int, str]:
    warnings = set(getattr(record, "warnings", []) or [])
    name = str(getattr(record, "source_filename", ""))
    file_size = int(getattr(record, "file_size_bytes", 0) or 0)
    return (
        1 if "filename_copy_marker" in warnings else 0,
        1 if "[" in name else 0,
        1 if "filename_source_revision_marker" in warnings else 0,
        -file_size,
        name,
    )


def _dedupe_report_dates(records: list[Any]) -> tuple[list[Any], int]:
    grouped: dict[tuple[str, str, str], list[Any]] = {}
    passthrough: list[Any] = []
    for record in records:
        if getattr(record, "rig", "") == "UtahForge" and getattr(record, "report_date", None):
            key = (
                str(getattr(record, "rig", "")),
                str(getattr(record, "wellbore", "")),
                str(getattr(record, "report_date", "")),
            )
            grouped.setdefault(key, []).append(record)
        else:
            passthrough.append(record)

    deduped = passthrough + [sorted(group, key=_canonical_record_sort_key)[0] for group in grouped.values()]
    deduped.sort(key=lambda record: (str(getattr(record, "report_date", "")), str(getattr(record, "source_filename", ""))))
    skipped = sum(len(group) - 1 for group in grouped.values())
    return deduped, skipped


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root)
    qc_out_dir = Path(args.qc_out_dir)
    audit = audit_raw_pdfs(
        args.raw_dir,
        block_id=str(args.block_id or ""),
        check_pdf_readability=not bool(args.skip_pdf_open_check),
    )
    write_audit_outputs(audit, qc_out_dir)

    if args.strict_qc and audit.blocking_issue_count > 0:
        print(f"Blocking QA/QC issues found: {audit.blocking_issue_count}. See {qc_out_dir}.")
        return 1

    candidates = [
        record
        for record in audit.records
        if record.parsed and not record.pdf_read_error and record.safe_doc_id
    ]
    duplicate_candidates_skipped = 0
    if not args.no_dedupe_report_dates:
        candidates, duplicate_candidates_skipped = _dedupe_report_dates(candidates)
    if args.limit and args.limit > 0:
        candidates = candidates[: int(args.limit)]

    print("DDR batch preprocessing")
    print(f"  Candidates: {len(candidates)}")
    if duplicate_candidates_skipped:
        print(f"  Duplicate report-date PDFs skipped: {duplicate_candidates_skipped}")
    print(f"  QA/QC warnings: {audit.warning_count}")
    print(f"  QA/QC blocking issues: {audit.blocking_issue_count}")
    print(f"  QC output: {qc_out_dir}")

    if args.dry_run:
        print("  Dry run only; no PDFs were preprocessed.")
        return 0 if audit.blocking_issue_count == 0 else 1

    results: list[dict[str, Any]] = []
    for index, record in enumerate(candidates, start=1):
        doc_dir = out_root / record.safe_doc_id
        chunks_path = doc_dir / "chunks.parquet"
        if not args.no_resume and chunks_path.exists():
            status = "skipped_existing"
            print(f"[{index}/{len(candidates)}] {record.safe_doc_id}: {status}")
            results.append(
                {
                    "safe_doc_id": record.safe_doc_id,
                    "source_filename": record.source_filename,
                    "status": status,
                    "returncode": 0,
                }
            )
            continue

        print(f"[{index}/{len(candidates)}] {record.safe_doc_id}: preprocessing")
        started = time.perf_counter()
        cmd = _preprocess_command(args, record.source_path, record.safe_doc_id)
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            text=True,
            capture_output=True,
            check=False,
        )
        elapsed = round(time.perf_counter() - started, 3)
        doc_dir.mkdir(parents=True, exist_ok=True)
        (doc_dir / "batch_preprocess.log").write_text(
            "\n".join(
                [
                    "COMMAND",
                    " ".join(cmd),
                    "",
                    "STDOUT",
                    proc.stdout or "",
                    "",
                    "STDERR",
                    proc.stderr or "",
                ]
            ),
            encoding="utf-8",
        )

        status = "processed" if proc.returncode == 0 else "failed"
        metadata = record.to_manifest_row()
        metadata.update(
            {
                "batch_status": status,
                "batch_returncode": proc.returncode,
                "batch_elapsed_seconds": elapsed,
            }
        )
        _write_json(doc_dir / "document_metadata.json", metadata)
        print(f"[{index}/{len(candidates)}] {record.safe_doc_id}: {status} ({elapsed}s)")
        results.append(
            {
                "safe_doc_id": record.safe_doc_id,
                "source_filename": record.source_filename,
                "status": status,
                "returncode": proc.returncode,
                "elapsed_seconds": elapsed,
            }
        )
        if proc.returncode != 0 and args.stop_on_error:
            break

    summary = {
        "processed_count": sum(1 for row in results if row["status"] == "processed"),
        "skipped_existing_count": sum(1 for row in results if row["status"] == "skipped_existing"),
        "failed_count": sum(1 for row in results if row["status"] == "failed"),
        "results": results,
    }
    _write_json(qc_out_dir / "batch_preprocess_summary.json", summary)

    if args.build_index and summary["failed_count"] == 0:
        cmd = [sys.executable, "scripts/build_index.py", "--data-dir", str(out_root)]
        proc = subprocess.run(cmd, cwd=str(repo_root), text=True, check=False)
        if proc.returncode != 0:
            return proc.returncode

    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
