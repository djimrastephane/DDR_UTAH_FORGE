from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable


def _build_filename_re() -> re.Pattern:
    try:
        from ddr_rag.ddr_profile import load_profile
        profile = load_profile("operator_alpha")
        if profile.filename_re is not None:
            return profile.filename_re
    except Exception:
        pass
    # Hardcoded fallback — always matches the Operator Alpha format
    return re.compile(
        r"^(?P<rig>[A-Za-z0-9_-]+)\s+"
        r"(?P<report_type>DDR)\s+"
        r"(?P<ddr_number>\d+)\s*"
        r"(?P<asset_or_project>[A-Za-z][A-Za-z0-9_-]*)\s+"
        r"(?P<wellbore_suffix>RB|R\d+)\s+"
        r"(?P<report_date>\d{2}\.\d{2}\.\d{4})"
        r"(?P<trailing_dots>\.*)\.pdf$",
        re.IGNORECASE,
    )


DDR_FILENAME_RE = _build_filename_re()
UTAH_FORGE_FILENAME_RE = re.compile(
    r"^Utah_Forge_"
    r"(?P<well_name>FORGE_16A_[\[(](?P<pad>\d+)[\])]-(?P<slot>\d+))_"
    r"(?P<operation_phase>[A-Za-z]+)-C_"
    r"(?P<date_blob>\d{6,16})"
    r"(?:_(?P<date_blob_2>\d{6,8}))?_"
    r"(?P<source_report_no>\d+)"
    r"(?P<source_revision>_\d+)?_"
    r"(?P<tmp_kind>reporttmp|tmp)"
    r"(?P<copy_marker> 2)?\.pdf$",
    re.IGNORECASE,
)
SAFE_DOC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")


@dataclass
class DDRFilenameRecord:
    source_path: str
    source_filename: str
    parsed: bool
    rig: str = ""
    report_type: str = "DDR"
    ddr_number: int | None = None
    asset_or_project: str = ""
    wellbore_suffix: str = ""
    wellbore: str = ""
    block_wellbore: str = ""
    report_date: date | None = None
    date_text: str = ""
    safe_doc_id: str = ""
    parse_error: str = ""
    warnings: list[str] = field(default_factory=list)
    file_size_bytes: int | None = None
    page_count: int | None = None
    pdf_read_error: str = ""

    @property
    def report_date_iso(self) -> str:
        return self.report_date.isoformat() if self.report_date is not None else ""

    def to_manifest_row(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_filename": self.source_filename,
            "parsed": self.parsed,
            "safe_doc_id": self.safe_doc_id,
            "rig": self.rig,
            "report_type": self.report_type,
            "ddr_number": self.ddr_number,
            "asset_or_project": self.asset_or_project,
            "wellbore_suffix": self.wellbore_suffix,
            "wellbore": self.wellbore,
            "block_wellbore": self.block_wellbore,
            "report_date": self.report_date_iso,
            "date_text": self.date_text,
            "file_size_bytes": self.file_size_bytes,
            "page_count": self.page_count,
            "warnings": ";".join(self.warnings),
            "parse_error": self.parse_error,
            "pdf_read_error": self.pdf_read_error,
        }


@dataclass(frozen=True)
class DDRQCIssue:
    severity: str
    issue_type: str
    message: str
    source_filename: str = ""
    safe_doc_id: str = ""
    ddr_number: int | None = None
    report_date: str = ""
    wellbore: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "issue_type": self.issue_type,
            "message": self.message,
            "source_filename": self.source_filename,
            "safe_doc_id": self.safe_doc_id,
            "ddr_number": self.ddr_number,
            "report_date": self.report_date,
            "wellbore": self.wellbore,
        }


@dataclass
class DDRFilenameAudit:
    raw_dir: Path
    records: list[DDRFilenameRecord]
    issues: list[DDRQCIssue]
    missing_reports: list[dict[str, Any]]

    @property
    def blocking_issue_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    def summary(self) -> dict[str, Any]:
        parsed = [record for record in self.records if record.parsed]
        dates = [record.report_date for record in parsed if record.report_date is not None]
        ddr_numbers = [record.ddr_number for record in parsed if record.ddr_number is not None]
        wellbore_counts = Counter(record.wellbore for record in parsed if record.wellbore)
        suffix_counts = Counter(record.wellbore_suffix for record in parsed if record.wellbore_suffix)
        return {
            "raw_dir": str(self.raw_dir),
            "pdf_count": len(self.records),
            "parsed_count": len(parsed),
            "unparsed_count": len(self.records) - len(parsed),
            "blocking_issue_count": self.blocking_issue_count,
            "warning_count": self.warning_count,
            "missing_report_count": len(self.missing_reports),
            "date_range_start": min(dates).isoformat() if dates else "",
            "date_range_end": max(dates).isoformat() if dates else "",
            "ddr_number_min": min(ddr_numbers) if ddr_numbers else None,
            "ddr_number_max": max(ddr_numbers) if ddr_numbers else None,
            "wellbore_counts": dict(sorted(wellbore_counts.items())),
            "wellbore_suffix_counts": dict(sorted(suffix_counts.items())),
            "issue_type_counts": dict(sorted(Counter(issue.issue_type for issue in self.issues).items())),
        }


def _slug_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "-", str(value or "")).strip("-")
    return token or "UNKNOWN"


def _parse_compact_mmddyyyy(token: str) -> date:
    if not token.isdigit() or len(token) < 6:
        raise ValueError(f"Invalid compact date token: {token}")

    year = int(token[-4:])
    prefix = token[:-4]
    candidates: list[date] = []
    for split in range(1, len(prefix)):
        month_text = prefix[:split]
        day_text = prefix[split:]
        try:
            candidates.append(date(year, int(month_text), int(day_text)))
        except ValueError:
            continue

    if not candidates:
        raise ValueError(f"Invalid compact date token: {token}")

    if year == 2020:
        for candidate in candidates:
            if candidate.month in {10, 11, 12}:
                return candidate
    if year == 2021:
        for candidate in candidates:
            if candidate.month == 1:
                return candidate

    return candidates[0]


def _parse_utah_forge_date(date_blob: str, date_blob_2: str | None) -> tuple[date, list[str]]:
    warnings: list[str] = []
    first_token = date_blob
    second_token = date_blob_2 or ""

    if not second_token and len(date_blob) % 2 == 0:
        half = len(date_blob) // 2
        left, right = date_blob[:half], date_blob[half:]
        if left == right:
            first_token = left
            second_token = right

    report_date = _parse_compact_mmddyyyy(first_token)
    if second_token:
        end_date = _parse_compact_mmddyyyy(second_token)
        if end_date != report_date:
            warnings.append("filename_start_end_date_mismatch")

    return report_date, warnings


def _parse_utah_forge_filename(pdf_path: Path) -> DDRFilenameRecord | None:
    filename = pdf_path.name
    match = UTAH_FORGE_FILENAME_RE.fullmatch(filename)
    if not match:
        return None

    groups = match.groupdict()
    try:
        report_date, warnings = _parse_utah_forge_date(groups["date_blob"], groups.get("date_blob_2"))
    except ValueError as exc:
        return DDRFilenameRecord(
            source_path=str(pdf_path),
            source_filename=filename,
            parsed=False,
            parse_error=f"invalid_utah_forge_date:{exc}",
        )

    source_hash = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:8]
    operation_phase = groups["operation_phase"].capitalize()
    wellbore = "FORGE-16A-78-32"
    report_no = int(groups["source_report_no"])
    safe_doc_id = (
        f"UtahForge-DDR-{wellbore}-{operation_phase}-"
        f"{report_date.isoformat()}-R{report_no:03d}-{source_hash}"
    )
    if not SAFE_DOC_ID_RE.fullmatch(safe_doc_id):
        raise ValueError(f"Generated unsafe doc_id: {safe_doc_id}")

    if groups.get("copy_marker"):
        warnings.append("filename_copy_marker")
    if groups.get("source_revision"):
        warnings.append("filename_source_revision_marker")

    campaign_day = (report_date - date(2020, 10, 21)).days + 1

    return DDRFilenameRecord(
        source_path=str(pdf_path),
        source_filename=filename,
        parsed=True,
        rig="UtahForge",
        report_type="DDR",
        ddr_number=campaign_day,
        asset_or_project="FORGE",
        wellbore_suffix="16A-78-32",
        wellbore=wellbore,
        block_wellbore=wellbore,
        report_date=report_date,
        date_text=groups["date_blob_2"]
        and f"{groups['date_blob']}_{groups['date_blob_2']}"
        or groups["date_blob"],
        safe_doc_id=safe_doc_id,
        warnings=warnings,
    )


def build_safe_doc_id(
    *,
    rig: str,
    ddr_number: int,
    asset_or_project: str,
    wellbore_suffix: str,
    report_date: date,
) -> str:
    doc_id = (
        f"{_slug_token(rig)}-DDR-{int(ddr_number):03d}-"
        f"{_slug_token(asset_or_project).upper()}-"
        f"{_slug_token(wellbore_suffix).upper()}-"
        f"{report_date.isoformat()}"
    )
    if not SAFE_DOC_ID_RE.fullmatch(doc_id):
        raise ValueError(f"Generated unsafe doc_id: {doc_id}")
    return doc_id


def parse_ddr_filename(path: str | Path, block_id: str = "30/07a") -> DDRFilenameRecord:
    pdf_path = Path(path)
    filename = pdf_path.name
    if pdf_path.suffix.lower() != ".pdf":
        return DDRFilenameRecord(
            source_path=str(pdf_path),
            source_filename=filename,
            parsed=False,
            parse_error="unsupported_file_extension",
        )

    utah_record = _parse_utah_forge_filename(pdf_path)
    if utah_record is not None:
        return utah_record

    match = DDR_FILENAME_RE.fullmatch(filename)
    if not match:
        return DDRFilenameRecord(
            source_path=str(pdf_path),
            source_filename=filename,
            parsed=False,
            parse_error="filename_does_not_match_expected_ddr_pattern",
        )

    groups = match.groupdict()
    report_date = datetime.strptime(groups["report_date"], "%d.%m.%Y").date()
    ddr_number = int(groups["ddr_number"])
    rig = groups["rig"]
    report_type = groups["report_type"].upper()
    asset_or_project = groups["asset_or_project"].upper()
    wellbore_suffix = groups["wellbore_suffix"].upper()
    wellbore = f"{asset_or_project}-{wellbore_suffix}"
    block_wellbore = f"{block_id}-{wellbore_suffix}" if block_id else ""
    warnings: list[str] = []

    if groups.get("trailing_dots"):
        warnings.append("extra_dot_before_pdf_extension")
    if re.search(r"\bDDR\s+\d+[A-Za-z]", filename):
        warnings.append("missing_space_after_ddr_number")

    safe_doc_id = build_safe_doc_id(
        rig=rig,
        ddr_number=ddr_number,
        asset_or_project=asset_or_project,
        wellbore_suffix=wellbore_suffix,
        report_date=report_date,
    )

    return DDRFilenameRecord(
        source_path=str(pdf_path),
        source_filename=filename,
        parsed=True,
        rig=rig,
        report_type=report_type,
        ddr_number=ddr_number,
        asset_or_project=asset_or_project,
        wellbore_suffix=wellbore_suffix,
        wellbore=wellbore,
        block_wellbore=block_wellbore,
        report_date=report_date,
        date_text=groups["report_date"],
        safe_doc_id=safe_doc_id,
        warnings=warnings,
    )


def _attach_pdf_file_quality(record: DDRFilenameRecord) -> None:
    path = Path(record.source_path)
    try:
        record.file_size_bytes = path.stat().st_size
        if record.file_size_bytes <= 0:
            record.pdf_read_error = "zero_byte_file"
            return
    except OSError as exc:
        record.pdf_read_error = f"stat_failed:{type(exc).__name__}:{exc}"
        return

    try:
        import pymupdf as fitz

        with fitz.open(path) as doc:
            record.page_count = int(doc.page_count)
            if record.page_count <= 0:
                record.pdf_read_error = "zero_page_pdf"
    except Exception as exc:
        record.pdf_read_error = f"pdf_open_failed:{type(exc).__name__}:{exc}"


def _duplicate_issues(
    records: Iterable[DDRFilenameRecord],
    *,
    field_name: str,
    issue_type: str,
    severity: str = "error",
) -> list[DDRQCIssue]:
    groups: dict[Any, list[DDRFilenameRecord]] = defaultdict(list)
    for record in records:
        groups[getattr(record, field_name)].append(record)

    issues: list[DDRQCIssue] = []
    for key, group in groups.items():
        if key in {"", None} or len(group) <= 1:
            continue
        filenames = ", ".join(record.source_filename for record in group)
        for record in group:
            issues.append(
                DDRQCIssue(
                    severity=severity,
                    issue_type=issue_type,
                    message=f"Duplicate {field_name}={key}: {filenames}",
                    source_filename=record.source_filename,
                    safe_doc_id=record.safe_doc_id,
                    ddr_number=record.ddr_number,
                    report_date=record.report_date_iso,
                    wellbore=record.wellbore,
                )
            )
    return issues


def _nearest_record(
    by_number: dict[int, DDRFilenameRecord],
    target_number: int,
    step: int,
    lower_bound: int,
    upper_bound: int,
) -> DDRFilenameRecord | None:
    number = target_number + step
    while lower_bound <= number <= upper_bound:
        if number in by_number:
            return by_number[number]
        number += step
    return None


def _missing_reports(parsed: list[DDRFilenameRecord]) -> tuple[list[dict[str, Any]], list[DDRQCIssue]]:
    by_number = {
        int(record.ddr_number): record
        for record in parsed
        if record.ddr_number is not None and record.report_date is not None
    }
    if not by_number:
        return [], []

    min_number = min(by_number)
    max_number = max(by_number)
    first_record = by_number[min_number]
    assert first_record.report_date is not None

    missing: list[dict[str, Any]] = []
    issues: list[DDRQCIssue] = []
    for number in range(min_number, max_number + 1):
        expected_date = first_record.report_date + timedelta(days=number - min_number)
        existing = by_number.get(number)
        if existing is not None:
            if existing.report_date != expected_date:
                issues.append(
                    DDRQCIssue(
                        severity="error",
                        issue_type="date_sequence_mismatch",
                        message=(
                            f"DDR {number} has date {existing.report_date_iso}; "
                            f"expected {expected_date.isoformat()} from sequence."
                        ),
                        source_filename=existing.source_filename,
                        safe_doc_id=existing.safe_doc_id,
                        ddr_number=number,
                        report_date=existing.report_date_iso,
                        wellbore=existing.wellbore,
                    )
                )
            continue

        previous_record = _nearest_record(by_number, number, -1, min_number, max_number)
        next_record = _nearest_record(by_number, number, 1, min_number, max_number)
        template = previous_record or next_record or first_record
        inferred_filename = (
            f"{template.rig} DDR {number} {template.asset_or_project} "
            f"{template.wellbore_suffix} {expected_date.strftime('%d.%m.%Y')}.pdf"
        )
        row = {
            "severity": "warning",
            "issue_type": "missing_report",
            "expected_ddr_number": number,
            "expected_report_date": expected_date.isoformat(),
            "rig": template.rig,
            "asset_or_project": template.asset_or_project,
            "wellbore_suffix": template.wellbore_suffix,
            "wellbore": template.wellbore,
            "block_wellbore": template.block_wellbore,
            "inferred_filename": inferred_filename,
            "previous_source_filename": previous_record.source_filename if previous_record else "",
            "next_source_filename": next_record.source_filename if next_record else "",
        }
        missing.append(row)
        issues.append(
            DDRQCIssue(
                severity="warning",
                issue_type="missing_report",
                message=f"Missing DDR {number} for {expected_date.isoformat()} ({template.wellbore}).",
                ddr_number=number,
                report_date=expected_date.isoformat(),
                wellbore=template.wellbore,
            )
        )
    return missing, issues


def audit_raw_pdfs(
    raw_dir: str | Path,
    *,
    block_id: str = "30/07a",
    check_pdf_readability: bool = True,
) -> DDRFilenameAudit:
    raw_path = Path(raw_dir)
    records = [
        parse_ddr_filename(path, block_id=block_id)
        for path in sorted(raw_path.glob("*.pdf"), key=lambda item: item.name)
    ]

    if check_pdf_readability:
        for record in records:
            _attach_pdf_file_quality(record)

    issues: list[DDRQCIssue] = []
    for record in records:
        if not record.parsed:
            issues.append(
                DDRQCIssue(
                    severity="error",
                    issue_type="unparsed_filename",
                    message=record.parse_error,
                    source_filename=record.source_filename,
                )
            )
        for warning in record.warnings:
            issues.append(
                DDRQCIssue(
                    severity="warning",
                    issue_type=warning,
                    message=warning.replace("_", " "),
                    source_filename=record.source_filename,
                    safe_doc_id=record.safe_doc_id,
                    ddr_number=record.ddr_number,
                    report_date=record.report_date_iso,
                    wellbore=record.wellbore,
                )
            )
        if record.pdf_read_error:
            issues.append(
                DDRQCIssue(
                    severity="error",
                    issue_type="pdf_file_quality",
                    message=record.pdf_read_error,
                    source_filename=record.source_filename,
                    safe_doc_id=record.safe_doc_id,
                    ddr_number=record.ddr_number,
                    report_date=record.report_date_iso,
                    wellbore=record.wellbore,
                )
            )

    parsed = [record for record in records if record.parsed]
    sequence_unique_records = [record for record in parsed if record.rig != "UtahForge"]
    sequence_duplicate_records = [record for record in parsed if record.rig == "UtahForge"]
    issues.extend(
        _duplicate_issues(
            sequence_unique_records,
            field_name="ddr_number",
            issue_type="duplicate_ddr_number",
        )
    )
    issues.extend(
        _duplicate_issues(
            sequence_unique_records,
            field_name="report_date",
            issue_type="duplicate_report_date",
        )
    )
    issues.extend(
        _duplicate_issues(
            sequence_duplicate_records,
            field_name="ddr_number",
            issue_type="duplicate_ddr_number",
            severity="warning",
        )
    )
    issues.extend(
        _duplicate_issues(
            sequence_duplicate_records,
            field_name="report_date",
            issue_type="duplicate_report_date",
            severity="warning",
        )
    )
    issues.extend(_duplicate_issues(parsed, field_name="safe_doc_id", issue_type="duplicate_safe_doc_id"))
    missing, missing_issues = _missing_reports(parsed)
    issues.extend(missing_issues)
    return DDRFilenameAudit(raw_dir=raw_path, records=records, issues=issues, missing_reports=missing)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_audit_outputs(audit: DDRFilenameAudit, out_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "raw_pdf_manifest.csv"
    issues_path = output_dir / "raw_pdf_issues.csv"
    missing_path = output_dir / "raw_pdf_missing_reports.csv"
    summary_path = output_dir / "raw_pdf_qc_summary.json"

    manifest_rows = [record.to_manifest_row() for record in audit.records]
    issue_rows = [issue.to_dict() for issue in audit.issues]
    missing_rows = list(audit.missing_reports)

    _write_csv(
        manifest_path,
        manifest_rows,
        [
            "source_path",
            "source_filename",
            "parsed",
            "safe_doc_id",
            "rig",
            "report_type",
            "ddr_number",
            "asset_or_project",
            "wellbore_suffix",
            "wellbore",
            "block_wellbore",
            "report_date",
            "date_text",
            "file_size_bytes",
            "page_count",
            "warnings",
            "parse_error",
            "pdf_read_error",
        ],
    )
    _write_csv(
        issues_path,
        issue_rows,
        [
            "severity",
            "issue_type",
            "message",
            "source_filename",
            "safe_doc_id",
            "ddr_number",
            "report_date",
            "wellbore",
        ],
    )
    _write_csv(
        missing_path,
        missing_rows,
        [
            "severity",
            "issue_type",
            "expected_ddr_number",
            "expected_report_date",
            "rig",
            "asset_or_project",
            "wellbore_suffix",
            "wellbore",
            "block_wellbore",
            "inferred_filename",
            "previous_source_filename",
            "next_source_filename",
        ],
    )
    summary_path.write_text(json.dumps(audit.summary(), indent=2), encoding="utf-8")

    return {
        "manifest": manifest_path,
        "issues": issues_path,
        "missing_reports": missing_path,
        "summary": summary_path,
    }
