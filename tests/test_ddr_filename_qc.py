from __future__ import annotations

from datetime import date
from pathlib import Path

from ddr_rag.filename_qc import audit_raw_pdfs, build_safe_doc_id, parse_ddr_filename


def test_parse_ddr_filename_builds_wellbore_metadata_and_safe_doc_id() -> None:
    record = parse_ddr_filename("data/raw/RigAlpha DDR 7 FLDX R2 18.04.2024.pdf")

    assert record.parsed
    assert record.rig == "RigAlpha"
    assert record.ddr_number == 7
    assert record.asset_or_project == "FLDX"
    assert record.wellbore_suffix == "R2"
    assert record.wellbore == "FLDX-R2"
    assert record.block_wellbore == "00/00a-R2"
    assert record.report_date == date(2024, 4, 18)
    assert record.safe_doc_id == "RigAlpha-DDR-007-FLDX-R2-2024-04-18"


def test_parse_ddr_filename_flags_formatting_warnings_without_failing() -> None:
    record = parse_ddr_filename("data/raw/RigAlpha DDR 107FLDX R2 27.07.2024..pdf")

    assert record.parsed
    assert record.ddr_number == 107
    assert record.safe_doc_id == "RigAlpha-DDR-107-FLDX-R2-2024-07-27"
    assert "missing_space_after_ddr_number" in record.warnings
    assert "extra_dot_before_pdf_extension" in record.warnings


def test_parse_utah_forge_filename_builds_stable_unique_doc_id() -> None:
    record = parse_ddr_filename(
        "data/raw/Utah_Forge_FORGE_16A_(78)-32_Drilling-C_01032021_01032021_15_1_reporttmp 2.pdf"
    )

    assert record.parsed
    assert record.rig == "UtahForge"
    assert record.ddr_number == 75
    assert record.asset_or_project == "FORGE"
    assert record.wellbore == "FORGE-16A-78-32"
    assert record.report_date == date(2021, 1, 3)
    assert record.safe_doc_id.startswith(
        "UtahForge-DDR-FORGE-16A-78-32-Drilling-2021-01-03-R015-"
    )
    assert "filename_copy_marker" in record.warnings
    assert "filename_source_revision_marker" in record.warnings


def test_parse_utah_forge_compact_unpadded_date() -> None:
    record = parse_ddr_filename(
        "data/raw/Utah_Forge_FORGE_16A_[78]-32_Drilling-C_11920201192020_11_reporttmp.pdf"
    )

    assert record.parsed
    assert record.report_date == date(2020, 11, 9)
    assert record.ddr_number == 20


def test_audit_allows_utah_forge_same_day_copies_without_blocking(tmp_path: Path) -> None:
    names = [
        "Utah_Forge_FORGE_16A_(78)-32_Drilling-C_01012021_01012021_15_reporttmp.pdf",
        "Utah_Forge_FORGE_16A_(78)-32_Drilling-C_01012021_01012021_15_reporttmp 2.pdf",
    ]
    for name in names:
        (tmp_path / name).write_bytes(b"not checked")

    audit = audit_raw_pdfs(tmp_path, check_pdf_readability=False)

    assert audit.blocking_issue_count == 0
    assert audit.summary()["parsed_count"] == 2
    assert len({record.safe_doc_id for record in audit.records}) == 2
    assert any(issue.issue_type == "duplicate_report_date" for issue in audit.issues)


def test_safe_doc_id_uses_three_digit_ddr_number() -> None:
    doc_id = build_safe_doc_id(
        rig="RigAlpha",
        ddr_number=1,
        asset_or_project="FLDX",
        wellbore_suffix="RB",
        report_date=date(2024, 4, 12),
    )

    assert doc_id == "RigAlpha-DDR-001-FLDX-RB-2024-04-12"


def test_audit_flags_missing_reports_as_warnings(tmp_path: Path) -> None:
    (tmp_path / "RigAlpha DDR 1 FLDX RB 12.04.2024.pdf").write_bytes(b"not checked")
    (tmp_path / "RigAlpha DDR 3 FLDX RB 14.04.2024.pdf").write_bytes(b"not checked")

    audit = audit_raw_pdfs(tmp_path, check_pdf_readability=False)

    assert audit.blocking_issue_count == 0
    assert audit.missing_reports == [
        {
            "severity": "warning",
            "issue_type": "missing_report",
            "expected_ddr_number": 2,
            "expected_report_date": "2024-04-13",
            "rig": "RigAlpha",
            "asset_or_project": "FLDX",
            "wellbore_suffix": "RB",
            "wellbore": "FLDX-RB",
            "block_wellbore": "00/00a-RB",
            "inferred_filename": "RigAlpha DDR 2 FLDX RB 13.04.2024.pdf",
            "previous_source_filename": "RigAlpha DDR 1 FLDX RB 12.04.2024.pdf",
            "next_source_filename": "RigAlpha DDR 3 FLDX RB 14.04.2024.pdf",
        }
    ]
    assert any(issue.issue_type == "missing_report" for issue in audit.issues)
