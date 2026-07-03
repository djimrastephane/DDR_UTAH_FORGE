from __future__ import annotations

from pathlib import Path

import pytest

from ddr_rag.ddr_extractor import extract_header_fields, extract_op_summary, run_ddr_extraction


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
EARLY_DRILLING_SAMPLE = RAW_DIR / "Utah_Forge_FORGE_16A_[78]-32_Drilling-C_1021202010212020_92_reporttmp.pdf"
DRILLING_SAMPLE = RAW_DIR / "Utah_Forge_FORGE_16A_(78)-32_Drilling-C_12012020_12012020_18_reporttmp.pdf"
COMPLETION_SAMPLE = RAW_DIR / "Utah_Forge_FORGE_16A_(78)-32_Completion-C_01062021_01062021_1_tmp.pdf"


pytestmark = pytest.mark.skipif(
    not DRILLING_SAMPLE.exists() or not COMPLETION_SAMPLE.exists(),
    reason="Utah FORGE raw sample PDFs are not present.",
)


def test_utah_forge_drilling_header_and_time_rows() -> None:
    header = extract_header_fields(DRILLING_SAMPLE)
    rows = extract_op_summary(DRILLING_SAMPLE)

    assert header["report_date"] == "2020-12-01"
    assert header["report_no"] == "43"
    assert header["wellbore"] == "FORGE-16A-78-32"
    assert header["rig_name"] == "Frontier Rig 16"
    assert header["end_depth_md_ft"] == "7390"
    assert header["morning_report_ops"] == "TRIP IN HOLE WITH GYRO BHA #25"

    assert len(rows) == 8
    assert sum(row["duration_hr"] for row in rows) == 24.0
    assert rows[0]["phase"] == "Production Drilling"
    assert rows[0]["op_code"] == "Reaming"
    assert "Ream from 6,263" in rows[0]["operation_text"]


@pytest.mark.skipif(not EARLY_DRILLING_SAMPLE.exists(), reason="Early Utah FORGE raw sample PDF is not present.")
def test_utah_forge_empty_depth_row_does_not_capture_next_label() -> None:
    header = extract_header_fields(EARLY_DRILLING_SAMPLE)
    rows = extract_op_summary(EARLY_DRILLING_SAMPLE)

    assert header["report_date"] == "2020-10-21"
    assert header["job_number"].endswith("Tangent")
    assert header["end_depth_md_ft"] == ""
    assert header["depth_progress_ft"] == ""

    assert len(rows) == 4
    assert sum(row["duration_hr"] for row in rows) == 24.0
    assert rows[0]["phase"] == "No Activity"
    assert rows[1]["phase"] == "Rig Move In"
    assert "Safety Meeting" in rows[1]["operation_text"]


def test_utah_forge_completion_header_and_time_rows() -> None:
    header = extract_header_fields(COMPLETION_SAMPLE)
    rows = extract_op_summary(COMPLETION_SAMPLE)

    assert header["report_date"] == "2021-01-06"
    assert header["report_no"] == "1"
    assert header["report_type"] == "COMPLETION"
    assert header["end_depth_md_ft"] == "10987"
    assert header["end_depth_tvd_ft"] == "8559"

    assert len(rows) == 6
    assert sum(row["duration_hr"] for row in rows) == 24.0
    assert rows[1]["phase"] == "Drillout"
    assert rows[1]["op_code"] == "Clean Out Hole"
    assert "Drill out cement" in rows[1]["operation_text"]


def test_utah_forge_run_ddr_extraction_schema() -> None:
    header_df, ops_df = run_ddr_extraction(
        DRILLING_SAMPLE,
        "UtahForge-DDR-FORGE-16A-78-32-Drilling-2020-12-01-R018-test",
        "test-corpus",
        "2026-05-31T00:00:00Z",
    )

    assert len(header_df) == 1
    assert len(ops_df) == 8
    assert header_df.loc[0, "doc_id"].startswith("UtahForge-DDR")
    assert set(["doc_id", "report_date", "phase", "operation_text"]).issubset(ops_df.columns)
