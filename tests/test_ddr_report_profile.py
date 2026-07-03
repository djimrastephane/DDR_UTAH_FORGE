from __future__ import annotations

from ddr_rag.report_profile import (
    classify_ddr_section,
    extract_ddr_header_fields,
    looks_like_daily_drilling_report,
)


def test_detects_daily_drilling_report() -> None:
    text = """
    Daily Well Operation
    30-07a-RB
    DRILLING ORIGINAL
    OPERATION SUMMARY
    """

    assert looks_like_daily_drilling_report(text)


def test_extracts_header_fields_from_ddr_text() -> None:
    text = """
    Daily Well Operation
    30-07a-RB
    DRILLING ORIGINAL
    14/04/2024
    Report No. 3
    Job Start: 01/04/2024
    Water Depth (ft) 358.90
    """

    fields = extract_ddr_header_fields(text)

    assert fields["well_name"] == "30-07a-RB"
    assert fields["report_date"] == "14/04/2024"
    assert fields["report_no"] == "3"
    assert fields["job_start"] == "01/04/2024"
    assert fields["report_status"] == "DRILLING ORIGINAL"
    assert fields["water_depth_ft"] == "358.90"


def test_classifies_known_sections() -> None:
    assert classify_ddr_section("OPERATION SUMMARY\nStart Time End Time") == "operation_summary"
    assert classify_ddr_section("PERSONNEL DATA\nCompany Function Count") == "personnel_data"
    assert classify_ddr_section("SUPPORT VESSELS\nVessel Type Vessel Name") == "support_vessels"

