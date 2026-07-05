from __future__ import annotations

from pathlib import Path

import pdfplumber
import pytest

import pandas as pd

from rag_pdf.table_extract import clean_table_dataframe
from rag_pdf.table_pdfplumber_region import (
    drop_trailing_summary_rows,
    extract_table_pdfplumber_region,
    extract_tables_for_page_region,
    find_anchor_top,
)


def _cleaner(df: pd.DataFrame) -> pd.DataFrame:
    return clean_table_dataframe(df, None)


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DRILLING_SAMPLE = RAW_DIR / "Utah_Forge_FORGE_16A_(78)-32_Drilling-C_12012020_12012020_18_reporttmp.pdf"
COMPLETION_SAMPLE = RAW_DIR / "Utah_Forge_FORGE_16A_(78)-32_Completion-C_01062021_01062021_1_tmp.pdf"


pytestmark = pytest.mark.skipif(
    not DRILLING_SAMPLE.exists(),
    reason="Utah FORGE raw sample PDFs are not present.",
)


def test_find_anchor_top_locates_consumables_heading() -> None:
    with pdfplumber.open(DRILLING_SAMPLE) as pdf:
        top = find_anchor_top(pdf.pages[0], "CONSUMABLES")

    assert top is not None
    assert 300 < top < 700


def test_find_anchor_top_returns_none_for_missing_anchor() -> None:
    with pdfplumber.open(DRILLING_SAMPLE) as pdf:
        top = find_anchor_top(pdf.pages[0], "NOT_A_REAL_HEADING_XYZ")

    assert top is None


def test_extract_table_pdfplumber_region_isolates_time_breakdown_table() -> None:
    with pdfplumber.open(DRILLING_SAMPLE) as pdf:
        df = extract_table_pdfplumber_region(pdf.pages[0])

    assert df is not None
    text = df.astype(str).apply(lambda col: col.str.cat(sep=" "), axis=0).str.cat(sep=" ")
    assert "TIME BREAKDOWN" in text
    assert "Production Drilling" in text
    # words within a cell must keep their spacing, unlike camelot's stream output
    assert "Lubricate Rig" in text
    assert all("LubricateRig" not in str(v) for v in df.values.flatten())


def test_extract_table_pdfplumber_region_returns_none_for_missing_anchor() -> None:
    with pdfplumber.open(DRILLING_SAMPLE) as pdf:
        df = extract_table_pdfplumber_region(pdf.pages[0], anchor_text="NOT_A_REAL_HEADING_XYZ")

    assert df is None


@pytest.mark.skipif(not COMPLETION_SAMPLE.exists(), reason="Completion sample PDF is not present.")
def test_find_anchor_top_matches_multi_word_heading() -> None:
    with pdfplumber.open(COMPLETION_SAMPLE) as pdf:
        page = pdf.pages[0]
        time_log_top = find_anchor_top(page, "TIME LOG")
        fluid_data_top = find_anchor_top(page, "FLUID DATA")

    assert time_log_top is not None
    assert fluid_data_top is not None
    assert time_log_top < fluid_data_top


@pytest.mark.skipif(not COMPLETION_SAMPLE.exists(), reason="Completion sample PDF is not present.")
def test_extract_table_pdfplumber_region_isolates_completion_time_log_table() -> None:
    # Completion-C uses a different section order than Drilling-C: the real
    # per-shift table sits under "TIME LOG", well before "CONSUMABLES", so it
    # needs an explicit end anchor rather than the Drilling-C default of
    # cropping to the page bottom.
    with pdfplumber.open(COMPLETION_SAMPLE) as pdf:
        df = extract_table_pdfplumber_region(
            pdf.pages[0], anchor_text="TIME LOG", end_anchor_text="FLUID DATA"
        )

    assert df is not None
    text = df.astype(str).apply(lambda col: col.str.cat(sep=" "), axis=0).str.cat(sep=" ")
    assert "Clean Out Hole" in text
    assert "Trip in hole picking up 3 1/2\" drill pipe" in text
    assert all("CleanOutHole" not in str(v) for v in df.values.flatten())


@pytest.mark.skipif(not COMPLETION_SAMPLE.exists(), reason="Completion sample PDF is not present.")
def test_extract_table_pdfplumber_region_returns_none_when_end_anchor_missing() -> None:
    with pdfplumber.open(COMPLETION_SAMPLE) as pdf:
        df = extract_table_pdfplumber_region(
            pdf.pages[0], anchor_text="TIME LOG", end_anchor_text="NOT_A_REAL_HEADING_XYZ"
        )

    assert df is None


def test_drop_trailing_summary_rows_removes_total_hrs_and_footer() -> None:
    with pdfplumber.open(DRILLING_SAMPLE) as pdf:
        df = extract_table_pdfplumber_region(pdf.pages[0])
    cleaned = clean_table_dataframe(df, None)

    filtered = drop_trailing_summary_rows(cleaned)

    assert len(filtered) == len(cleaned) - 2
    text = filtered.astype(str).apply(lambda col: col.str.cat(sep=" "), axis=0).str.cat(sep=" ")
    assert "total hrs" not in text.lower()
    assert "wellez.com" not in text.lower()
    assert "Production Drilling" in text


@pytest.mark.skipif(not COMPLETION_SAMPLE.exists(), reason="Completion sample PDF is not present.")
def test_drop_trailing_summary_rows_removes_total_hours_wording_variant() -> None:
    with pdfplumber.open(COMPLETION_SAMPLE) as pdf:
        df = extract_table_pdfplumber_region(
            pdf.pages[0], anchor_text="TIME LOG", end_anchor_text="FLUID DATA"
        )
    cleaned = clean_table_dataframe(df, None)

    filtered = drop_trailing_summary_rows(cleaned)

    assert len(filtered) == len(cleaned) - 1
    text = filtered.astype(str).apply(lambda col: col.str.cat(sep=" "), axis=0).str.cat(sep=" ")
    assert "total hours" not in text.lower()


def test_drop_trailing_summary_rows_handles_none_and_empty() -> None:
    assert drop_trailing_summary_rows(None) is None
    assert len(drop_trailing_summary_rows(pd.DataFrame())) == 0


def test_extract_tables_for_page_region_picks_consumables_for_drilling_template() -> None:
    with pdfplumber.open(DRILLING_SAMPLE) as pdf:
        results = extract_tables_for_page_region(pdf, 1, cleaner=_cleaner)

    assert len(results) == 1
    assert results[0].flavor == "pdfplumber_region[CONSUMABLES]"
    assert "Production Drilling" in results[0].dataframe.astype(str).to_string()


@pytest.mark.skipif(not COMPLETION_SAMPLE.exists(), reason="Completion sample PDF is not present.")
def test_extract_tables_for_page_region_picks_time_log_for_completion_template() -> None:
    # Regression test: "CONSUMABLES" also exists on the Completion-C page (it
    # just bounds a small unrelated table there), so a naive "first anchor
    # that finds anything" strategy would wrongly pick it over the real
    # "TIME LOG" table. The selection must compare cleaned table sizes.
    with pdfplumber.open(COMPLETION_SAMPLE) as pdf:
        results = extract_tables_for_page_region(pdf, 1, cleaner=_cleaner)

    assert len(results) == 1
    assert results[0].flavor == "pdfplumber_region[TIME LOG]"
    assert "Clean Out Hole" in results[0].dataframe.astype(str).to_string()


def test_extract_tables_for_page_region_returns_empty_list_for_invalid_page() -> None:
    with pdfplumber.open(DRILLING_SAMPLE) as pdf:
        results = extract_tables_for_page_region(pdf, 999, cleaner=_cleaner)

    assert results == []
