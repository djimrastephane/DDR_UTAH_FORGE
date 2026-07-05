from __future__ import annotations

from pathlib import Path

import pdfplumber
import pytest

from rag_pdf.table_pdfplumber_region import extract_table_pdfplumber_region, find_anchor_top


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
def test_extract_table_pdfplumber_region_on_completion_template_is_unvalidated() -> None:
    with pdfplumber.open(COMPLETION_SAMPLE) as pdf:
        top = find_anchor_top(pdf.pages[0], "CONSUMABLES")

    # The Completion-C template has not been validated against this anchor;
    # this test documents the current (unsupported) behavior rather than
    # asserting a specific result.
    assert top is None or isinstance(top, float)
