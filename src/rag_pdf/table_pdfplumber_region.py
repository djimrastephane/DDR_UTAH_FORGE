from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional, Union

import pandas as pd

from rag_pdf.config import DEFAULT_CONFIG

TABLE_EXTRACT_CFG = DEFAULT_CONFIG.TABLE_EXTRACT


@dataclass
class TableResult:
    page_no: int
    flavor: str
    dataframe: pd.DataFrame
    parsing_report: dict[str, Union[float, int, str]]
    logs: list[str]


TRAILING_ROW_PATTERNS = (
    re.compile(r"total\s+ho?u?rs?", re.IGNORECASE),
    re.compile(r"wellez\.com", re.IGNORECASE),
)


def find_anchor_top(page, anchor_text: str, line_tolerance_pt: float = 2.0) -> Optional[float]:
    target = " ".join(anchor_text.strip().upper().split())
    words = sorted(page.extract_words(), key=lambda w: (w["top"], w["x0"]))
    lines: list[list[dict]] = []
    for word in words:
        if lines and abs(word["top"] - lines[-1][-1]["top"]) <= line_tolerance_pt:
            lines[-1].append(word)
        else:
            lines.append([word])
    for line in lines:
        line_text = " ".join(w["text"].strip().upper().rstrip(":") for w in line)
        if target in line_text:
            return float(min(w["top"] for w in line))
    return None


def extract_table_pdfplumber_region(
    page,
    anchor_text: Optional[str] = None,
    end_anchor_text: Optional[str] = None,
    footer_buffer_pt: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    # Read TABLE_EXTRACT_CFG here (not as a default arg) so runtime config
    # overrides applied via module attribute patching - see
    # scripts/preprocess_hybrid.py's _apply_config_overrides - take effect.
    if anchor_text is None:
        anchor_text = TABLE_EXTRACT_CFG.PDFPLUMBER_REGION_ANCHOR_TEXT
    if footer_buffer_pt is None:
        footer_buffer_pt = TABLE_EXTRACT_CFG.PDFPLUMBER_REGION_FOOTER_BUFFER_PT
    try:
        top = find_anchor_top(page, anchor_text)
        if top is None:
            return None
        if end_anchor_text:
            end_top = find_anchor_top(page, end_anchor_text)
            if end_top is None or end_top <= top:
                return None
            bottom = end_top - 2.0
        else:
            bottom = page.height - footer_buffer_pt
        if bottom <= top:
            return None
        cropped = page.crop((0, max(0.0, top - 2.0), page.width, bottom))
        tables = cropped.find_tables()
        if not tables:
            return None
        candidates = []
        for t in tables:
            rows = t.extract()
            if not rows:
                continue
            n_cols = max((len(r) for r in rows), default=0)
            if n_cols == 0:
                continue
            candidates.append((len(rows) * n_cols, rows))
        if not candidates:
            return None
        candidates.sort(reverse=True, key=lambda c: c[0])
        return pd.DataFrame(candidates[0][1])
    except Exception:
        return None


def drop_trailing_summary_rows(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or len(df) == 0:
        return df

    def _is_trailing(row: pd.Series) -> bool:
        text = " ".join(str(v) for v in row if str(v).strip())
        return any(p.search(text) for p in TRAILING_ROW_PATTERNS)

    mask = df.apply(_is_trailing, axis=1)
    return df.loc[~mask].reset_index(drop=True)


def extract_tables_for_page_region(
    pdf_plumber,
    page_no: int,
    *,
    cleaner: Callable[[pd.DataFrame], Optional[pd.DataFrame]],
) -> list[TableResult]:
    try:
        page = pdf_plumber.pages[page_no - 1]
    except Exception:
        return []

    # Different Utah FORGE report templates put the real data table in
    # different places (Drilling-C: "CONSUMABLES" through the footer;
    # Completion-C: "TIME LOG" through "FLUID DATA"). There is no per-page
    # signal telling us which template a page came from, and some anchors
    # exist on both templates but bound the wrong region on one of them
    # (e.g. "CONSUMABLES" exists on Completion-C pages too, cropping to a
    # small unrelated table padded with blank ruled-grid rows that make its
    # *raw* size misleadingly close to the real table's) - so try every
    # known anchor pair, clean each candidate, and keep the largest cleaned
    # result rather than the first or largest-raw one.
    anchor_candidates: tuple[tuple[str, Optional[str]], ...] = (
        (TABLE_EXTRACT_CFG.PDFPLUMBER_REGION_ANCHOR_TEXT, None),
        ("TIME LOG", "FLUID DATA"),
    )

    best: Optional[tuple[int, str, pd.DataFrame]] = None
    for anchor_text, end_anchor_text in anchor_candidates:
        raw = extract_table_pdfplumber_region(page, anchor_text=anchor_text, end_anchor_text=end_anchor_text)
        raw = drop_trailing_summary_rows(raw)
        if raw is None or len(raw) == 0:
            continue
        cleaned = cleaner(raw)
        if cleaned is None or len(cleaned) == 0:
            continue
        size = cleaned.shape[0] * cleaned.shape[1]
        if best is None or size > best[0]:
            best = (size, anchor_text, cleaned)

    if best is None:
        return []

    _, best_anchor, best_cleaned = best

    return [
        TableResult(
            page_no=page_no,
            flavor=f"pdfplumber_region[{best_anchor}]",
            dataframe=best_cleaned,
            parsing_report={"accuracy": 0.0, "whitespace": 0.0, "order": 0, "page": str(page_no)},
            logs=[f"page {page_no}: pdfplumber_region succeeded with anchor {best_anchor!r}"],
        )
    ]
