from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from rag_pdf.config import DEFAULT_CONFIG

TABLE_EXTRACT_CFG = DEFAULT_CONFIG.TABLE_EXTRACT

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
    anchor_text: str = TABLE_EXTRACT_CFG.PDFPLUMBER_REGION_ANCHOR_TEXT,
    end_anchor_text: Optional[str] = None,
    footer_buffer_pt: float = TABLE_EXTRACT_CFG.PDFPLUMBER_REGION_FOOTER_BUFFER_PT,
) -> Optional[pd.DataFrame]:
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
