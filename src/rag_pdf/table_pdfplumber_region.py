from __future__ import annotations

from typing import Optional

import pandas as pd

from rag_pdf.config import DEFAULT_CONFIG

TABLE_EXTRACT_CFG = DEFAULT_CONFIG.TABLE_EXTRACT


def find_anchor_top(page, anchor_text: str) -> Optional[float]:
    target = anchor_text.strip().upper()
    for word in page.extract_words():
        if word["text"].strip().upper().rstrip(":") == target:
            return float(word["top"])
    return None


def extract_table_pdfplumber_region(
    page,
    anchor_text: str = TABLE_EXTRACT_CFG.PDFPLUMBER_REGION_ANCHOR_TEXT,
    footer_buffer_pt: float = TABLE_EXTRACT_CFG.PDFPLUMBER_REGION_FOOTER_BUFFER_PT,
) -> Optional[pd.DataFrame]:
    try:
        top = find_anchor_top(page, anchor_text)
        if top is None:
            return None
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
