from __future__ import annotations

import math
from typing import Any, Optional


def _to_int_if_whole(x: Any) -> Optional[int]:
    # Handles type coercion from parquet readers: int, float (2.0), str ("2").
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if math.isfinite(x) and float(x).is_integer():
            return int(x)
        return None
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        if s.isdigit():
            return int(s)
        try:
            f = float(s)
            if math.isfinite(f) and f.is_integer():
                return int(f)
        except Exception:
            return None
    return None


def build_pages_from_span(page_start: Any, page_end: Any) -> list[int]:
    ps = _to_int_if_whole(page_start)
    pe = _to_int_if_whole(page_end)
    if ps is None or pe is None:
        return []
    if ps <= pe:
        return list(range(ps, pe + 1))
    return list(range(pe, ps + 1))


def build_page_list_struct(pages: list[int]) -> list[dict]:
    return [{"element": int(p)} for p in pages]


def make_chunk_id_global(doc_id: str, chunk_id: str) -> str:
    return f"{doc_id}:{chunk_id}"
