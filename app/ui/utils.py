from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go


def _apply_chart_theme(fig: go.Figure) -> go.Figure:
    _TICK  = dict(size=11.5, color="rgb(40,40,40)",  family="Arial, sans-serif")
    _TITLE = dict(size=12,   color="rgb(20,20,20)",  family="Arial, sans-serif")
    _GRID  = "rgba(175,175,175,0.35)"
    _ZERO  = "rgba(110,110,110,0.55)"
    _LINE  = "rgba(110,110,110,0.40)"

    fig.update_xaxes(
        tickfont=_TICK,
        title_font=_TITLE,
        gridcolor=_GRID,
        zerolinecolor=_ZERO,
        linecolor=_LINE,
        tickcolor="rgba(80,80,80,0.6)",
    )
    fig.update_yaxes(
        tickfont=_TICK,
        title_font=_TITLE,
        gridcolor=_GRID,
        zerolinecolor=_ZERO,
        linecolor=_LINE,
        tickcolor="rgba(80,80,80,0.6)",
    )
    fig.update_layout(
        font=dict(size=11.5, color="rgb(40,40,40)", family="Arial, sans-serif"),
        legend=dict(
            font=dict(size=11, color="rgb(40,40,40)"),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(150,150,150,0.35)",
            borderwidth=1,
        ),
    )
    return fig


def _ddr_citation(doc_id: str, page: object, shift: str = "") -> str:
    m   = re.search(r"DDR-?(\d+)", str(doc_id), re.I)
    ddr = f"DDR-{m.group(1)}" if m else (str(doc_id)[:20] or "—")
    p   = f" · p.{int(page)}" if pd.notna(page) else ""
    s   = f" · {shift}" if shift else ""
    return f"{ddr}{p}{s}"


def _ddr_citation_row(row: dict) -> str:
    st  = str(row.get("start_time") or "").strip()
    et  = str(row.get("end_time") or "").strip()
    time_str = (
        f"{st}–{et}" if (st and et)
        else (st or str(row.get("shift_block") or ""))
    )
    return _ddr_citation(row.get("doc_id", ""), row.get("page"), time_str)


def _sea_state(wave_ft: float | None) -> str:
    if wave_ft is None or (isinstance(wave_ft, float) and wave_ft != wave_ft):
        return "—"
    if wave_ft < 2:
        return "Calm"
    if wave_ft < 5:
        return "Slight"
    if wave_ft < 8:
        return "Moderate"
    if wave_ft < 13:
        return "Rough"
    return "Very Rough"


def _beaufort_colour(wind_kn: float | None) -> str:
    if wind_kn is None:
        return "#90CAF9"
    if wind_kn < 7:
        return "#B3E5FC"   # light blue — light
    if wind_kn < 14:
        return "#29B6F6"   # blue       — moderate
    if wind_kn < 22:
        return "#F9A825"   # amber      — fresh
    if wind_kn < 28:
        return "#EF6C00"   # orange     — strong
    return "#B71C1C"


def _phase_date_ranges(ops: pd.DataFrame) -> dict[str, tuple]:
    ranges: dict[str, tuple] = {}
    for phase, grp in ops.dropna(subset=["report_date_parsed"]).groupby("phase"):
        ranges[phase] = (
            grp["report_date_parsed"].min(),
            grp["report_date_parsed"].max(),
        )
    return ranges


def _well_label(well_id: str, meta: dict) -> str:
    m   = meta.get(well_id, {})
    rig = m.get("rig", "")
    yr  = (m.get("spud_date") or "")[:4]
    return f"{well_id}  ({rig}, {yr})" if rig else well_id


def _t2h(t: str) -> float:
    try:
        parts = str(t).strip().split(":")
        return int(parts[0]) + int(parts[1]) / 60.0
    except Exception:
        return 0.0
