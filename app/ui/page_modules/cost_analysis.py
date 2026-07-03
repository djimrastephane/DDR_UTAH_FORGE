from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from .utils import _apply_chart_theme
except ImportError:
    from utils import _apply_chart_theme  # type: ignore[no-redef]

_ui_root = Path(__file__).resolve().parents[3]
if str(_ui_root / "src") not in sys.path:
    sys.path.insert(0, str(_ui_root / "src"))

from ddr_rag.vocab import label_phase
from ddr_rag.npt_classifier import CATEGORY_LABELS, CATEGORY_COLOURS


_STRIP_RE = re.compile(r"[^\d.]")


def _parse_cost_num(s) -> float | None:
    try:
        return float(_STRIP_RE.sub("", str(s).split()[0]))
    except Exception:
        return None


def _render_kpis(
    total_cost: float,
    afe_val: float | None,
    avg_day: float,
    npt_cost: float,
) -> None:
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Well Cost", f"£{total_cost/1e6:.1f}M")
    k2.metric("AFE",             f"£{afe_val/1e6:.1f}M" if afe_val else "—")
    k3.metric("vs AFE",          f"+£{(total_cost-afe_val)/1e6:.1f}M" if afe_val else "—",
              delta_color="inverse" if afe_val and total_cost > afe_val else "normal")
    k4.metric("Avg Daily Cost",  f"£{avg_day:,.0f}")
    k5.metric("Est. NPT Cost",   f"£{npt_cost/1e6:.1f}M",
              help="NPT hours × average daily rate")


def _render_tab_npt_pareto(npt_ops: pd.DataFrame) -> None:
    if npt_ops.empty:
        st.info("No NPT rows found.")
        return

    cat_cost = (
        npt_ops.groupby("npt_category")["row_cost"]
        .sum().sort_values(ascending=False)
    )
    cat_labels  = [CATEGORY_LABELS.get(c, c) for c in cat_cost.index]
    cat_colours = [CATEGORY_COLOURS.get(c, "#9E9E9E") for c in cat_cost.index]
    cum_pct     = (cat_cost.cumsum() / cat_cost.sum() * 100).values

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cat_labels, y=cat_cost.values / 1e6,
        marker_color=cat_colours,
        text=[f"£{v/1e6:.2f}M" for v in cat_cost.values],
        textposition="outside",
        name="Cost (£M)",
    ))
    fig.add_trace(go.Scatter(
        x=cat_labels, y=cum_pct,
        mode="lines+markers", name="Cumulative %",
        line=dict(color="#D32F2F", width=2),
        yaxis="y2",
    ))
    fig.update_layout(
        height=420, plot_bgcolor="white",
        yaxis=dict(title="Cost (£M)", showgrid=True,
                   gridcolor="rgba(175,175,175,0.35)"),
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right",
                    range=[0, 110], showgrid=False, ticksuffix="%"),
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=10, r=10, t=40, b=80),
    )
    _apply_chart_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    npt_hrs_by_cat = npt_ops.groupby("npt_category")["duration_hr"].sum().reindex(cat_cost.index)
    fig_tbl = go.Figure(go.Table(
        columnwidth=[20, 10, 12, 10],
        header=dict(
            values=["<b>NPT Category</b>", "<b>Cost (£M)</b>",
                    "<b>% of NPT Cost</b>", "<b>NPT Hours</b>"],
            fill_color="#1565C0", font=dict(color="white", size=11),
            align="left", height=28,
        ),
        cells=dict(
            values=[
                cat_labels,
                [f"£{v/1e6:.2f}M" for v in cat_cost.values],
                [f"{v:.1f}%" for v in (cat_cost / cat_cost.sum() * 100).values],
                [f"{h:.0f}" for h in npt_hrs_by_cat.values],
            ],
            fill_color="white", font=dict(color="#1A202C", size=10.5),
            align="left", height=24,
        ),
    ))
    fig_tbl.update_layout(
        margin=dict(l=0, r=0, t=5, b=0),
        height=max(180, len(cat_labels) * 26 + 50),
    )
    st.plotly_chart(fig_tbl, use_container_width=True)


def _render_tab_phase_breakdown(ops_hdr: pd.DataFrame) -> None:
    phase_cost = (
        ops_hdr.groupby("phase")
        .agg(
            total_cost=("row_cost", "sum"),
            npt_cost=("row_cost", lambda x: x[ops_hdr.loc[x.index, "is_npt"]].sum()),
            total_hrs=("duration_hr", "sum"),
        )
        .reset_index()
    )
    phase_cost["prod_cost"]    = phase_cost["total_cost"] - phase_cost["npt_cost"]
    phase_cost["phase_label"]  = phase_cost["phase"].map(label_phase).fillna(phase_cost["phase"])
    phase_cost = phase_cost.sort_values("total_cost", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=phase_cost["phase_label"], x=phase_cost["prod_cost"] / 1e6,
        name="Productive Cost", orientation="h", marker_color="#4CAF50",
        text=[f"£{v/1e6:.1f}M" if v > 1e5 else "" for v in phase_cost["prod_cost"]],
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        y=phase_cost["phase_label"], x=phase_cost["npt_cost"] / 1e6,
        name="NPT Cost", orientation="h", marker_color="#EF5350",
        text=[f"£{v/1e6:.1f}M" if v > 1e5 else "" for v in phase_cost["npt_cost"]],
        textposition="inside",
    ))
    fig.update_layout(
        barmode="stack", height=350, plot_bgcolor="white",
        xaxis=dict(title="Cost (£M)", showgrid=True,
                   gridcolor="rgba(175,175,175,0.35)"),
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=10, r=10, t=30, b=20),
    )
    _apply_chart_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    ph_labels = phase_cost["phase_label"].tolist()
    fig_tbl = go.Figure(go.Table(
        columnwidth=[12, 10, 10, 12, 10],
        header=dict(
            values=["<b>Phase</b>", "<b>Total Cost</b>", "<b>NPT Cost</b>",
                    "<b>Productive Cost</b>", "<b>Total Hours</b>"],
            fill_color="#1565C0", font=dict(color="white", size=11),
            align="left", height=28,
        ),
        cells=dict(
            values=[
                ph_labels,
                [f"£{v/1e6:.2f}M" for v in phase_cost["total_cost"]],
                [f"£{v/1e6:.2f}M" for v in phase_cost["npt_cost"]],
                [f"£{v/1e6:.2f}M" for v in phase_cost["prod_cost"]],
                [f"{v:.0f} hrs" for v in phase_cost["total_hrs"]],
            ],
            fill_color="white", font=dict(color="#1A202C", size=10.5),
            align="left", height=24,
        ),
    ))
    fig_tbl.update_layout(
        margin=dict(l=0, r=0, t=5, b=0),
        height=max(160, len(ph_labels) * 26 + 50),
    )
    st.plotly_chart(fig_tbl, use_container_width=True)


def _render_tab_daily_burn(hdr2: pd.DataFrame, afe_val: float | None) -> None:
    rolling = hdr2["cost_num"].rolling(7, min_periods=1).median()
    bulk    = hdr2["cost_num"] > 3 * rolling

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hdr2["dt"], y=hdr2["cost_num"] / 1e3,
        name="Daily Cost",
        marker_color=[("#D32F2F" if b else "#1565C0") for b in bulk],
        hovertemplate="Date: %{x|%d %b %Y}<br>Cost: £%{y:,.0f}K<extra></extra>",
    ))
    if afe_val:
        fig.add_hline(
            y=afe_val / 1e3 / len(hdr2),
            line=dict(color="#FF8F00", width=1.5, dash="dot"),
            annotation_text="AFE daily rate",
            annotation_position="top right",
        )
    fig.update_layout(
        height=380, plot_bgcolor="white",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Cost (£k)", showgrid=True,
                   gridcolor="rgba(175,175,175,0.35)"),
        margin=dict(l=10, r=10, t=20, b=20),
    )
    _apply_chart_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    if bulk.any():
        st.caption(
            f"🔴 Red bars = likely bulk/mobilisation charges (>3× 7-day rolling median). "
            f"{bulk.sum()} day(s) flagged."
        )


def page_cost_analysis(ops: pd.DataFrame, hdr: pd.DataFrame) -> None:
    _rig   = hdr["rig_name"].dropna().mode()
    _field = hdr["field_name"].dropna().mode()
    _ctx   = " · ".join(x for x in [
        _field.iloc[0].title() if not _field.empty else "",
        _rig.iloc[0].title()   if not _rig.empty   else "",
    ] if x)
    st.header(f"Cost Analysis — {_ctx}" if _ctx else "Cost Analysis")
    st.caption("Well cost breakdown by phase and NPT category. "
               "Costs sourced from DDR daily cost fields.")

    hdr2 = hdr.copy()
    hdr2["cost_num"] = hdr2["daily_cost"].apply(_parse_cost_num)
    hdr2["dt"]       = pd.to_datetime(hdr2["report_date"], dayfirst=True, errors="coerce")
    hdr2 = hdr2.dropna(subset=["cost_num", "dt"]).sort_values("dt")

    ops2 = ops.copy()
    ops2["dt"] = pd.to_datetime(ops2["report_date"], dayfirst=True, errors="coerce")

    total_cost = hdr2["cost_num"].sum()
    afe        = hdr2["afe_amt"].apply(_parse_cost_num).dropna()
    afe_val    = afe.iloc[0] if not afe.empty else None
    avg_day    = hdr2["cost_num"].mean()
    npt_days   = ops2[ops2["is_npt"]]["duration_hr"].sum() / 24
    npt_cost   = npt_days * avg_day

    _render_kpis(total_cost, afe_val, avg_day, npt_cost)
    st.divider()

    ops_hdr = ops2.merge(
        hdr2[["dt", "cost_num"]].rename(columns={"cost_num": "day_cost"}),
        on="dt", how="left",
    )
    ops_hdr["row_cost"] = (
        ops_hdr["duration_hr"].fillna(0) / 24
    ) * ops_hdr["day_cost"].fillna(avg_day)

    t1, t2, t3 = st.tabs(["💰 NPT Cost Pareto", "📊 Phase Breakdown", "📈 Daily Burn"])

    with t1:
        _render_tab_npt_pareto(ops_hdr[ops_hdr["is_npt"]].copy())

    with t2:
        _render_tab_phase_breakdown(ops_hdr)

    with t3:
        _render_tab_daily_burn(hdr2, afe_val)
