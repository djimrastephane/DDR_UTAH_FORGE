from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from .constants import PHASE_ORDER
    from .loaders import load_ditch_magnets
    from .utils import _apply_chart_theme
except ImportError:
    from constants import PHASE_ORDER                   # type: ignore[no-redef]
    from loaders import load_ditch_magnets              # type: ignore[no-redef]
    from utils import _apply_chart_theme                # type: ignore[no-redef]

_ui_root = Path(__file__).resolve().parents[3]
if str(_ui_root / "src") not in sys.path:
    sys.path.insert(0, str(_ui_root / "src"))

from ddr_rag.vocab import label_phase
from ddr_rag.npt_classifier import CATEGORY_LABELS, CATEGORY_COLOURS


def _render_kpis(total_h: float, npt_h: float) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total hours", f"{total_h:.0f}h")
    c2.metric("NPT hours",   f"{npt_h:.0f}h")
    c3.metric("NPT %",       f"{100*npt_h/total_h:.0f}%" if total_h else "—")


def _render_category_breakdown(npt_ops: pd.DataFrame, npt_h: float) -> None:
    st.subheader("NPT breakdown by cause")
    if npt_ops.empty:
        st.info("No operation rows are currently flagged as NPT for this selection.")
        return

    cat_stats = (
        npt_ops.groupby("npt_category")["duration_hr"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    cat_stats["label"]  = cat_stats["npt_category"].map(CATEGORY_LABELS).fillna("Other NPT")
    cat_stats["colour"] = cat_stats["npt_category"].map(CATEGORY_COLOURS).fillna("#9E9E9E")
    cat_stats["pct"]    = (100 * cat_stats["duration_hr"] / npt_h).round(1) if npt_h else 0

    fig = go.Figure(go.Bar(
        y=cat_stats["label"],
        x=cat_stats["duration_hr"],
        orientation="h",
        marker_color=cat_stats["colour"].tolist(),
        text=cat_stats["pct"].map("{:.0f}%".format),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:.0f}h (%{text})<extra></extra>",
    ))
    fig.update_layout(
        height=max(280, len(cat_stats) * 32),
        margin=dict(l=10, r=80, t=10, b=10),
        xaxis_title="NPT hours",
        plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA",
        showlegend=False,
    )
    st.plotly_chart(_apply_chart_theme(fig), use_container_width=True, config={"displayModeBar": False})


def _render_findings(phase_ops: pd.DataFrame, npt_ops: pd.DataFrame) -> None:
    st.subheader("Classification notes")
    st.caption(
        "Utah FORGE operation summaries do not provide a P-T-X code. "
        "NPT rows shown here are text-derived flags from explicit repair, fishing, "
        "wellbore-condition, waiting, or downhole-tool trouble evidence."
    )

    if npt_ops.empty:
        return

    cat_stats = (
        npt_ops.groupby("npt_cat_label")["duration_hr"]
        .sum()
        .sort_values(ascending=False)
    )
    total_h = float(phase_ops["duration_hr"].sum())
    for label, hrs in cat_stats.head(4).items():
        pct = 100 * float(hrs) / total_h if total_h else 0.0
        st.markdown(f"- **{label or 'Unlabelled'}**: {hrs:.1f}h ({pct:.1f}% of selected time)")

    top = npt_ops.nlargest(3, "duration_hr")
    if not top.empty:
        with st.expander("Largest flagged rows", expanded=False):
            for _, row in top.iterrows():
                st.markdown(
                    f"**{row.get('report_date', '')} · {row.get('duration_hr', 0):.1f}h · "
                    f"{label_phase(row.get('phase', ''))}**  "
                    f"{str(row.get('operation_text', ''))[:260]}"
                )


def _render_event_table(npt_ops: pd.DataFrame) -> None:
    st.subheader("NPT events")
    if npt_ops.empty:
        return

    disp = npt_ops.rename(columns={
        "report_date":    "Date",
        "phase_label":    "Phase",
        "op_code_label":  "Op type",
        "start_time":     "Start",
        "end_time":       "End",
        "duration_hr":    "Dur (h)",
        "npt_cat_label":  "NPT type",
        "operation_text": "Operation",
    })
    cols = ["Date", "Phase", "Op type", "Start", "End", "Dur (h)", "NPT type", "Operation"]
    cols = [c for c in cols if c in disp.columns]
    st.dataframe(
        disp[cols],
        hide_index=True,
        use_container_width=True,
        height=380,
        column_config={
            "Operation": st.column_config.TextColumn(width="large"),
            "Dur (h)":   st.column_config.NumberColumn(format="%.2f"),
        },
    )


def _render_debris_chart(phase: str) -> None:
    dm = load_ditch_magnets()
    if dm.empty or phase not in ("ALL", "PROD1", "INTRM1", "INTRM2", "COND1"):
        return

    dm_active = dm[dm["has_ditch_magnet"] & dm["daily_grams"].notna()].copy()
    if dm_active.empty:
        return

    st.subheader("Metallic Debris Recovery — Ditch Magnet Readings")
    st.caption(
        "Daily metallic debris recovered from the wellbore (ditch magnet readings "
        "from General Notes section). "
        "Debris originates from drillout of float equipment, BHA wear, and casing shoe track. "
        "Elevated readings indicate active debris in the wellbore requiring magnet/junk mill runs."
    )

    _section_colours = {
        '16" Section Total':     "#90CAF9",
        '12-1/4" Section Total': "#42A5F5",
        '8-1/2" Section Total':  "#E53935",
    }
    _section_label = {
        '16" Section Total':     '16" (Conductor)',
        '12-1/4" Section Total': '12¼" (Intermediate)',
        '8-1/2" Section Total':  '8½" (Production)',
    }

    fig_dm = go.Figure()

    sec8 = dm_active[dm_active["section_name"].str.contains('8-1/2', na=False)]
    if not sec8.empty:
        fig_dm.add_trace(go.Scatter(
            x=sec8["report_date"],
            y=sec8["section_total_grams"] / 1000,
            name='Running total 8½" (kg)',
            mode="lines",
            line=dict(color="rgba(183,28,28,0.5)", width=1.5, dash="dot"),
            yaxis="y2",
            hovertemplate="<b>%{x|%d %b}</b><br>Cumulative: %{y:.2f} kg<extra></extra>",
        ))

    for section_name, grp in dm_active.groupby("section_name"):
        colour = _section_colours.get(section_name, "#999")
        label  = _section_label.get(section_name, section_name)
        fig_dm.add_trace(go.Bar(
            x=grp["report_date"],
            y=grp["daily_grams"],
            name=label,
            marker_color=colour,
            yaxis="y",
            hovertemplate=(
                "<b>%{x|%d %b}</b><br>"
                f"{label}<br>"
                "Daily: %{y:,.0f}g<extra></extra>"
            ),
            customdata=grp[["daily_qualifier", "section_total_grams"]].values,
        ))

    peak_row = dm_active.loc[dm_active["daily_grams"].idxmax()]
    fig_dm.add_annotation(
        x=peak_row["report_date"], y=peak_row["daily_grams"],
        text=f"Peak {peak_row['daily_grams']:,.0f}g",
        showarrow=True, arrowhead=2, arrowcolor="#B71C1C",
        ax=0, ay=-32,
        font=dict(size=9, color="#B71C1C"),
        bgcolor="rgba(255,255,255,0.88)", bordercolor="#B71C1C", borderwidth=1,
    )
    fig_dm.update_layout(
        barmode="stack", height=320,
        margin=dict(l=10, r=60, t=10, b=10),
        yaxis=dict(title="Daily recovery (g)"),
        yaxis2=dict(title="Cumulative (kg)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=-0.25),
        plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA",
    )
    st.plotly_chart(_apply_chart_theme(fig_dm), use_container_width=True, config={"displayModeBar": False})

    peak_total  = dm_active["section_total_grams"].max()
    peak_day    = dm_active["daily_grams"].max()
    active_days = (dm_active["daily_grams"] > 0).sum()
    m1, m2, m3 = st.columns(3)
    m1.metric('Peak cumulative (8½")', f"{peak_total/1000:.2f} kg",
              help='Running total of metallic debris in the 8-1/2" production section')
    m2.metric("Peak single day",       f"{peak_day:,.0f} g",
              help="Highest single-day ditch magnet recovery")
    m3.metric("Days with recovery > 0", str(active_days),
              help="Days where non-zero metallic debris was captured by ditch magnets")


def page_npt_intelligence(ops: pd.DataFrame) -> None:
    st.header("NPT Intelligence")

    phase_opts = sorted(
        ops["phase"].dropna().unique(),
        key=lambda x: PHASE_ORDER.index(x) if x in PHASE_ORDER else 99,
    )
    phase = st.selectbox(
        "Phase", ["ALL"] + phase_opts,
        format_func=lambda x: "All phases" if x == "ALL" else label_phase(x),
        key="npt_intel_phase",
    )

    phase_ops = ops if phase == "ALL" else ops[ops["phase"] == phase]
    npt_ops   = phase_ops[phase_ops["is_npt"]]
    total_h   = float(phase_ops["duration_hr"].sum())
    npt_h     = float(npt_ops["duration_hr"].sum())

    _render_kpis(total_h, npt_h)
    st.divider()
    _render_category_breakdown(npt_ops, npt_h)
    _render_findings(phase_ops, npt_ops)
    _render_event_table(npt_ops)
    _render_debris_chart(phase)
