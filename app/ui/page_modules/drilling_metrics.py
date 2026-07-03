from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from .constants import PHASE_ORDER, PHASE_COLOURS
    from .loaders import load_drilling_metrics, load_field_headers
    from .utils import _apply_chart_theme
except ImportError:
    from constants import PHASE_ORDER, PHASE_COLOURS          # type: ignore[no-redef]
    from loaders import load_drilling_metrics, load_field_headers  # type: ignore[no-redef]
    from utils import _apply_chart_theme                       # type: ignore[no-redef]

_root = Path(__file__).resolve().parents[3]
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from ddr_rag.vocab import label_phase

_METRIC_META: dict[str, dict] = {
    "trip_speed":    {"label": "Tripping Speed",      "unit": "ft/min",    "icon": "🔄"},
    "rop_inst":      {"label": "Inst. ROP",            "unit": "ft/hr",     "icon": "⛏"},
    "rop_avg":       {"label": "Avg ROP",              "unit": "ft/hr",     "icon": "⛏"},
    "rop_gen":       {"label": "ROP (generic)",        "unit": "ft/hr",     "icon": "⛏"},
    "casing_speed":  {"label": "Casing Running Speed", "unit": "joints/hr", "icon": "🔩"},
    "tubing_speed":  {"label": "Tubing Running Speed", "unit": "joints/hr", "icon": "🔩"},
    "running_speed": {"label": "Running Speed",        "unit": "joints/hr", "icon": "🔩"},
    "wob":           {"label": "Weight on Bit",        "unit": "klbs",      "icon": "⚖"},
    "flow_rate":     {"label": "Flow Rate",            "unit": "gpm",       "icon": "💧"},
}


def _box_by_phase(sub: pd.DataFrame, title: str, unit: str, split_hole: bool = False) -> go.Figure:
    fig = go.Figure()
    phases = [p for p in PHASE_ORDER if p in sub["phase"].values]

    if split_hole:
        for ht, col, sym in [("OH", "#1565C0", "circle"), ("CH", "#E65100", "diamond")]:
            grp = sub[sub["hole_type"] == ht]
            if grp.empty:
                continue
            fig.add_trace(go.Box(
                x=[label_phase(p) for p in grp["phase"]],
                y=grp["value"],
                name=ht,
                marker_color=col,
                marker_symbol=sym,
                boxpoints="all", jitter=0.35, pointpos=0, line_width=1.5,
                hovertemplate=(
                    f"<b>%{{x}}</b><br>{ht}<br>Value: %{{y:.1f}} {unit}<extra></extra>"
                ),
            ))
    else:
        for ph in phases:
            g = sub[sub["phase"] == ph]
            fig.add_trace(go.Box(
                x=[label_phase(ph)] * len(g),
                y=g["value"],
                name=label_phase(ph),
                marker_color=PHASE_COLOURS.get(ph, "#9E9E9E"),
                boxpoints="all", jitter=0.3, pointpos=0, line_width=1.5,
                hovertemplate=f"<b>%{{x}}</b><br>Value: %{{y:.1f}} {unit}<extra></extra>",
            ))

    fig.update_layout(
        title_text=title, height=380, plot_bgcolor="white",
        yaxis=dict(title=unit, showgrid=True, gridcolor="rgba(175,175,175,0.35)"),
        xaxis=dict(showgrid=False),
        showlegend=split_hole,
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=10, r=10, t=50, b=20),
    )
    return _apply_chart_theme(fig)


def _trend_chart(sub: pd.DataFrame, title: str, unit: str, color_by: str = "phase") -> go.Figure:
    fig = go.Figure()
    groups  = sub["phase"].unique() if color_by == "phase" else sub["hole_type"].unique()
    palette = PHASE_COLOURS if color_by == "phase" else {
        "OH": "#1565C0", "CH": "#E65100", "unknown": "#9E9E9E"
    }
    for grp_val in groups:
        g   = sub[sub[color_by] == grp_val]
        lbl = label_phase(grp_val) if color_by == "phase" else grp_val
        fig.add_trace(go.Scatter(
            x=g["report_date_dt"], y=g["value"],
            mode="markers",
            marker=dict(size=7, color=palette.get(grp_val, "#9E9E9E"),
                        opacity=0.75, line=dict(width=0.5, color="white")),
            name=lbl,
            hovertemplate=(
                f"<b>%{{x|%d %b %Y}}</b><br>{lbl}<br>"
                f"Value: %{{y:.1f}} {unit}<extra></extra>"
            ),
        ))
    fig.update_layout(
        title_text=title, height=320, plot_bgcolor="white",
        yaxis=dict(title=unit, showgrid=True, gridcolor="rgba(175,175,175,0.35)"),
        xaxis=dict(title="Date", showgrid=False),
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=10, r=10, t=50, b=20),
    )
    return _apply_chart_theme(fig)


def _stats_table(sub: pd.DataFrame, unit: str) -> go.Figure:
    rows = []
    for ph in [p for p in PHASE_ORDER if p in sub["phase"].values]:
        g = sub[sub["phase"] == ph]
        hole_types = (["OH", "CH"]
                      if ("OH" in g["hole_type"].values or "CH" in g["hole_type"].values)
                      else ["all"])
        for ht in hole_types:
            s = g[g["hole_type"] == ht] if ht != "all" else g
            if s.empty:
                continue
            v = s["value"]
            rows.append({
                "Phase":  label_phase(ph),
                "Hole":   ht,
                "n":      len(s),
                "Min":    f"{v.min():.1f}",
                "Mean":   f"{v.mean():.1f}",
                "p25":    f"{v.quantile(0.25):.1f}",
                "Median": f"{v.median():.1f}",
                "p75":    f"{v.quantile(0.75):.1f}",
                "p90":    f"{v.quantile(0.9):.1f}",
                "Max":    f"{v.max():.1f}",
            })
    if not rows:
        return go.Figure()
    fig = go.Figure(go.Table(
        columnwidth=[14, 6, 5, 7, 7, 7, 8, 7, 7, 7],
        header=dict(
            values=[f"<b>{c}</b> ({unit})" if c in ("Median", "Mean") else f"<b>{c}</b>"
                    for c in rows[0]],
            fill_color="#455A64", font=dict(color="white", size=11),
            align="left", height=28,
        ),
        cells=dict(
            values=[[r[k] for r in rows] for k in rows[0]],
            fill_color="white",
            font=dict(color="#1A202C", size=10.5),
            align=["left", "center", "center"] + ["right"] * 7,
            height=24,
        ),
    ))
    fig.update_layout(margin=dict(l=0, r=0, t=5, b=0),
                      height=max(150, len(rows) * 26 + 60))
    return fig


def _render_kpis(df: pd.DataFrame) -> None:
    trip = df[df["metric_type"] == "trip_speed"]["value"]
    rop  = df[df["metric_type"].isin(["rop_inst", "rop_avg"])]["value"]
    cas  = df[df["metric_type"] == "casing_speed"]["value"]
    tub  = df[df["metric_type"] == "tubing_speed"]["value"]

    def _mmm(s: pd.Series, fmt: str = ".0f") -> str:
        return f"{s.min():{fmt}} / {s.mean():{fmt}} / {s.max():{fmt}}" if not s.empty else "—"

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Metric records",           f"{len(df):,}")
    k2.metric("Trip speed  min/mean/max", (_mmm(trip) + " ft/min")  if not trip.empty else "—",
              help="All hole types combined")
    k3.metric("ROP  min/mean/max",        (_mmm(rop)  + " ft/hr")   if not rop.empty  else "—",
              help="Instantaneous + average combined")
    k4.metric("Casing speed  min/mean/max", (_mmm(cas, ".1f") + " jts/hr") if not cas.empty else "—")
    k5.metric("Tubing speed  min/mean/max", (_mmm(tub) + " jts/hr") if not tub.empty else "—")
    st.divider()


def _render_tab_tripping(df: pd.DataFrame) -> None:
    sub = df[df["metric_type"] == "trip_speed"].copy()
    if sub.empty:
        st.info("No tripping speed data extracted.")
        return

    st.subheader(f"Tripping Speed — {len(sub)} data points")
    st.caption(
        "Speed at which pipe was pulled out of (POOH) or run in hole (RIH).  "
        "Open Hole (OH) speeds are typically lower due to pack-off risk; "
        "Cased Hole (CH) speeds are higher."
    )

    oh = sub[sub["hole_type"] == "OH"]["value"]
    ch = sub[sub["hole_type"] == "CH"]["value"]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Min",   f"{sub['value'].min():.0f} ft/min")
    c2.metric("Mean",  f"{sub['value'].mean():.0f} ft/min")
    c3.metric("Max",   f"{sub['value'].max():.0f} ft/min")
    c4.metric("OH  min/mean/max",
              f"{oh.min():.0f} / {oh.mean():.0f} / {oh.max():.0f}" if not oh.empty else "—")
    c5.metric("CH  min/mean/max",
              f"{ch.min():.0f} / {ch.mean():.0f} / {ch.max():.0f}" if not ch.empty else "—")
    c6.metric("Records", len(sub))

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            _box_by_phase(sub, "Tripping Speed by Phase — OH vs CH", "ft/min", split_hole=True),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            _trend_chart(sub, "Tripping Speed over Campaign", "ft/min", color_by="hole_type"),
            use_container_width=True,
        )

    st.markdown("**Statistics by phase:**")
    st.plotly_chart(_stats_table(sub, "ft/min"), use_container_width=True)

    with st.expander("Source DDR citations — top 20 highest speeds"):
        top = sub.nlargest(20, "value")[
            ["report_date", "phase", "activity_code", "hole_type",
             "value", "depth_from_ft", "depth_to_ft", "ddr_citation", "raw_snippet"]
        ].copy()
        top.columns = ["Date", "Phase", "Activity", "Hole", "ft/min",
                       "From (ft)", "To (ft)", "DDR Citation", "Context"]
        top["Phase"] = top["Phase"].map(label_phase)
        st.dataframe(top, use_container_width=True, hide_index=True)


def _render_tab_rop(df: pd.DataFrame) -> None:
    sub = df[df["metric_type"].isin(["rop_inst", "rop_avg", "rop_gen"])].copy()
    if sub.empty:
        st.info("No ROP data extracted.")
        return

    st.subheader(f"Rate of Penetration — {len(sub)} data points")
    st.caption(
        "Instantaneous ROP = peak ft/hr at a given moment.  "
        "Average ROP = mean over a drilled interval."
    )

    inst = sub[sub["metric_type"] == "rop_inst"]["value"]
    avg  = sub[sub["metric_type"] == "rop_avg"]["value"]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Inst. ROP — Min",  f"{inst.min():.0f} ft/hr"  if not inst.empty else "—")
    c2.metric("Inst. ROP — Mean", f"{inst.mean():.0f} ft/hr" if not inst.empty else "—")
    c3.metric("Inst. ROP — Max",  f"{inst.max():.0f} ft/hr"  if not inst.empty else "—")
    c4.metric("Avg ROP — Min",    f"{avg.min():.0f} ft/hr"   if not avg.empty  else "—")
    c5.metric("Avg ROP — Mean",   f"{avg.mean():.0f} ft/hr"  if not avg.empty  else "—")
    c6.metric("Avg ROP — Max",    f"{avg.max():.0f} ft/hr"   if not avg.empty  else "—")

    inst_df = sub[sub["metric_type"] == "rop_inst"]
    avg_df  = sub[sub["metric_type"] == "rop_avg"]

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            _box_by_phase(inst_df, "Instantaneous ROP by Phase", "ft/hr"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            _box_by_phase(avg_df, "Average ROP by Phase", "ft/hr"),
            use_container_width=True,
        )

    sub2 = sub.dropna(subset=["depth_from_ft"])
    if not sub2.empty:
        fig = go.Figure()
        for mtype, col, sym in [("rop_inst", "#1565C0", "circle"), ("rop_avg", "#43A047", "diamond")]:
            g = sub2[sub2["metric_type"] == mtype]
            if g.empty:
                continue
            fig.add_trace(go.Scatter(
                x=g["depth_from_ft"], y=g["value"],
                mode="markers",
                marker=dict(size=8, color=col, symbol=sym, opacity=0.75),
                name=_METRIC_META[mtype]["label"],
                hovertemplate="<b>Depth: %{x:,.0f} ft</b><br>ROP: %{y:.1f} ft/hr<extra></extra>",
            ))
        fig.update_xaxes(title="Depth from (ft MD)", tickformat=",")
        fig.update_yaxes(title="ROP (ft/hr)", showgrid=True,
                         gridcolor="rgba(175,175,175,0.35)")
        fig.update_layout(
            height=320, plot_bgcolor="white", title_text="ROP vs Depth",
            legend=dict(orientation="h", y=1.08),
            margin=dict(l=10, r=10, t=50, b=20),
        )
        st.plotly_chart(_apply_chart_theme(fig), use_container_width=True)

    st.markdown("**Statistics by phase:**")
    st.plotly_chart(_stats_table(sub, "ft/hr"), use_container_width=True)

    with st.expander("Source DDR citations — top 20 highest ROP"):
        top = sub.nlargest(20, "value")[
            ["report_date", "phase", "metric_type", "value",
             "depth_from_ft", "depth_to_ft", "ddr_citation", "raw_snippet"]
        ].copy()
        top.columns = ["Date", "Phase", "Type", "ft/hr",
                       "From (ft)", "To (ft)", "DDR Citation", "Context"]
        top["Phase"] = top["Phase"].map(label_phase)
        st.dataframe(top, use_container_width=True, hide_index=True)


def _render_tab_running(df: pd.DataFrame) -> None:
    cas_s = df[df["metric_type"] == "casing_speed"]
    tub_s = df[df["metric_type"].isin(["tubing_speed", "running_speed"])]

    st.subheader("Pipe Running Speeds")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Casing Running Speed**")
        if cas_s.empty:
            st.info("No casing speed data extracted.")
        else:
            v = cas_s["value"]
            st.caption(
                f"{len(cas_s)} records  ·  "
                f"min {v.min():.1f}  ·  mean {v.mean():.1f}  ·  max {v.max():.1f} joints/hr"
            )
            st.plotly_chart(
                _box_by_phase(cas_s, "Casing Running Speed", "joints/hr"),
                use_container_width=True,
            )
            st.plotly_chart(_stats_table(cas_s, "joints/hr"), use_container_width=True)

    with col2:
        st.markdown("**Tubing / Wash-Pipe Running Speed**")
        if tub_s.empty:
            st.info("No tubing speed data extracted.")
        else:
            v = tub_s["value"]
            st.caption(
                f"{len(tub_s)} records  ·  "
                f"min {v.min():.1f}  ·  mean {v.mean():.1f}  ·  max {v.max():.1f} joints/hr"
            )
            st.plotly_chart(
                _box_by_phase(tub_s, "Tubing Running Speed", "joints/hr"),
                use_container_width=True,
            )
            st.plotly_chart(_stats_table(tub_s, "joints/hr"), use_container_width=True)

    run_all = pd.concat([cas_s, tub_s])
    if not run_all.empty:
        with st.expander("All running speed records with DDR citations"):
            view = run_all.sort_values("value", ascending=False)[
                ["report_date", "phase", "metric_type", "value",
                 "depth_from_ft", "depth_to_ft", "ddr_citation", "raw_snippet"]
            ].copy()
            view["metric_type"] = view["metric_type"].map(
                lambda x: _METRIC_META.get(x, {}).get("label", x)
            )
            view["phase"] = view["phase"].map(label_phase)
            view.columns = ["Date", "Phase", "Metric", "joints/hr",
                            "From (ft)", "To (ft)", "DDR Citation", "Context"]
            st.dataframe(view, use_container_width=True, hide_index=True)


def _render_tab_flow_wob(df: pd.DataFrame) -> None:
    flow = df[df["metric_type"] == "flow_rate"]
    wob  = df[df["metric_type"] == "wob"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Circulation Flow Rate (gpm)**")
        if flow.empty:
            st.info("No flow rate data.")
        else:
            st.caption(
                f"{len(flow)} records  ·  "
                f"p50 {flow['value'].median():.0f} gpm  ·  max {flow['value'].max():.0f} gpm"
            )
            st.plotly_chart(
                _box_by_phase(flow, "Flow Rate by Phase", "gpm"),
                use_container_width=True,
            )

    with col2:
        st.markdown("**Weight on Bit (klbs)**")
        if wob.empty:
            st.info("No WOB data extracted.")
        else:
            st.caption(
                f"{len(wob)} records  ·  "
                f"p50 {wob['value'].median():.0f} klbs  ·  max {wob['value'].max():.0f} klbs"
            )
            st.plotly_chart(
                _box_by_phase(wob, "Weight on Bit by Phase", "klbs"),
                use_container_width=True,
            )


def page_drilling_metrics() -> None:
    df = load_drilling_metrics()

    hdr = load_field_headers()
    rig = hdr["rig_name"].dropna().mode().iloc[0] if not hdr.empty and "rig_name" in hdr.columns else "Rig"

    st.header(f"Drilling Metrics — {rig}")
    st.caption(
        f"Performance KPIs extracted from DDR operational text ({len(df):,} records).  "
        "Each data point traces back to a specific DDR report and page via the DDR Citation column."
    )

    if df.empty:
        st.warning(
            "No metrics file found. Run:  \n"
            "`python scripts/extract_drilling_metrics.py`",
            icon="⚠️",
        )
        return

    _render_kpis(df)

    tab_trip, tab_rop, tab_run, tab_other = st.tabs([
        "🔄 Tripping Speed",
        "⛏ Rate of Penetration",
        "🔩 Running Speeds",
        "💧 Flow Rate & WOB",
    ])

    with tab_trip:
        _render_tab_tripping(df)

    with tab_rop:
        _render_tab_rop(df)

    with tab_run:
        _render_tab_running(df)

    with tab_other:
        _render_tab_flow_wob(df)
