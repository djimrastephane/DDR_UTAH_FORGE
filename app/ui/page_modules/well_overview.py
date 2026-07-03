from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from .constants import PHASE_ORDER, PHASE_COLOURS, PROCESSED_DIR
    from .loaders import load_personnel, generate_well_narrative
    from .utils import (
        _apply_chart_theme, _sea_state, _beaufort_colour, _phase_date_ranges, _t2h,
    )
except ImportError:
    from constants import PHASE_ORDER, PHASE_COLOURS, PROCESSED_DIR         # type: ignore[no-redef]
    from loaders import load_personnel, generate_well_narrative              # type: ignore[no-redef]
    from utils import (                                                      # type: ignore[no-redef]
        _apply_chart_theme, _sea_state, _beaufort_colour, _phase_date_ranges, _t2h,
    )

_ui_root = Path(__file__).resolve().parents[3]
if str(_ui_root / "src") not in sys.path:
    sys.path.insert(0, str(_ui_root / "src"))

from ddr_rag.vocab import label_phase
from ddr_rag.npt_classifier import CATEGORY_LABELS, CATEGORY_COLOURS


def build_npt_interval_chart(day_ops: pd.DataFrame, date_label: str) -> "go.Figure":
    if day_ops.empty:
        fig = go.Figure()
        fig.add_annotation(text="No operations for this date.",
                           showarrow=False, font=dict(size=13))
        return fig

    # Only add 24 when adjusted_start is >12h behind adjusted_prev_end
    # (genuine midnight rollover, not re-triggering on each op).
    rows = day_ops.reset_index(drop=True).copy()

    starts, ends, labels = [], [], []
    adj_prev_end = 0.0
    offset       = 0.0

    for _, r in rows.iterrows():
        sh = _t2h(str(r.get("start_time", "00:00")))
        eh = _t2h(str(r.get("end_time",   "00:00")))

        sh_adj = sh + offset
        # If adjusted start is >12h behind previous adjusted end → midnight crossed
        if sh_adj + 12 < adj_prev_end:
            offset += 24
            sh_adj  = sh + offset

        eh_adj = (24.0 + offset) if eh == 0.0 else (eh + offset)

        starts.append(sh_adj)
        ends.append(eh_adj)
        adj_prev_end = eh_adj

        labels.append(f"{r.get('start_time','')}–{r.get('end_time','')}")

    rows["_start_h"] = starts
    rows["_end_h"]   = ends
    rows["_dur_h"]   = rows["_end_h"] - rows["_start_h"]
    rows["_label"]   = labels
    rows["_row"]     = range(len(rows))   # y-axis position

    x_max = max(30.0, float(rows["_end_h"].max()) + 0.5)

    fig = go.Figure()

    used_categories: dict[str, str] = {}
    for _, r in rows.iterrows():
        is_npt = bool(r.get("is_npt", False))
        cat  = str(r.get("npt_category", "other_npt"))

        if is_npt:
            colour = CATEGORY_COLOURS.get(cat, "#F44336")
            used_categories[cat] = colour
        else:
            colour = "#43A047"

        op_label = str(r.get("op_code_label", r.get("op_code", "")))
        npt_label = str(r.get("npt_cat_label", ""))
        op_text  = str(r.get("operation_text", ""))[:180]
        class_label = "Flagged NPT" if is_npt else "Normal operation"

        fig.add_trace(go.Bar(
            x=[float(r["_dur_h"])],
            y=[int(r["_row"])],
            base=[float(r["_start_h"])],
            orientation="h",
            marker_color=colour,
            marker_line_width=0.5,
            marker_line_color="rgba(255,255,255,0.4)",
            showlegend=False,
            hovertemplate=(
                f"<b>{r['_label']}</b>  ({r['_dur_h']:.2f}h)<br>"
                f"Op type: {op_label}<br>"
                f"Classification: {class_label}"
                + (f"<br>NPT type: {npt_label}" if is_npt and npt_label else "")
                + f"<br><i>{op_text}</i>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="overlay",
        height=max(280, len(rows) * 22 + 60),
        margin=dict(l=10, r=10, t=40, b=40),
        title=dict(text=f"Operations — {date_label}", font=dict(size=13)),
        xaxis=dict(
            title="Hour of day",
            range=[0, x_max],
            tickvals=list(range(0, int(x_max) + 1, 3)),
            ticktext=[f"{h % 24:02d}:00" for h in range(0, int(x_max) + 1, 3)],
        ),
        yaxis=dict(
            tickvals=list(range(len(rows))),
            ticktext=rows["_label"].tolist(),
            tickfont=dict(size=9),
            autorange="reversed",
        ),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FAFAFA",
    )

    if x_max > 24:
        fig.add_vline(x=24, line_dash="dash", line_color="#999",
                      annotation_text="Midnight", annotation_position="top",
                      annotation_font=dict(size=9, color="#999"))

    legend_items = {"Normal operation": "#43A047"}
    legend_items.update({
        CATEGORY_LABELS.get(cat, cat): colour
        for cat, colour in sorted(used_categories.items())
    })
    for name, col in legend_items.items():
        fig.add_trace(go.Bar(
            x=[None], y=[None], orientation="h",
            name=name, marker_color=col, showlegend=True,
        ))
    fig.update_layout(
        legend=dict(orientation="h", y=-0.12, font=dict(size=10)),
    )

    return fig


def _render_kpis(ops: pd.DataFrame, hdr: pd.DataFrame,
                 planned_time: pd.DataFrame | None) -> None:
    total_h  = float(ops["duration_hr"].sum())
    npt_h    = float(ops.loc[ops["is_npt"], "duration_hr"].sum())
    npt_pct  = 100 * npt_h / total_h if total_h else 0
    max_depth  = float(hdr["end_depth_num"].dropna().max() or 0)
    n_days     = int(hdr["report_date_parsed"].nunique())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reporting Days",   f"{n_days}")
    c2.metric("Operation Hours",  f"{total_h:.0f}h")
    c3.metric("Max Depth (MD)",   f"{max_depth:,.0f} ft")
    c4.metric("Flagged NPT",      f"{npt_h:.0f}h", f"{npt_pct:.1f}% of reported time")

    narrative = generate_well_narrative(ops, hdr, planned_time)
    if narrative:
        st.info(narrative)

    st.divider()


def _render_npt_phase_bar(ops: pd.DataFrame) -> None:
    left, right = st.columns([3, 2])

    phase_rows = []
    for phase in PHASE_ORDER:
        grp = ops[ops["phase"] == phase]
        if grp.empty:
            continue
        t  = float(grp["duration_hr"].sum())
        nt = float(grp.loc[grp["is_npt"], "duration_hr"].sum())
        phase_rows.append(dict(Phase=label_phase(phase), prod=t-nt, npt=nt, total=t,
                               npt_pct=round(100*nt/t, 1) if t else 0))
    ps = pd.DataFrame(phase_rows)

    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(y=ps["Phase"], x=ps["prod"], name="Productive",
                             orientation="h", marker_color="#4CAF50", opacity=0.85))
        fig.add_trace(go.Bar(y=ps["Phase"], x=ps["npt"],  name="NPT",
                             orientation="h", marker_color="#F44336", opacity=0.85))
        fig.update_layout(barmode="stack", height=300, margin=dict(l=10,r=10,t=30,b=10),
                          xaxis_title="Hours", title="Hours by Phase",
                          legend=dict(orientation="h", y=-0.3),
                          plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA")
        st.plotly_chart(_apply_chart_theme(fig), use_container_width=True)

    with right:
        ps_d = ps[["Phase","total","npt","npt_pct"]].sort_values(
            "npt_pct", ascending=False
        ).rename(columns={"total":"Total h", "npt":"NPT h", "npt_pct":"NPT pct"})
        ps_d["Total h"] = ps_d["Total h"].map("{:.0f}h".format)
        ps_d["NPT h"]   = ps_d["NPT h"].map("{:.0f}h".format)
        ps_d["NPT pct"] = ps_d["NPT pct"].map("{:.0f}%".format)
        st.caption("Phase NPT ranking")
        st.dataframe(ps_d, hide_index=True, use_container_width=True)


def _render_npt_trajectory(ops: pd.DataFrame, planned_time: pd.DataFrame) -> None:
    # Filter out implausibly high late-well NPT% (parse artefacts where
    # only T hours were captured, making the ratio look like 99%+)
    pt = planned_time[
        planned_time["cumulative_npt_pct"].notna() &
        ~((planned_time["cumulative_npt_pct"] > 95) &
          (planned_time["cumulative_hrs"].fillna(0) > 500))
    ].copy()
    if pt.empty:
        return

    st.subheader("Cumulative NPT Trajectory")
    st.caption(
        "Running well efficiency from job start. "
        "Lower is better. Shaded bands show drilling phase."
    )

    phase_ranges_pt = _phase_date_ranges(ops)
    phase_colours_pt = {
        "MIRU":   "rgba(158,158,158,0.13)", "COND1":  "rgba(121,85,72,0.13)",
        "INTRM1": "rgba(33,150,243,0.10)",  "INTRM2": "rgba(33,150,243,0.16)",
        "PROD1":  "rgba(244,67,54,0.09)",   "COMPZN": "rgba(76,175,80,0.09)",
    }
    fig_pt = go.Figure()

    _pt_abbrev = {
        "MIRU":   "MIRU", "COND1":  "COND",
        "INTRM1": "INT1", "INTRM2": "INT2",
        "PROD1":  "PROD", "COMPZN": "COMP",
    }
    for phase_key in PHASE_ORDER:
        if phase_key not in phase_ranges_pt:
            continue
        p0, p1 = phase_ranges_pt[phase_key]
        fig_pt.add_vrect(
            x0=p0, x1=p1,
            fillcolor=phase_colours_pt.get(phase_key, "rgba(200,200,200,0.1)"),
            layer="below", line_width=0,
        )
        mid = p0 + (p1 - p0) / 2
        fig_pt.add_annotation(
            x=mid, y=1.04, yref="paper",
            text=_pt_abbrev.get(phase_key, phase_key),
            showarrow=False, xanchor="center",
            font=dict(size=7, color="#888"),
        )

    fig_pt.add_hline(y=50, line_dash="dot", line_color="#9E9E9E",
                     opacity=0.6,
                     annotation_text="50%", annotation_position="right",
                     annotation_font=dict(size=9, color="#9E9E9E"))

    fig_pt.add_trace(go.Scatter(
        x=pt["report_date"],
        y=pt["cumulative_npt_pct"],
        mode="lines",
        line=dict(width=2.5, color="#F44336"),
        fill="tozeroy",
        fillcolor="rgba(244,67,54,0.08)",
        name="Cumulative NPT %",
        hovertemplate="<b>%{x|%d %b}</b><br>Cumulative NPT: %{y:.1f}%<extra></extra>",
    ))

    _phase_ranges_pt = _phase_date_ranges(ops)
    best_row      = pt.loc[pt["cumulative_npt_pct"].idxmin()]
    above50       = pt[pt["cumulative_npt_pct"] > 50]
    prod1_start   = _phase_ranges_pt.get("PROD1", (None, None))[0]
    comp_start    = _phase_ranges_pt.get("COMPZN", (None, None))[0]
    cond1_ops     = ops[ops["phase"] == "COND1"]
    cond1_worst_day = (
        cond1_ops.groupby("report_date_parsed")["is_npt"].mean().idxmax()
        if not cond1_ops.empty else None
    )

    _event_style = dict(
        showarrow=True, arrowhead=2, arrowsize=0.8,
        bgcolor="rgba(255,255,255,0.88)", borderwidth=1,
        font=dict(size=9),
    )

    fig_pt.add_annotation(
        x=best_row["report_date"], y=best_row["cumulative_npt_pct"],
        text=f"▼ Best: {best_row['cumulative_npt_pct']:.0f}%",
        arrowcolor="#2E7D32", bordercolor="#2E7D32",
        ax=0, ay=-36, font=dict(size=10, color="#2E7D32"),
        **{k:v for k,v in _event_style.items() if k not in ("font",)},
    )
    if prod1_start is not None:
        row_p1 = pt[pt["report_date"] >= pd.Timestamp(prod1_start)]
        if not row_p1.empty:
            fig_pt.add_annotation(
                x=prod1_start,
                y=float(row_p1.iloc[0]["cumulative_npt_pct"]),
                text="Metallic debris recovery<br>(magnet/junk mill runs)",
                arrowcolor="#E65100", bordercolor="#E65100",
                ax=0, ay=32, font=dict(size=9, color="#E65100"),
                **{k:v for k,v in _event_style.items() if k not in ("font",)},
            )
    if comp_start is not None:
        row_cp = pt[pt["report_date"] >= pd.Timestamp(comp_start)]
        if not row_cp.empty:
            fig_pt.add_annotation(
                x=comp_start,
                y=float(row_cp.iloc[0]["cumulative_npt_pct"]),
                text="Completion start",
                arrowcolor="#1565C0", bordercolor="#1565C0",
                ax=0, ay=-36, font=dict(size=9, color="#1565C0"),
                **{k:v for k,v in _event_style.items() if k not in ("font",)},
            )
    if cond1_worst_day is not None:
        row_c1 = pt[pt["report_date"].dt.date == cond1_worst_day.date()]
        if not row_c1.empty:
            fig_pt.add_annotation(
                x=cond1_worst_day,
                y=float(row_c1.iloc[0]["cumulative_npt_pct"]),
                text="Conductor NPT peak",
                arrowcolor="#795548", bordercolor="#795548",
                ax=0, ay=32, font=dict(size=9, color="#795548"),
                **{k:v for k,v in _event_style.items() if k not in ("font",)},
            )

    fig_pt.update_layout(
        height=320,
        margin=dict(l=10, r=80, t=10, b=10),
        yaxis=dict(title="Cumulative NPT %", range=[0, 110]),
        plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA",
        showlegend=False,
    )
    st.plotly_chart(_apply_chart_theme(fig_pt), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Best efficiency",
              f"{best_row['cumulative_npt_pct']:.0f}% NPT",
              f"{best_row['report_date'].strftime('%d %b %Y')}")
    final_npt_pct = 100 * ops.loc[ops["is_npt"], "duration_hr"].sum() / max(ops["duration_hr"].sum(), 1)
    final_npt_h   = ops.loc[ops["is_npt"], "duration_hr"].sum()
    c2.metric("Final well NPT",
              f"{final_npt_pct:.0f}%",
              f"{final_npt_h:.0f}h NPT of {ops['duration_hr'].sum():.0f}h total")
    above50_count = (pt["cumulative_npt_pct"] > 50).sum()
    c3.metric("Days above 50% NPT",
              f"{above50_count} days",
              f"{100*above50_count/len(pt):.0f}% of reporting period",
              delta_color="inverse")


def _render_daily_cost(ops: pd.DataFrame, hdr: pd.DataFrame) -> None:
    cost_df = hdr[hdr["daily_cost_num"].notna() & (hdr["daily_cost_num"] > 0)].copy()
    if cost_df.empty:
        return

    st.subheader("Daily Cost — with Major Operational Events")

    avg_cost     = float(cost_df["daily_cost_num"].mean())
    peak_cost    = float(cost_df["daily_cost_num"].max())
    peak_cost_dt = cost_df.loc[cost_df["daily_cost_num"].idxmax(), "report_date_parsed"]
    spike_threshold = avg_cost * 1.6

    use_log = st.checkbox("Log scale (reduces spike compression)", value=False,
                          key="cost_log",
                          help="Use logarithmic y-axis to make smaller anomalies visible alongside the Aug spike")

    daily_npt_dom = (
        ops[ops["is_npt"]]
        .groupby(["report_date_parsed", "npt_category"])["duration_hr"].sum()
        .reset_index()
        .sort_values("duration_hr", ascending=False)
        .drop_duplicates("report_date_parsed")
        .set_index("report_date_parsed")
    )

    fig2 = go.Figure()

    _phase_ranges_cost = _phase_date_ranges(ops)
    _cost_phase_bg = {
        "MIRU":   "rgba(158,158,158,0.11)", "COND1":  "rgba(121,85,72,0.11)",
        "INTRM1": "rgba(33,150,243,0.08)",  "INTRM2": "rgba(33,150,243,0.12)",
        "PROD1":  "rgba(244,67,54,0.08)",   "COMPZN": "rgba(76,175,80,0.08)",
    }
    for phase_key in PHASE_ORDER:
        if phase_key not in _phase_ranges_cost:
            continue
        p0, p1 = _phase_ranges_cost[phase_key]
        fig2.add_vrect(
            x0=p0, x1=p1,
            fillcolor=_cost_phase_bg.get(phase_key, "rgba(200,200,200,0.08)"),
            layer="below", line_width=0,
            annotation_text=label_phase(phase_key).split("/")[0].strip(),
            annotation_position="top left",
            annotation_font=dict(size=8, color="#777"),
        )

    fig2.add_trace(go.Scatter(
        x=cost_df["report_date_parsed"], y=cost_df["daily_cost_num"] / 1000,
        mode="lines", fill="tozeroy",
        line=dict(color="#1565C0", width=1.8), fillcolor="rgba(33,150,243,0.12)",
        name="Daily cost",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>£%{y:,.0f}k<extra></extra>",
    ))
    fig2.add_hline(
        y=avg_cost / 1000, line_dash="dot", line_color="#FF8F00", line_width=1.5,
        annotation_text=f"Avg £{avg_cost/1000:,.0f}k/day",
        annotation_position="right",
        annotation_font=dict(size=10, color="#FF8F00"),
    )

    _daily_npt_hrs   = ops[ops["is_npt"]].groupby("report_date_parsed")["duration_hr"].sum()
    _daily_total_hrs = ops.groupby("report_date_parsed")["duration_hr"].sum()
    _cost_series     = cost_df.set_index("report_date_parsed")["daily_cost_num"]
    _rolling_med     = _cost_series.rolling(7, center=True, min_periods=3).median()

    _peak_npt_h   = float(_daily_npt_hrs.get(peak_cost_dt, 0))
    _peak_total   = float(_daily_total_hrs.get(peak_cost_dt, 0))
    _peak_npt_pct = int(100 * _peak_npt_h / max(_peak_total, 1))
    _peak_rolling = float(_rolling_med.get(peak_cost_dt, avg_cost))
    _peak_is_bulk = peak_cost > 3 * _peak_rolling
    _peak_note    = "⚠ Likely bulk/mobilisation charge" if _peak_is_bulk else f"NPT: {_peak_npt_pct}%"
    fig2.add_annotation(
        x=peak_cost_dt, y=peak_cost / 1000,
        text=f"Peak £{peak_cost/1e6:.1f}M<br><sub>{_peak_note}</sub>",
        showarrow=True, arrowhead=2, arrowcolor="#B71C1C",
        ax=0, ay=-42,
        font=dict(size=10, color="#B71C1C"),
        bgcolor="rgba(255,255,255,0.92)", bordercolor="#B71C1C", borderwidth=1.5,
    )

    shown_events: set[str] = set()
    for _, row in cost_df.nlargest(40, "daily_cost_num").iterrows():
        if row["daily_cost_num"] < spike_threshold:
            continue
        dt = row["report_date_parsed"]
        if dt not in daily_npt_dom.index:
            continue
        cat    = daily_npt_dom.loc[dt, "npt_category"]
        icon   = CATEGORY_LABELS.get(cat, cat.replace("_", " ").title())
        border = CATEGORY_COLOURS.get(cat, "#666666")
        key    = f"{cat}_{dt.month}"
        if key in shown_events or dt == peak_cost_dt:
            continue

        _npt_h   = float(_daily_npt_hrs.get(dt, 0))
        _total_h = float(_daily_total_hrs.get(dt, 0))
        _npt_pct = int(100 * _npt_h / max(_total_h, 1))
        _rolling_val = float(_rolling_med.get(dt, avg_cost))
        _bulk_flag   = row["daily_cost_num"] > 3 * _rolling_val
        _note = "⚠ Possible bulk charge" if _bulk_flag else f"NPT: {_npt_pct}% ({_npt_h:.0f}h)"
        _ann_text = f"{icon}<br><sub>{_note}</sub>"

        fig2.add_annotation(
            x=dt, y=row["daily_cost_num"] / 1000,
            text=_ann_text, showarrow=True, arrowhead=2, arrowsize=0.8,
            arrowcolor=border, ax=0, ay=-32,
            font=dict(size=10), bgcolor="rgba(255,255,255,0.90)",
            bordercolor=border, borderwidth=1.5,
        )
        shown_events.add(key)

    _y_type = "log" if use_log else "linear"
    fig2.update_layout(
        height=320, margin=dict(l=10, r=100, t=10, b=10),
        yaxis=dict(title="£k / day", type=_y_type),
        plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA",
    )
    st.plotly_chart(_apply_chart_theme(fig2), use_container_width=True)

    _compzn_dates = _phase_ranges_cost.get("COMPZN")
    if _compzn_dates:
        _comp_start_dt = pd.Timestamp(_compzn_dates[0])
        _comp_cost  = cost_df[cost_df["report_date_parsed"] >= _comp_start_dt]["daily_cost_num"].mean()
        _drill_cost = cost_df[cost_df["report_date_parsed"] <  _comp_start_dt]["daily_cost_num"].mean()
        _comp_label = label_phase("COMPZN").split("/")[0].strip()
        _comp_start_label = _comp_start_dt.strftime("%b %Y")
        st.caption(
            f"Each annotation shows the dominant NPT category and daily NPT hours for that spike. "
            f"**⚠ Likely bulk/mobilisation charge** flags days where cost is >3× the surrounding 7-day median — "
            f"these spikes probably represent equipment mobilisation or batch invoice posting rather than daily operational burn. "
            f"Average cost during {_comp_label} phase ({_comp_start_label} onwards): "
            f"£{_comp_cost/1000:,.0f}k/day vs £{_drill_cost/1000:,.0f}k/day pre-completion."
        )
    else:
        st.caption(
            "Each annotation shows the dominant NPT category and daily NPT hours. "
            "⚠ Likely bulk/mobilisation charge flags days where cost is >3× the surrounding 7-day median."
        )


def _render_depth_chart(ops: pd.DataFrame, hdr: pd.DataFrame) -> None:
    depth_df = (
        hdr[hdr["report_date_parsed"].notna() & hdr["end_depth_num"].notna() & (hdr["end_depth_num"] > 0)]
        .sort_values("report_date_parsed")
        .groupby("report_date_parsed", as_index=False)
        .agg(
            end_depth_num=("end_depth_num", "max"),
            report_type=("report_type", "last"),
            morning_report_ops=("morning_report_ops", "last"),
        )
    )
    if depth_df.empty:
        return

    st.subheader("Measured Depth Progression")

    phase_ranges = _phase_date_ranges(ops)
    max_depth = float(depth_df["end_depth_num"].max())
    depth_df["depth_delta"] = depth_df["end_depth_num"].diff()
    depth_df["advance_ft"] = depth_df["depth_delta"].clip(lower=0).fillna(0)
    advance_days = int((depth_df["advance_ft"] > 0.1).sum())
    hold_days = int(((depth_df["depth_delta"].fillna(0) <= 0.1) & (depth_df.index > 0)).sum())
    first_depth_dt = depth_df["report_date_parsed"].min()
    max_depth_dt = depth_df.loc[depth_df["end_depth_num"].idxmax(), "report_date_parsed"]

    def _rgba(hex_colour: str, alpha: float = 0.12) -> str:
        colour = str(hex_colour or "#BDBDBD").lstrip("#")
        if len(colour) != 6:
            return f"rgba(189,189,189,{alpha})"
        r, g, b = (int(colour[i:i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"

    fig = go.Figure()
    for phase in PHASE_ORDER:
        if phase not in phase_ranges:
            continue
        p_start, p_end = phase_ranges[phase]
        if pd.isna(p_start) or pd.isna(p_end):
            continue
        fig.add_vrect(
            x0=p_start,
            x1=p_end + pd.Timedelta(days=1),
            fillcolor=_rgba(PHASE_COLOURS.get(phase, "#BDBDBD"), 0.10),
            layer="below",
            line_width=0,
            annotation_text=label_phase(phase),
            annotation_position="top left",
            annotation_font=dict(size=9, color=PHASE_COLOURS.get(phase, "#555")),
        )

    fig.add_trace(go.Scatter(
        x=depth_df["report_date_parsed"],
        y=depth_df["end_depth_num"],
        mode="lines+markers",
        line=dict(color="#263238", width=3, shape="hv"),
        marker=dict(size=6, color="#00838F", line=dict(color="#263238", width=0.8)),
        customdata=depth_df[["advance_ft", "morning_report_ops"]].values,
        hovertemplate=(
            "<b>%{x|%d %b %Y}</b><br>"
            "End depth: %{y:,.0f} ft MD<br>"
            "Daily advance: %{customdata[0]:,.0f} ft<br>"
            "<i>%{customdata[1]}</i><extra></extra>"
        ),
        name="End depth",
    ))

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=20, t=10, b=10),
        yaxis=dict(title="Measured depth (ft MD)", range=[max_depth + 600, 0]),
        xaxis=dict(title="", tickformat="%d %b"),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FAFAFA",
        showlegend=False,
    )
    st.plotly_chart(_apply_chart_theme(fig), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("First Depth Report", first_depth_dt.strftime("%d %b %Y"))
    c2.metric("Max Depth", f"{max_depth:,.0f} ft MD", max_depth_dt.strftime("%d %b %Y"))
    c3.metric("Advance Days", f"{advance_days}")
    c4.metric("Hold Days", f"{hold_days}")
    st.caption(
        "Depth is plotted only from DDR header end-depth fields. Early rig move-in reports without "
        "reported MD are omitted from the line rather than interpolated."
    )


def _render_weather(ops: pd.DataFrame, weather: pd.DataFrame) -> None:
    st.subheader("Weather Conditions — Wind & Wave (Apr–Jun 2024)")
    _wx_count    = len(weather)
    _total_count = len([d for d in PROCESSED_DIR.iterdir()
                        if d.is_dir() and (d / "ddr_facts.parquet").exists()])
    st.caption(
        f"Weather measurements recorded in {_wx_count} of {_total_count} DDRs. "
        "Reporting stopped when drilling entered the deep production section in July 2024."
    )
    wx_col1, wx_col2 = st.columns([1, 1])
    show_swell = wx_col1.checkbox("Show swell height", value=False,
                                   help="Dotted line for ocean swell — hide to reduce clutter")
    show_npt   = wx_col2.checkbox("Overlay daily NPT%", value=True,
                                   help="Faint red bars showing NPT% — allows weather-NPT correlation")

    daily_info = (
        ops.groupby("report_date_parsed")
        .apply(lambda g: pd.Series({
            "npt_pct":  100 * g.loc[g["is_npt"], "duration_hr"].sum()
                        / max(g["duration_hr"].sum(), 1),
            "phase":    g["phase"].mode()[0] if not g["phase"].empty else "",
        }), include_groups=False)
        .reset_index()
        .rename(columns={"report_date_parsed": "report_date"})
    )
    w_npt = weather.merge(daily_info, on="report_date", how="left")
    w_npt["phase_label"] = w_npt["phase"].apply(
        lambda p: label_phase(p) if p else "—"
    )

    fig_wx = go.Figure()

    phase_ranges_wx = _phase_date_ranges(ops)
    phase_colours_wx = {
        "MIRU":   "rgba(158,158,158,0.13)",
        "COND1":  "rgba(121,85,72,0.13)",
        "INTRM1": "rgba(33,150,243,0.10)",
        "INTRM2": "rgba(33,150,243,0.16)",
        "PROD1":  "rgba(244,67,54,0.09)",
    }
    wx_start = weather["report_date"].min()
    wx_end   = weather["report_date"].max()
    for phase_key in ["MIRU", "COND1", "INTRM1", "INTRM2", "PROD1"]:
        if phase_key not in phase_ranges_wx:
            continue
        p0, p1 = phase_ranges_wx[phase_key]
        p0 = max(p0, wx_start)
        p1 = min(p1, wx_end)
        if p0 >= p1:
            continue
        fig_wx.add_vrect(
            x0=p0, x1=p1,
            fillcolor=phase_colours_wx.get(phase_key, "rgba(200,200,200,0.1)"),
            layer="below", line_width=0,
            annotation_text=label_phase(phase_key).split("/")[0].strip(),
            annotation_position="top left",
            annotation_font=dict(size=8, color="#666"),
        )

    if show_npt:
        fig_wx.add_trace(go.Bar(
            x=w_npt["report_date"],
            y=w_npt["npt_pct"],
            name="NPT %",
            marker_color="rgba(244,67,54,0.18)",
            yaxis="y3",
            hovertemplate="NPT: %{y:.0f}%<extra></extra>",
        ))

    fig_wx.add_trace(go.Bar(
        x=w_npt["report_date"],
        y=w_npt["wind_speed_kn"],
        name="Wind speed (kn)",
        marker_color=[_beaufort_colour(v) for v in w_npt["wind_speed_kn"]],
        yaxis="y",
        customdata=w_npt[["phase_label", "npt_pct"]].values,
        hovertemplate=(
            "<b>%{x|%d %b}</b><br>"
            "Phase: %{customdata[0]}<br>"
            "Wind: %{y:.0f} kn<br>"
            "NPT: %{customdata[1]:.0f}%"
            "<extra></extra>"
        ),
    ))

    fig_wx.add_trace(go.Scatter(
        x=w_npt["report_date"],
        y=w_npt["wave_height_ft"],
        name="Wave height (ft)",
        mode="lines+markers",
        line=dict(color="#0288D1", width=2),
        marker=dict(size=4),
        yaxis="y2",
        customdata=w_npt[["phase_label"]].values,
        hovertemplate=(
            "Wave: %{y:.1f} ft  [%{customdata[0]}]"
            "<extra></extra>"
        ),
    ))

    if show_swell and "swell_height_ft" in weather.columns:
        fig_wx.add_trace(go.Scatter(
            x=w_npt["report_date"],
            y=w_npt["swell_height_ft"],
            name="Swell (ft)",
            mode="lines",
            line=dict(color="#01579B", width=1.5, dash="dot"),
            yaxis="y2",
            hovertemplate="Swell: %{y:.1f} ft<extra></extra>",
        ))

    fig_wx.add_hline(y=8, line_dash="dash", line_color="#0288D1",
                     opacity=0.4, yref="y2",
                     annotation_text="Rough (8 ft)", annotation_position="right",
                     annotation_font=dict(size=9, color="#0288D1"))
    fig_wx.add_hline(y=28, line_dash="dash", line_color="#EF6C00",
                     opacity=0.4,
                     annotation_text="Gale (28 kn)", annotation_position="right",
                     annotation_font=dict(size=9, color="#EF6C00"))

    fig_wx.update_layout(
        barmode="overlay",
        height=330,
        margin=dict(l=10, r=90, t=10, b=10),
        plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA",
        legend=dict(orientation="h", y=-0.22),
        yaxis=dict(title="Wind speed (kn)", range=[0, 55]),
        yaxis2=dict(title="Wave / Swell (ft)", overlaying="y",
                    side="right", range=[0, 25]),
        yaxis3=dict(title="NPT %", overlaying="y", side="right",
                    range=[0, 300], showticklabels=False, showgrid=False),
        hovermode="x unified",
    )
    st.plotly_chart(_apply_chart_theme(fig_wx), use_container_width=True)

    phase_wx = (
        w_npt[w_npt["phase"].notna() & (w_npt["phase"] != "")]
        .groupby("phase")
        .agg(
            Days      =("report_date", "count"),
            Avg_wind  =("wind_speed_kn", "mean"),
            Max_wind  =("wind_speed_kn", "max"),
            Avg_wave  =("wave_height_ft", "mean"),
            Max_wave  =("wave_height_ft", "max"),
            Avg_NPT   =("npt_pct", "mean"),
        )
        .reset_index()
    )
    phase_wx["Phase"] = phase_wx["phase"].apply(label_phase)
    phase_wx = phase_wx.drop(columns="phase").rename(columns={
        "Days": "Days", "Avg_wind": "Avg wind (kn)", "Max_wind": "Max wind (kn)",
        "Avg_wave": "Avg wave (ft)", "Max_wave": "Max wave (ft)", "Avg_NPT": "Avg NPT %",
    })
    phase_order_map = {label_phase(p): i for i, p in enumerate(PHASE_ORDER)}
    phase_wx["_ord"] = phase_wx["Phase"].map(phase_order_map).fillna(99)
    phase_wx = phase_wx.sort_values("_ord").drop(columns="_ord")
    for col in ["Avg wind (kn)", "Max wind (kn)"]:
        phase_wx[col] = phase_wx[col].round(1)
    for col in ["Avg wave (ft)", "Max wave (ft)", "Avg NPT %"]:
        phase_wx[col] = phase_wx[col].round(1)

    st.caption("Weather summary by drilling phase:")
    st.dataframe(phase_wx, hide_index=True, use_container_width=True)

    st.caption(
        "Bar colour: "
        "<span style='color:#B3E5FC'>■</span> Light (<7kn) &nbsp;"
        "<span style='color:#29B6F6'>■</span> Moderate (7–13kn) &nbsp;"
        "<span style='color:#F9A825'>■</span> Fresh (14–21kn) &nbsp;"
        "<span style='color:#EF6C00'>■</span> Strong (22–27kn) &nbsp;"
        "<span style='color:#B71C1C'>■</span> Gale+ (28kn+) &nbsp;"
        "· Red bars = NPT% (faint) · Blue line = wave height",
        unsafe_allow_html=True,
    )


def _render_vessels(vessels: pd.DataFrame) -> None:
    st.subheader("Vessel Logistics")
    st.caption(
        "Supply, standby, and helicopter traffic by month. "
        "Coverage matches DDRs where the Support Vessels section was parseable (104 DDRs)."
    )

    v = vessels.copy()
    v["month"] = v["report_date"].dt.to_period("M").astype(str)
    monthly = (
        v.groupby(["month", "vessel_type"])
        .size()
        .reset_index(name="count")
    )

    type_colour = {
        "Supply Vessel":   "#2196F3",
        "Standby Vessel":  "#4CAF50",
        "Helicopter":      "#FF9800",
        "SAR Helicopter":  "#9C27B0",
    }
    months_all = sorted(monthly["month"].unique())

    fig_v = go.Figure()
    for vtype in ["Supply Vessel", "Standby Vessel", "Helicopter", "SAR Helicopter"]:
        sub = monthly[monthly["vessel_type"] == vtype]
        if sub.empty:
            continue
        sub_full = (
            pd.DataFrame({"month": months_all})
            .merge(sub[["month", "count"]], on="month", how="left")
            .fillna(0)
        )
        fig_v.add_trace(go.Bar(
            x=sub_full["month"],
            y=sub_full["count"],
            name=vtype,
            marker_color=type_colour.get(vtype, "#999"),
            hovertemplate=f"<b>%{{x}}</b><br>{vtype}: %{{y}}<extra></extra>",
        ))

    fig_v.update_layout(
        barmode="stack",
        height=260,
        margin=dict(l=10, r=10, t=10, b=40),
        xaxis_title="Month",
        yaxis_title="Vessel visits / helicopter calls",
        plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA",
        legend=dict(orientation="h", y=-0.3),
    )
    st.plotly_chart(_apply_chart_theme(fig_v), use_container_width=True)

    left_v, right_v = st.columns(2)
    with left_v:
        supply = (
            v[v["vessel_type"] == "Supply Vessel"]
            .groupby("vessel_name")
            .size()
            .sort_values(ascending=False)
            .reset_index(name="calls")
            .reset_index(drop=True)
        )
        supply.columns = ["Vessel", "Calls"]
        st.caption("Supply vessels — total calls")
        st.table(supply.set_index("Vessel"))
    with right_v:
        standby = (
            v[v["vessel_type"] == "Standby Vessel"]
            .groupby("vessel_name")
            .size()
            .sort_values(ascending=False)
            .reset_index(name="days")
            .reset_index(drop=True)
        )
        standby.columns = ["Vessel", "Days"]
        st.caption("Standby vessels — days on station")
        st.table(standby.set_index("Vessel"))


def _render_personnel(ops: pd.DataFrame, hdr: pd.DataFrame) -> None:
    _pers_df    = load_personnel()
    _hc = hdr[["report_date_parsed", "head_count"]].copy()
    _hc["head_count_num"] = pd.to_numeric(_hc["head_count"], errors="coerce")
    _hc = _hc.dropna(subset=["head_count_num"]).sort_values("report_date_parsed")
    if _hc.empty and _pers_df.empty:
        return

    st.subheader("Personnel on Board (POB)")
    st.caption(
        "Daily total headcount from DDR header and company-level breakdown from "
        "personnel data table. Useful for crew planning on future wells."
    )

    _phase_ranges = _phase_date_ranges(ops)

    _phase_day = (
        ops.groupby("report_date_parsed")["phase"]
        .agg(lambda x: x.mode().iloc[0] if len(x) else None)
        .reset_index()
    )
    _hc = _hc.merge(_phase_day, on="report_date_parsed", how="left")

    _PHASE_COL = {
        "MIRU":   "#78909C", "COND1": "#8D6E63", "INTRM1": "#42A5F5",
        "INTRM2": "#66BB6A", "PROD1": "#FFA726", "COMPZN": "#AB47BC",
    }

    col_chart, col_table = st.columns([3, 2], gap="large")

    with col_chart:
        fig_pob = go.Figure()

        for ph, (d0, d1) in _phase_ranges.items():
            fig_pob.add_vrect(
                x0=d0, x1=d1,
                fillcolor=_PHASE_COL.get(ph, "#aaa"), opacity=0.06,
                layer="below", line_width=0,
            )
            mid = d0 + (d1 - d0) / 2
            fig_pob.add_annotation(
                x=mid, xref="x", y=1.04, yref="paper",
                text=f"<b>{label_phase(ph).split('/')[0].strip()}</b>",
                showarrow=False,
                font=dict(size=8, color=_PHASE_COL.get(ph, "#555")),
            )

        for ph, grp in _hc.groupby("phase"):
            if grp.empty:
                continue
            avg = grp["head_count_num"].mean()
            d0, d1 = _phase_ranges.get(ph, (grp["report_date_parsed"].min(),
                                             grp["report_date_parsed"].max()))
            fig_pob.add_shape(
                type="line", x0=d0, x1=d1,
                y0=avg, y1=avg,
                line=dict(color=_PHASE_COL.get(ph, "#aaa"),
                          width=1.5, dash="dot"),
                layer="above",
            )

        fig_pob.add_trace(go.Scatter(
            x=_hc["report_date_parsed"],
            y=_hc["head_count_num"],
            mode="lines+markers",
            name="Total POB",
            line=dict(color="#1565C0", width=2),
            marker=dict(size=4, color=_hc["phase"].map(_PHASE_COL).fillna("#999")),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>POB: %{y:.0f}<extra></extra>",
        ))

        fig_pob.update_layout(
            height=300,
            yaxis=dict(title="Personnel on Board", rangemode="tozero"),
            xaxis_title="",
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False,
            margin=dict(l=50, r=20, t=40, b=40),
        )
        _apply_chart_theme(fig_pob)
        st.plotly_chart(fig_pob, use_container_width=True)

    with col_table:
        st.markdown("**Phase POB summary — planning reference**")
        plan_rows = []
        for ph in PHASE_ORDER:
            grp = _hc[_hc["phase"] == ph]
            if grp.empty:
                continue
            peak_row = grp.loc[grp["head_count_num"].idxmax()]
            plan_rows.append({
                "Phase":    label_phase(ph).split("/")[0].strip(),
                "Avg POB":  f"{grp['head_count_num'].mean():.0f}",
                "Min":      f"{grp['head_count_num'].min():.0f}",
                "Peak":     f"{grp['head_count_num'].max():.0f}",
                "Peak date": peak_row["report_date_parsed"].strftime("%d %b"),
                "Days":     len(grp),
            })
        if plan_rows:
            st.dataframe(
                pd.DataFrame(plan_rows),
                hide_index=True,
                use_container_width=True,
            )

    if not _pers_df.empty:
        st.markdown("**Contractor presence by phase**")
        st.caption(
            "Company daily headcount aggregated per phase. "
            "Shows which contractors arrive and depart at phase transitions."
        )

        _phase_day_r = _phase_day.rename(columns={"report_date_parsed": "report_date_dt"})
        _pers_ph = _pers_df.merge(_phase_day_r, on="report_date_dt", how="left")

        top_cos = (
            _pers_ph.groupby("company")["count"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .index.tolist()
        )

        pivot = (
            _pers_ph[_pers_ph["company"].isin(top_cos)]
            .groupby(["phase", "company"])["count"]
            .mean()
            .unstack(fill_value=0)
            .reindex([p for p in PHASE_ORDER if p in _pers_ph["phase"].unique()])
        )
        pivot.index = [label_phase(p).split("/")[0].strip() for p in pivot.index]

        _CO_COLOURS = [
            "#1565C0", "#2E7D32", "#F57C00", "#7B1FA2", "#C62828",
            "#00695C", "#4527A0", "#558B2F", "#6D4C41", "#0277BD",
        ]

        fig_co = go.Figure()
        for i, co in enumerate(top_cos):
            if co not in pivot.columns:
                continue
            fig_co.add_trace(go.Bar(
                name=co,
                x=pivot.index,
                y=pivot[co],
                marker_color=_CO_COLOURS[i % len(_CO_COLOURS)],
                hovertemplate=f"<b>{co}</b><br>%{{x}}<br>Avg daily: %{{y:.1f}}<extra></extra>",
            ))

        fig_co.update_layout(
            barmode="stack",
            height=300,
            yaxis_title="Avg daily headcount",
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(
                orientation="h", y=-0.35,
                font=dict(size=10, color="rgb(40,40,40)"),
            ),
            margin=dict(l=50, r=20, t=10, b=100),
        )
        _apply_chart_theme(fig_co)
        st.plotly_chart(fig_co, use_container_width=True)


def _render_drill_down(ops: pd.DataFrame, hdr: pd.DataFrame,
                       weather: pd.DataFrame | None) -> None:
    st.subheader("Daily Drill-Down")
    st.caption("Select a date to see what drove cost and NPT on that day.")

    date_options = sorted(ops["report_date_parsed"].dropna().dt.date.unique())
    sel_date = st.selectbox("Select date", date_options,
                            index=len(date_options)//2,
                            format_func=lambda d: d.strftime("%d %b %Y"),
                            key="well_overview_sel_date")

    day_ops = ops[ops["report_date_parsed"].dt.date == sel_date]
    if day_ops.empty:
        st.info("No operations found for selected date.")
        return

    d_total = day_ops["duration_hr"].sum()
    d_npt   = day_ops.loc[day_ops["is_npt"], "duration_hr"].sum()
    d_cost  = hdr.loc[hdr["report_date_parsed"].dt.date == sel_date, "daily_cost_num"]
    d_cost_val = f"£{d_cost.values[0]/1000:,.0f}k" if not d_cost.empty and not pd.isna(d_cost.values[0]) else "—"

    d_wx = (weather[weather["report_date"].dt.date == sel_date].iloc[0]
            if weather is not None and not weather.empty
            and any(weather["report_date"].dt.date == sel_date) else None)

    if d_wx is not None:
        cc1, cc2, cc3, cc4, cc5, cc6 = st.columns(6)
        cc1.metric("Daily cost",   d_cost_val)
        cc2.metric("Total hours",  f"{d_total:.1f}h")
        cc3.metric("NPT hours",    f"{d_npt:.1f}h")
        cc4.metric("Phase",        label_phase(day_ops["phase"].mode()[0]) if not day_ops["phase"].empty else "—")
        cc5.metric("Wind speed",   f"{d_wx.get('wind_speed_kn', '—'):.0f} kn"
                   if pd.notna(d_wx.get("wind_speed_kn")) else "—")
        cc6.metric("Wave height",  f"{d_wx.get('wave_height_ft', '—'):.1f} ft  ({_sea_state(d_wx.get('wave_height_ft'))})"
                   if pd.notna(d_wx.get("wave_height_ft")) else "—")
    else:
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Daily cost",   d_cost_val)
        cc2.metric("Total hours",  f"{d_total:.1f}h")
        cc3.metric("NPT hours",    f"{d_npt:.1f}h")
        cc4.metric("Phase",        label_phase(day_ops["phase"].mode()[0]) if not day_ops["phase"].empty else "—")

    fig_gantt = build_npt_interval_chart(day_ops, sel_date.strftime("%d %b %Y"))
    st.plotly_chart(_apply_chart_theme(fig_gantt), use_container_width=True)

    with st.expander("Show operations table", expanded=False):
        show_cols = ["start_time", "end_time", "duration_hr", "op_code_label",
                     "is_npt", "npt_cat_label", "operation_text"]
        show_cols = [c for c in show_cols if c in day_ops.columns]
        renamed = {
            "start_time": "Start", "end_time": "End", "duration_hr": "Dur (h)",
            "op_code_label": "Op type", "is_npt": "Classification",
            "npt_cat_label": "NPT type", "operation_text": "Operation",
        }
        disp = day_ops[show_cols].rename(columns=renamed).copy()
        if "Classification" in disp.columns:
            disp["Classification"] = disp["Classification"].map(
                {True: "Flagged NPT", False: "Normal operation"}
            )

        def _row_colour(row: pd.Series) -> list[str]:
            if row.get("Classification") == "Flagged NPT":
                return ["background-color:#FFCDD2; color:#212121"] * len(row)
            return [""] * len(row)

        st.dataframe(
            disp.style.apply(_row_colour, axis=1),
            hide_index=True, use_container_width=True, height=300,
            column_config={"Operation": st.column_config.TextColumn(width="large"),
                           "Dur (h)": st.column_config.NumberColumn(format="%.2f")},
        )


def page_well_overview(ops: pd.DataFrame, hdr: pd.DataFrame,
                       weather: pd.DataFrame | None = None,
                       planned_time: pd.DataFrame | None = None,
                       vessels: pd.DataFrame | None = None) -> None:
    _rig   = hdr["rig_name"].dropna().mode()
    _field = hdr["field_name"].dropna().mode()
    _ctx   = " · ".join(x for x in [
        _field.iloc[0].title() if not _field.empty else "",
        _rig.iloc[0].title()   if not _rig.empty   else "",
    ] if x)
    st.header(f"Well Overview — {_ctx}" if _ctx else "Well Overview")

    if hdr.empty or ops.empty:
        st.warning("No data found.")
        return

    _render_kpis(ops, hdr, planned_time)
    _render_npt_phase_bar(ops)

    if planned_time is not None and not planned_time.empty:
        _render_npt_trajectory(ops, planned_time)

    _render_daily_cost(ops, hdr)
    _render_depth_chart(ops, hdr)

    if weather is not None and not weather.empty:
        _render_weather(ops, weather)

    if vessels is not None and not vessels.empty:
        _render_vessels(vessels)

    _render_personnel(ops, hdr)
    _render_drill_down(ops, hdr, weather)
