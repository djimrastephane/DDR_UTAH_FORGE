from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from .constants import PHASE_ORDER, PHASE_COLOURS
    from .loaders import load_casing, load_fit_lot
    from .utils import _apply_chart_theme, _ddr_citation_row
except ImportError:
    from constants import PHASE_ORDER, PHASE_COLOURS                   # type: ignore[no-redef]
    from loaders import load_casing, load_fit_lot                      # type: ignore[no-redef]
    from utils import _apply_chart_theme, _ddr_citation_row            # type: ignore[no-redef]

_ui_root = Path(__file__).resolve().parents[3]
if str(_ui_root / "src") not in sys.path:
    sys.path.insert(0, str(_ui_root / "src"))

from ddr_rag.vocab import label_phase


_PHASE_LABELS: dict[str, str] = {
    "MIRU":   "Move In / Rig Up",
    "COND1":  "Conductor",
    "INTRM1": "Intermediate 1",
    "INTRM2": "Intermediate 2",
    "PROD1":  "Production / Reservoir",
    "COMPZN": "Completion / Zonal",
}

_SEV_COLOUR: dict[str, str] = {
    "total":         "#D32F2F",
    "partial":       "#F57C00",
    "seepage":       "#F9A825",
    "lcm_treatment": "#7B1FA2",
}

_OP_SCALE = [
    [0.0,  "#1976D2"],
    [0.35, "#7B1FA2"],
    [0.65, "#E65100"],
    [1.0,  "#B71C1C"],
]

_SHOE_LABELS: dict[str, str] = {
    "26\" x 20\" Conductor":          "20\" Csg shoe",
    "13-3/8\" Intermediate Casing":   "13⅜\" shoe",
    "9-7/8\" Production Casing":      "9⅞\" shoe",
    "7\" x 5-1/2\" Production Liner": "7\" liner top",
}

_FIT_STYLE: dict[str, tuple[str, str]] = {
    "pass":       ("#1565C0", "dash"),
    "initiation": ("#B71C1C", "dot"),
    "unknown":    ("#757575", "longdash"),
}


def _render_kpis(events: pd.DataFrame) -> None:
    res  = events[events["event_type"] == "restriction"]
    op_  = events[events["event_type"] == "overpull"]
    loss = events[events["event_type"] == "mud_loss"]
    form = events[events["event_type"] == "formation"]

    max_op   = op_["force_klbs"].max()
    max_loss = loss["loss_rate_bbl_hr"].max()
    max_ecd  = (
        form.loc[form["sub_type"] == "high_ecd", "ecd_ppge"].max()
        if not form.empty else None
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tight Spots / Restrictions", str(len(res)),
              help="Depth-referenced restrictions and packoffs")
    c2.metric("Overpull Events", str(len(op_)),
              f"Peak {max_op:.0f} klbs" if pd.notna(max_op) else None)
    c3.metric("Mud Loss Events", str(len(loss)),
              f"Peak {max_loss:.0f} bbl/hr" if pd.notna(max_loss) else None)
    c4.metric("Formation Challenges", str(len(form)),
              f"Peak ECD {max_ecd:.2f} ppge" if pd.notna(max_ecd) else None)


def _render_tab_mechanical(ev: pd.DataFrame, ops: pd.DataFrame) -> None:
    mech = ev[ev["event_type"].isin(["overpull", "restriction"])].copy()
    mech = mech.dropna(subset=["event_depth_ft_md", "report_date_dt"])

    if mech.empty:
        st.info("No depth-tagged mechanical events for the selected phases.")
        return

    phase_bands = (
        ops.groupby("phase")["report_date_parsed"]
        .agg(["min", "max"])
        .reset_index()
    )

    casing = load_casing()
    shoe_depths: dict[str, float] = {}
    if not casing.empty:
        for _, cr in casing.drop_duplicates("casing_description").iterrows():
            if pd.notna(cr.get("set_depth_ft")):
                shoe_depths[str(cr["casing_description"])[:30]] = float(cr["set_depth_ft"])

    y_max = mech["event_depth_ft_md"].max() * 1.05
    fig = go.Figure()

    for _, pb in phase_bands.iterrows():
        ph = pb["phase"]
        fig.add_vrect(
            x0=pb["min"], x1=pb["max"],
            fillcolor=PHASE_COLOURS.get(ph, "#aaa"), opacity=0.04,
            layer="below", line_width=0,
        )

    for bx in phase_bands.sort_values("min")["min"].tolist()[1:]:
        fig.add_vline(
            x=bx, line_dash="dash",
            line_color="rgba(100,100,100,0.35)", line_width=1,
        )

    for _, pb in phase_bands.iterrows():
        ph     = pb["phase"]
        mid_x  = pb["min"] + (pb["max"] - pb["min"]) / 2
        fig.add_annotation(
            x=mid_x, xref="x",
            y=1.04, yref="paper",
            text=f"<b>{_PHASE_LABELS.get(ph, ph)}</b>",
            showarrow=False,
            font=dict(size=9, color=PHASE_COLOURS.get(ph, "#555"), family="Arial"),
            align="center",
        )

    for desc, depth in shoe_depths.items():
        if depth <= y_max:
            short = next(
                (v for k, v in _SHOE_LABELS.items() if k[:12] in desc), f"{depth:.0f} ft"
            )
            fig.add_hline(
                y=depth, line_dash="dot",
                line_color="rgba(80,80,80,0.55)", line_width=1.2,
            )
            fig.add_annotation(
                x=1.01, xref="paper", y=depth, yref="y",
                text=short, showarrow=False,
                font=dict(size=8, color="rgb(80,80,80)"),
                xanchor="left",
            )

    fig.add_trace(go.Histogram2dContour(
        x=mech["report_date_dt"],
        y=mech["event_depth_ft_md"],
        colorscale=[
            [0,   "rgba(255,100,0,0)"],
            [0.4, "rgba(255,100,0,0.06)"],
            [1.0, "rgba(255,60,0,0.18)"],
        ],
        showscale=False,
        contours=dict(coloring="fill", showlines=False),
        nbinsx=16, nbinsy=20,
        hoverinfo="skip",
        showlegend=False,
    ))

    for etype, marker_sym, label in [
        ("overpull",    "circle",  "Overpull"),
        ("restriction", "diamond", "Restriction / Packoff"),
    ]:
        sub = mech[mech["event_type"] == etype].copy()
        if sub.empty:
            continue

        forces = sub["force_klbs"].fillna(10)
        log_f  = np.log1p(forces.clip(lower=1))
        sizes  = (log_f / max(float(log_f.max()), 1) * 14 + 9).clip(9, 24)

        sub["_excerpt"]   = (
            sub["full_op_text"].fillna("")
            .str.replace(r"\s+", " ", regex=True)
            .str[:220]
        )
        sub["_npt_flag"]  = sub["is_npt"].map({True: "⚠ NPT", False: "Productive"})
        sub["_dur"]       = sub["duration_hr"].fillna(0).round(1)
        sub["_phase_lbl"] = sub["phase"].map(label_phase).fillna(sub["phase"])
        sub["_citation"]  = sub.apply(_ddr_citation_row, axis=1)

        customdata = sub[[
            "_phase_lbl", "force_klbs", "hole_type",
            "_npt_flag", "_dur", "_excerpt", "_citation",
        ]].values

        if etype == "overpull":
            marker_kw = dict(
                symbol=marker_sym, size=sizes,
                color=forces, colorscale=_OP_SCALE,
                cmin=5, cmax=max(float(forces.max()), 60),
                colorbar=dict(
                    title=dict(text="Force (klbs)", font=dict(size=11)),
                    thickness=14, len=0.6, x=1.08,
                    tickfont=dict(size=10, color="rgb(40,40,40)"),
                    outlinewidth=0,
                ),
                showscale=True, opacity=0.88,
                line=dict(width=1.5, color="white"),
            )
        else:
            marker_kw = dict(
                symbol=marker_sym, size=sizes,
                color="#F57F17", opacity=0.90,
                line=dict(width=1.8, color="white"),
            )

        fig.add_trace(go.Scatter(
            x=sub["report_date_dt"],
            y=sub["event_depth_ft_md"],
            mode="markers",
            name=label,
            marker=marker_kw,
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b>  ·  %{x|%d %b %Y}<br>"
                "Depth: <b>%{y:,.0f} ft MD</b><br>"
                "Force: <b>%{customdata[1]:.0f} klbs</b>  ·  "
                "Hole: %{customdata[2]}<br>"
                "%{customdata[3]}  ·  %{customdata[4]:.1f} hrs<br>"
                "<span style='font-size:10px;color:#888'>Source: %{customdata[6]}</span><br>"
                "<br>"
                "<span style='font-size:11px;color:#444'>%{customdata[5]}</span>"
                "<extra></extra>"
            ),
        ))

    fig.update_yaxes(autorange="reversed", title="Depth (ft MD)", tickformat=",")
    fig.update_xaxes(title="")
    fig.update_layout(
        title=dict(text="Mechanical Events — Depth vs Date",
                   font=dict(size=14, color="rgb(30,30,30)")),
        height=560,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(
            orientation="h", y=-0.12,
            font=dict(size=11, color="rgb(40,40,40)"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(180,180,180,0.5)", borderwidth=1,
        ),
        margin=dict(l=70, r=120, t=80, b=60),
    )
    _apply_chart_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "**Circle size and colour** = overpull force (blue → purple → crimson with increasing severity).  "
        "**Amber diamond** = tight spot / restriction. "
        "Dashed horizontal lines = casing shoes / liner top."
    )

    st.subheader("Highest-force events")
    top = (
        mech.dropna(subset=["force_klbs"])
        .nlargest(15, "force_klbs")
        [["report_date", "phase", "event_type", "hole_type",
          "event_depth_ft_md", "force_klbs", "doc_id", "page",
          "shift_block", "raw_snippet"]]
        .rename(columns={
            "report_date":      "Date",
            "phase":            "Phase",
            "event_type":       "Type",
            "hole_type":        "Hole",
            "event_depth_ft_md": "Depth (ft MD)",
            "force_klbs":       "Force (klbs)",
            "raw_snippet":      "Context",
        })
    )
    top["Source"]       = top.apply(_ddr_citation_row, axis=1)
    top                 = top.drop(columns=["doc_id", "page", "shift_block"])
    top["Depth (ft MD)"] = top["Depth (ft MD)"].apply(
        lambda v: f"{v:,.0f}" if pd.notna(v) else "—"
    )
    top["Force (klbs)"] = top["Force (klbs)"].apply(
        lambda v: f"{v:.0f}" if pd.notna(v) else "—"
    )
    st.dataframe(
        top, use_container_width=True, hide_index=True,
        column_config={"Context": st.column_config.TextColumn(width="large")},
    )


def _render_tab_losses_formation(ev: pd.DataFrame) -> None:
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Mud Loss Events")
        loss_ev = ev[ev["event_type"] == "mud_loss"].copy()
        if loss_ev.empty:
            st.info("No mud loss events for the selected phases.")
        else:
            fig_l = go.Figure()
            for sev in ["total", "partial", "seepage", "lcm_treatment"]:
                sub = loss_ev[loss_ev["severity"] == sev]
                if sub.empty:
                    continue
                rates = sub["loss_rate_bbl_hr"].fillna(0)
                sub = sub.copy()
                sub["_citation"] = sub.apply(_ddr_citation_row, axis=1)
                fig_l.add_trace(go.Bar(
                    x=sub["report_date_dt"],
                    y=rates.where(rates > 0, 1),
                    name=sev.replace("_", " ").title(),
                    marker_color=_SEV_COLOUR.get(sev, "#999"),
                    customdata=sub[["phase", "severity", "raw_snippet", "_citation"]].values,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b> · %{x|%d %b %Y}<br>"
                        "Severity: %{customdata[1]}<br>"
                        "Rate: %{y:.0f} bbl/hr<br>"
                        "<span style='font-size:10px;color:#888'>Source: %{customdata[3]}</span><br>"
                        "<i>%{customdata[2]}</i><extra></extra>"
                    ),
                ))
            fig_l.update_layout(
                barmode="stack", height=350,
                yaxis_title="Loss rate (bbl/hr)", xaxis_title="Date",
                margin=dict(l=50, r=20, t=30, b=50),
                legend=dict(orientation="h", y=-0.25),
            )
            _apply_chart_theme(fig_l)
            st.plotly_chart(fig_l, use_container_width=True)

            loss_tbl = loss_ev[["report_date", "phase", "severity",
                                 "loss_rate_bbl_hr", "event_depth_ft_md",
                                 "doc_id", "page", "shift_block", "raw_snippet"]].copy()
            loss_tbl["Source"] = loss_tbl.apply(_ddr_citation_row, axis=1)
            loss_tbl = loss_tbl.drop(columns=["doc_id", "page", "shift_block"])
            st.dataframe(
                loss_tbl.rename(columns={
                    "report_date":       "Date",
                    "phase":             "Phase",
                    "severity":          "Severity",
                    "loss_rate_bbl_hr":  "Rate (bbl/hr)",
                    "event_depth_ft_md": "Depth (ft MD)",
                    "raw_snippet":       "Context",
                }),
                use_container_width=True, hide_index=True,
            )

    with col_r:
        st.subheader("Formation Challenges")
        form_ev = ev[ev["event_type"] == "formation"].copy()
        if form_ev.empty:
            st.info("No formation events for the selected phases.")
        else:
            ecd_ev = form_ev[form_ev["sub_type"] == "high_ecd"].dropna(subset=["ecd_ppge"])
            if not ecd_ev.empty:
                fig_ecd = go.Figure()
                for ph in ecd_ev["phase"].unique():
                    sub = ecd_ev[ecd_ev["phase"] == ph].sort_values("report_date_dt").copy()
                    sub["_citation"] = sub.apply(_ddr_citation_row, axis=1)
                    fig_ecd.add_trace(go.Scatter(
                        x=sub["report_date_dt"],
                        y=sub["ecd_ppge"],
                        mode="markers+lines",
                        name=_PHASE_LABELS.get(ph, ph),
                        marker=dict(size=6, color=PHASE_COLOURS.get(ph, "#999")),
                        line=dict(color=PHASE_COLOURS.get(ph, "#999"), width=1.5),
                        customdata=sub[["_citation"]].values,
                        hovertemplate=(
                            f"<b>{_PHASE_LABELS.get(ph, ph)}</b> · %{{x|%d %b %Y}}<br>"
                            "ECD: %{y:.2f} ppge<br>"
                            "<span style='font-size:10px;color:#888'>"
                            "Source: %{customdata[0]}</span>"
                            "<extra></extra>"
                        ),
                    ))

                fit_lot = load_fit_lot()
                if not fit_lot.empty:
                    limits = (
                        fit_lot.dropna(subset=["limit_ppge"])
                        .drop_duplicates("limit_ppge")
                        .sort_values("limit_ppge", ascending=False)
                    )
                    for _, lr in limits.iterrows():
                        col, dash = _FIT_STYLE.get(lr.get("result", "unknown"), ("#757575", "dash"))
                        citation  = _ddr_citation_row(lr)
                        shoe      = f" @ {lr['casing_shoe']}" if lr.get("casing_shoe") else ""
                        label     = f"{lr['test_type']}{shoe}: {lr['limit_ppge']:.2f} ppge ({citation})"
                        fig_ecd.add_hline(
                            y=lr["limit_ppge"],
                            line_dash=dash, line_color=col, line_width=1.5,
                            annotation_text=label,
                            annotation_position="top right",
                            annotation_font_size=9, annotation_font_color=col,
                        )
                else:
                    fig_ecd.add_hline(
                        y=14.75,
                        line_dash="dash", line_color="#D32F2F", line_width=1.5,
                        annotation_text="Est. frac gradient 14.75 ppge",
                        annotation_position="top right",
                        annotation_font_size=9, annotation_font_color="#D32F2F",
                    )
                fig_ecd.update_layout(
                    height=280, yaxis_title="ECD (ppge)",
                    xaxis_title="Date", title="ECD Trend (PWD)",
                    margin=dict(l=50, r=20, t=40, b=50),
                    legend=dict(orientation="h", y=-0.3),
                )
                _apply_chart_theme(fig_ecd)
                st.plotly_chart(fig_ecd, use_container_width=True)

            other_form = form_ev[form_ev["sub_type"] != "high_ecd"]
            if not other_form.empty:
                st.markdown("**Other formation events**")
                form_tbl = other_form[["report_date", "phase", "sub_type",
                                        "doc_id", "page", "shift_block", "raw_snippet"]].copy()
                form_tbl["Source"] = form_tbl.apply(_ddr_citation_row, axis=1)
                form_tbl = form_tbl.drop(columns=["doc_id", "page", "shift_block"])
                st.dataframe(
                    form_tbl.rename(columns={
                        "report_date": "Date",
                        "phase":       "Phase",
                        "sub_type":    "Type",
                        "raw_snippet": "Context",
                    }),
                    use_container_width=True, hide_index=True,
                    column_config={"Context": st.column_config.TextColumn(width="large")},
                )

    fit_lot_tbl = load_fit_lot()
    if not fit_lot_tbl.empty:
        st.divider()
        st.subheader("FIT / LOT Test Results")
        fit_tbl = fit_lot_tbl.copy()
        fit_tbl["Source"] = fit_tbl.apply(_ddr_citation_row, axis=1)
        fit_tbl = fit_tbl[["report_date", "phase", "test_type",
                            "limit_ppge", "result", "casing_shoe", "Source"]]
        fit_tbl = fit_tbl.rename(columns={
            "report_date": "Date", "phase": "Phase",
            "test_type":   "Test", "limit_ppge": "Limit (ppge)",
            "result":      "Result", "casing_shoe": "Casing Shoe",
        })
        fit_tbl["Limit (ppge)"] = fit_tbl["Limit (ppge)"].apply(
            lambda v: f"{v:.2f}" if pd.notna(v) else "—"
        )
        st.dataframe(fit_tbl, use_container_width=True, hide_index=True)


def _render_tab_log(ev: pd.DataFrame) -> None:
    st.subheader(f"All Events ({len(ev):,})")

    filt_type = st.multiselect(
        "Event type",
        options=sorted(ev["event_type"].unique()),
        default=sorted(ev["event_type"].unique()),
    )
    filt_hole = st.multiselect(
        "Hole type",
        options=sorted(ev["hole_type"].dropna().unique()),
        default=sorted(ev["hole_type"].dropna().unique()),
    )
    log = ev[ev["event_type"].isin(filt_type) & ev["hole_type"].isin(filt_hole)]

    display_cols = [
        "report_date", "phase", "event_type", "sub_type", "hole_type",
        "event_depth_ft_md", "force_klbs", "loss_rate_bbl_hr",
        "ecd_ppge", "severity", "is_npt", "doc_id", "page",
        "shift_block", "raw_snippet",
    ]
    display_cols = [c for c in display_cols if c in log.columns]
    out = log[display_cols].copy()
    out["Source"] = out.apply(_ddr_citation_row, axis=1)
    out = out.drop(columns=[c for c in ["doc_id", "page", "shift_block"] if c in out.columns])
    out = out.rename(columns={
        "report_date":       "Date",
        "phase":             "Phase",
        "event_type":        "Event Type",
        "sub_type":          "Sub-type",
        "hole_type":         "Hole",
        "event_depth_ft_md": "Depth (ft MD)",
        "force_klbs":        "Force (klbs)",
        "loss_rate_bbl_hr":  "Rate (bbl/hr)",
        "ecd_ppge":          "ECD (ppge)",
        "severity":          "Severity",
        "is_npt":            "NPT",
        "raw_snippet":       "Context",
    })
    for col in ["Depth (ft MD)", "Force (klbs)", "Rate (bbl/hr)", "ECD (ppge)"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(1)

    st.dataframe(
        out, use_container_width=True, hide_index=True,
        column_config={"Context": st.column_config.TextColumn(width="large")},
    )
    st.download_button(
        "Download CSV",
        data=out.to_csv(index=False),
        file_name="ddr_wellbore_events.csv",
        mime="text/csv",
    )


def page_wellbore_events(events: pd.DataFrame, ops: pd.DataFrame) -> None:
    n_ddrs = ops["report_date"].nunique()
    st.header("Wellbore Events")
    st.caption(
        f"Tight spots, overpull, mud losses and formation challenges extracted from "
        f"operational text across {n_ddrs} DDRs. Depths are measured depth (ft MD)."
    )

    if events.empty:
        st.warning("No wellbore events data found. Run `scripts/extract_wellbore_events.py` first.")
        return

    _render_kpis(events)
    st.divider()

    available_phases = [p for p in PHASE_ORDER if p in events["phase"].unique()]
    sel_phases = st.multiselect(
        "Filter by phase",
        options=available_phases,
        default=available_phases,
        format_func=lambda p: _PHASE_LABELS.get(p, p),
    )
    ev = events[events["phase"].isin(sel_phases)] if sel_phases else events

    tab1, tab2, tab3 = st.tabs(["Overpull & Restrictions", "Losses & Formation", "Full Event Log"])
    with tab1:
        _render_tab_mechanical(ev, ops)
    with tab2:
        _render_tab_losses_formation(ev)
    with tab3:
        _render_tab_log(ev)
