from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from .constants import PHASE_ORDER, PHASE_COLOURS
    from .loaders import (
        load_all_ops, load_all_headers, load_casing, load_fit_lot,
        load_wellbore_events, load_completion_string, load_frac_sleeve_status,
        _parse_num,
    )
    from .utils import _apply_chart_theme, _phase_date_ranges
except ImportError:
    from constants import PHASE_ORDER, PHASE_COLOURS  # type: ignore[no-redef]
    from loaders import (                              # type: ignore[no-redef]
        load_all_ops, load_all_headers, load_casing, load_fit_lot,
        load_wellbore_events, load_completion_string, load_frac_sleeve_status,
        _parse_num,
    )
    from utils import _apply_chart_theme, _phase_date_ranges  # type: ignore[no-redef]

_root = Path(__file__).resolve().parents[3]
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from ddr_rag.vocab import label_phase
from ddr_rag.npt_classifier import classify_ops_df, CATEGORY_LABELS, CATEGORY_COLOURS


def _section(title: str, icon: str = "") -> None:
    st.markdown(
        f"<h3 style='color:#1565C0;border-bottom:2px solid #1565C0;"
        f"padding-bottom:4px;margin-top:28px'>{icon}  {title}</h3>",
        unsafe_allow_html=True,
    )


def _field(label: str, value: str) -> str:
    return (
        f"<tr><td style='font-weight:600;color:#455A64;width:200px;"
        f"padding:4px 12px 4px 0'>{label}</td>"
        f"<td style='color:#1A202C;padding:4px 0'>{value}</td></tr>"
    )


def _render_eowr_section1(ctx: SimpleNamespace) -> None:
    _section("1. Executive Summary", "📋")

    st.markdown("#### 1.1  Well Objectives")
    st.info(
        "The well was drilled to evaluate and complete the production interval "
        "in Field Block A. Objectives included: setting 9-7/8\" production casing, "
        "running a 5-1/2\" production liner with 18 multi-stage frac sleeves, "
        "stimulating the reservoir and completing the well for production.",
    )

    st.markdown("#### 1.2  Results & Final Status")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Campaign Duration",  f"{ctx.campaign_days} days")
    k2.metric("Total Depth",        f"{ctx.td_ft:,.0f} ft MD")
    k3.metric("Well Cost",          f"£{ctx.total_cost/1e6:.1f}M")
    k4.metric("vs. AFE",
              f"+£{(ctx.total_cost-ctx.afe)/1e6:.1f}M" if ctx.afe else "—",
              delta_color="inverse")

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Overall NPT",        f"{ctx.npt_pct:.1f}%")
    k6.metric("NPT Hours",          f"{ctx.npt_h:,.0f} h")
    k7.metric("Total Operations",   f"{len(ctx.ops):,}")
    k8.metric("Final Status",       "Producer")

    st.markdown("#### 1.3  Key Findings")
    top_npt_cat = (
        ctx.ops[ctx.ops["is_npt"]].groupby("npt_category")["duration_hr"]
        .sum().idxmax()
    ) if ctx.ops["is_npt"].any() else "—"
    top_npt_lbl = CATEGORY_LABELS.get(top_npt_cat, top_npt_cat)
    top_npt_h   = ctx.ops[ctx.ops["npt_category"] == top_npt_cat]["duration_hr"].sum()

    findings = [
        f"Well reached TD of **{ctx.td_ft:,.0f} ft MD** on "
        f"{ctx.hdr.loc[ctx.hdr['depth_n']==ctx.td_ft,'dt'].iloc[0].strftime('%d %b %Y') if ctx.hdr['depth_n'].notna().any() else '—'}.",
        (f"Total well cost **£{ctx.total_cost/1e6:.1f}M** against AFE of £{ctx.afe/1e6:.1f}M "
         f"(+{100*(ctx.total_cost-ctx.afe)/ctx.afe:.0f}% over budget)." if ctx.afe else ""),
        f"Overall NPT was **{ctx.npt_pct:.1f}%** ({ctx.npt_h:,.0f} hrs). "
        f"Primary NPT driver: **{top_npt_lbl}** ({top_npt_h:.0f} hrs).",
        f"Production / Reservoir Section had the highest NPT at "
        f"**{100*ctx.ops[ctx.ops['phase']=='PROD1']['is_npt'].mean():.0f}%** "
        f"driven by MPD casing programme and metallics recovery campaign.",
        f"18 NCS multi-stage frac sleeves installed; 13 sleeves successfully "
        f"stimulated during COMPZN phase.",
        f"Upper completion installed with 4-1/2\" × 5-1/2\" string, "
        f"Halliburton DHSV at ~871 ft MD and tubing hanger landed at 120.8 ft MD.",
    ]
    for f in findings:
        if f:
            st.markdown(f"• {f}")


def _render_eowr_section2(ctx: SimpleNamespace) -> None:
    _section("2. General Information", "ℹ️")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 2.1  Well Details")
        rows = [
            ("Well / Wellbore",    "Field Block A — Well A-W1 (R2 sidetrack)"),
            ("License / Block",    "Block A (anonymised)"),
            ("Operator",           "Operator Alpha"),
            ("Field",              ctx.hdr["field_name"].dropna().mode().iloc[0].title()
                                   if "field_name" in ctx.hdr.columns and ctx.hdr["field_name"].notna().any()
                                   else "—"),
            ("API / UWI",          str(ctx.hdr["api_uwi"].dropna().iloc[0])
                                   if "api_uwi" in ctx.hdr.columns and ctx.hdr["api_uwi"].notna().any()
                                   else "—"),
            ("AFE No.",            str(ctx.hdr["afe_no"].dropna().iloc[0])
                                   if "afe_no" in ctx.hdr.columns and ctx.hdr["afe_no"].notna().any()
                                   else "—"),
            ("Total Depth",        f"{ctx.td_ft:,.0f} ft MD"),
            ("KB Elevation",       f"{float(ctx.kb_elev):.0f} ft" if ctx.kb_elev else "—"),
            ("Water Depth",        f"{float(ctx.water_depth):.0f} ft" if ctx.water_depth else "—"),
            ("Spud Date",          ctx.campaign_start.strftime("%d %b %Y")),
            ("TD / Rig Release",   ctx.campaign_end.strftime("%d %b %Y")),
            ("Campaign Duration",  f"{ctx.campaign_days} days"),
        ]
        html = "<table style='border-collapse:collapse;width:100%'>"
        for label, val in rows:
            html += _field(label, val)
        html += "</table>"
        st.markdown(html, unsafe_allow_html=True)

    with col2:
        st.markdown("#### 2.2  Rig Information")
        _rig = ctx.hdr["rig_name"].dropna().mode()
        _rig_str = _rig.iloc[0].title() if not _rig.empty else "—"
        rows2 = [
            ("Rig Name",          _rig_str),
            ("Rig Type",          "Jack-Up"),
            ("Contractor",        "Rig Contractor Alpha"),
            ("Mobilisation",      ctx.job_start if ctx.job_start else "—"),
            ("Demobilisation",    ctx.campaign_end.strftime("%d %b %Y")),
            ("Total Rig Days",    f"{ctx.campaign_days} days"),
            ("Avg Daily Cost",    f"£{ctx.hdr['cost_n'].mean():,.0f}"),
            ("Peak Daily Cost",   f"£{ctx.hdr['cost_n'].max():,.0f}"),
        ]
        html2 = "<table style='border-collapse:collapse;width:100%'>"
        for label, val in rows2:
            html2 += _field(label, val)
        html2 += "</table>"
        st.markdown(html2, unsafe_allow_html=True)

    st.markdown("#### 2.3  Time-Depth Curve")
    depth_df = ctx.hdr.dropna(subset=["depth_n"]).copy()
    if not depth_df.empty:
        fig = go.Figure()
        phase_ranges = _phase_date_ranges(ctx.ops)
        for ph, (d0, d1) in phase_ranges.items():
            col = PHASE_COLOURS.get(ph, "#9E9E9E")
            fig.add_vrect(x0=d0, x1=d1, fillcolor=col, opacity=0.06, line_width=0)
            fig.add_annotation(
                x=d0+(d1-d0)/2, y=depth_df["depth_n"].max()*0.02,
                text=label_phase(ph), showarrow=False,
                font=dict(size=9, color=col),
            )
        fig.add_trace(go.Scatter(
            x=depth_df["dt"], y=depth_df["depth_n"],
            mode="lines", line=dict(color="#1565C0", width=2.5),
            name="Actual depth",
        ))
        fig.add_trace(go.Scatter(
            x=[depth_df["dt"].iloc[0], depth_df["dt"].iloc[-1]],
            y=[depth_df["depth_n"].iloc[0], depth_df["depth_n"].iloc[-1]],
            mode="lines", line=dict(color="#9E9E9E", width=1.5, dash="dot"),
            name="Theoretical (no NPT)",
        ))
        fig.update_yaxes(autorange="reversed", title="Depth (ft MD)",
                         tickformat=",", showgrid=True,
                         gridcolor="rgba(175,175,175,0.35)")
        fig.update_xaxes(title="Date", showgrid=False)
        fig.update_layout(
            height=420, plot_bgcolor="white",
            legend=dict(orientation="h", y=1.04),
            margin=dict(l=10, r=10, t=30, b=20),
        )
        _apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)


def _render_eowr_section3(ctx: SimpleNamespace) -> None:
    _section("3. Drilling Operations Summary", "⛏")

    st.markdown("#### 3.1  Phase Chronology")
    phase_rows = []
    for ph in PHASE_ORDER:
        g = ctx.ops[ctx.ops["phase"] == ph]
        if g.empty:
            continue
        tot_h = g["duration_hr"].sum()
        npt_h = g[g["is_npt"]]["duration_hr"].sum()
        phase_rows.append({
            "Phase":      label_phase(ph),
            "Start":      g["dt"].min().strftime("%d %b %Y"),
            "End":        g["dt"].max().strftime("%d %b %Y"),
            "Days":       g["dt"].nunique(),
            "Total (h)":  f"{tot_h:.0f}",
            "NPT (h)":    f"{npt_h:.0f}",
            "NPT%":       f"{100*npt_h/tot_h:.0f}%" if tot_h else "—",
            "Primary NPT": CATEGORY_LABELS.get(
                g[g["is_npt"]].groupby("npt_category")["duration_hr"]
                .sum().idxmax() if g["is_npt"].any() else "", "—"),
        })

    if phase_rows:
        def _hex_to_rgba(hex_col: str, alpha: float = 0.15) -> str:
            h = hex_col.lstrip("#")
            r, g2, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
            return f"rgba({r},{g2},{b},{alpha})"

        fig_ph = go.Figure(go.Table(
            columnwidth=[16, 8, 8, 5, 8, 8, 6, 18],
            header=dict(
                values=[f"<b>{c}</b>" for c in phase_rows[0].keys()],
                fill_color="#1565C0", font=dict(color="white", size=11),
                align="left", height=28,
            ),
            cells=dict(
                values=[[r[k] for r in phase_rows] for k in phase_rows[0].keys()],
                fill_color="white",
                font=dict(color="#1A202C", size=10.5),
                align="left", height=26,
            ),
        ))
        fig_ph.update_layout(margin=dict(l=0,r=0,t=5,b=0),
                             height=len(phase_rows)*28+70)
        st.plotly_chart(fig_ph, use_container_width=True)

    st.markdown("#### 3.2  Casing Programme")
    if not ctx.cas.empty:
        cas_uniq = (
            ctx.cas.drop_duplicates(["casing_description","od_in","set_depth_ft"])
            .sort_values("set_depth_ft")
            [["casing_description","od_in","set_depth_ft",
              "top_depth_ft","weight_lb_per_ft","run_date"]]
            .rename(columns={
                "casing_description":"String",
                "od_in":"OD (in)",
                "set_depth_ft":"Set Depth (ft)",
                "top_depth_ft":"Top Depth (ft)",
                "weight_lb_per_ft":"Weight (ppf)",
                "run_date":"Run Date",
            })
        )
        fig_c = go.Figure(go.Table(
            columnwidth=[22,8,10,10,10,10],
            header=dict(
                values=[f"<b>{c}</b>" for c in cas_uniq.columns],
                fill_color="#455A64", font=dict(color="white",size=11),
                align="left", height=28,
            ),
            cells=dict(
                values=[cas_uniq[c].fillna("—").astype(str).tolist()
                        for c in cas_uniq.columns],
                fill_color="white",
                font=dict(color="#1A202C",size=10.5),
                align="left", height=24,
            ),
        ))
        fig_c.update_layout(margin=dict(l=0,r=0,t=5,b=0),
                            height=max(180, len(cas_uniq)*26+60))
        st.plotly_chart(fig_c, use_container_width=True)

    st.markdown("#### 3.3  NPT Analysis by Category")
    npt_by_cat = (
        ctx.ops[ctx.ops["is_npt"]]
        .groupby("npt_category")["duration_hr"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    npt_by_cat["label"] = npt_by_cat["npt_category"].map(CATEGORY_LABELS)
    npt_by_cat["pct"]   = npt_by_cat["duration_hr"] / ctx.npt_h * 100

    if not npt_by_cat.empty:
        fig_n = go.Figure(go.Bar(
            x=npt_by_cat["label"],
            y=npt_by_cat["duration_hr"],
            text=npt_by_cat.apply(
                lambda r: f"{r['duration_hr']:.0f}h ({r['pct']:.0f}%)", axis=1),
            textposition="outside",
            marker_color=[CATEGORY_COLOURS.get(c,"#9E9E9E")
                          for c in npt_by_cat["npt_category"]],
        ))
        fig_n.update_layout(
            height=340, plot_bgcolor="white",
            yaxis=dict(title="Hours", showgrid=True,
                       gridcolor="rgba(175,175,175,0.35)"),
            xaxis=dict(tickangle=-25),
            margin=dict(l=10,r=10,t=20,b=80),
        )
        _apply_chart_theme(fig_n)
        st.plotly_chart(fig_n, use_container_width=True)


def _render_eowr_section4(ctx: SimpleNamespace) -> None:
    _section("4. Geological & Subsurface Data", "🪨")

    st.markdown("#### 4.1  Formation Integrity Tests (FIT / LOT / DFIT)")
    if not ctx.fit_lot.empty:
        fit_disp = ctx.fit_lot.rename(columns={
            "report_date":"Date","phase":"Phase","test_type":"Test",
            "limit_ppge":"EMW (ppge)","result":"Result","casing_shoe":"Casing Shoe",
        })
        if "Phase" in fit_disp.columns:
            fit_disp["Phase"] = fit_disp["Phase"].map(label_phase)
        if "EMW (ppge)" in fit_disp.columns:
            fit_disp["EMW (ppge)"] = fit_disp["EMW (ppge)"].apply(
                lambda v: f"{v:.2f}" if pd.notna(v) else "—")
        cols = [c for c in ["Date","Phase","Test","EMW (ppge)","Result","Casing Shoe"]
                if c in fit_disp.columns]
        fig_fit = go.Figure(go.Table(
            columnwidth=[10,14,8,10,8,14],
            header=dict(values=[f"<b>{c}</b>" for c in cols],
                        fill_color="#1565C0",font=dict(color="white",size=11),
                        align="left",height=28),
            cells=dict(values=[fit_disp[c].astype(str).tolist() for c in cols],
                       fill_color="white",font=dict(color="#1A202C",size=10.5),
                       align="left",height=24),
        ))
        fig_fit.update_layout(margin=dict(l=0,r=0,t=5,b=0),
                              height=max(140,len(fit_disp)*26+60))
        st.plotly_chart(fig_fit, use_container_width=True)
    else:
        st.info("No FIT/LOT results available.")

    st.markdown("#### 4.2  Formation Pressure & ECD Data")
    if not ctx.events.empty:
        ecd_ev = ctx.events[
            (ctx.events["event_type"]=="formation") &
            (ctx.events["sub_type"]=="high_ecd")
        ].dropna(subset=["ecd_ppge"])
        if not ecd_ev.empty:
            c1,c2,c3 = st.columns(3)
            c1.metric("High-ECD Events", len(ecd_ev))
            c2.metric("Max ECD",  f"{ecd_ev['ecd_ppge'].max():.2f} ppge")
            c3.metric("Min ECD",  f"{ecd_ev['ecd_ppge'].min():.2f} ppge")

            fig_ecd = go.Figure(go.Scatter(
                x=ecd_ev["report_date"],
                y=ecd_ev["ecd_ppge"],
                mode="markers",
                marker=dict(size=8, color="#E65100", opacity=0.75),
                hovertemplate="<b>%{x}</b><br>ECD: %{y:.2f} ppge<extra></extra>",
            ))
            fig_ecd.add_hline(
                y=14.75, line=dict(color="#D32F2F", width=1.5, dash="dot"),
                annotation_text="Frac gradient 14.75 ppge",
                annotation_position="top right",
            )
            fig_ecd.update_layout(
                height=300, plot_bgcolor="white",
                yaxis=dict(title="ECD (ppge)", showgrid=True,
                           gridcolor="rgba(175,175,175,0.35)"),
                xaxis=dict(title="Date"),
                margin=dict(l=10,r=10,t=20,b=20),
            )
            _apply_chart_theme(fig_ecd)
            st.plotly_chart(fig_ecd, use_container_width=True)

    st.markdown("#### 4.3  Mud Loss Events")
    if not ctx.events.empty:
        losses = ctx.events[ctx.events["event_type"]=="mud_loss"].copy()
        if not losses.empty:
            c1,c2,c3 = st.columns(3)
            c1.metric("Loss Events",   len(losses))
            c2.metric("Max Rate",      f"{losses['loss_rate_bbl_hr'].max():.0f} bbl/hr"
                      if losses["loss_rate_bbl_hr"].notna().any() else "—")
            c3.metric("Primary Phase", label_phase(losses["phase"].mode().iloc[0]))
            for _, r in losses.iterrows():
                st.markdown(
                    f"<div style='border-left:3px solid #1565C0;padding:5px 10px;"
                    f"margin:3px 0;background:#F0F4F8;color:#1A202C;"
                    f"border-radius:0 4px 4px 0;font-size:0.88em'>"
                    f"<b>{r['report_date']}</b>  ·  {label_phase(r['phase'])}  "
                    f"·  {r.get('loss_rate_bbl_hr','?'):.0f} bbl/hr  "
                    f"·  {r.get('severity','?')}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.success("No mud loss events recorded.")


def _render_eowr_section5(ctx: SimpleNamespace) -> None:
    _section("5. Well Testing & Evaluation", "🔬")

    st.markdown("#### 5.1  Stimulation Summary")
    slv = load_frac_sleeve_status()
    if not slv.empty:
        opened = slv[slv["status"]=="OPENED"]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Frac Sleeves Installed", 18)
        c2.metric("Sleeves Opened",         len(opened))
        c3.metric("Not Fracked (#1-5)",      5,
                  help="Deepest sleeves not reached in campaign")
        prop = slv["proppant_lbs_total"].dropna().sum()
        c4.metric("Total Proppant",
                  f"{prop/1e6:.2f}M lbs" if prop > 0 else "—")
        st.markdown("**Sleeve status:**")
        for _, r in slv.sort_values("sleeve_no").iterrows():
            col_map = {"OPENED":"#43A047","LOCATED":"#FDD835",
                       "NO INDICATION":"#FB8C00","NOT FRACKED":"#9E9E9E"}
            col = col_map.get(r["status"], "#9E9E9E")
            prop_str = (f" · {r['proppant_lbs_total']/1e3:.0f}k lbs"
                        if pd.notna(r.get("proppant_lbs_total")) else "")
            psi_str  = (f" · open at {r['open_psi']:.0f} psi"
                        if pd.notna(r.get("open_psi")) else "")
            st.markdown(
                f"<span style='color:{col}'>●</span> "
                f"**Sleeve #{int(r['sleeve_no'])}** — {r['status']}"
                f"{psi_str}{prop_str}",
                unsafe_allow_html=True,
            )
    else:
        st.info("Run `scripts/extract_frac_sleeve_status.py` to populate frac sleeve data.")

    st.markdown("#### 5.2  Well Cleanup & Flow Test")
    cleanup_ops = ctx.ops[
        (ctx.ops["phase"] == "COMPZN") &
        ctx.ops["operation_text"].str.contains(
            r"clean.?up|flow period|well test|flaring|proppant return",
            case=False, na=False,
        )
    ].sort_values("dt")
    if not cleanup_ops.empty:
        st.success(
            f"Well cleanup / flow test conducted from "
            f"**{cleanup_ops['dt'].min().strftime('%d %b %Y')}** to "
            f"**{cleanup_ops['dt'].max().strftime('%d %b %Y')}** "
            f"({cleanup_ops['dt'].nunique()} days, "
            f"{cleanup_ops['duration_hr'].sum():.0f} hrs)."
        )
        for _, r in cleanup_ops.head(5).iterrows():
            st.markdown(
                f"<div style='border-left:3px solid #00838F;padding:5px 10px;"
                f"margin:3px 0;background:#E0F2F1;color:#1A202C;"
                f"border-radius:0 4px 4px 0;font-size:0.87em'>"
                f"<b>{r['report_date']}</b>  ·  "
                f"{str(r['operation_text'])[:220]}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No cleanup / flow test operations identified in DDRs.")


def _render_eowr_section6(ctx: SimpleNamespace) -> None:
    _section("6. Well Completion & Status", "🧵")

    st.markdown("#### 6.1  Final Well Status")
    st.success(
        "**Well A-W1** is a producer. Upper completion installed Sep 2024. "
        "Tubing hanger landed and locked at 120.8 ft MD. Well handed over "
        "to production operations following successful cleanup flow test.",
    )

    st.markdown("#### 6.2  Completion String Summary")
    if not ctx.comp.empty:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Components", len(ctx.comp))
        c2.metric("Completion TD",    f"{ctx.comp['depth_ft'].max():,.0f} ft MD")
        c3.metric("Min Bore (ID)",    f"{ctx.comp['id_in'].dropna().min():.3f}\"")
        c4.metric("ID Restrictions",  int(ctx.comp["is_id_restriction"].sum()))

        key_comps = ctx.comp[~ctx.comp["component_type"].isin(["frac_sleeve"])].copy()
        tbl_data = key_comps[
            ["component","component_type","depth_ft","od_in","id_in",
             "vendor","depth_source"]
        ].copy()
        tbl_data["depth_ft"] = tbl_data["depth_ft"].apply(
            lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
        tbl_data["od_in"] = tbl_data["od_in"].apply(
            lambda v: f"{v:.3f}" if pd.notna(v) else "—")
        tbl_data["id_in"] = tbl_data["id_in"].apply(
            lambda v: f"{v:.3f}" if pd.notna(v) else "—")
        tbl_data.columns = ["Component","Type","Depth (ft MD)","OD (in)",
                             "ID (in)","Vendor","Source"]

        fig_comp = go.Figure(go.Table(
            columnwidth=[22,14,10,8,8,12,10],
            header=dict(
                values=[f"<b>{c}</b>" for c in tbl_data.columns],
                fill_color="#1565C0",font=dict(color="white",size=11),
                align="left",height=28),
            cells=dict(
                values=[tbl_data[c].tolist() for c in tbl_data.columns],
                fill_color=[
                    ["#FFCDD2" if r["is_id_restriction"] else
                     "#FFF8E1" if r["depth_source"] in ("calculated","estimated") else "white"
                     for _, r in key_comps.iterrows()]
                ] * len(tbl_data.columns),
                font=dict(color="#1A202C",size=10.5),
                align="left",height=24),
        ))
        fig_comp.update_layout(
            margin=dict(l=0,r=0,t=5,b=0),
            height=max(250, len(tbl_data)*26+60),
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        st.caption(
            "🔴 Red = ID restriction  ·  🟡 Amber = depth calculated from running sequence  "
            "·  See 🧵 Completion String page for full bore profile and pass-through analysis."
        )
    else:
        st.info("Run `scripts/extract_completion_string.py` to populate.")


def _render_eowr_section7(ctx: SimpleNamespace) -> None:
    _section("7. Appendices & Export", "📎")

    st.markdown("#### 7.1  Source Pages")
    links = [
        ("🔩 Well Overview",       "Detailed cost, depth, NPT trajectory and daily drill-down"),
        ("⚡ Wellbore Events",      "221 mechanical and fluid events with depth, force and DDR citations"),
        ("🧵 Completion String",    "29-component completion string, bore profile, frac sleeve status"),
        ("📊 Drilling Metrics",     "920 performance metrics: ROP, trip speed, running speeds"),
        ("🔄 Operation Sequence",   "255-step programme table with NPT highlighting"),
        ("💰 Cost Analysis",        "NPT cost Pareto, phase cost breakdown, daily burn chart"),
        ("📝 Lessons Learned",      "Auto-generated per-phase recommendations for next well"),
        ("🔗 Cross-Phase Causality","PROD1→COMPZN causal chain and leading indicators"),
    ]
    for page_name, desc in links:
        st.markdown(f"**{page_name}** — {desc}")

    st.markdown("#### 7.2  Data Artifacts Generated")
    artifacts = [
        ("ddr_completion_string.parquet", "Completion string components with OD/ID/citations"),
        ("ddr_frac_sleeve_status.parquet","Frac sleeve status, pressures and proppant volumes"),
        ("ddr_wellbore_events.parquet",   "221 wellbore events with source DDR citations"),
        ("ddr_drilling_metrics.parquet",  "920 performance metrics extracted from DDR text"),
        ("ddr_fit_lot_results.parquet",   "6 FIT/LOT formation integrity test results"),
        ("ddr_casing.parquet",            "Casing programme from DDR casing sections"),
        ("ddr_weather.parquet",           "Weather data (68 DDRs, Apr–Jun 2024)"),
    ]
    for fname, desc in artifacts:
        st.markdown(f"• `{fname}` — {desc}")

    st.divider()
    st.markdown("#### 7.3  Export EOWR as Markdown")
    st.info(
        "The Markdown export can be opened in any text editor, converted to "
        "Word via pandoc, or pasted into your company EOWR template.",
        icon="💡",
    )

    md = _build_markdown_export(
        ctx.campaign_start, ctx.campaign_end, ctx.campaign_days, ctx.td_ft,
        ctx.total_cost, ctx.afe, ctx.npt_pct, ctx.npt_h,
        ctx.ops, ctx.cas, ctx.fit_lot, ctx.comp,
    )
    st.download_button(
        "⬇ Download EOWR as Markdown",
        data=md,
        file_name="end_of_well_report.md",
        mime="text/markdown",
    )


def _build_markdown_export(
    campaign_start, campaign_end, campaign_days, td_ft,
    total_cost, afe, npt_pct, npt_h, ops, cas, fit_lot, comp,
) -> str:
    lines = [
        "# End of Well Report",
        f"*Auto-generated from DDR Intelligence Platform*  ",
        f"*Generated: {pd.Timestamp.now().strftime('%d %b %Y')}*",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "### Key Statistics",
        "| Parameter | Value |",
        "|---|---|",
        f"| Campaign Duration | {campaign_days} days |",
        f"| Spud Date | {campaign_start.strftime('%d %b %Y')} |",
        f"| Rig Release | {campaign_end.strftime('%d %b %Y')} |",
        f"| Total Depth | {td_ft:,.0f} ft MD |",
        f"| Total Cost | £{total_cost/1e6:.1f}M |",
        (f"| AFE | £{afe/1e6:.1f}M |" if afe else "| AFE | — |"),
        f"| Overall NPT | {npt_pct:.1f}% ({npt_h:,.0f} hrs) |",
        f"| Final Status | Producer |",
        "",
        "---",
        "",
        "## 3. Drilling Operations Summary",
        "",
        "### Phase Chronology",
        "| Phase | Start | End | Days | Total (h) | NPT (h) | NPT% |",
        "|---|---|---|---|---|---|---|",
    ]
    for ph in PHASE_ORDER:
        g = ops[ops["phase"] == ph]
        if g.empty:
            continue
        tot = g["duration_hr"].sum()
        npt = g[g["is_npt"]]["duration_hr"].sum()
        lines.append(
            f"| {label_phase(ph)} | {g['dt'].min().strftime('%d %b %Y')} | "
            f"{g['dt'].max().strftime('%d %b %Y')} | {g['dt'].nunique()} | "
            f"{tot:.0f} | {npt:.0f} | {100*npt/tot:.0f}% |"
        )

    lines += ["", "### Casing Programme",
              "| String | OD (in) | Set Depth (ft) | Run Date |",
              "|---|---|---|---|"]
    if not cas.empty:
        for _, r in (cas.drop_duplicates(["casing_description","set_depth_ft"])
                     .sort_values("set_depth_ft").iterrows()):
            lines.append(
                f"| {r['casing_description']} | {r['od_in']} | "
                f"{r['set_depth_ft']:,.0f} | {r['run_date']} |"
            )

    lines += ["", "### NPT by Category",
              "| Category | Hours | % of Total NPT |", "|---|---|---|"]
    npt_ops = (ops[ops["is_npt"]].groupby("npt_category")["duration_hr"]
               .sum().sort_values(ascending=False))
    for cat, h in npt_ops.items():
        lines.append(f"| {CATEGORY_LABELS.get(cat,cat)} | {h:.0f} | {100*h/npt_h:.1f}% |")

    lines += ["", "---", "", "## 4. Geological & Subsurface Data",
              "", "### Formation Integrity Tests"]
    if not fit_lot.empty:
        lines += ["| Date | Phase | Test | EMW (ppge) | Result |", "|---|---|---|---|---|"]
        for _, r in fit_lot.iterrows():
            ppge = f"{r['limit_ppge']:.2f}" if pd.notna(r.get("limit_ppge")) else "—"
            lines.append(
                f"| {r.get('report_date','—')} | {label_phase(r.get('phase',''))} | "
                f"{r.get('test_type','—')} | {ppge} | {r.get('result','—')} |"
            )

    lines += ["", "---", "", "## 6. Well Completion & Status",
              "", "### Completion String",
              "| Component | Depth (ft MD) | OD (in) | ID (in) | Source |",
              "|---|---|---|---|---|"]
    if not comp.empty:
        for _, r in comp.sort_values("depth_ft").iterrows():
            depth = f"{r['depth_ft']:,.0f}" if pd.notna(r.get("depth_ft")) else "—"
            od    = f"{r['od_in']:.3f}" if pd.notna(r.get("od_in")) else "—"
            idd   = f"{r['id_in']:.3f}" if pd.notna(r.get("id_in")) else "—"
            restr = " ◄ ID restriction" if r.get("is_id_restriction") else ""
            lines.append(
                f"| {r['component']}{restr} | {depth} | {od} | {idd} | "
                f"{r.get('depth_source','—')} |"
            )

    lines += ["", "---", "", "*End of Report*"]
    return "\n".join(lines)


def page_eowr() -> None:
    st.header("End of Well Report")

    with st.spinner("Compiling EOWR…"):
        ops     = load_all_ops()
        hdr     = load_all_headers()
        cas     = load_casing()
        fit_lot = load_fit_lot()
        events  = load_wellbore_events()
        comp    = load_completion_string()

    ops["dt"]          = pd.to_datetime(ops["report_date"], dayfirst=True, errors="coerce")
    ops["npt_category"] = classify_ops_df(ops)
    hdr["dt"]          = pd.to_datetime(hdr["report_date"], dayfirst=True, errors="coerce")
    hdr = hdr.sort_values("dt")

    _date_range = (
        f"{hdr['dt'].min():%b %Y} – {hdr['dt'].max():%b %Y}"
        if hdr["dt"].notna().any() else "—"
    )
    st.caption(
        f"Auto-generated from {len(hdr)} DDRs ({_date_range}). "
        "All data extracted from Daily Drilling Reports."
    )

    hdr["depth_n"] = hdr["end_depth_md_ft"].apply(_parse_num)
    hdr["cost_n"]  = hdr["daily_cost"].apply(_parse_num)
    hdr["cum_n"]   = hdr["cumulative_cost"].apply(_parse_num)
    hdr["afe_n"]   = hdr["afe_amt"].apply(_parse_num)

    td_ft          = hdr["depth_n"].max()
    total_cost     = hdr["cum_n"].max()
    afe            = hdr["afe_n"].dropna().iloc[0] if hdr["afe_n"].notna().any() else None
    kb_elev        = (hdr["kb_elevation_ft"].dropna().iloc[0]
                     if "kb_elevation_ft" in hdr.columns and hdr["kb_elevation_ft"].notna().any()
                     else None)
    water_depth    = (hdr["water_depth_ft"].dropna().iloc[0]
                     if "water_depth_ft" in hdr.columns and hdr["water_depth_ft"].notna().any()
                     else None)
    job_start      = hdr["job_start"].dropna().iloc[0] if hdr["job_start"].notna().any() else None
    campaign_start = hdr["dt"].min()
    campaign_end   = hdr["dt"].max()
    campaign_days  = (campaign_end - campaign_start).days + 1

    total_h = ops["duration_hr"].sum()
    npt_h   = ops[ops["is_npt"]]["duration_hr"].sum()
    npt_pct = 100 * npt_h / total_h if total_h > 0 else 0

    ctx = SimpleNamespace(
        ops=ops, hdr=hdr, cas=cas, fit_lot=fit_lot, events=events, comp=comp,
        td_ft=td_ft, total_cost=total_cost, afe=afe, npt_pct=npt_pct, npt_h=npt_h,
        campaign_start=campaign_start, campaign_end=campaign_end, campaign_days=campaign_days,
        kb_elev=kb_elev, water_depth=water_depth, job_start=job_start,
    )

    tabs = st.tabs([
        "1 · Executive Summary",
        "2 · General Information",
        "3 · Drilling Operations",
        "4 · Geological Data",
        "5 · Well Testing",
        "6 · Completion & Status",
        "7 · Appendices & Export",
    ])

    with tabs[0]: _render_eowr_section1(ctx)
    with tabs[1]: _render_eowr_section2(ctx)
    with tabs[2]: _render_eowr_section3(ctx)
    with tabs[3]: _render_eowr_section4(ctx)
    with tabs[4]: _render_eowr_section5(ctx)
    with tabs[5]: _render_eowr_section6(ctx)
    with tabs[6]: _render_eowr_section7(ctx)
