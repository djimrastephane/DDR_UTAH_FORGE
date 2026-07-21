from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from .constants import _COMP_COLOURS, _COMP_LABELS
    from .loaders import load_completion_string, load_frac_sleeve_status
    from .utils import _apply_chart_theme
except ImportError:
    from constants import _COMP_COLOURS, _COMP_LABELS  # type: ignore[no-redef]
    from loaders import load_completion_string, load_frac_sleeve_status  # type: ignore[no-redef]
    from utils import _apply_chart_theme  # type: ignore[no-redef]


def _render_tab_component_table(df: pd.DataFrame) -> None:
    restrictions = df[df["is_id_restriction"]].sort_values("depth_ft")
    if not restrictions.empty:
        parts = []
        for _, r in restrictions.iterrows():
            src = " _(calc.)_" if r["depth_source"] in ("calculated", "estimated") else ""
            parts.append(f"**{r['component']}** · {r['id_in']:.3f}\" bore · {r['depth_ft']:,.0f} ft MD{src}")
        st.error("**ID Restrictions:**  " + "   |   ".join(parts))

    calc_rows = df[df["depth_source"] == "calculated"]
    if not calc_rows.empty and "review_note" in df.columns:
        with st.expander(
            f"⚠️  {len(calc_rows)} component(s) with calculated depths — engineer review required",
            expanded=True,
        ):
            st.warning(
                "The following component depths are **not explicitly stated in the DDR**. "
                "They are derived from the completion running sequence and are provided as "
                "**indicative only**. Verify against the original completion tally document "
                "before any workover or intervention planning.",
                icon="⚠️",
            )
            for _, r in calc_rows.sort_values("depth_ft").iterrows():
                note = str(r.get("review_note") or "")
                if note:
                    st.markdown(
                        f"**{r['component']}** — indicated depth {r['depth_ft']:,.0f} ft MD  \n"
                        f"<span style='color:#B45309;font-size:0.88em'>{note}</span>",
                        unsafe_allow_html=True,
                    )

    row_fill, font_colours = [], []
    for _, r in df.iterrows():
        if r["is_id_restriction"]:
            row_fill.append("#FFCDD2")
            font_colours.append("#B71C1C")
        elif r["depth_source"] in ("calculated", "estimated"):
            row_fill.append("#FFF8E1")
            font_colours.append("#5D4037")
        else:
            row_fill.append("white")
            font_colours.append("#212121")

    flag_col  = ["ID RESTRICT" if r else ("est." if s in ("calculated","estimated") else "")
                 for r, s in zip(df["is_id_restriction"], df["depth_source"])]
    depth_col = [f"{v:,.1f}" if pd.notna(v) else "—" for v in df["depth_ft"]]
    od_col    = [f"{v:.3f}" if pd.notna(v) else "—" for v in df["od_in"]]
    id_col    = [f"{v:.3f}" if pd.notna(v) else "—" for v in df["id_in"]]
    drift_col = [f"{v:.3f}" if pd.notna(v) else "—" for v in df["drift_in"]]
    wt_col    = [f"{v:.1f}" if pd.notna(v) else "—" for v in df["weight_lbft"]]
    cite_col  = df["ddr_citation"].fillna("—").tolist() if "ddr_citation" in df.columns else ["—"] * len(df)

    review_col = []
    for _, r in df.iterrows():
        if r["depth_source"] == "calculated" and "review_note" in df.columns:
            note = str(r.get("review_note") or "")
            review_col.append("⚠ Depth not in DDR — indicative only. Verify vs tally." if note else "")
        else:
            review_col.append("")

    fig_tbl = go.Figure(go.Table(
        columnwidth=[6, 22, 10, 9, 7, 7, 7, 6, 8, 13, 10, 7, 14, 20],
        header=dict(
            values=["<b>Flag</b>", "<b>Component</b>", "<b>Type</b>",
                    "<b>Depth (ft MD)</b>", "<b>OD (in)</b>", "<b>ID (in)</b>",
                    "<b>Drift (in)</b>", "<b>Wt (ppf)</b>", "<b>Grade</b>",
                    "<b>Connection</b>", "<b>Vendor</b>", "<b>Source</b>",
                    "<b>DDR Citation</b>", "<b>Engineer Review</b>"],
            fill_color="#1565C0", font=dict(color="white", size=11),
            align="left", height=28,
        ),
        cells=dict(
            values=[
                flag_col,
                df["component"].tolist(),
                [_COMP_LABELS.get(t, t) for t in df["component_type"]],
                depth_col, od_col, id_col, drift_col, wt_col,
                df["grade"].fillna("—").tolist(),
                df["connection"].fillna("—").tolist(),
                df["vendor"].fillna("—").tolist(),
                df["depth_source"].tolist(),
                cite_col,
                review_col,
            ],
            fill_color=[row_fill] * 14,
            font=dict(color=[font_colours] * 14, size=10.5),
            align="left", height=24,
        ),
    ))
    fig_tbl.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=780)
    st.plotly_chart(fig_tbl, use_container_width=True)

    lc1, lc2, lc3 = st.columns(3)
    lc1.markdown("🔴 **ID RESTRICT** — bore < tubing ID (3.958\")")
    lc2.markdown("🟡 **est.** — depth from running-sequence calculation")
    lc3.markdown("⬜ _(blank)_ — depth confirmed in DDR text")

    csv_rows = pd.DataFrame({
        "Component": df["component"], "Type": df["component_type"],
        "Depth_ft_MD": df["depth_ft"], "OD_in": df["od_in"],
        "ID_in": df["id_in"], "Drift_in": df["drift_in"],
        "Weight_lbft": df["weight_lbft"], "Grade": df["grade"],
        "Connection": df["connection"], "Vendor": df["vendor"],
        "Depth_Source": df["depth_source"], "DDR_Citation": df.get("ddr_citation", ""),
        "ID_Restriction": df["is_id_restriction"],
    })
    st.download_button("⬇ Download CSV", data=csv_rows.to_csv(index=False),
                       file_name="completion_string.csv", mime="text/csv")


def _render_tab_depth_track(df: pd.DataFrame) -> None:
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.30, 0.70],
        shared_yaxes=True,
        subplot_titles=["String Schematic", "Bore Profile (ID vs Depth)"],
        horizontal_spacing=0.04,
    )

    max_depth = df["depth_ft"].max()

    CASINGS = [
        (20.0,    0,      1_812,     "#B0BEC5", "20\" Conductor",      1_400),
        (13.375,  0,      5_704,     "#90A4AE", "13-3/8\" Casing",     4_800),
        (9.875,   0,      12_067,    "#78909C", "9-7/8\" Casing",     10_800),
        (7.0,     11_631, max_depth, "#607D8B", "7\" × 5-1/2\" Liner", 13_500),
    ]
    for od, top, bot, col, label, lbl_y in CASINGS:
        hw = od / 2.0
        fig.add_shape(type="rect", row=1, col=1,
            x0=-hw, x1=-(hw - 0.45), y0=top, y1=bot,
            fillcolor=col, opacity=0.55, line_width=0)
        fig.add_shape(type="rect", row=1, col=1,
            x0=hw - 0.45, x1=hw, y0=top, y1=bot,
            fillcolor=col, opacity=0.55, line_width=0)
        fig.add_annotation(row=1, col=1, x=hw + 0.3, y=lbl_y,
            text=label, showarrow=False, xanchor="left",
            font=dict(size=8.5, color="#455A64"))

    TUBE_TYPES = {
        "tubing":              ("#1565C0", 0.7),
        "liner":               ("#546E7A", 0.6),
        "liner_hanger_packer": ("#F57F17", 0.5),
    }
    for _, r in df[df["component_type"].isin(TUBE_TYPES)].iterrows():
        col, w = TUBE_TYPES[r["component_type"]]
        top     = r.get("depth_top_ft") if pd.notna(r.get("depth_top_ft")) else 0
        bot     = r["depth_ft"]
        id_half = (r["id_in"] / 2.0) if pd.notna(r.get("id_in")) else w * 0.55
        od_half = (r["od_in"] / 2.0) if pd.notna(r.get("od_in")) else w
        fig.add_shape(type="rect", row=1, col=1,
            x0=-od_half, x1=-id_half, y0=top, y1=bot,
            fillcolor=col, opacity=0.70, line_width=0)
        fig.add_shape(type="rect", row=1, col=1,
            x0=id_half, x1=od_half, y0=top, y1=bot,
            fillcolor=col, opacity=0.70, line_width=0)

    POINT_TYPES = ["tubing_hanger","dhsv","crossover","gauge_mandrel",
                   "production_packer","frac_sleeve","float_shoe"]
    for _, r in df[df["component_type"].isin(POINT_TYPES)].iterrows():
        colour  = _COMP_COLOURS.get(r["component_type"], "#78909C")
        is_est  = r["depth_source"] in ("calculated","estimated")
        od_half = (r["od_in"] / 2.0) if pd.notna(r.get("od_in")) else 2.75
        id_half = (r["id_in"] / 2.0) if pd.notna(r.get("id_in")) else od_half * 0.85
        d       = r["depth_ft"]
        fig.add_shape(type="rect", row=1, col=1,
            x0=-od_half, x1=od_half, y0=d - 30, y1=d + 30,
            fillcolor=colour, opacity=0.80, line_width=0)
        fig.add_shape(type="rect", row=1, col=1,
            x0=-id_half, x1=id_half, y0=d - 30, y1=d + 30,
            fillcolor="white", opacity=1.0, line_width=0)
        fig.add_trace(go.Scatter(
            x=[0], y=[d], mode="markers",
            marker=dict(size=1, color=colour),
            hovertemplate=(
                f"<b>{r['component']}</b><br>"
                f"Depth: {d:,.1f} ft MD<br>"
                + (f"OD: {r['od_in']:.3f}\"<br>" if pd.notna(r.get('od_in')) else "")
                + (f"ID: {r['id_in']:.3f}\"<br>" if pd.notna(r.get('id_in')) else "")
                + (f"Source: {r['depth_source']}<br>" if is_est else "")
                + (f"DDR: {r['ddr_citation']}" if r.get('ddr_citation') else "")
                + "<extra></extra>"
            ),
            showlegend=False,
        ), row=1, col=1)

    bore_pts = (df[df["id_in"].notna()]
                .sort_values("depth_ft")
                [["depth_ft","id_in","component","is_id_restriction","depth_source"]]
                .copy())

    depths_step, ids_step, hover_step = [], [], []
    prev_id = None
    for _, r in bore_pts.iterrows():
        d   = r["depth_ft"]
        iid = r["id_in"]
        if prev_id is not None and prev_id != iid:
            depths_step.append(d)
            ids_step.append(prev_id)
            hover_step.append(f"transition to {r['component']}")
        depths_step.append(d)
        ids_step.append(iid)
        hover_step.append(r["component"])
        prev_id = iid
    if depths_step:
        depths_step.append(max_depth)
        ids_step.append(ids_step[-1])
        hover_step.append("TD")

    min_tubing_id = df.loc[df["component_type"] == "tubing", "id_in"].min()
    for i in range(len(depths_step) - 1):
        if ids_step[i] < min_tubing_id:
            fig.add_shape(type="rect", row=1, col=2,
                x0=ids_step[i] - 0.05, x1=ids_step[i] + 0.02,
                y0=depths_step[i], y1=depths_step[i + 1],
                fillcolor="#FFCDD2", opacity=0.55, line_width=0)

    for ref_id, ref_label, ref_col in [
        (3.958, "4-1/2\" tubing ID", "#1565C0"),
        (4.892, "5-1/2\" tubing / liner ID", "#546E7A"),
    ]:
        fig.add_vline(x=ref_id, row=1, col=2,
            line=dict(color=ref_col, width=1, dash="dot"))
        fig.add_annotation(row=1, col=2, x=ref_id, y=500,
            text=ref_label, showarrow=False, xanchor="center",
            font=dict(size=8, color=ref_col), textangle=-90)

    fig.add_trace(go.Scatter(
        x=ids_step, y=depths_step,
        mode="lines",
        line=dict(color="#1565C0", width=2.5, shape="hv"),
        fill="tozerox",
        fillcolor="rgba(21,101,192,0.08)",
        name="Bore ID",
        hovertemplate="Depth: %{y:,.0f} ft MD<br>ID: %{x:.3f}\"<extra></extra>",
    ), row=1, col=2)

    prev_id_ann = None
    for _, r in bore_pts.iterrows():
        iid = r["id_in"]
        d   = r["depth_ft"]
        if prev_id_ann is not None and iid != prev_id_ann:
            is_restr = iid < min_tubing_id
            fig.add_annotation(
                row=1, col=2,
                x=3.1, y=d,
                text=f"<b>{d:,.0f} ft</b>",
                showarrow=True, arrowhead=0, arrowwidth=1,
                ax=60, ay=0,
                arrowcolor="rgba(100,100,100,0.4)",
                xanchor="left",
                font=dict(size=8.5, color="#B71C1C" if is_restr else "#546E7A"),
            )
        prev_id_ann = iid

    for _, r in df[df["is_id_restriction"]].iterrows():
        fig.add_trace(go.Scatter(
            x=[r["id_in"]], y=[r["depth_ft"]],
            mode="markers+text",
            marker=dict(symbol="diamond", size=12,
                        color="#D32F2F", line=dict(width=1.5, color="white")),
            text=f"  {r['id_in']:.3f}\"  ◄",
            textposition="middle right",
            textfont=dict(size=10, color="#B71C1C"),
            name=r["component"],
            hovertemplate=(
                f"<b>ID Restriction: {r['component']}</b><br>"
                f"Bore: {r['id_in']:.3f}\"<br>"
                f"Depth: {r['depth_ft']:,.0f} ft MD<br>"
                f"Source: {r['depth_source']}"
                "<extra></extra>"
            ),
            showlegend=False,
        ), row=1, col=2)

    fig.update_yaxes(autorange="reversed", tickformat=",",
                     title_text="Depth (ft MD)", row=1, col=1)
    fig.update_yaxes(autorange="reversed", tickformat=",",
                     showticklabels=False, row=1, col=2)
    fig.update_xaxes(title_text="OD (in)", range=[-12, 18],
                     showgrid=False, zeroline=False, row=1, col=1)
    fig.update_xaxes(title_text="ID / Bore (in)", range=[3.0, 6.0],
                     showgrid=True, gridcolor="rgba(175,175,175,0.35)",
                     row=1, col=2)
    fig.update_layout(
        height=950, margin=dict(l=10, r=20, t=40, b=20),
        plot_bgcolor="white", showlegend=False,
        font=dict(size=11),
    )
    _apply_chart_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "**Left:** Completion string schematic — pipe walls proportional to OD/ID; "
        "grey = casing strings; blue = 4-1/2\" tubing; dark = 5-1/2\" liner.  "
        "**Right:** Bore profile step chart — available pass-through ID at every depth; "
        "red diamonds = ID restrictions; red shading = bore narrower than tubing ID (3.958\")."
    )


def _render_tab_frac_sleeve_status(df: pd.DataFrame) -> None:
    slv = load_frac_sleeve_status()
    if slv.empty:
        st.warning("Run `scripts/extract_frac_sleeve_status.py` to generate sleeve data.")
        return

    depth_map = df[df["component_type"] == "frac_sleeve"].set_index(
        df[df["component_type"] == "frac_sleeve"]["component"].str.extract(r"#(\d+)")[0].astype(int)
    )["depth_ft"].to_dict()
    slv["depth_ft"] = slv["sleeve_no"].map(depth_map)

    STATUS_COL = {
        "OPENED":        "#E8F5E9",
        "LOCATED":       "#FFF9C4",
        "NO INDICATION": "#FFF3E0",
        "NOT FRACKED":   "#F5F5F5",
    }
    REVIEW_COL  = "#FFE0B2"
    REVIEW_FONT = "#BF360C"

    n_review = int(slv.get("review_flag", pd.Series(False)).sum()) if "review_flag" in slv.columns else 0

    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    kc1.metric("Sleeves Opened",        int((slv["status"]=="OPENED").sum()))
    kc2.metric("Located (not open)",    int((slv["status"]=="LOCATED").sum()))
    kc3.metric("No Indication",         int((slv["status"]=="NO INDICATION").sum()))
    kc4.metric("Not Fracked (deepest)", int((slv["status"]=="NOT FRACKED").sum()))
    kc5.metric("Needs Review",          n_review,
               help="Rows where extracted data is internally contradictory")

    if n_review:
        st.warning(
            f"**{n_review} sleeve(s) flagged for engineer review** — "
            "highlighted in amber below. Data extracted from DDR text may be incomplete "
            "or contradictory; verify against original DDR reports before use.",
            icon="⚠️",
        )

    st.divider()

    status_counts = slv["status"].value_counts().reindex(
        ["OPENED","LOCATED","NO INDICATION","NOT FRACKED"], fill_value=0
    )
    fig_bar = go.Figure(go.Bar(
        x=status_counts.index,
        y=status_counts.values,
        marker_color=["#43A047","#FDD835","#FB8C00","#9E9E9E"],
        text=status_counts.values,
        textposition="outside",
    ))
    fig_bar.update_layout(
        height=220, margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor="white", showlegend=False,
        yaxis=dict(showgrid=True, gridcolor="rgba(175,175,175,0.35)"),
    )
    _apply_chart_theme(fig_bar)
    st.plotly_chart(fig_bar, use_container_width=True)

    tbl_data = []
    fill_cols, font_cols = [], []
    for _, r in slv.sort_values("sleeve_no").iterrows():
        is_review = bool(r.get("review_flag", False))
        fill_cols.append(REVIEW_COL if is_review else STATUS_COL.get(r["status"], "white"))
        font_cols.append(REVIEW_FONT if is_review else "#212121")

        prop_lbs   = r.get("proppant_lbs_total")
        review_txt = r.get("review_reason", "") or ""
        status_icon = r["status"]

        tbl_data.append({
            "Sleeve":           f"#{int(r['sleeve_no'])}",
            "Depth (ft MD)":    f"{r['depth_ft']:,.0f}" if pd.notna(r.get("depth_ft")) else "est.",
            "Status":           ("⚠ " if is_review else "") + status_icon,
            "Open (psi)":       f"{r['open_psi']:,.0f}" if pd.notna(r.get("open_psi")) else "—",
            "Brkover (psi)":    f"{r['breakover_psi']:,.0f}" if pd.notna(r.get("breakover_psi")) else "—",
            "ISIP (psi)":       f"{r['isip_psi']:,.0f}" if pd.notna(r.get("isip_psi")) else "—",
            "Proppant (lbs)":   f"{prop_lbs:,.0f}" if pd.notna(prop_lbs) else "—",
            "Vol (bbl)":        f"{r['vol_injected_bbl']:,.0f}" if pd.notna(r.get("vol_injected_bbl")) else "—",
            "First Op Date":    r.get("first_date") or "—",
            "DDR Citation":     r.get("ddr_citation") or "—",
            "Review Comment":   review_txt,
        })

    tbl_df = pd.DataFrame(tbl_data)
    fig_slv = go.Figure(go.Table(
        columnwidth=[5,9,13,9,9,8,10,7,10,14,22],
        header=dict(
            values=[f"<b>{c}</b>" for c in tbl_df.columns],
            fill_color="#1565C0", font=dict(color="white", size=11),
            align="left", height=28,
        ),
        cells=dict(
            values=[tbl_df[c].tolist() for c in tbl_df.columns],
            fill_color=[fill_cols] * len(tbl_df.columns),
            font=dict(color=[font_cols] * len(tbl_df.columns), size=10.5),
            align="left", height=26,
        ),
    ))
    fig_slv.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=580)
    st.plotly_chart(fig_slv, use_container_width=True)

    ll1, ll2, ll3, ll4 = st.columns(4)
    ll1.markdown("🟢 **OPENED** — confirmed open")
    ll2.markdown("🟡 **LOCATED** — packer set, no open confirm")
    ll3.markdown("🟠 **NO INDICATION** — profile not found")
    ll4.markdown("🟧 **Amber** — contradictory data, engineer review")

    st.caption(
        "Opening pressure = pressure at which sleeve shifted open.  "
        "Breakover = injection rate increased sharply (formation acceptance).  "
        "ISIP = Instantaneous Shut-In Pressure.  "
        "Proppant = total mass injected across all frac runs for that sleeve.  "
        "Sleeves #1–5 not reached during campaign (deepest, below fraced interval)."
    )


def _render_tab_pass_through(df: pd.DataFrame) -> None:
    st.subheader("Pass-Through Calculator")
    st.caption(
        "Enter a tool OD to find the maximum depth it can reach before hitting a "
        "restriction in the completion string."
    )
    st.warning(
        "**Disclaimer — bore ID alone does not guarantee tool access.**  "
        "Physical passage also depends on: **BHA length** (long assemblies may not navigate "
        "high-dogleg sections); **well trajectory** (doglegs, build/drop rates can prevent "
        "tool entry regardless of bore clearance); **drag and torque** in highly deviated "
        "wells; **centraliser ODs** if present.  "
        "This calculator checks bore restrictions only — consult the directional survey and "
        "BHA tally before committing to any intervention.",
        icon="⚠️",
    )

    tool_od = st.slider(
        "Tool OD (inches)", min_value=1.0, max_value=5.5,
        value=3.5, step=0.0625,
        format="%.4f\"",
    )

    bore_df = df[df["id_in"].notna()].sort_values("depth_ft")[
        ["depth_ft", "id_in", "component", "component_type", "ddr_citation"]
    ].copy()

    blocked = bore_df[bore_df["id_in"] <= tool_od].sort_values("depth_ft")
    clear   = bore_df[bore_df["id_in"] > tool_od].sort_values("depth_ft")

    if blocked.empty:
        max_depth = df["depth_ft"].max()
        st.success(
            f"**{tool_od:.3f}\" OD tool can reach TD ({max_depth:,.0f} ft MD)** "
            f"— no restrictions in the completion string."
        )
    else:
        first_block = blocked.iloc[0]
        max_reach   = first_block["depth_ft"]
        st.error(
            f"**{tool_od:.3f}\" OD tool is blocked at {max_reach:,.0f} ft MD** "
            f"by **{first_block['component']}** (bore {first_block['id_in']:.3f}\")"
        )

    sorted_bore = bore_df.sort_values("depth_ft")
    x_min = max(2.8, sorted_bore["id_in"].min() - 0.3)
    x_max = 5.6

    depths_s, ids_s = [], []
    prev_id = None
    for _, r in sorted_bore.iterrows():
        if prev_id is not None and prev_id != r["id_in"]:
            depths_s.append(r["depth_ft"])
            ids_s.append(prev_id)
        depths_s.append(r["depth_ft"])
        ids_s.append(r["id_in"])
        prev_id = r["id_in"]
    depths_s.append(df["depth_ft"].max())
    ids_s.append(ids_s[-1])

    fig_pt = go.Figure()

    fig_pt.add_trace(go.Scatter(
        x=ids_s + [x_max] * len(ids_s),
        y=depths_s + list(reversed(depths_s)),
        fill="toself", fillcolor="rgba(21,101,192,0.07)",
        line=dict(width=0), hoverinfo="skip", showlegend=False,
    ))

    for i in range(len(depths_s) - 1):
        if ids_s[i] <= tool_od:
            fig_pt.add_shape(
                type="rect",
                x0=x_min, x1=ids_s[i],
                y0=depths_s[i], y1=depths_s[i + 1],
                fillcolor="#FFCDD2", opacity=0.55, line_width=0,
            )

    fig_pt.add_trace(go.Scatter(
        x=ids_s, y=depths_s,
        mode="lines", line=dict(color="#1565C0", width=3, shape="hv"),
        name="Bore ID",
        hovertemplate="Depth: %{y:,.0f} ft MD<br>Bore: %{x:.3f}\"<extra></extra>",
    ))

    prev_id, seg_top, seg_name = None, 0, ""
    for _, r in sorted_bore.iterrows():
        cid = r["id_in"]
        if cid != prev_id:
            if prev_id is not None:
                mid_depth  = (seg_top + r["depth_ft"]) / 2
                is_blocked = prev_id <= tool_od
                fig_pt.add_annotation(
                    x=prev_id + 0.04, y=mid_depth,
                    text=f"<b>{prev_id:.3f}\"</b>  {seg_name[:28]}",
                    showarrow=False, xanchor="left",
                    font=dict(size=9.5,
                              color="#B71C1C" if is_blocked else "#1565C0"),
                )
                fig_pt.add_annotation(
                    x=x_min + 0.02, y=r["depth_ft"],
                    text=f"<b>{r['depth_ft']:,.0f} ft</b>",
                    showarrow=True, arrowhead=0, arrowwidth=1,
                    ax=55, ay=0,
                    arrowcolor="rgba(120,120,120,0.45)",
                    xanchor="left",
                    font=dict(size=8.5,
                              color="#B71C1C" if cid <= tool_od else "#455A64"),
                )
            seg_top  = r["depth_ft"]
            seg_name = r["component"].replace("4-1/2\"", "4½\"").replace("5-1/2\"", "5½\"")
            prev_id  = cid

    if prev_id is not None:
        mid_depth  = (seg_top + df["depth_ft"].max()) / 2
        is_blocked = prev_id <= tool_od
        fig_pt.add_annotation(
            x=prev_id + 0.04, y=mid_depth,
            text=f"<b>{prev_id:.3f}\"</b>  {seg_name[:28]}",
            showarrow=False, xanchor="left",
            font=dict(size=9.5,
                      color="#B71C1C" if is_blocked else "#1565C0"),
        )

    fig_pt.add_vline(
        x=tool_od,
        line=dict(color="#D32F2F", width=2, dash="dash"),
        annotation_text=f"Tool OD {tool_od:.4f}\"",
        annotation_position="top left",
        annotation_font=dict(color="#D32F2F", size=11),
    )

    for _, r in blocked.iterrows():
        fig_pt.add_trace(go.Scatter(
            x=[r["id_in"]], y=[r["depth_ft"]],
            mode="markers",
            marker=dict(symbol="x", size=13, color="#D32F2F",
                        line=dict(width=2.5)),
            hovertemplate=(
                f"<b>BLOCKED: {r['component']}</b><br>"
                f"Bore: {r['id_in']:.3f}\"<br>"
                f"Depth: {r['depth_ft']:,.0f} ft MD"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    fig_pt.update_yaxes(autorange="reversed", title="Depth (ft MD)",
                        tickformat=",", showgrid=True,
                        gridcolor="rgba(175,175,175,0.35)")
    fig_pt.update_xaxes(title="ID / Bore (in)", range=[x_min, x_max],
                        showgrid=True, gridcolor="rgba(175,175,175,0.35)",
                        dtick=0.25)
    fig_pt.update_layout(
        height=750, margin=dict(l=10, r=10, t=30, b=20),
        plot_bgcolor="white", showlegend=False,
    )
    _apply_chart_theme(fig_pt)
    st.plotly_chart(fig_pt, use_container_width=True)

    if not blocked.empty:
        st.markdown("**Restrictions encountered (shallowest first):**")
        restr_tbl = blocked[["component", "depth_ft", "id_in", "ddr_citation"]].copy()
        restr_tbl.columns = ["Component", "Depth (ft MD)", "Bore (in)", "DDR Citation"]
        restr_tbl["Depth (ft MD)"] = restr_tbl["Depth (ft MD)"].apply(lambda v: f"{v:,.0f}")
        restr_tbl["Bore (in)"]     = restr_tbl["Bore (in)"].apply(lambda v: f"{v:.3f}")
        st.dataframe(restr_tbl, use_container_width=True, hide_index=True)


def _render_tab_string_volumes(df: pd.DataFrame) -> None:
    st.subheader("Completion String Volumes")
    st.caption(
        "Calculated from pipe internal dimensions (ID) and interval lengths. "
        "1 bbl = 9,702 in³. Useful for kill-pill sizing, displacement, bullheading."
    )

    BBL = 9702.0

    vol_rows = []
    tubing_intervals = df[df["component_type"].isin(["tubing", "liner"])].sort_values("depth_ft")

    for _, r in tubing_intervals.iterrows():
        top = r.get("depth_top_ft")
        bot = r["depth_ft"]
        if not pd.notna(top) or not pd.notna(bot) or not pd.notna(r.get("id_in")):
            continue
        length_ft = bot - top
        length_in = length_ft * 12.0
        id_in     = r["id_in"]
        area_in2  = math.pi / 4.0 * id_in ** 2
        vol_bbl   = area_in2 * length_in / BBL

        vol_rows.append({
            "Section":        r["component"],
            "From (ft MD)":   f"{top:,.0f}",
            "To (ft MD)":     f"{bot:,.0f}",
            "Length (ft)":    f"{length_ft:,.0f}",
            "ID (in)":        f"{id_in:.3f}",
            "Vol (bbl)":      round(vol_bbl, 1),
            "Vol (m³)":       round(vol_bbl * 0.158987, 2),
        })

    if not vol_rows:
        st.info("No tubing interval data available.")
        return

    vol_df = pd.DataFrame(vol_rows)
    total_bbl = vol_df["Vol (bbl)"].sum()
    total_m3  = vol_df["Vol (m³)"].sum()

    vc1, vc2, vc3, vc4 = st.columns(4)
    vc1.metric("Total String Volume", f"{total_bbl:,.0f} bbl")
    vc2.metric("Total String Volume", f"{total_m3:,.1f} m³")
    tub_bbl   = vol_df[vol_df["Section"].str.contains("Tubing", case=False)]["Vol (bbl)"].sum()
    liner_bbl = vol_df[vol_df["Section"].str.contains("Liner|liner", case=False)]["Vol (bbl)"].sum()
    vc3.metric("Tubing Volume",  f"{tub_bbl:,.0f} bbl")
    vc4.metric("Liner Volume",   f"{liner_bbl:,.0f} bbl")

    st.divider()

    fig_vol = go.Figure(go.Bar(
        x=vol_df["Section"].str[:35],
        y=vol_df["Vol (bbl)"],
        text=vol_df["Vol (bbl)"].apply(lambda v: f"{v:,.0f}"),
        textposition="outside",
        marker_color=["#1565C0" if "Tubing" in s else "#546E7A" for s in vol_df["Section"]],
    ))
    fig_vol.update_layout(
        height=280, margin=dict(l=0, r=0, t=20, b=80),
        plot_bgcolor="white", showlegend=False,
        xaxis=dict(tickangle=-25),
        yaxis=dict(title="Volume (bbl)", showgrid=True,
                   gridcolor="rgba(175,175,175,0.35)"),
    )
    _apply_chart_theme(fig_vol)
    st.plotly_chart(fig_vol, use_container_width=True)

    vol_df["Vol (bbl)"] = vol_df["Vol (bbl)"].apply(lambda v: f"{v:,.1f}")
    vol_df["Vol (m³)"]  = vol_df["Vol (m³)"].apply(lambda v: f"{v:,.2f}")
    st.dataframe(vol_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**Annular Volumes (tubing OD → casing/liner ID)**")

    CASING_ID = {
        (0,     1_812):   ("20\" Conductor",   18.376),
        (0,     5_704):   ("13-3/8\" Casing",  12.347),
        (0,     12_067):  ("9-7/8\" Casing",    8.755),
        (11_631, 19_117): ("7\" Liner",          6.184),
    }

    ann_rows = []
    for _, r in tubing_intervals.iterrows():
        top = r.get("depth_top_ft")
        bot = r["depth_ft"]
        if not pd.notna(top) or not pd.notna(r.get("od_in")):
            continue
        for (cs_top, cs_bot), (cs_name, cs_id) in CASING_ID.items():
            ov_top = max(top, cs_top)
            ov_bot = min(bot, cs_bot)
            if ov_bot <= ov_top:
                continue
            length_in = (ov_bot - ov_top) * 12.0
            ann_area  = math.pi / 4.0 * (cs_id**2 - r["od_in"]**2)
            if ann_area <= 0:
                continue
            ann_bbl = ann_area * length_in / BBL
            ann_rows.append({
                "Interval":       f"{r['component'][:28]} in {cs_name}",
                "From (ft MD)":   f"{ov_top:,.0f}",
                "To (ft MD)":     f"{ov_bot:,.0f}",
                "Tubing OD (in)": f"{r['od_in']:.3f}",
                "Casing ID (in)": f"{cs_id:.3f}",
                "Ann Vol (bbl)":  f"{ann_bbl:,.1f}",
                "Ann Vol (m³)":   f"{ann_bbl*0.158987:,.2f}",
            })

    if ann_rows:
        ann_df = pd.DataFrame(ann_rows)
        total_ann = sum(float(r["Ann Vol (bbl)"].replace(",","")) for r in ann_rows)
        st.metric("Total Annular Volume", f"{total_ann:,.0f} bbl")
        st.dataframe(ann_df, use_container_width=True, hide_index=True)


def page_completion_string() -> None:
    df = load_completion_string()
    if df.empty:
        st.warning("Completion string data not found. Run `scripts/extract_completion_string.py`.")
        return

    st.header("Completion String")
    st.caption(
        "Completion components extracted from processed DDRs. "
        "Depth source (confirmed / calculated / estimated) is noted per component."
    )

    min_id = df["id_in"].dropna().min()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Components", len(df))
    c2.metric("Completion TD", f"{df['depth_ft'].max():,.0f} ft MD")
    c3.metric("Min Bore (ID)", f"{min_id:.3f}\"")
    c4.metric("Frac Sleeves", int((df["component_type"] == "frac_sleeve").sum()))
    c5.metric("ID Restrictions", int(df["is_id_restriction"].sum()))

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Component Table", "📊 Depth Track",
        "🌡 Frac Sleeve Status", "🔧 Pass-Through Calculator", "🧮 String Volumes",
    ])

    with tab1:
        _render_tab_component_table(df)

    with tab2:
        _render_tab_depth_track(df)

    with tab3:
        _render_tab_frac_sleeve_status(df)

    with tab4:
        _render_tab_pass_through(df)

    with tab5:
        _render_tab_string_volumes(df)
