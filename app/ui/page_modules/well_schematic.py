from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from .loaders import load_wellbore_events, load_casing, load_fit_lot, load_all_headers
    from .utils import _apply_chart_theme, _ddr_citation_row
except ImportError:
    from loaders import load_wellbore_events, load_casing, load_fit_lot, load_all_headers  # type: ignore[no-redef]
    from utils import _apply_chart_theme, _ddr_citation_row              # type: ignore[no-redef]


_COMP_COLOURS: dict[str, str] = {
    "tubing_hanger":       "#1565C0",
    "tubing":              "#90CAF9",
    "dhsv":                "#D32F2F",
    "crossover":           "#7B1FA2",
    "gauge_mandrel":       "#00838F",
    "production_packer":   "#E65100",
    "liner_hanger_packer": "#F57F17",
    "liner":               "#B0BEC5",
    "frac_sleeve":         "#FF8F00",
    "float_shoe":          "#455A64",
}

_COMP_LABELS: dict[str, str] = {
    "tubing_hanger":       "Tubing Hanger",
    "tubing":              "Tubing",
    "dhsv":                "DHSV",
    "crossover":           "Crossover",
    "gauge_mandrel":       "Gauge Mandrel",
    "production_packer":   "Production Packer",
    "liner_hanger_packer": "Liner Hanger Packer",
    "liner":               "Liner",
    "frac_sleeve":         "Frac Sleeve",
    "float_shoe":          "Float Shoe",
}


def _parse_od(val: object) -> float | None:
    s = str(val).strip()
    m = re.match(r"^(\d+)\s+(\d+)/(\d+)$", s)
    if m:
        return int(m.group(1)) + int(m.group(2)) / int(m.group(3))
    try:
        return float(s)
    except ValueError:
        return None


def _prepare_casing_df(casing: pd.DataFrame) -> pd.DataFrame:
    df = (
        casing.dropna(subset=["set_depth_ft"])
        .copy()
        .drop_duplicates(subset=["set_depth_ft"], keep="last")
        .sort_values("set_depth_ft")
        .reset_index(drop=True)
    )
    df["od_in"] = df["od_in"].apply(_parse_od)
    df.loc[
        df["casing_description"].str.contains('7"', na=False, regex=False)
        & df["od_in"].isna(), "od_in"
    ] = 7.0
    df = df.dropna(subset=["od_in"])
    for i in df.index:
        if pd.isna(df.at[i, "top_depth_ft"]) and i > 0:
            df.at[i, "top_depth_ft"] = df.at[i - 1, "set_depth_ft"]
    df["top_depth_ft"] = df["top_depth_ft"].fillna(0.0)
    return df


_BIT_SIZES = [(0.0, 1812.2, 26.0), (1812.2, 5704.0, 17.5),
              (5704.0, 12067.0, 12.25), (12067.0, None, 8.5)]

_MAX_X = 14.5
_WALL  = 0.65
_CTYPE_COL = {
    "Conductor":         "#455A64",
    "Intermediate":      "#6D4C41",
    "Production casing": "#1565C0",
    "Liner":             "#6A1B9A",
}
_CEMENT = "rgba(255,243,160,0.80)"
_FORM   = "rgba(193,154,107,0.30)"
_HOLE   = "rgba(248,250,255,0.96)"

_PHASE_COL_S = {
    "COND1": "#8D6E63", "INTRM1": "#1E88E5",
    "INTRM2": "#43A047", "PROD1": "#FB8C00",
}
_PHASE_LBL_S = {
    "COND1": "Conductor", "INTRM1": "Intermediate 1",
    "INTRM2": "Intermediate 2", "PROD1": "Production",
}
_FIT_RES_COL = {"pass": "#1565C0", "initiation": "#B71C1C", "unknown": "#757575"}

_FORM_STYLE = {
    "high_ecd":      ("#7B1FA2", "star",               "High ECD"),
    "gas_influx":    ("#C62828", "star-triangle-up",   "Gas Influx"),
    "instability":   ("#E65100", "star-diamond",       "Instability"),
    "ballooning":    ("#00695C", "star-square",        "Ballooning"),
    "diff_sticking": ("#1A237E", "star-triangle-down", "Diff. Sticking"),
}


def _fill(x0, x1, y0, y1, fillcolor, line_color="rgba(0,0,0,0)", lw=0.4,
          name="", showlegend=False, legendgroup="", hovertemplate="<extra></extra>"):
    return go.Scatter(
        x=[x0, x1, x1, x0, x0], y=[y0, y0, y1, y1, y0],
        fill="toself", fillcolor=fillcolor,
        line=dict(color=line_color, width=lw),
        mode="lines", name=name,
        showlegend=showlegend, legendgroup=legendgroup,
        hovertemplate=hovertemplate,
    )


def _build_schematic_left_panel(
    fig: go.Figure,
    df: pd.DataFrame,
    td: float,
    fit_lot: pd.DataFrame,
    anonymise: bool = False,
) -> None:
    def _y(ft: float) -> float:
        return ft / td * 100 if anonymise else ft

    def _hole_half(depth_ft: float) -> float:
        for frm, to, bit in _BIT_SIZES:
            if to is None:
                to = td
            if frm <= depth_ft <= to:
                return bit / 2
        return 4.25

    y_bottom = _y(td)
    fig.add_trace(_fill(-_MAX_X, _MAX_X, 0, y_bottom, _FORM), row=1, col=1)

    for frm, to, bit in _BIT_SIZES:
        if to is None:
            to = td
        fig.add_trace(
            _fill(-bit / 2, bit / 2, _y(frm), _y(to), _HOLE,
                  "rgba(150,150,150,0.20)", 0.4),
            row=1, col=1,
        )

    for _, row_c in df.iterrows():
        od   = row_c["od_in"]
        shoe = row_c["set_depth_ft"]
        top  = row_c["top_depth_ft"]
        half_hole = _hole_half(shoe)
        half_cas  = od / 2 + _WALL
        fig.add_trace(_fill(-half_hole, -half_cas, _y(top), _y(shoe),
                            _CEMENT, "rgba(180,160,80,0.20)", 0.3), row=1, col=1)
        fig.add_trace(_fill( half_cas,  half_hole, _y(top), _y(shoe),
                            _CEMENT, "rgba(180,160,80,0.20)", 0.3), row=1, col=1)

    for idx, row_c in df.iterrows():
        od    = row_c["od_in"]
        shoe  = row_c["set_depth_ft"]
        top   = row_c["top_depth_ft"]
        ctype = row_c["casing_type"]
        name  = row_c["casing_description"]
        col   = _CTYPE_COL.get(ctype, "#555")
        half  = od / 2
        first = bool(idx == df[df["casing_type"] == ctype].index[0])

        if anonymise:
            shoe_label = f"~{round(shoe/1000)}k ft" if round(shoe/1000) > 0 else "Surface"
        else:
            shoe_label = f"{shoe:,.0f} ft"

        ht = (f"<b>{name}</b><br>"
              f"OD: {od}\"  ·  Top: {_y(top):.0f}{'%' if anonymise else ' ft'}  "
              f"·  Shoe: {shoe_label}"
              "<extra></extra>")
        for x0, x1 in [(-(half + _WALL), -half), (half, half + _WALL)]:
            fig.add_trace(_fill(x0, x1, _y(top), _y(shoe), col, col, 0.8,
                                name=name, showlegend=first, legendgroup=name,
                                hovertemplate=ht), row=1, col=1)
            first = False
        fig.add_shape(type="line",
            x0=-(half + _WALL + 1.8), x1=(half + _WALL + 1.8),
            y0=_y(shoe), y1=_y(shoe),
            line=dict(color=col, width=2.2),
            row=1, col=1)
        fig.add_annotation(
            x=_MAX_X - 0.3, y=_y(shoe),
            xref="x", yref="y",
            text=f"<b>{od}\" shoe</b>  {shoe_label}",
            showarrow=False, font=dict(size=8, color=col),
            xanchor="right", yanchor="bottom",
        )

    phase_bands = [
        ("COND1",   0.0,     1812.2),
        ("INTRM1",  1812.2,  5704.0),
        ("INTRM2",  5704.0, 12067.0),
        ("PROD1",  12067.0,     td),
    ]
    for phase, top_ft, bot_ft in phase_bands:
        mid_y  = (_y(top_ft) + _y(bot_ft)) / 2
        col_ph = _PHASE_COL_S[phase]
        if top_ft > 0:
            fig.add_shape(type="line", x0=-_MAX_X, x1=_MAX_X,
                          y0=_y(top_ft), y1=_y(top_ft),
                          line=dict(color=col_ph, width=0.9, dash="dot"), row=1, col=1)
        fig.add_annotation(
            x=-_MAX_X + 0.4, y=mid_y, xref="x", yref="y",
            text=f"<b>{_PHASE_LBL_S[phase]}</b>",
            showarrow=False, textangle=-90,
            font=dict(size=8, color=col_ph, family="Arial"),
            xanchor="left", yanchor="middle",
        )

    if not fit_lot.empty:
        fit_shoe_depths = {
            '20" shoe': 1812.2, '13-3/8" shoe': 5704.0, '9-7/8" shoe': 12067.0,
        }
        for _, lr in fit_lot.drop_duplicates("limit_ppge").iterrows():
            dep_ft = fit_shoe_depths.get(str(lr.get("casing_shoe", "")))
            if dep_ft is None:
                continue
            col_f = _FIT_RES_COL.get(lr.get("result", "unknown"), "#757575")
            if anonymise:
                symbol = "✓" if lr.get("result") == "pass" else "⚠"
                ann_text = f"{lr['test_type']} {symbol}"
                ann_y = _y(dep_ft) + 2
            else:
                cit = _ddr_citation_row(lr)
                ann_text = f"{lr['test_type']}: <b>{lr['limit_ppge']:.2f} ppge</b>  ({cit})"
                ann_y = dep_ft + 350
            fig.add_annotation(
                x=0.5, y=ann_y, xref="x", yref="y",
                text=ann_text,
                showarrow=True, arrowhead=2, arrowsize=0.9,
                arrowwidth=1.2, arrowcolor=col_f,
                ax=60 if anonymise else 70, ay=0,
                font=dict(size=8 if not anonymise else 9, color=col_f),
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor=col_f, borderwidth=1, borderpad=2,
            )

    last_od = float(df["od_in"].min()) / 2 + _WALL
    fig.add_shape(type="line", x0=-last_od, x1=last_od,
                  y0=_y(td), y1=_y(td),
                  line=dict(color="#B71C1C", width=3.5), row=1, col=1)
    td_label = "<b>TD  ≈ 19k ft</b>" if anonymise else f"<b>TD  {td:,.0f} ft MD</b>"
    ann_y_td = _y(td) + (2 if anonymise else 500)
    fig.add_annotation(
        x=0, y=ann_y_td, xref="x", yref="y",
        text=td_label,
        showarrow=False, font=dict(size=9, color="#B71C1C"), xanchor="center",
    )


def page_well_schematic() -> None:
    casing  = load_casing()
    events  = load_wellbore_events()
    fit_lot = load_fit_lot()
    n_ddrs  = len(load_all_headers())

    if casing.empty:
        st.warning("No casing data. Run `scripts/extract_well_sections.py` first.")
        return

    df = _prepare_casing_df(casing)
    TD = float(df["set_depth_ft"].max())

    st.header("Well Schematic")
    st.caption(f"Casing programme: {len(df)} strings · TD {TD:,.0f} ft MD")
    N_EV = len(events) if not events.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Depth", f"{TD:,.0f} ft MD")
    c2.metric("Casing Strings", str(len(df)))
    c3.metric("Wellbore Events", str(N_EV))
    c4.metric("FIT / LOT Tests", str(len(fit_lot)) if not fit_lot.empty else "0")

    if not events.empty:
        _ev_d = events.dropna(subset=["event_depth_ft_md"])
        _prod_pct = int(_ev_d["event_depth_ft_md"].between(12000, TD).sum()
                        / max(len(_ev_d), 1) * 100)
        _peak_op  = events["force_klbs"].max()
        _bins     = pd.cut(_ev_d["event_depth_ft_md"],
                           bins=range(0, int(TD) + 1001, 1000)).value_counts()
        _dense_lo = int(_bins.index[0].left)
        _dense_hi = int(_bins.index[0].right)
        st.info(
            f"**{len(events):,} wellbore events** across {n_ddrs} DDRs  ·  "
            f"**{_prod_pct}% in the production section** (12,000 ft – TD)  ·  "
            f"Peak overpull **{_peak_op:.0f} klbs**  ·  "
            f"Highest density **{_dense_lo:,}–{_dense_hi:,} ft MD**"
        )
    st.divider()

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.38, 0.62],
        shared_yaxes=True,
        subplot_titles=["Well Schematic (depth reference)", "Operational Timeline by Depth"],
        horizontal_spacing=0.04,
    )

    _build_schematic_left_panel(fig, df, TD, fit_lot, anonymise=False)

    if not events.empty:
        ev = events.copy()
        ev["report_date_dt"] = pd.to_datetime(ev["report_date_dt"])
        ev["_citation"] = ev.apply(lambda r: _ddr_citation_row(r), axis=1)
        ev["_excerpt"] = (
            ev["full_op_text"].fillna("").str.replace(r"\s+", " ", regex=True).str[:180]
        )
        ev["_npt"] = ev["is_npt"].map({True: "⚠ NPT", False: "Productive"}).fillna("")

        def _severity_size(forces, lo=6, hi=22):
            f = forces.fillna(5).clip(1, 400)
            return (np.sqrt(f / 5) * 5.5 + lo).clip(lo, hi)

        _op = ev[ev["event_type"] == "overpull"].dropna(subset=["event_depth_ft_md"])
        if not _op.empty:
            _f = _op["force_klbs"].fillna(5)
            fig.add_trace(go.Scatter(
                x=_op["report_date_dt"], y=_op["event_depth_ft_md"],
                mode="markers", name="Overpull",
                marker=dict(symbol="circle", size=_severity_size(_f),
                            color=_f, colorscale=[[0,"#1976D2"],[0.4,"#7B1FA2"],
                                                   [0.7,"#E65100"],[1.0,"#B71C1C"]],
                            cmin=5, cmax=max(float(_f.max()), 50),
                            colorbar=dict(title=dict(text="Force (klbs)", font=dict(size=10)),
                                          thickness=12, len=0.45, x=1.02,
                                          tickfont=dict(size=9), outlinewidth=0),
                            showscale=True, opacity=0.85, line=dict(width=1, color="white")),
                customdata=_op[["_citation", "_excerpt", "force_klbs", "hole_type", "_npt"]].values,
                hovertemplate=(
                    "<b>Overpull</b>  ·  %{x|%d %b %Y}<br>"
                    "Depth: <b>%{y:,.0f} ft MD</b>  ·  %{customdata[4]}<br>"
                    "Force: <b>%{customdata[2]:.0f} klbs</b>  ·  %{customdata[3]}<br>"
                    "<span style='font-size:10px;color:#888'>%{customdata[0]}</span><br>"
                    "<span style='font-size:11px;color:#444'>%{customdata[1]}</span>"
                    "<extra></extra>"
                ),
            ), row=1, col=2)

        _rs = ev[ev["event_type"] == "restriction"].dropna(subset=["event_depth_ft_md"])
        if not _rs.empty:
            _f = _rs["force_klbs"].fillna(10)
            fig.add_trace(go.Scatter(
                x=_rs["report_date_dt"], y=_rs["event_depth_ft_md"],
                mode="markers", name="Restriction",
                marker=dict(symbol="diamond", size=_severity_size(_f, 9, 22),
                            color="#F57C00", opacity=0.92, line=dict(width=1.5, color="white")),
                customdata=_rs[["_citation", "_excerpt", "force_klbs", "hole_type", "_npt"]].values,
                hovertemplate=(
                    "<b>Restriction / Packoff</b>  ·  %{x|%d %b %Y}<br>"
                    "Depth: <b>%{y:,.0f} ft MD</b>  ·  %{customdata[4]}<br>"
                    "Force: <b>%{customdata[2]:.0f} klbs</b>  ·  %{customdata[3]}<br>"
                    "<span style='font-size:10px;color:#888'>%{customdata[0]}</span><br>"
                    "<span style='font-size:11px;color:#444'>%{customdata[1]}</span>"
                    "<extra></extra>"
                ),
            ), row=1, col=2)

        _ml = ev[ev["event_type"] == "mud_loss"].dropna(subset=["event_depth_ft_md"])
        if not _ml.empty:
            _rate = _ml["loss_rate_bbl_hr"].fillna(10)
            _sz   = _severity_size(_rate, 10, 22)
            fig.add_trace(go.Scatter(
                x=_ml["report_date_dt"], y=_ml["event_depth_ft_md"],
                mode="markers", name="Mud Loss",
                marker=dict(symbol="square", size=_sz,
                            color="#1565C0", opacity=0.90, line=dict(width=1.5, color="white")),
                customdata=_ml[["_citation", "_excerpt", "loss_rate_bbl_hr", "severity", "_npt"]].values,
                hovertemplate=(
                    "<b>Mud Loss</b>  ·  %{x|%d %b %Y}<br>"
                    "Depth: <b>%{y:,.0f} ft MD</b>  ·  %{customdata[4]}<br>"
                    "Rate: <b>%{customdata[2]:.0f} bbl/hr</b>  ·  %{customdata[3]}<br>"
                    "<span style='font-size:10px;color:#888'>%{customdata[0]}</span><br>"
                    "<span style='font-size:11px;color:#444'>%{customdata[1]}</span>"
                    "<extra></extra>"
                ),
            ), row=1, col=2)

        _fm = ev[ev["event_type"] == "formation"].dropna(subset=["event_depth_ft_md"])
        for stype, (col_f, sym_f, lbl_f) in _FORM_STYLE.items():
            _sub = _fm[_fm["sub_type"] == stype]
            if _sub.empty:
                continue
            _cd = _sub[["_citation", "_excerpt", "_npt"]].copy()
            _cd["_ecd"] = (
                _sub["ecd_ppge"].apply(lambda v: f"{v:.2f} ppge" if pd.notna(v) else "—")
                if "ecd_ppge" in _sub.columns else "—"
            )
            fig.add_trace(go.Scatter(
                x=_sub["report_date_dt"], y=_sub["event_depth_ft_md"],
                mode="markers", name=lbl_f,
                marker=dict(symbol=sym_f, size=14,
                            color=col_f, opacity=0.88, line=dict(width=1.2, color="white")),
                customdata=_cd[["_citation", "_excerpt", "_npt", "_ecd"]].values,
                hovertemplate=(
                    f"<b>{lbl_f}</b>  ·  %{{x|%d %b %Y}}<br>"
                    "Depth: <b>%{y:,.0f} ft MD</b>  ·  %{customdata[2]}<br>"
                    "ECD: %{customdata[3]}<br>"
                    "<span style='font-size:10px;color:#888'>%{customdata[0]}</span><br>"
                    "<span style='font-size:11px;color:#444'>%{customdata[1]}</span>"
                    "<extra></extra>"
                ),
            ), row=1, col=2)

        for _, row_c in df.iterrows():
            _col = _CTYPE_COL.get(row_c["casing_type"], "#aaa")
            fig.add_hline(y=row_c["set_depth_ft"],
                line_dash="dot", line_color=_col, line_width=1.0, row=1, col=2)

        _phase_starts = (ev.groupby("phase")["report_date_dt"].min()
                         .reindex(["COND1","INTRM1","INTRM2","PROD1","COMPZN"])
                         .dropna())
        for phase, start_dt in _phase_starts.items():
            if phase == "COND1":
                continue
            _cp  = _PHASE_COL_S.get(phase, "#555")
            _iso = start_dt.isoformat()
            fig.add_shape(type="line",
                x0=_iso, x1=_iso, y0=0, y1=1, yref="paper", xref="x2",
                line=dict(color=_cp, width=1.3, dash="dash"))
            fig.add_annotation(
                x=_iso, y=0.98, xref="x2", yref="paper",
                text=f"<b>{_PHASE_LBL_S.get(phase, phase)}</b>",
                showarrow=False, textangle=-90,
                font=dict(size=9, color=_cp), xanchor="left", yanchor="top",
            )

        _top_op = (_op.dropna(subset=["force_klbs"]).nlargest(1, "force_klbs")
                   if not _op.empty else pd.DataFrame())
        if not _top_op.empty:
            _r = _top_op.iloc[0]
            fig.add_annotation(
                x=_r["report_date_dt"], y=_r["event_depth_ft_md"],
                xref="x2", yref="y",
                text=f"⚠ Peak overpull at depth<br>{_r['force_klbs']:.0f} klbs",
                showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor="#B71C1C",
                ax=-65, ay=-35,
                font=dict(size=9, color="#B71C1C"),
                bgcolor="rgba(255,255,255,0.90)",
                bordercolor="#B71C1C", borderwidth=1, borderpad=3,
            )

        _total_loss = (_ml[_ml.get("severity", pd.Series()) == "total"]
                       .sort_values("report_date_dt") if not _ml.empty else pd.DataFrame())
        if not _total_loss.empty:
            _rl = _total_loss.iloc[0]
            fig.add_annotation(
                x=_rl["report_date_dt"], y=_rl["event_depth_ft_md"],
                xref="x2", yref="y",
                text="💧 Total losses<br>51 bbl/hr",
                showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor="#1565C0",
                ax=65, ay=30,
                font=dict(size=9, color="#1565C0"),
                bgcolor="rgba(255,255,255,0.90)",
                bordercolor="#1565C0", borderwidth=1, borderpad=3,
            )

        if not _fm.empty:
            _fbins = pd.cut(_fm["event_depth_ft_md"],
                            bins=range(0, int(TD) + 2001, 2000)).value_counts()
            _fb    = _fbins.index[0]
            fig.add_hrect(
                y0=_fb.left, y1=_fb.right,
                fillcolor="rgba(123,31,162,0.07)", line_width=0,
                annotation_text="Formation risk interval",
                annotation_position="top right",
                annotation_font_size=9, annotation_font_color="#7B1FA2",
                row=1, col=2,
            )

    fig.update_yaxes(
        autorange="reversed", title_text="Depth (ft MD)", tickformat=",",
        row=1, col=1,
    )
    fig.update_yaxes(autorange="reversed", tickformat=",", row=1, col=2)
    fig.update_xaxes(
        showticklabels=False, showgrid=False, zeroline=False,
        range=[-_MAX_X - 0.5, _MAX_X + 0.5], row=1, col=1,
    )
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_layout(
        height=960,
        plot_bgcolor="white", paper_bgcolor="white",
        title=dict(
            text="Operational Events by Time and Depth",
            font=dict(size=15, color="rgb(30,30,30)"),
            x=0.5, xanchor="center",
        ),
        legend=dict(
            orientation="h", y=-0.06, font=dict(size=10),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(180,180,180,0.5)", borderwidth=1,
        ),
        margin=dict(l=60, r=90, t=80, b=70),
    )
    _apply_chart_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Left panel acts as depth reference. "
        "Right panel: circle = overpull (sized by force), "
        "diamond = restriction, square = mud loss, star = formation event. "
        "Dashed verticals = phase transitions."
    )

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.subheader("Casing Programme")
        tbl_c = df[["casing_description", "casing_type", "od_in",
                     "top_depth_ft", "set_depth_ft", "weight_lb_per_ft", "run_date"]].copy()
        tbl_c["od_in"]            = tbl_c["od_in"].apply(lambda v: f'{v}"' if pd.notna(v) else "—")
        tbl_c["top_depth_ft"]     = tbl_c["top_depth_ft"].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
        tbl_c["set_depth_ft"]     = tbl_c["set_depth_ft"].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
        tbl_c["weight_lb_per_ft"] = tbl_c["weight_lb_per_ft"].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
        st.dataframe(
            tbl_c.rename(columns={
                "casing_description": "String", "casing_type": "Type",
                "od_in": "OD", "top_depth_ft": "Top (ft MD)",
                "set_depth_ft": "Shoe (ft MD)", "weight_lb_per_ft": "Weight (lb/ft)",
                "run_date": "Run Date",
            }),
            use_container_width=True, hide_index=True,
        )

    with col_t2:
        st.subheader("Formation Integrity Tests")
        if not fit_lot.empty:
            tbl_f = fit_lot.copy()
            tbl_f["Source"] = tbl_f.apply(lambda r: _ddr_citation_row(r), axis=1)
            tbl_f = tbl_f[["report_date", "phase", "test_type",
                            "limit_ppge", "result", "casing_shoe", "Source"]]
            tbl_f["limit_ppge"] = tbl_f["limit_ppge"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
            st.dataframe(
                tbl_f.rename(columns={
                    "report_date": "Date", "phase": "Phase", "test_type": "Test",
                    "limit_ppge": "Limit (ppge)", "result": "Result",
                    "casing_shoe": "Casing Shoe",
                }),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No FIT/LOT data found.")


def page_well_schematic_linkedin() -> None:
    st.header("Well Schematic — Anonymised for Publication")
    st.caption(
        "Offshore extended-reach development well · 4-string casing programme · "
        "All depths and dates normalised · Force values categorised · "
        "Suitable for LinkedIn / conference use"
    )

    casing  = load_casing()
    events  = load_wellbore_events()
    fit_lot = load_fit_lot()

    if casing.empty:
        st.warning("No casing data available.")
        return

    df = _prepare_casing_df(casing)
    TD = float(df["set_depth_ft"].max())

    def _d(ft: float) -> float:
        return ft / TD * 100

    def _force_cat(klbs) -> str:
        if pd.isna(klbs) or klbs <= 0:
            return "Low"
        if klbs <= 10:
            return "Low"
        if klbs <= 30:
            return "Medium"
        return "High"

    FORCE_SIZE = {"Low": 8,        "Medium": 14,      "High": 20}
    FORCE_COL  = {"Low": "#1976D2","Medium": "#E65100","High": "#B71C1C"}

    ev = pd.DataFrame()
    start_day = None
    if not events.empty:
        ev = events.copy()
        ev["report_date_dt"] = pd.to_datetime(ev["report_date_dt"])
        start_day = ev["report_date_dt"].min()
        ev["_day"]       = (ev["report_date_dt"] - start_day).dt.days + 1
        ev["_depth_pct"] = ev["event_depth_ft_md"].apply(
            lambda d: _d(d) if pd.notna(d) else float("nan"))
        ev["_force_cat"] = ev["force_klbs"].apply(_force_cat)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Depth", "≈ 19k ft")
    c2.metric("Casing Strings", str(len(df)))
    c3.metric("Wellbore Events", str(len(events)) if not events.empty else "0")
    c4.metric("Integrity Tests", str(len(fit_lot)) if not fit_lot.empty else "0")

    if not ev.empty:
        _ev_d = ev.dropna(subset=["event_depth_ft_md"])
        _prod_pct = int(_ev_d["event_depth_ft_md"].between(12000, TD).sum()
                        / max(len(_ev_d), 1) * 100)
        _dense_lo = int(_ev_d["_depth_pct"].dropna().quantile(0.40))
        _dense_hi = int(_ev_d["_depth_pct"].dropna().quantile(0.60))
        st.info(
            f"**{len(events):,} wellbore events** extracted automatically from daily drilling reports  ·  "
            f"**{_prod_pct}% concentrated in the production section**  ·  "
            "High-severity overpull recorded  ·  "
            f"Highest event density between **{_dense_lo}%–{_dense_hi}% of well depth**"
        )
    st.divider()

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.38, 0.62],
        shared_yaxes=True,
        subplot_titles=["Well Schematic (depth reference)", "Operational Timeline"],
        horizontal_spacing=0.04,
    )

    _build_schematic_left_panel(fig, df, TD, fit_lot, anonymise=True)

    if not ev.empty and start_day is not None:
        _op = ev[ev["event_type"] == "overpull"].dropna(subset=["_depth_pct"])
        for cat in ["Low", "Medium", "High"]:
            _sub = _op[_op["_force_cat"] == cat]
            if _sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=_sub["_day"], y=_sub["_depth_pct"],
                mode="markers", name=f"Overpull – {cat}", legendgroup="overpull",
                marker=dict(symbol="circle", size=FORCE_SIZE[cat],
                            color=FORCE_COL[cat], opacity=0.85,
                            line=dict(width=1, color="white")),
                hovertemplate=(
                    f"<b>Overpull – {cat}</b>  ·  Day %{{x}}<br>"
                    "Depth: <b>%{y:.0f}%</b> of TD<extra></extra>"
                ),
            ), row=1, col=2)

        _rs = ev[ev["event_type"] == "restriction"].dropna(subset=["_depth_pct"])
        if not _rs.empty:
            fig.add_trace(go.Scatter(
                x=_rs["_day"], y=_rs["_depth_pct"],
                mode="markers", name="Restriction",
                marker=dict(symbol="diamond",
                            size=_rs["_force_cat"].map(FORCE_SIZE).fillna(10),
                            color="#F57C00", opacity=0.92,
                            line=dict(width=1.5, color="white")),
                hovertemplate=(
                    "<b>Restriction</b>  ·  Day %{x}<br>"
                    "Depth: <b>%{y:.0f}%</b> of TD<extra></extra>"
                ),
            ), row=1, col=2)

        _ml = ev[ev["event_type"] == "mud_loss"].dropna(subset=["_depth_pct"])
        if not _ml.empty:
            fig.add_trace(go.Scatter(
                x=_ml["_day"], y=_ml["_depth_pct"],
                mode="markers", name="Mud Loss",
                marker=dict(symbol="square", size=16, color="#1565C0",
                            opacity=0.90, line=dict(width=1.5, color="white")),
                hovertemplate=(
                    "<b>Mud Loss</b>  ·  Day %{x}<br>"
                    "Depth: <b>%{y:.0f}%</b> of TD<extra></extra>"
                ),
            ), row=1, col=2)

        _fm = ev[ev["event_type"] == "formation"].dropna(subset=["_depth_pct"])
        for stype, (col_f, sym_f, lbl_f) in _FORM_STYLE.items():
            _sub = _fm[_fm["sub_type"] == stype]
            if _sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=_sub["_day"], y=_sub["_depth_pct"],
                mode="markers", name=lbl_f,
                marker=dict(symbol=sym_f, size=14, color=col_f,
                            opacity=0.88, line=dict(width=1.2, color="white")),
                hovertemplate=(
                    f"<b>{lbl_f}</b>  ·  Day %{{x}}<br>"
                    "Depth: <b>%{y:.0f}%</b> of TD<extra></extra>"
                ),
            ), row=1, col=2)

        for _, rc in df.iterrows():
            _c = _CTYPE_COL.get(rc["casing_type"], "#aaa")
            fig.add_hline(y=_d(rc["set_depth_ft"]),
                line_dash="dot", line_color=_c, line_width=1.0, row=1, col=2)

        _phase_days = (ev.groupby("phase")["_day"].min()
                       .reindex(["COND1","INTRM1","INTRM2","PROD1","COMPZN"])
                       .dropna())
        for phase, day0 in _phase_days.items():
            if phase == "COND1":
                continue
            _cp = _PHASE_COL_S.get(phase, "#555")
            fig.add_shape(type="line",
                x0=int(day0), x1=int(day0), y0=0, y1=1,
                yref="paper", xref="x2",
                line=dict(color=_cp, width=1.3, dash="dash"))
            fig.add_annotation(
                x=int(day0), y=0.98, xref="x2", yref="paper",
                text=f"<b>{_PHASE_LBL_S.get(phase, phase)}</b>",
                showarrow=False, textangle=-90,
                font=dict(size=9, color=_cp), xanchor="left", yanchor="top",
            )

        _top_op = _op.dropna(subset=["force_klbs"]).nlargest(1, "force_klbs")
        if not _top_op.empty:
            _r = _top_op.iloc[0]
            fig.add_annotation(
                x=int(_r["_day"]), y=_r["_depth_pct"],
                xref="x2", yref="y",
                text="⚠ Severe overpull event",
                showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor="#B71C1C",
                ax=-70, ay=-35, font=dict(size=9, color="#B71C1C"),
                bgcolor="rgba(255,255,255,0.90)",
                bordercolor="#B71C1C", borderwidth=1, borderpad=3,
            )

        if not _ml.empty:
            _tl = _ml[_ml.get("severity", pd.Series()) == "total"].sort_values("_day")
            if not _tl.empty:
                _rl = _tl.iloc[0]
                fig.add_annotation(
                    x=int(_rl["_day"]), y=_rl["_depth_pct"],
                    xref="x2", yref="y",
                    text="💧 Total mud losses",
                    showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor="#1565C0",
                    ax=70, ay=30, font=dict(size=9, color="#1565C0"),
                    bgcolor="rgba(255,255,255,0.90)",
                    bordercolor="#1565C0", borderwidth=1, borderpad=3,
                )

        if not _fm.empty:
            _fbins = pd.cut(_fm["_depth_pct"], bins=range(0, 105, 10)).value_counts()
            _fb    = _fbins.index[0]
            fig.add_hrect(
                y0=_fb.left, y1=_fb.right,
                fillcolor="rgba(123,31,162,0.07)", line_width=0,
                annotation_text="Formation risk interval",
                annotation_position="top right",
                annotation_font_size=9, annotation_font_color="#7B1FA2",
                row=1, col=2,
            )

    _y_vals = [0, _d(1812.2), _d(5704.0), _d(12067.0), 100]
    _y_lbls = ["Surface", "~2k ft", "~6k ft", "~12k ft", "TD ≈ 19k ft"]

    fig.update_yaxes(autorange="reversed", title_text="Well Depth (% of TD)",
                     tickvals=_y_vals, ticktext=_y_lbls, row=1, col=1)
    fig.update_yaxes(autorange="reversed",
                     tickvals=_y_vals, ticktext=_y_lbls, row=1, col=2)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False,
                     range=[-_MAX_X - 0.5, _MAX_X + 0.5], row=1, col=1)
    fig.update_xaxes(title_text="Relative Campaign Day", row=1, col=2)
    fig.update_layout(
        height=960,
        plot_bgcolor="white", paper_bgcolor="white",
        title=dict(
            text="Automated Drilling Event Extraction — Well Architecture & Operational Risk",
            font=dict(size=14, color="rgb(30,30,30)"),
            x=0.5, xanchor="center",
        ),
        legend=dict(orientation="h", y=-0.06, font=dict(size=10),
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="rgba(180,180,180,0.5)", borderwidth=1),
        margin=dict(l=60, r=60, t=80, b=70),
    )
    _apply_chart_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Depths normalised to % of well TD (≈ 19k ft).  "
        "Dates shown as relative campaign days.  "
        "Force: Low ≤ 10 klbs · Medium 10–30 klbs · High > 30 klbs.  "
        "Integrity test values not disclosed."
    )

    st.subheader("Casing Programme")
    tbl_c = df[["casing_description", "casing_type", "od_in"]].copy()
    tbl_c["od_in"]    = tbl_c["od_in"].apply(lambda v: f'{v}"' if pd.notna(v) else "—")
    tbl_c["shoe_pct"] = df["set_depth_ft"].apply(lambda v: f"{_d(v):.0f}% of TD")
    tbl_c["approx"]   = df["set_depth_ft"].apply(
        lambda v: f"~{round(v/1000)}k ft" if round(v/1000) > 0 else "Surface"
    )
    st.dataframe(
        tbl_c.rename(columns={
            "casing_description": "String", "casing_type": "Type",
            "od_in": "OD", "shoe_pct": "Shoe Depth", "approx": "Approx.",
        }),
        use_container_width=True, hide_index=True,
    )
