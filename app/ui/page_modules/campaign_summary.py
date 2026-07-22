from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from .constants import WELL_COLOURS, PHASE_ORDER, PHASE_COLOURS
    from .loaders import load_field_ops, load_field_headers, load_well_metadata
    from .utils import _apply_chart_theme
except ImportError:
    from constants import WELL_COLOURS, PHASE_ORDER, PHASE_COLOURS           # type: ignore[no-redef]
    from loaders import load_field_ops, load_field_headers, load_well_metadata  # type: ignore[no-redef]
    from utils import _apply_chart_theme                                      # type: ignore[no-redef]

_ui_root = Path(__file__).resolve().parents[3]
if str(_ui_root / "src") not in sys.path:
    sys.path.insert(0, str(_ui_root / "src"))

from ddr_rag.vocab import label_phase
from ddr_rag.npt_classifier import CATEGORY_LABELS, classify_equipment_subtype


_EQUIPMENT_ACTION_TEMPLATES: dict[str, str] = {
    "Pump":               "Verify pump maintenance schedule and spares availability before the next well.",
    "Top Drive":          "Verify top drive service history and spares before the next well.",
    "Hydraulic System":   "Inspect hydraulic lines/connections and stage spares before the next well.",
    "Motor / Electrical": "Verify motor/electrical spares and inspection schedule before the next well.",
    "Valve":              "Verify valve maintenance and spares before the next well.",
    "Cable":              "Inspect cable/connector condition and stage spares before the next well.",
    "Crane":              "Verify crane inspection and maintenance schedule before the next well.",
}


def _first_words(text: str, max_chars: int = 120) -> str:
    text = re.sub(r"^\d{2}:\d{2}\s[\d.]+.*\n?", "", str(text or ""), flags=re.M).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def _derive_caption(hdr: pd.DataFrame) -> str:
    rig   = hdr["rig_name"].dropna().mode() if "rig_name" in hdr.columns else pd.Series(dtype=object)
    field = hdr["field_name"].dropna().mode() if "field_name" in hdr.columns else pd.Series(dtype=object)
    rng   = hdr["report_date_parsed"].dropna() if "report_date_parsed" in hdr.columns else pd.Series(dtype=object)
    rig_s   = rig.iloc[0].title()                 if not rig.empty   else ""
    field_s = field.iloc[0].title() + " Field"    if not field.empty else ""
    date_s  = (
        f"{rng.min().strftime('%b %Y')} – {rng.max().strftime('%b %Y')}"
        if not rng.empty else ""
    )
    return " · ".join(x for x in [field_s, rig_s, date_s] if x)


def _render_kpis(ops: pd.DataFrame, hdr: pd.DataFrame,
                 events: pd.DataFrame, npt_h: float, npt_pct: float) -> None:
    n_days    = int(ops["report_date_parsed"].dropna().nunique())
    max_depth = float(hdr["end_depth_num"].dropna().max() or 0)
    total_h   = float(ops["duration_hr"].sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Well Duration",    f"{n_days} days")
    k2.metric("Reported Time",    f"{total_h:,.0f} hrs")
    k3.metric("Max Depth (MD)",   f"{max_depth:,.0f} ft")
    k4.metric("Flagged NPT",      f"{npt_h:,.0f} hrs", f"{npt_pct:.1f}% of well time")


def _render_superintendent_takeaways(ops: pd.DataFrame) -> None:
    npt_ops = ops[ops["is_npt"]].copy()
    if npt_ops.empty:
        return

    with st.container(border=True):
        st.markdown("##### 🎯 Superintendent Takeaways")

        cat_h = npt_ops.groupby("npt_cat_label")["duration_hr"].sum().sort_values(ascending=False)
        top_cat_label = str(cat_h.index[0])
        top_cat_h = float(cat_h.iloc[0])
        st.markdown(f"**Primary NPT driver:** {top_cat_label}, {top_cat_h:.0f} h")

        top_event = npt_ops.nlargest(1, "duration_hr").iloc[0]
        event_desc = _first_words(str(top_event.get("operation_text") or ""))
        st.markdown(
            f"**Largest single event:** {event_desc} "
            f"— {float(top_event['duration_hr']):.1f} h "
            f"({top_event.get('report_date', '')})"
        )

        # Require a meaningful denominator (>=24h) so a tiny phase with one
        # bad hour doesn't dominate the ranking.
        phase_rows = []
        for ph in PHASE_ORDER:
            g = ops[ops["phase"] == ph]
            tot = float(g["duration_hr"].sum())
            if tot < 24:
                continue
            npt = float(g.loc[g["is_npt"], "duration_hr"].sum())
            phase_rows.append((ph, 100 * npt / tot))
        if phase_rows:
            top_phase, top_phase_pct = max(phase_rows, key=lambda r: r[1])
            st.markdown(f"**Highest-risk phase:** {label_phase(top_phase)}, {top_phase_pct:.0f}% NPT")

        top_cat_code = str(
            npt_ops.groupby("npt_category")["duration_hr"].sum()
            .sort_values(ascending=False).index[0]
        )
        if top_cat_code == "equipment":
            eq = npt_ops[npt_ops["npt_category"] == "equipment"].copy()
            eq["subtype"] = eq["operation_text"].apply(classify_equipment_subtype)
            sub_stats = (
                eq[eq["subtype"] != "Unspecified"]
                .groupby("subtype")["duration_hr"].agg(["sum", "count"])
                .sort_values("sum", ascending=False)
            )
            if not sub_stats.empty and sub_stats.iloc[0]["count"] >= 2:
                subtype = str(sub_stats.index[0])
                s_h, s_n = float(sub_stats.iloc[0]["sum"]), int(sub_stats.iloc[0]["count"])
                st.markdown(f"**Main repeat issue:** {subtype}-related failures ({s_n} events, {s_h:.1f} h)")
                action = _EQUIPMENT_ACTION_TEMPLATES.get(subtype)
                if action:
                    st.markdown(f"**Planning action:** {action}")

        st.caption(
            "Derived directly from classified DDR operation rows — not a plan or "
            "offset-well comparison (neither is available for this well)."
        )


def _render_phase_performance(ops: pd.DataFrame) -> None:
    phase_stats = []
    for ph in PHASE_ORDER:
        g = ops[ops["phase"] == ph]
        if g.empty:
            continue
        tot = float(g["duration_hr"].sum())
        npt = float(g.loc[g["is_npt"], "duration_hr"].sum())
        phase_stats.append({
            "phase":   ph,
            "label":   label_phase(ph),
            "total_h": tot,
            "npt_h":   npt,
            "prod_h":  tot - npt,
            "npt_pct": round(100 * npt / max(tot, 1), 1),
        })
    ps = pd.DataFrame(phase_stats)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ps["label"], y=ps["prod_h"],
        name="Productive time",
        marker_color=[PHASE_COLOURS.get(p, "#aaa") for p in ps["phase"]],
        opacity=0.85,
        hovertemplate="%{x}<br>Productive: %{y:,.0f} hrs<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=ps["label"], y=ps["npt_h"],
        name="NPT",
        marker_color="#EF5350",
        hovertemplate="%{x}<br>NPT: %{y:,.0f} hrs (%{customdata:.1f}%)<extra></extra>",
        customdata=ps["npt_pct"],
    ))
    for _, row in ps.iterrows():
        if row["npt_pct"] >= 5:
            fig.add_annotation(
                x=row["label"], y=row["total_h"] + 15,
                text=f"{row['npt_pct']:.0f}%",
                showarrow=False,
                font=dict(size=11, color="rgb(30,30,30)", family="Arial"),
            )
    fig.update_layout(
        barmode="stack", height=300,
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title="Hours",
        legend=dict(orientation="h", y=-0.25, font=dict(size=11)),
        margin=dict(l=50, r=20, t=10, b=60),
    )
    _apply_chart_theme(fig)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_risk_summary(ops: pd.DataFrame, events: pd.DataFrame,
                          npt_h: float) -> None:
    npt_ops = ops[ops["is_npt"]].copy()
    if not npt_ops.empty:
        top_cats = (
            npt_ops.groupby("npt_cat_label")["duration_hr"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
        for cat, hrs in top_cats.items():
            pct = 100 * hrs / npt_h
            st.markdown(
                f"**{cat}** &nbsp; `{hrs:,.0f} h` &nbsp; "
                f"<span style='color:#888;font-size:13px'>{pct:.1f}% of NPT</span>",
                unsafe_allow_html=True,
            )
            st.progress(min(pct / 100, 1.0))
    else:
        st.caption("No NPT rows were flagged in the extracted operation summary.")

    if not events.empty:
        st.divider()
        max_op  = events.loc[events["event_type"] == "overpull", "force_klbs"].max()
        max_ecd = events.loc[
            (events["event_type"] == "formation") & (events["sub_type"] == "high_ecd"),
            "ecd_ppge",
        ].max()
        gas_ev  = events[
            (events["event_type"] == "formation") & (events["sub_type"] == "gas_influx")
        ]
        loss_ev = events[
            (events["event_type"] == "mud_loss") & (events["severity"] == "total")
        ]

        flags: list[str] = []
        if pd.notna(max_op) and max_op > 100:
            flags.append(f"🔴 Stuck pipe — {max_op:.0f} klbs peak overpull (COND1)")
        if pd.notna(max_ecd) and max_ecd > 14.5:
            flags.append(f"🟠 High ECD — peak {max_ecd:.2f} ppge (narrow margin)")
        if not loss_ev.empty:
            flags.append(f"🟠 Total mud losses — {len(loss_ev)} events in PROD1 / COMPZN")
        if not gas_ev.empty:
            flags.append(f"🟡 Gas influx — {len(gas_ev)} events in COMPZN (managed)")
        if not flags:
            flags = ["🟢 No critical mechanical risk flags"]
        for flag in flags:
            st.markdown(flag)


def _render_major_events(ops: pd.DataFrame) -> None:
    major_npt = (
        ops[ops["is_npt"]]
        .groupby(["report_date_parsed", "phase", "npt_cat_label"])["duration_hr"]
        .sum()
        .reset_index()
        .sort_values("duration_hr", ascending=False)
        .head(8)
    )
    if major_npt.empty:
        return
    major_npt["Date"]         = major_npt["report_date_parsed"].dt.strftime("%d %b %Y")
    major_npt["Phase"]        = major_npt["phase"].map(label_phase).fillna(major_npt["phase"])
    major_npt["NPT Category"] = major_npt["npt_cat_label"]
    major_npt["Hours"]        = major_npt["duration_hr"].round(1)
    st.dataframe(
        major_npt[["Date", "Phase", "NPT Category", "Hours"]],
        hide_index=True,
        use_container_width=True,
    )


def _render_cross_well(field_ops: pd.DataFrame, field_hdr: pd.DataFrame,
                        well_meta: dict) -> None:
    if field_ops.empty:
        return

    st.subheader("Cross-Well Comparison")
    wells     = sorted(field_ops["well_id"].unique())
    well_cols = {w: WELL_COLOURS.get(w, "#555") for w in wells}

    w_stats: dict[str, dict] = {}
    for w in wells:
        wo = field_ops[field_ops["well_id"] == w]
        wh = field_hdr[field_hdr["well_id"] == w]
        tot_h  = float(wo["duration_hr"].sum())
        npt_h  = float(wo.loc[wo["is_npt"], "duration_hr"].sum())
        cost   = float(wh["cum_cost_num"].dropna().max() or 0)
        n_days = int(wh["report_date_parsed"].nunique())
        m      = well_meta.get(w, {})
        w_stats[w] = dict(
            tot_h=tot_h, npt_h=npt_h,
            npt_pct=100 * npt_h / max(tot_h, 1),
            cost=cost, n_days=n_days,
            rig=m.get("rig", ""), spud=(m.get("spud_date") or "")[:7],
        )

    if len(wells) >= 2:
        w1, w2     = wells[0], wells[1]
        s1, s2     = w_stats[w1], w_stats[w2]
        days_delta = s1["n_days"]  - s2["n_days"]
        cost_delta = s1["cost"]    - s2["cost"]
        npt_delta  = s1["npt_pct"] - s2["npt_pct"]

        best_phase, best_save = None, 0.0
        for ph in PHASE_ORDER:
            ph1 = field_ops[(field_ops["well_id"] == w1) & (field_ops["phase"] == ph)]
            ph2 = field_ops[(field_ops["well_id"] == w2) & (field_ops["phase"] == ph)]
            if ph1.empty or ph2.empty:
                continue
            save = (float(ph1.loc[ph1.is_npt, "duration_hr"].sum()) -
                    float(ph2.loc[ph2.is_npt, "duration_hr"].sum()))
            if save > best_save:
                best_save, best_phase = save, ph

        direction = "improvement" if npt_delta > 0 else "increase"
        cost_str  = (f"saving **£{cost_delta/1e6:.1f}M**"
                     if cost_delta > 0 else f"costing **£{abs(cost_delta)/1e6:.1f}M** more")
        days_str  = (f"**{abs(days_delta)} days faster**"
                     if days_delta > 0 else f"**{abs(days_delta)} days longer**")

        parts = [
            f"**{w2}** ({s2['rig']}, {s2['spud']}) completed in **{s2['n_days']} days** — "
            f"{days_str} than **{w1}** ({s1['rig']}, {s1['spud']}), {cost_str}.",
            f"Overall NPT moved from **{s1['npt_pct']:.0f}%** to **{s2['npt_pct']:.0f}%** "
            f"({abs(npt_delta):.0f} pp {direction}).",
        ]
        if best_phase:
            parts.append(
                f"Largest gain: **{label_phase(best_phase)}** — "
                f"{best_save:.0f} h of NPT avoided on {w2}."
            )
        st.info("  ".join(parts))

    metric_cols = st.columns(len(wells))
    for col, w in zip(metric_cols, wells):
        s = w_stats[w]
        col.markdown(
            f"<div style='border-left:4px solid {well_cols[w]};"
            f"padding-left:10px'><b>{w}</b><br>"
            f"<span style='color:#888;font-size:12px'>{s['rig']} · {s['spud']}</span></div>",
            unsafe_allow_html=True,
        )
        col.metric("Campaign",    f"{s['n_days']} days")
        col.metric("Overall NPT", f"{s['npt_pct']:.0f}%", f"{s['npt_h']:.0f} h")
        col.metric("Total cost",
                   f"£{s['cost']/1e6:.1f}M" if s["cost"] > 1e6
                   else f"£{s['cost']:,.0f}")

    st.divider()

    col_chart, col_insight = st.columns([3, 2], gap="large")

    with col_chart:
        st.markdown("**Phase NPT comparison**")
        phase_rows = []
        for ph in PHASE_ORDER:
            for w in wells:
                g = field_ops[(field_ops["well_id"] == w) & (field_ops["phase"] == ph)]
                if g.empty:
                    continue
                tot = float(g["duration_hr"].sum())
                npt = float(g.loc[g.is_npt, "duration_hr"].sum())
                phase_rows.append({
                    "Phase": label_phase(ph).split("/")[0].strip(),
                    "Well":  w,
                    "NPT %": round(100 * npt / max(tot, 1), 1),
                    "NPT h": round(npt, 0),
                    "Tot h": round(tot, 0),
                    "_ord":  PHASE_ORDER.index(ph),
                })
        ph_df = pd.DataFrame(phase_rows).sort_values("_ord")

        fig_cw = go.Figure()
        for w in wells:
            sub = ph_df[ph_df["Well"] == w]
            fig_cw.add_trace(go.Bar(
                name=w,
                x=sub["Phase"], y=sub["NPT %"],
                marker_color=well_cols[w],
                text=sub["NPT %"].map("{:.0f}%".format),
                textposition="outside",
                hovertemplate=(
                    f"<b>{w}</b> · %{{x}}<br>"
                    "NPT: %{y:.0f}%  "
                    "(%{customdata[0]:.0f}h / %{customdata[1]:.0f}h)<extra></extra>"
                ),
                customdata=sub[["NPT h", "Tot h"]].values,
            ))
        fig_cw.update_layout(
            barmode="group", height=300,
            yaxis=dict(title="NPT %", range=[0, 115]),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.3, font=dict(size=11)),
            margin=dict(l=50, r=20, t=10, b=60),
        )
        _apply_chart_theme(fig_cw)
        st.plotly_chart(fig_cw, use_container_width=True, config={"displayModeBar": False})

    with col_insight:
        st.markdown("**Recurring vs resolved NPT challenges**")
        if len(wells) >= 2:
            w1, w2   = wells[0], wells[1]
            all_cats = field_ops[field_ops["is_npt"]]["npt_category"].dropna().unique()
            cat_rows = []
            for cat in all_cats:
                h1 = float(field_ops[
                    (field_ops["well_id"] == w1) & field_ops["is_npt"] &
                    (field_ops["npt_category"] == cat)
                ]["duration_hr"].sum())
                h2 = float(field_ops[
                    (field_ops["well_id"] == w2) & field_ops["is_npt"] &
                    (field_ops["npt_category"] == cat)
                ]["duration_hr"].sum())
                if h1 < 5 and h2 < 5:
                    continue
                cat_rows.append({
                    "cat": CATEGORY_LABELS.get(cat, cat),
                    "h1": h1, "h2": h2,
                    "recurring": h1 > 5 and h2 > 5,
                })
            cat_df = pd.DataFrame(cat_rows).sort_values("h1", ascending=False)

            recurring = cat_df[cat_df["recurring"]]
            resolved  = cat_df[(cat_df["h1"] > 5) & ~cat_df["recurring"]]
            new_w2    = cat_df[(cat_df["h2"] > 5) & ~cat_df["recurring"] & (cat_df["h1"] <= 5)]

            if not recurring.empty:
                st.markdown("🔴 **Recurring** (both wells)")
                for _, r in recurring.head(4).iterrows():
                    st.markdown(
                        f"&nbsp;&nbsp;{r['cat']}<br>"
                        f"&nbsp;&nbsp;<span style='color:#888;font-size:11px'>"
                        f"{w1}: {r['h1']:.0f}h → {w2}: {r['h2']:.0f}h</span>",
                        unsafe_allow_html=True,
                    )
            if not resolved.empty:
                st.markdown("✅ **Resolved on W2**")
                for _, r in resolved.head(3).iterrows():
                    saved = r["h1"] - r["h2"]
                    st.markdown(
                        f"&nbsp;&nbsp;{r['cat']} "
                        f"<span style='color:#2E7D32;font-size:11px'>−{saved:.0f}h</span>",
                        unsafe_allow_html=True,
                    )
            if not new_w2.empty:
                st.markdown(f"\U0001f195 **New on {w2}**")
                for _, r in new_w2.head(3).iterrows():
                    st.markdown(f"&nbsp;&nbsp;{r['cat']} ({r['h2']:.0f}h)",
                                unsafe_allow_html=True)


def page_executive_summary(ops: pd.DataFrame, hdr: pd.DataFrame,
                            events: pd.DataFrame) -> None:
    st.header("Campaign Summary")
    st.caption(_derive_caption(hdr))

    if ops.empty or hdr.empty:
        st.info(
            "No processed DDR data found yet. The raw Utah FORGE PDFs are staged in "
            "`data/raw/`; run `python scripts/batch_preprocess_raw_ddrs.py --build-index` "
            "to populate this dashboard."
        )
        return

    total_h = float(ops["duration_hr"].sum())
    npt_h   = float(ops.loc[ops["is_npt"], "duration_hr"].sum())
    npt_pct = 100 * npt_h / total_h if total_h else 0

    _render_kpis(ops, hdr, events, npt_h, npt_pct)
    st.divider()

    _render_superintendent_takeaways(ops)
    st.divider()

    col_left, col_right = st.columns([3, 2], gap="large")
    with col_left:
        st.subheader("Phase Performance")
        _render_phase_performance(ops)
    with col_right:
        st.subheader("Risk Summary")
        _render_risk_summary(ops, events, npt_h)

    st.divider()
    st.subheader("Major Events")
    _render_major_events(ops)

    st.divider()
    field_ops = load_field_ops()
    if not field_ops.empty and field_ops.get("well_id", pd.Series(dtype=object)).nunique() > 1:
        st.divider()
        _render_cross_well(field_ops, load_field_headers(), load_well_metadata())
