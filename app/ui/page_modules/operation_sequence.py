from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from .constants import PHASE_ORDER, PHASE_COLOURS, repo_root
    from .loaders import load_all_ops, load_all_headers, _parse_num, _parse_report_dates
    from .utils import _apply_chart_theme, _phase_date_ranges, _report_hour
except ImportError:
    from constants import PHASE_ORDER, PHASE_COLOURS, repo_root  # type: ignore[no-redef]
    from loaders import load_all_ops, load_all_headers, _parse_num, _parse_report_dates  # type: ignore[no-redef]
    from utils import _apply_chart_theme, _phase_date_ranges, _report_hour  # type: ignore[no-redef]

_root = Path(__file__).resolve().parents[3]
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from ddr_rag.vocab import label_phase, label_op_code
from ddr_rag.npt_classifier import (
    apply_corpus_npt_rules, classify_ops_df, classify_equipment_subtype,
    CATEGORY_LABELS, CATEGORY_COLOURS,
)

# Operation types that structurally produce no depth gain and are a normal
# part of a drilling programme (casing, cementing, logging, pressure tests,
# rig moves). No planned-time programme exists for this well, so this is a
# structural classification of the op_code field, not a comparison to a plan.
_EXPECTED_FLAT_OP_CODES: frozenset[str] = frozenset({
    "Run Csg & Cement", "Wire Line Logs", "Test B.O.P.", "Nipple Up B.O.P.",
    "Wait on Cement", "Rig Up & Tear Down", "Cut off Drill Line",
    "Rig Service", "Lubricate Rig", "Clean Out Hole",
})

# Flat-time op_codes that are normal operational support/overhead (trips,
# mud conditioning, directional surveys) rather than a non-drilling
# activity or an unexplained gap.
_OPERATIONAL_FLAT_OP_CODES: frozenset[str] = frozenset({
    "Trips", "Cond Mud & Circ", "Circulating", "Reaming", "Dir Work", "Coring",
})


def _classify_flat_day(day_ops: pd.DataFrame) -> str:
    """Bucket one no-depth-gain, non-NPT-flagged day by its dominant op_code."""
    if day_ops.empty or day_ops["op_code"].isna().all():
        return "Unclassified — needs review"
    dominant = str(
        day_ops.groupby("op_code")["duration_hr"].sum().idxmax() or ""
    ).strip()
    if dominant in _EXPECTED_FLAT_OP_CODES:
        return "Expected (non-drilling)"
    if dominant in _OPERATIONAL_FLAT_OP_CODES:
        return "Operational / support"
    return "Unclassified — needs review"

def _first_sentence(text: str, max_chars: int = 180) -> str:
    if not text:
        return ""
    # Remove leading time-stamp lines (e.g. "14:00 45 330 3.9")
    text = re.sub(r"^\d{2}:\d{2}\s[\d.]+.*\n?", "", text, flags=re.M).strip()
    for sep in (".", "•", "\n"):
        idx = text.find(sep)
        if 20 < idx < max_chars:
            return text[:idx].strip()
    return text[:max_chars].strip()


def _build_blocks(ops: pd.DataFrame) -> pd.DataFrame:
    # Sort by hours-since-06:00, not raw "HH:MM" — DDR reporting days run
    # 06:00 -> 06:00 next day, so a plain string sort would put a report's
    # early-morning tail-end rows (00:00-05:59) before its actual 06:00 start.
    ops = ops.copy()
    ops["_report_hour"] = ops["start_time"].apply(_report_hour)
    ops = ops.sort_values(["dt", "_report_hour"]).drop(columns=["_report_hour"]).reset_index(drop=True)

    # New step when phase, op_code OR date changes — daily boundaries ensure
    # frac stages, cement jobs and multi-day campaigns appear as individual steps.
    ops["_block"] = (
        (ops["phase"]   != ops["phase"].shift()) |
        (ops["op_code"] != ops["op_code"].shift()) |
        (ops["dt"]      != ops["dt"].shift())
    ).cumsum()

    def _agg(g: pd.DataFrame) -> pd.Series:
        npt_rows  = g[g["is_npt"]]
        total_h   = g["duration_hr"].sum()
        npt_h     = npt_rows["duration_hr"].sum()
        dom_cat   = (
            npt_rows.groupby("npt_category")["duration_hr"].sum().idxmax()
            if not npt_rows.empty else ""
        )
        best_text = max(
            (str(t) for t in g["operation_text"].dropna()),
            key=len, default="",
        )
        return pd.Series({
            "phase":          g["phase"].iloc[0],
            "op_code":        g["op_code"].iloc[0],
            "start_dt":       g["dt"].iloc[0],
            "end_dt":         g["dt"].iloc[-1],
            "total_h":        total_h,
            "npt_h":          npt_h,
            "npt_pct":        100.0 * npt_h / total_h if total_h > 0 else 0.0,
            "npt_category":   dom_cat,
            "description":    best_text,
            "n_ops":          len(g),
        })

    blocks = (
        ops.groupby("_block", sort=True)
        .apply(_agg, include_groups=False)
        .reset_index(drop=True)
    )
    blocks.index.name = None
    blocks["step"] = range(1, len(blocks) + 1)

    blocks["duration_days"] = (
        (blocks["end_dt"] - blocks["start_dt"]).dt.days + 1
    ).clip(lower=1)

    return blocks


def _compute_daily_npt(ops: pd.DataFrame) -> pd.DataFrame:
    daily_npt = (
        ops.groupby("dt")
        .apply(lambda g: pd.Series({
            "total_h": g["duration_hr"].sum(),
            "npt_h":   g.loc[g["is_npt"], "duration_hr"].sum(),
        }), include_groups=False)
        .reset_index()
    )
    daily_npt["npt_pct"] = daily_npt["npt_h"] / daily_npt["total_h"].replace(0, 1) * 100
    return daily_npt


def page_operation_sequence() -> None:
    hdr = load_all_headers()
    _rig = hdr["rig_name"].dropna().mode()
    st.header(
        f"Operation Sequence — {_rig.iloc[0].title()}" if not _rig.empty
        else "Operation Sequence"
    )

    with st.spinner("Loading data…"):
        ops = load_all_ops()

    if ops.empty:
        st.warning("No ops data found.")
        return

    ops = apply_corpus_npt_rules(ops)
    ops["dt"] = (
        ops["report_date_parsed"]
        if "report_date_parsed" in ops.columns
        else _parse_report_dates(ops["report_date"])
    )
    hdr["dt"] = (
        hdr["report_date_parsed"]
        if "report_date_parsed" in hdr.columns
        else _parse_report_dates(hdr["report_date"])
    )
    ops["npt_category"] = classify_ops_df(ops)

    blocks = _build_blocks(ops)

    n_ddrs   = ops["report_date"].nunique()
    n_ops    = len(ops)
    date_min = ops["dt"].dropna().min()
    date_max = ops["dt"].dropna().max()
    date_rng = (
        f"{date_min.strftime('%b %Y')} – {date_max.strftime('%b %Y')}"
        if pd.notna(date_min) else ""
    )
    st.caption(
        f"Full operational programme reconstructed from {n_ddrs} DDRs"
        + (f" ({date_rng})" if date_rng else "")
        + f" · {n_ops:,} individual operations condensed into {len(blocks)} steps."
    )

    total_h   = blocks["total_h"].sum()
    npt_h     = blocks["npt_h"].sum()
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Total Steps",       len(blocks))
    k2.metric("Campaign Duration",  f"{(ops['dt'].max()-ops['dt'].min()).days+1} days")
    k3.metric("Total Hours",        f"{total_h:,.0f} h")
    k4.metric("NPT Hours",          f"{npt_h:,.0f} h ({100*npt_h/total_h:.1f}%)")
    k5.metric("High-NPT Steps",
              int((blocks["npt_pct"] > 50).sum()),
              help="Steps where >50% of hours are flagged as NPT")

    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "📋 Programme Steps",
        "📈 Well Performance Chart",
        "🔍 Improvement Opportunities",
    ])

    with tab1:
        _render_programme_table(blocks, ops)

    with tab2:
        _render_well_performance(ops, hdr, blocks)

    with tab3:
        _render_improvement_analysis(ops, blocks)



def _root_cause_key(row: pd.Series) -> str:
    """Grouping key used to flag steps that share a repeated root cause.
    Drills into the equipment sub-type (Phase 1's classifier) when the
    dominant NPT category is "equipment"; falls back to the category
    itself otherwise."""
    cat = str(row.get("npt_category") or "").strip()
    if not cat:
        return ""
    if cat == "equipment":
        subtype = classify_equipment_subtype(str(row.get("description") or ""))
        if subtype != "Unspecified":
            return f"equipment:{subtype}"
    return cat


def _render_programme_table(blocks: pd.DataFrame, ops: pd.DataFrame) -> None:
    fc1, fc2 = st.columns([2, 2])
    with fc1:
        phase_opts = ["All"] + [p for p in PHASE_ORDER if p in blocks["phase"].values]
        sel_phase  = st.selectbox(
            "Filter by phase",
            phase_opts,
            format_func=lambda x: "All phases" if x == "All" else label_phase(x),
        )
    with fc2:
        min_npt = st.slider(
            "Show steps with NPT% ≥",
            min_value=0, max_value=100, value=0, step=5,
            help="Set to 50 to show only high-NPT steps",
        )

    fc3, fc4, fc5, fc6 = st.columns(4)
    with fc3:
        op_opts = sorted({c for c in blocks["op_code"].astype(str).str.strip() if c})
        sel_ops = st.multiselect(
            "Operation type", op_opts, format_func=lambda c: label_op_code(c) or c
        )
    with fc5:
        min_dur = st.number_input("Min step duration (h)", min_value=0.0, value=0.0, step=1.0)
    with fc6:
        exceptions_only = st.checkbox(
            "Exception rows only", value=False, help="Only steps with NPT hours > 0"
        )
    with fc4:
        cat_opts = sorted({c for c in blocks["npt_category"].astype(str).str.strip() if c})
        sel_cats = st.multiselect(
            "NPT category", cat_opts, format_func=lambda c: CATEGORY_LABELS.get(c, c)
        )

    min_d, max_d = blocks["start_dt"].min().date(), blocks["end_dt"].max().date()
    date_range = st.date_input(
        "Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d,
    )

    view = blocks.copy()
    if sel_phase != "All":
        view = view[view["phase"] == sel_phase]
    if min_npt > 0:
        view = view[view["npt_pct"] >= min_npt]
    if sel_ops:
        view = view[view["op_code"].isin(sel_ops)]
    if sel_cats:
        view = view[view["npt_category"].isin(sel_cats)]
    if min_dur > 0:
        view = view[view["total_h"] >= min_dur]
    if exceptions_only:
        view = view[view["npt_h"] > 0]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        d0, d1 = date_range
        view = view[(view["start_dt"].dt.date <= d1) & (view["end_dt"].dt.date >= d0)]

    if view.empty:
        st.info("No steps match the selected filters.")
        return

    view = view.copy()
    view["_root_cause"] = view.apply(_root_cause_key, axis=1)
    repeat_counts = view["_root_cause"].value_counts()
    view["repeat"] = view["_root_cause"].map(
        lambda k: "🔁" if k and repeat_counts.get(k, 0) >= 2 else ""
    )

    st.caption(
        f"Showing {len(view)} of {len(blocks)} steps  ·  "
        f"🔴 red = NPT>50%  ·  🟡 amber = NPT 25-50%  ·  ⬜ white = on-programme  ·  "
        f"🔁 = repeated root cause within this view"
    )

    def _status_icon(npt_pct: float) -> str:
        if npt_pct > 50:
            return "🔴"
        if npt_pct >= 25:
            return "🟡"
        return "⬜"

    disp = pd.DataFrame({
        "Status":         view["npt_pct"].map(_status_icon),
        "Step":           view["step"],
        "Repeat":         view["repeat"],
        "Phase":          view["phase"].map(label_phase),
        "Operation Type": view["op_code"].map(lambda c: label_op_code(c) or c),
        "Start":          view["start_dt"].dt.strftime("%d %b %Y"),
        "End":            view["end_dt"].dt.strftime("%d %b %Y"),
        "Duration (h)":   view["total_h"].round(0).astype(int),
        "NPT (h)":        view["npt_h"].round(0).astype(int),
        "NPT%":           view["npt_pct"].round(0).astype(int).astype(str) + "%",
        "NPT Category":   view["npt_category"].map(lambda c: CATEGORY_LABELS.get(c, "—") if c else "—"),
        "Description":    view["description"].map(_first_sentence),
    })

    # Plain values only (no pandas Styler) — a Styler + this many filter
    # widgets on one page reproducibly crashed with a React error on
    # rerun; the status icon column carries the same red/amber/white
    # signal without it.
    st.dataframe(disp, hide_index=True, height=560)

    st.download_button(
        "⬇ Download programme CSV",
        data=disp.to_csv(index=False),
        file_name="operation_sequence.csv",
        mime="text/csv",
    )



def _render_well_performance(
    ops: pd.DataFrame,
    hdr: pd.DataFrame,
    blocks: pd.DataFrame,
) -> None:
    st.subheader("Depth vs. Time — Well Performance Chart")
    st.caption(
        "Steep slope = fast drilling progress.  "
        "Flat sections = no depth gained (NPT or waiting).  "
        "Red shading = days with >50% NPT."
    )

    hdr2 = hdr.copy()
    hdr2["depth_ft"] = hdr2["end_depth_md_ft"].apply(_parse_num)
    depth_df = hdr2.dropna(subset=["depth_ft", "dt"]).sort_values("dt")

    if depth_df.empty:
        st.info("No depth data available.")
        return

    daily_npt = _compute_daily_npt(ops)

    fig = go.Figure()

    phase_ranges = _phase_date_ranges(ops)
    y_top = depth_df["depth_ft"].max() * 0.015
    for ph, (d0, d1) in phase_ranges.items():
        col = PHASE_COLOURS.get(ph, "#9E9E9E")
        fig.add_vrect(x0=d0, x1=d1, fillcolor=col, opacity=0.06, line_width=0)
        fig.add_annotation(
            x=d0 + (d1 - d0) / 2, y=y_top,
            text=label_phase(ph), showarrow=False,
            font=dict(size=9, color=col), yanchor="bottom",
        )

    for d in daily_npt.loc[daily_npt["npt_pct"] > 50, "dt"]:
        fig.add_vrect(
            x0=d - pd.Timedelta(hours=12),
            x1=d + pd.Timedelta(hours=12),
            fillcolor="#D32F2F", opacity=0.10, line_width=0,
        )

    fig.add_trace(go.Scatter(
        x=depth_df["dt"], y=depth_df["depth_ft"],
        mode="lines", line=dict(color="#1565C0", width=2.5),
        name="Actual depth",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Depth: %{y:,.0f} ft MD<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=[depth_df["dt"].iloc[0], depth_df["dt"].iloc[-1]],
        y=[depth_df["depth_ft"].iloc[0], depth_df["depth_ft"].iloc[-1]],
        mode="lines",
        line=dict(color="#9E9E9E", width=1.5, dash="dot"),
        name="Theoretical (no NPT)",
        hoverinfo="skip",
    ))

    fig.update_yaxes(
        autorange="reversed", title="Depth (ft MD)",
        tickformat=",", showgrid=True, gridcolor="rgba(175,175,175,0.35)",
    )
    fig.update_xaxes(title="Date", showgrid=False)
    fig.update_layout(
        height=550, plot_bgcolor="white",
        legend=dict(orientation="h", y=1.04),
        margin=dict(l=10, r=10, t=40, b=20),
    )
    _apply_chart_theme(fig)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    df_m = depth_df.merge(daily_npt[["dt","npt_pct"]], on="dt", how="left")
    df_m["depth_delta"] = df_m["depth_ft"].diff().abs()
    flat  = df_m[(df_m["depth_delta"] < 1) & (df_m.index > 0)]

    st.markdown("**Flat Time Reconciliation**")
    st.caption(
        "No planned-time programme exists for this well, so \"Expected\" below means "
        "the day's operations were structurally non-drilling (casing, cementing, "
        "logging, BOP tests, rig moves) — not a comparison to an actual plan."
    )

    confirmed_npt = flat[flat["npt_pct"] > 50]
    flat_other    = flat[flat["npt_pct"] <= 50].copy()
    flat_other["bucket"] = [
        _classify_flat_day(ops[ops["dt"] == d]) for d in flat_other["dt"]
    ]
    n_expected     = int((flat_other["bucket"] == "Expected (non-drilling)").sum())
    n_operational  = int((flat_other["bucket"] == "Operational / support").sum())
    n_unclassified = int((flat_other["bucket"] == "Unclassified — needs review").sum())

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Days with no depth gain", len(flat))
    r2.metric("Confirmed NPT", len(confirmed_npt))
    r3.metric("Expected / operational support", n_expected + n_operational,
              help=f"{n_expected} expected non-drilling (casing/cement/logging/BOP/rig move) "
                   f"+ {n_operational} operational support (trips/circulating/conditioning/surveys)")
    r4.metric("Unclassified — needs review", n_unclassified,
              help="No depth progress, not flagged as NPT, and not dominated by an expected "
                   "non-drilling or support operation — review the DDR text before treating "
                   "as avoidable time.")

    review_days = flat_other[flat_other["bucket"] == "Unclassified — needs review"]
    if not review_days.empty:
        with st.expander(f"Days needing review ({len(review_days)})"):
            rows = []
            for d in review_days["dt"]:
                day_ops = ops[ops["dt"] == d]
                dominant = (
                    day_ops.groupby("op_code")["duration_hr"].sum().idxmax()
                    if not day_ops.empty and not day_ops["op_code"].isna().all() else "—"
                )
                rows.append({
                    "Date": d.strftime("%d %b %Y"),
                    "Dominant op type": label_op_code(dominant) if dominant != "—" else "—",
                    "Hours logged": f"{day_ops['duration_hr'].sum():.1f}h",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)



def _render_improvement_analysis(
    ops: pd.DataFrame,
    blocks: pd.DataFrame,
) -> None:
    st.subheader("Improvement Opportunities")
    st.caption(
        f"Six patterns auto-detected from {len(ops):,} DDR operation rows. "
        "Each finding includes phase attribution and DDR text evidence."
    )

    with st.expander("1 · Sustained NPT Streaks — consecutive steps >50% NPT", expanded=True):
        streaks, cur = [], []
        for _, r in blocks.iterrows():
            if r["npt_pct"] > 50:
                cur.append(r)
            else:
                if len(cur) >= 2:
                    streaks.append(cur)
                cur = []
        if cur and len(cur) >= 2:
            streaks.append(cur)

        if not streaks:
            st.success("No sustained NPT streaks (≥2 consecutive high-NPT steps).")
        else:
            total_streak_h = sum(r["total_h"] for s in streaks for r in s)
            st.warning(
                f"**{len(streaks)} streak(s)** covering "
                f"**{sum(len(s) for s in streaks)} steps** "
                f"({total_streak_h:.0f} hrs) with >50% NPT. "
                f"These sustained periods represent the highest-priority targets.",
                icon="⚠️",
            )
            for streak in sorted(streaks, key=lambda s: -sum(r["npt_h"] for r in s)):
                npt_h  = sum(r["npt_h"]   for r in streak)
                tot_h  = sum(r["total_h"] for r in streak)
                d0     = streak[0]["start_dt"].strftime("%d %b %Y")
                d1     = streak[-1]["end_dt"].strftime("%d %b %Y")
                ph     = streak[0]["phase"]
                dom_cat = (
                    ops[(ops["dt"] >= streak[0]["start_dt"]) &
                        (ops["dt"] <= streak[-1]["end_dt"]) &
                        ops["is_npt"]]
                    .groupby("npt_category")["duration_hr"].sum()
                    .idxmax() if not ops[
                        (ops["dt"] >= streak[0]["start_dt"]) &
                        ops["is_npt"]].empty else ""
                )
                ph_col = PHASE_COLOURS.get(ph, "#546E7A")
                st.markdown(
                    f"<div style='border-left:4px solid {ph_col};"
                    f"padding:8px 14px;margin:4px 0;background:#FFF8E1;"
                    f"color:#1A202C;border-radius:0 4px 4px 0'>"
                    f"<b>Steps {streak[0]['step']}–{streak[-1]['step']}</b>  "
                    f"({d0} – {d1})  ·  Phase: <b>{label_phase(ph)}</b>  "
                    f"·  {len(streak)} steps  ·  {npt_h:.0f}/{tot_h:.0f} hrs NPT<br>"
                    f"Primary cause: <b>{CATEGORY_LABELS.get(dom_cat, dom_cat)}</b></div>",
                    unsafe_allow_html=True,
                )

    with st.expander("2 · Top 10 Individual Time Sinks — largest single NPT events"):
        npt_top = ops[ops["is_npt"]].nlargest(10, "duration_hr")
        if npt_top.empty:
            st.info("No NPT events found.")
        else:
            total_npt = ops[ops["is_npt"]]["duration_hr"].sum()
            top3_h    = npt_top.iloc[:3]["duration_hr"].sum()
            st.info(
                f"Top 10 events = **{npt_top['duration_hr'].sum():.0f} hrs** "
                f"({100*npt_top['duration_hr'].sum()/total_npt:.1f}% of all NPT).  "
                f"Eliminating the worst 3 alone would recover ~{top3_h:.0f} hrs.",
                icon="💡",
            )
            for rank, (_, r) in enumerate(npt_top.iterrows(), 1):
                cat_col = CATEGORY_COLOURS.get(r["npt_category"], "#9E9E9E")
                cat_lbl = CATEGORY_LABELS.get(r["npt_category"], r["npt_category"])
                st.markdown(
                    f"<div style='border-left:4px solid {cat_col};padding:6px 12px;"
                    f"margin:3px 0;background:#FAFAFA;color:#1A202C;"
                    f"border-radius:0 4px 4px 0'>"
                    f"<b>#{rank}  {r['report_date']}  ·  {label_phase(r['phase'])}  "
                    f"·  {r['activity_code']}  ·  {r['duration_hr']:.1f} hrs</b>  "
                    f"<span style='color:{cat_col}'>● {cat_lbl}</span><br>"
                    f"<span style='font-size:0.85em'>"
                    f"{_first_sentence(str(r['operation_text']), 250)}</span></div>",
                    unsafe_allow_html=True,
                )

    with st.expander("3 · Repeated Trip Cycles — POOH/RIH without depth gain"):
        trip_per_day = (
            ops[ops["activity_code"].isin(["TRIP","POOH","RIH","TDTRIP"])]
            .groupby("dt").size().rename("n")
        )
        heavy = trip_per_day[trip_per_day >= 5]
        if heavy.empty:
            st.success("No days with 5+ trip operations detected.")
        else:
            st.warning(
                f"**{len(heavy)} days** had 5+ trip-type operations — "
                f"back-reaming, stuck-pipe recovery or BHA change-out cycles.",
                icon="⚠️",
            )
            for dt, cnt in heavy.nlargest(5).items():
                day_ops = ops[(ops["dt"] == dt) &
                              ops["activity_code"].isin(["TRIP","POOH","RIH","TDTRIP"])]
                ph  = day_ops["phase"].mode().iloc[0] if not day_ops.empty else "—"
                txt = str(day_ops["operation_text"].iloc[0]) if not day_ops.empty else ""
                st.markdown(
                    f"<div style='border-left:3px solid #E65100;padding:5px 10px;"
                    f"margin:3px 0;background:#FFF3E0;color:#1A202C;"
                    f"border-radius:0 4px 4px 0'>"
                    f"<b>{dt.strftime('%d %b %Y')}</b> — {cnt} trip ops  "
                    f"· {label_phase(ph)}<br>"
                    f"<span style='font-size:0.85em'>{_first_sentence(txt, 180)}</span></div>",
                    unsafe_allow_html=True,
                )

    with st.expander("4 · Flat Time Not Flagged as NPT"):
        hdr_d = load_all_headers().copy()
        hdr_d["dt"]       = (
            hdr_d["report_date_parsed"]
            if "report_date_parsed" in hdr_d.columns
            else _parse_report_dates(hdr_d["report_date"])
        )
        hdr_d["depth_ft"] = hdr_d["end_depth_md_ft"].apply(_parse_num)
        hdr_d = hdr_d.dropna(subset=["dt","depth_ft"]).sort_values("dt")
        hdr_d["dd"] = hdr_d["depth_ft"].diff().abs()

        daily_npt = _compute_daily_npt(ops)[["dt", "npt_pct"]]
        flat_p = (
            hdr_d[hdr_d["dd"] < 1]
            .merge(daily_npt, on="dt", how="left")
            .query("npt_pct <= 50")
        )
        p_hrs = ops[ops["dt"].isin(flat_p["dt"]) & ~ops["is_npt"]]["duration_hr"].sum()
        if flat_p.empty:
            st.success("No unflagged flat-time days detected.")
        else:
            st.warning(
                f"**{len(flat_p)} days** showed zero depth progress while the operation rows "
                f"were predominantly not flagged as NPT — **{p_hrs:.0f} hrs** to review.",
                icon="⚠️",
            )

    with st.expander("5 · Recurring NPT Categories — same issue across 3+ phases"):
        cat_phase = (
            ops[ops["is_npt"]]
            .groupby(["npt_category","phase"])["duration_hr"]
            .sum().unstack(fill_value=0.0)
        )
        recurring = cat_phase[(cat_phase > 0).sum(axis=1) >= 3]
        if recurring.empty:
            st.success("No NPT category appears in 3+ phases.")
        else:
            st.warning(
                f"**{len(recurring)} category(ies)** present in 3+ phases — "
                f"systemic issues, not phase-specific anomalies.",
                icon="⚠️",
            )
            for cat in recurring.index:
                col = CATEGORY_COLOURS.get(cat, "#9E9E9E")
                lbl = CATEGORY_LABELS.get(cat, cat)
                phases_p = [p for p in PHASE_ORDER if recurring.loc[cat].get(p,0) > 0]
                tot = recurring.loc[cat].sum()
                st.markdown(
                    f"<div style='border-left:4px solid {col};padding:6px 12px;"
                    f"margin:3px 0;background:#F3F4F6;color:#1A202C;"
                    f"border-radius:0 4px 4px 0'>"
                    f"<b style='color:{col}'>{lbl}</b>  ·  {tot:.0f} hrs  ·  "
                    f"{len(phases_p)} phases<br>"
                    f"<span style='font-size:0.82em;color:#555'>"
                    f"{'  →  '.join(label_phase(p) for p in phases_p)}"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )

    with st.expander("6 · Safety / JSA Overhead — time on safety briefings per phase"):
        sfty = ops[ops["activity_code"] == "SFTY"].groupby("phase")["duration_hr"].sum()
        tot  = ops.groupby("phase")["duration_hr"].sum()
        rows = [{"Phase": label_phase(p),
                 "SFTY (h)": f"{sfty.get(p,0):.0f}",
                 "Total (h)": f"{tot.get(p,0):.0f}",
                 "SFTY%": f"{100*sfty.get(p,0)/tot.get(p,1):.1f}%"}
                for p in PHASE_ORDER if tot.get(p,0) > 0]
        fig_s = go.Figure(go.Table(
            columnwidth=[20,10,10,10],
            header=dict(values=["<b>Phase</b>","<b>SFTY (h)</b>",
                                "<b>Total (h)</b>","<b>SFTY%</b>"],
                        fill_color="#455A64", font=dict(color="white",size=11),
                        align="left", height=28),
            cells=dict(
                values=[[r["Phase"] for r in rows],
                        [r["SFTY (h)"] for r in rows],
                        [r["Total (h)"] for r in rows],
                        [r["SFTY%"] for r in rows]],
                fill_color="white", font=dict(size=10.5, color="#1A202C"),
                align="left", height=24),
        ))
        fig_s.update_layout(margin=dict(l=0,r=0,t=5,b=0), height=220)
        st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})
        st.caption("Benchmark: 1–3% per phase. Higher values suggest frequent operation changes requiring re-briefing.")
