from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ui_root = Path(__file__).resolve().parents[3]
if str(_ui_root / "src") not in sys.path:
    sys.path.insert(0, str(_ui_root / "src"))

from ddr_rag.vocab import label_phase
from ddr_rag.npt_classifier import CATEGORY_LABELS, CATEGORY_COLOURS


_DECISION_RE = re.compile(
    r"recommend|decision\s+made|agreed\s+(?:to|plan)|going\s+forward|"
    r"action\s+item|mitigat|root\s+cause|lesson|next\s+time|prevent",
    re.I,
)

_PHASE_ORDER = ["MIRU", "COND1", "INTRM1", "INTRM2", "PROD1", "COMPZN"]

_PHASE_COLOUR = {
    "MIRU":   "#1565C0",
    "COND1":  "#2E7D32",
    "INTRM1": "#6A1B9A",
    "INTRM2": "#4527A0",
    "PROD1":  "#E65100",
    "COMPZN": "#00695C",
}

_EV_TYPES = {
    "overpull":    ("🔴 Overpull",    "force_klbs",       "klbs",   "Max force"),
    "restriction": ("🟠 Restrictions", "force_klbs",       "klbs",   "Max force"),
    "mud_loss":    ("🔵 Mud Losses",   "loss_rate_bbl_hr", "bbl/hr", "Max rate"),
    "formation":   ("🟣 Formation",    "ecd_ppge",         "ppge",   "Max ECD"),
}


def _render_npt_summary(ops_f: pd.DataFrame) -> tuple[float, float, float, pd.Series]:
    total_hrs = ops_f["duration_hr"].sum()
    npt_hrs   = ops_f[ops_f["is_npt"]]["duration_hr"].sum()
    npt_pct   = (npt_hrs / total_hrs * 100) if total_hrs > 0 else 0.0

    st.subheader("📊 NPT Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Operations", f"{total_hrs:,.0f} hrs")
    c2.metric("NPT Hours",        f"{npt_hrs:,.0f} hrs")
    c3.metric("NPT %",            f"{npt_pct:.1f}%")

    npt_ops = ops_f[ops_f["is_npt"]].copy()
    cat_hrs = npt_ops.groupby("npt_category")["duration_hr"].sum().sort_values(ascending=False)
    if not cat_hrs.empty:
        st.markdown("**Top NPT Categories:**")
        for cat, hrs in cat_hrs.head(5).items():
            pct   = hrs / npt_hrs * 100 if npt_hrs > 0 else 0
            label = CATEGORY_LABELS.get(cat, cat)
            col   = CATEGORY_COLOURS.get(cat, "#9E9E9E")
            st.markdown(
                f"<span style='color:{col}'>●</span> **{label}** — "
                f"{hrs:.0f} hrs ({pct:.1f}% of phase NPT)",
                unsafe_allow_html=True,
            )

    return total_hrs, npt_hrs, npt_pct, cat_hrs


def _render_wellbore_challenges(events_f: pd.DataFrame) -> None:
    st.subheader("⚠️ Key Wellbore Challenges")
    if events_f.empty:
        st.info("No wellbore events for this selection.")
        return

    cols = st.columns(len(_EV_TYPES))
    for col, (etype, (label, val_col, unit, val_label)) in zip(cols, _EV_TYPES.items()):
        grp = events_f[events_f["event_type"] == etype]
        if grp.empty:
            col.metric(label, "0 events")
        else:
            peak = grp[val_col].dropna().max() if val_col in grp.columns else None
            peak_str = f"{peak:.1f} {unit}" if pd.notna(peak) else "—"
            col.metric(label, f"{len(grp)} events",
                       delta=f"{val_label}: {peak_str}", delta_color="off")

    for etype, (label, val_col, unit, val_label) in _EV_TYPES.items():
        grp = events_f[events_f["event_type"] == etype].sort_values("report_date")
        if grp.empty:
            continue
        with st.expander(f"{label} — {len(grp)} events", expanded=False):
            sort_col = val_col if val_col in grp.columns and grp[val_col].notna().any() else "depth_ft"
            if sort_col == "depth_ft" and "event_depth_ft_md" in grp.columns:
                sort_col = "event_depth_ft_md"
            top = grp.nlargest(5, sort_col) if sort_col in grp.columns else grp.head(5)
            for _, r in top.iterrows():
                depth     = r.get("event_depth_ft_md") or r.get("header_depth_ft_md")
                peak      = r.get(val_col)
                depth_str = f"{depth:,.0f} ft MD" if pd.notna(depth) else "—"
                peak_str  = f"{peak:.1f} {unit}" if pd.notna(peak) else "—"
                phase_lbl = label_phase(r.get("phase", ""))
                st.markdown(
                    f"<div style='border-left:3px solid #546E7A;padding:5px 10px;"
                    f"margin:3px 0;background:#F0F4F8;color:#1A202C;"
                    f"border-radius:0 4px 4px 0;font-size:0.88em'>"
                    f"<b>{r['report_date']}</b> · {phase_lbl} · {depth_str} · "
                    f"{val_label}: <b>{peak_str}</b></div>",
                    unsafe_allow_html=True,
                )


def _render_decisions(
    ops_f: pd.DataFrame, sel_phase: str, show_raw: bool
) -> tuple[pd.DataFrame, int]:
    st.subheader("🎯 Key Decisions & Recommendations")

    decision_rows = ops_f[ops_f["operation_text"].str.contains(_DECISION_RE, na=False)].copy()
    decision_rows["dt"] = pd.to_datetime(decision_rows["report_date"], dayfirst=True, errors="coerce")

    if decision_rows.empty:
        st.info("No explicit decision/recommendation text found in this selection.")
        return decision_rows, 0

    phases_present = [p for p in _PHASE_ORDER if p in decision_rows["phase"].values]
    if sel_phase != "ALL":
        phases_present = [sel_phase] if sel_phase in phases_present else []

    total_shown = 0
    for ph in phases_present:
        ph_rows  = decision_rows[decision_rows["phase"] == ph].sort_values("dt")
        if ph_rows.empty:
            continue
        ph_label  = label_phase(ph)
        ph_colour = _PHASE_COLOUR.get(ph, "#455A64")
        with st.expander(
            f"**{ph_label}** — {len(ph_rows)} entr{'y' if len(ph_rows) == 1 else 'ies'}",
            expanded=(sel_phase != "ALL" or len(phases_present) <= 2),
        ):
            for _, r in ph_rows.iterrows():
                text      = str(r["operation_text"])
                sentences = [
                    s.strip() for s in re.split(r"[.•\n]", text)
                    if _DECISION_RE.search(s) and len(s.strip()) > 20
                ]
                for sent in sentences[:2]:
                    st.markdown(
                        f"<div style='border-left:3px solid {ph_colour};"
                        f"padding:6px 12px;margin:4px 0;"
                        f"background:#F0F4F8;color:#1A202C;"
                        f"border-radius:0 4px 4px 0'>"
                        f"<span style='font-size:0.82em;color:#555'>"
                        f"{r['report_date']}</span><br>{sent.strip()}</div>",
                        unsafe_allow_html=True,
                    )
                if show_raw:
                    st.markdown(
                        f"<details style='margin:4px 0 8px 12px'>"
                        f"<summary style='font-size:0.82em;color:#555;cursor:pointer'>"
                        f"Full text — {r['report_date']}</summary>"
                        f"<pre style='font-size:0.80em;white-space:pre-wrap;"
                        f"background:#F0F4F8;color:#1A202C;padding:8px;"
                        f"border-radius:4px;border:1px solid #CBD5E0;"
                        f"margin:4px 0'>{text[:600]}</pre></details>",
                        unsafe_allow_html=True,
                    )
                total_shown += 1

    st.caption(f"{total_shown} entries shown across {len(phases_present)} phase(s).")
    return decision_rows, total_shown


def _render_recommendations(
    ops_f: pd.DataFrame,
    events_f: pd.DataFrame,
    npt_pct: float,
    npt_hrs: float,
    cat_hrs: pd.Series,
    decision_rows: pd.DataFrame,
) -> list[str]:
    st.subheader("📋 Recommendations for Next Well")
    st.info(
        "Auto-generated from NPT patterns — review and edit before including in handover report.",
        icon="💡",
    )

    recs: list[str] = []
    if npt_pct > 30:
        top_cat = cat_hrs.index[0] if not cat_hrs.empty else None
        if top_cat:
            top_label = CATEGORY_LABELS.get(top_cat, top_cat)
            recs.append(
                f"NPT was {npt_pct:.0f}% of phase time. Primary driver: **{top_label}** "
                f"({cat_hrs.iloc[0]:.0f} hrs). Review mitigation plan before spudding next well."
            )
    if not events_f.empty:
        overpull = events_f[events_f["event_type"] == "overpull"]
        if not overpull.empty and overpull["force_klbs"].max() > 200:
            recs.append(
                f"Peak overpull {overpull['force_klbs'].max():.0f} klbs observed. "
                "Review BHA design, centralisation and mud programme for next well."
            )
        losses = events_f[events_f["event_type"] == "mud_loss"]
        if not losses.empty:
            recs.append(
                f"{len(losses)} mud loss event(s) detected "
                f"(peak {losses['loss_rate_bbl_hr'].max():.0f} bbl/hr). "
                "Pre-blend LCM and confirm MPD standing procedures before entering similar interval."
            )
        high_ecd = events_f[(events_f["event_type"] == "formation") &
                            (events_f["sub_type"] == "high_ecd")]
        if not high_ecd.empty:
            recs.append(
                f"High ECD events recorded (max {high_ecd['ecd_ppge'].max():.2f} ppge). "
                "Confirm hydraulics model and MPD contingency plan on next well."
            )
    if len(decision_rows) > 0:
        recs.append(
            f"{len(decision_rows)} operational decisions recorded in DDRs — "
            "review for lessons that transfer to the next well offset programme."
        )
    if not recs:
        recs.append("No specific risks identified — review detailed sections above.")

    for i, rec in enumerate(recs, 1):
        st.markdown(f"**{i}.** {rec}")

    return recs


def _render_export(
    sel_phase: str,
    total_hrs: float,
    npt_hrs: float,
    npt_pct: float,
    cat_hrs: pd.Series,
    recs: list[str],
) -> None:
    st.divider()
    md_lines = [
        f"# Lessons Learned — {sel_phase if sel_phase != 'ALL' else 'Full Well'}\n",
        "## NPT Summary\n",
        f"- Total operations: {total_hrs:,.0f} hrs",
        f"- NPT: {npt_hrs:,.0f} hrs ({npt_pct:.1f}%)",
        "",
        "## Top NPT Categories\n",
    ]
    for cat, hrs in cat_hrs.head(5).items():
        md_lines.append(f"- {CATEGORY_LABELS.get(cat, cat)}: {hrs:.0f} hrs")
    md_lines += ["", "## Recommendations\n"]
    for i, rec in enumerate(recs, 1):
        clean = re.sub(r"\*\*(.+?)\*\*", r"\1", rec)
        md_lines.append(f"{i}. {clean}")
    st.download_button(
        "⬇ Export as Markdown",
        data="\n".join(md_lines),
        file_name=f"lessons_learned_{sel_phase.lower()}.md",
        mime="text/markdown",
    )


def page_lessons_learned(ops: pd.DataFrame, events: pd.DataFrame) -> None:
    n_ddrs = ops["report_date"].nunique()
    st.header("Lessons Learned")
    st.caption(
        f"Auto-generated from {n_ddrs} DDRs. Each section is derived from structured NPT data, "
        "wellbore events and decision/recommendation text extracted from operational narratives."
    )

    phases = [p for p in _PHASE_ORDER if p in ops["phase"].unique()]
    sel_phase = st.selectbox("Phase", ["ALL"] + phases)
    show_raw  = st.checkbox("Show source DDR excerpts", value=False)

    ops_f    = ops if sel_phase == "ALL" else ops[ops["phase"] == sel_phase]
    events_f = (events if sel_phase == "ALL"
                else events[events["phase"] == sel_phase] if not events.empty
                else events)

    st.divider()

    total_hrs, npt_hrs, npt_pct, cat_hrs = _render_npt_summary(ops_f)
    _render_wellbore_challenges(events_f)
    decision_rows, _ = _render_decisions(ops_f, sel_phase, show_raw)
    recs = _render_recommendations(ops_f, events_f, npt_pct, npt_hrs, cat_hrs, decision_rows)
    _render_export(sel_phase, total_hrs, npt_hrs, npt_pct, cat_hrs, recs)
