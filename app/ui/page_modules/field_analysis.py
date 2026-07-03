from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from .constants import PHASE_COLOURS, PHASE_ORDER
    from .loaders import (
        load_field_ops, load_field_headers,
        load_corpus_gaps, _run_search, load_global_search,
    )
    from .utils import _apply_chart_theme
except ImportError:
    from constants import PHASE_COLOURS, PHASE_ORDER              # type: ignore[no-redef]
    from loaders import (                                          # type: ignore[no-redef]
        load_field_ops, load_field_headers,
        load_corpus_gaps, _run_search, load_global_search,
    )
    from utils import _apply_chart_theme                           # type: ignore[no-redef]

_ui_root = Path(__file__).resolve().parents[3]
if str(_ui_root / "src") not in sys.path:
    sys.path.insert(0, str(_ui_root / "src"))

from ddr_rag.vocab import label_phase
from ddr_rag.npt_classifier import classify_ops_df, CATEGORY_LABELS

_WB_COLOURS: dict[str, str] = {
    "30-07a-RB": "#1976D2",
    "30-07a-R2": "#E65100",
}


def _derive_field_name(hdr: pd.DataFrame) -> str:
    if "field_name" in hdr.columns:
        vals = hdr["field_name"].dropna()
        if not vals.empty:
            name = vals.mode().iloc[0]
            return str(name).title()
    return "Field"


def _wellbore_label(wb: str) -> str:
    return wb.replace("-", "/", 2)      # 30-07a-RB → 30/07a-RB


def _render_campaign_overview(
    ops: pd.DataFrame,
    hdr: pd.DataFrame,
    wellbores: list[str],
    wb_cols: dict[str, str],
) -> None:
    total_cost  = float(hdr["cum_cost_num"].max() or 0)
    total_h     = float(ops["duration_hr"].sum())
    npt_h       = float(ops.loc[ops["is_npt"], "duration_hr"].sum())
    npt_pct     = 100 * npt_h / max(total_h, 1)
    max_td      = float(hdr["end_depth_num"].dropna().max() or 0)
    campaign_days = int(hdr["report_date_dt"].nunique())
    rig         = hdr["rig_name"].dropna().mode().iloc[0] if not hdr["rig_name"].dropna().empty else ""

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total cost",     f"£{total_cost/1e6:.1f} M")
    k2.metric("Overall NPT",    f"{npt_pct:.0f}%",
              f"{npt_h:.0f} h of {total_h:.0f} h")
    k3.metric("Campaign",       f"{campaign_days} days")
    k4.metric("Max TD (MD)",    f"{max_td:,.0f} ft")
    k5.metric("Rig",            rig)

    st.divider()

    st.subheader("Phase Timeline")
    ops_dt = ops.copy()
    ops_dt["report_date_dt"] = pd.to_datetime(
        ops_dt["report_date"], dayfirst=True, errors="coerce"
    )
    phase_ranges = (
        ops_dt.groupby(["wellbore", "phase"])["report_date_dt"]
        .agg(start="min", end="max")
        .reset_index()
    )
    phase_ranges["end"] = phase_ranges["end"] + pd.Timedelta(days=1)

    fig = go.Figure()
    y_labels: list[str] = []
    y_pos = 0
    for ph in reversed(PHASE_ORDER):
        for wb in reversed(wellbores):
            row = phase_ranges[
                (phase_ranges["phase"] == ph) & (phase_ranges["wellbore"] == wb)
            ]
            if row.empty:
                continue
            lbl = f"{_wellbore_label(wb)} · {label_phase(ph).split('/')[0].strip()}"
            y_labels.append(lbl)
            start = row["start"].iloc[0]
            end   = row["end"].iloc[0]
            colour = PHASE_COLOURS.get(ph, "#555")
            fig.add_trace(go.Bar(
                y=[lbl],
                x=[(end - start).days],
                base=[start.timestamp() * 1000],
                orientation="h",
                marker_color=colour,
                name=label_phase(ph).split("/")[0].strip(),
                showlegend=(wb == wellbores[0]),
                legendgroup=ph,
                hovertemplate=(
                    f"<b>{lbl}</b><br>"
                    f"{start.strftime('%d %b')} – {(end - pd.Timedelta(days=1)).strftime('%d %b %Y')}"
                    f" ({(end-start).days} days)<extra></extra>"
                ),
            ))
            y_pos += 1

    fig.update_layout(
        barmode="stack", height=max(220, y_pos * 36 + 80),
        margin=dict(l=10, r=10, t=10, b=40),
        xaxis=dict(
            type="date",
            title="Date",
            tickformat="%b %Y",
        ),
        yaxis=dict(title=None),
        legend=dict(orientation="h", y=-0.18, title_text="Phase"),
        plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA",
    )
    st.plotly_chart(_apply_chart_theme(fig), use_container_width=True)

    st.subheader("Wellbore Summary")
    rows = []
    for wb in wellbores:
        wo = ops[ops["wellbore"] == wb]
        wh = hdr[hdr["wellbore"] == wb]
        tot = float(wo["duration_hr"].sum())
        npt = float(wo.loc[wo["is_npt"], "duration_hr"].sum())
        cost = float(wh["cum_cost_num"].dropna().max() or 0)
        n_days = int(wh["report_date_dt"].nunique())
        max_d  = float(wh["end_depth_num"].dropna().max() or 0)
        phases = ", ".join(
            label_phase(p).split("/")[0].strip()
            for p in PHASE_ORDER if p in wo["phase"].values
        )
        rows.append({
            "Wellbore": _wellbore_label(wb),
            "DDRs":     n_days,
            "Phases":   phases,
            "NPT %":    f"{100*npt/max(tot,1):.0f}%",
            "NPT h":    round(npt, 0),
            "Max TD (ft MD)": f"{max_d:,.0f}",
            "Cost":     f"£{cost/1e6:.1f} M",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_phase_analysis(
    ops: pd.DataFrame,
    wellbores: list[str],
    wb_cols: dict[str, str],
) -> None:
    st.markdown(
        "Phase-by-phase NPT breakdown for each wellbore. "
        "Only phases present on that wellbore are shown."
    )

    rows = []
    for ph in PHASE_ORDER:
        for wb in wellbores:
            g = ops[(ops["wellbore"] == wb) & (ops["phase"] == ph)]
            if g.empty:
                continue
            tot = float(g["duration_hr"].sum())
            npt = float(g.loc[g["is_npt"], "duration_hr"].sum())
            n_days = int(g["report_date"].nunique())
            rows.append({
                "Phase": label_phase(ph).split("/")[0].strip(),
                "Wellbore": _wellbore_label(wb),
                "NPT %": round(100 * npt / max(tot, 1), 1),
                "NPT h": round(npt, 0),
                "Total h": round(tot, 0),
                "Days": n_days,
                "_order": PHASE_ORDER.index(ph),
                "_wb": wb,
            })
    pd_df = pd.DataFrame(rows).sort_values("_order")

    fig = go.Figure()
    for wb in wellbores:
        sub = pd_df[pd_df["_wb"] == wb]
        fig.add_trace(go.Bar(
            name=_wellbore_label(wb),
            x=sub["Phase"],
            y=sub["NPT %"],
            marker_color=wb_cols.get(wb, "#555"),
            text=sub["NPT %"].map("{:.0f}%".format),
            textposition="outside",
            hovertemplate=(
                f"<b>{_wellbore_label(wb)}</b><br>"
                "%{x}<br>"
                "NPT %{y:.0f}%  (%{customdata[0]:.0f} h / %{customdata[1]:.0f} h total)"
                "<extra></extra>"
            ),
            customdata=sub[["NPT h", "Total h"]].values,
        ))
    fig.update_layout(
        barmode="group", height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="NPT %", range=[0, 115]),
        legend=dict(orientation="h", y=-0.2),
        plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA",
    )
    st.plotly_chart(_apply_chart_theme(fig), use_container_width=True)

    shared = [ph for ph in PHASE_ORDER
              if all(not ops[(ops["wellbore"] == wb) & (ops["phase"] == ph)].empty
                     for wb in wellbores)]
    if len(wellbores) >= 2 and shared:
        wb1, wb2 = wellbores[0], wellbores[1]
        st.subheader(f"Shared-phase comparison  ·  {_wellbore_label(wb1)} vs {_wellbore_label(wb2)}")
        delta_rows = []
        for ph in shared:
            s1 = pd_df[(pd_df["_wb"] == wb1) & (pd_df["_order"] == PHASE_ORDER.index(ph))]
            s2 = pd_df[(pd_df["_wb"] == wb2) & (pd_df["_order"] == PHASE_ORDER.index(ph))]
            if s1.empty or s2.empty:
                continue
            npt1, npt2 = float(s1["NPT %"].iloc[0]), float(s2["NPT %"].iloc[0])
            h1,   h2   = float(s1["NPT h"].iloc[0]),  float(s2["NPT h"].iloc[0])
            delta_pct  = npt2 - npt1
            signal = (
                "🔴 Higher NPT" if delta_pct > 10 else
                "🟢 Lower NPT"  if delta_pct < -10 else
                "≈ Similar"
            )
            delta_rows.append({
                "Phase": label_phase(ph).split("/")[0].strip(),
                f"{_wellbore_label(wb1)} NPT %": f"{npt1:.0f}%",
                f"{_wellbore_label(wb1)} h":     f"{s1['Total h'].iloc[0]:.0f} h",
                f"{_wellbore_label(wb2)} NPT %": f"{npt2:.0f}%",
                f"{_wellbore_label(wb2)} h":     f"{s2['Total h'].iloc[0]:.0f} h",
                "Δ NPT %":  f"{delta_pct:+.0f}pp",
                "Signal":   signal,
            })
        if delta_rows:
            st.dataframe(pd.DataFrame(delta_rows), hide_index=True, use_container_width=True)

    with st.expander("All phase details"):
        disp = pd_df.drop(columns=["_order", "_wb"]).rename(
            columns={"NPT %": "NPT %", "Total h": "Total h", "Days": "DDRs"}
        )
        st.dataframe(disp, hide_index=True, use_container_width=True)


def _render_recurring_challenges(
    ops: pd.DataFrame,
    wellbores: list[str],
) -> None:
    st.markdown(
        "Which NPT categories are **recurring across both wellbores** "
        "vs unique to one — the key signal for lessons-learned effectiveness."
    )

    if "npt_category" not in ops.columns:
        st.info("Classifying operations…")
        ops = ops.copy()
        ops["npt_category"] = classify_ops_df(ops)

    npt_ops = ops[ops["is_npt"] & ops["npt_category"].notna()]
    all_cats = npt_ops["npt_category"].unique()

    cat_rows = []
    for cat in all_cats:
        row: dict = {"NPT Category": CATEGORY_LABELS.get(cat, cat)}
        total_h = 0.0
        for wb in wellbores:
            h = float(
                npt_ops[(npt_ops["wellbore"] == wb) & (npt_ops["npt_category"] == cat)]["duration_hr"].sum()
            )
            row[_wellbore_label(wb)] = round(h, 0)
            total_h += h
        row["Total (h)"] = round(total_h, 0)
        n_wb = sum(1 for wb in wellbores if row.get(_wellbore_label(wb), 0) > 5)
        row["Recurrence"] = (
            "🔴 Both wellbores"         if n_wb == len(wellbores) else
            f"🟡 {n_wb}/{len(wellbores)} wellbores" if n_wb > 0 else "—"
        )
        cat_rows.append(row)

    cat_df = (
        pd.DataFrame(cat_rows)
        .sort_values("Total (h)", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(cat_df, hide_index=True, use_container_width=True, height=400)

    if len(wellbores) >= 2:
        wb1, wb2 = wellbores[0], wellbores[1]
        l1, l2 = _wellbore_label(wb1), _wellbore_label(wb2)
        recurring  = cat_df[cat_df["Recurrence"].str.startswith("🔴")]
        only_wb1   = cat_df[(cat_df[l1] > 5) & (cat_df[l2] <= 5)]
        only_wb2   = cat_df[(cat_df[l2] > 5) & (cat_df[l1] <= 5)]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**🔴 Recurring — both wellbores**")
            st.caption("Priority for future wells — not eliminated by wellbore change.")
            for _, r in recurring.iterrows():
                st.markdown(f"- {r['NPT Category']} &nbsp;({r['Total (h)']} h combined)")
        with c2:
            st.markdown(f"**🔵 Only on {l1}**")
            st.caption("Eliminated or absent on the second wellbore.")
            for _, r in only_wb1.head(6).iterrows():
                st.markdown(f"- {r['NPT Category']} &nbsp;({r[l1]} h)")
        with c3:
            st.markdown(f"**🟠 Only on {l2}**")
            st.caption("New challenge not seen on first wellbore.")
            for _, r in only_wb2.head(6).iterrows():
                st.markdown(f"- {r['NPT Category']} &nbsp;({r[l2]} h)")


def _render_cross_ddr_search(
    ops: pd.DataFrame,
    wellbores: list[str],
    wb_cols: dict[str, str],
) -> None:
    st.markdown(
        "Search across all ingested DDRs. Useful for comparing how a specific "
        "challenge or operation was handled across different dates or phases."
    )

    question = st.text_input(
        "Ask a question across all DDRs",
        placeholder="e.g. How was overpull managed during liner operations?",
        key="field_search_q",
    )
    sel_wb = st.selectbox(
        "Filter by wellbore",
        ["All wellbores"] + [_wellbore_label(wb) for wb in wellbores],
        key="field_search_wb",
    )
    field_k = st.slider("Results", 5, 20, 10, 5, key="field_k")

    if not question:
        st.markdown(
            "**Example questions:**\n"
            "- What caused the extended PROD1 NPT on 30/07a-R2?\n"
            "- How was overpull at frac sleeves managed?\n"
            "- What MPD issues occurred during INTRM1?\n"
            "- Describe the COND1 liner loss events.\n"
            "- How was well control managed during COMPZN?"
        )
        return

    if not load_global_search():
        st.warning("Global index not found. Run: `python scripts/build_global_index.py`")
        return

    doc_filter = None
    if sel_wb != "All wellbores":
        for wb in wellbores:
            if _wellbore_label(wb) in sel_wb:
                doc_filter = wb
                break

    with st.spinner("Searching…"):
        results, search_err = _run_search(question, k=field_k * 2, doc_filter=doc_filter)
    if search_err:
        st.error(f"Search failed: {search_err}")
        return

    results = results[:field_k]
    if not results:
        st.warning("No results found.")
        return

    st.markdown(f"**{len(results)} results**")
    for r in results:
        doc_id  = str(r.get("doc_id", ""))
        wb_match = next((wb for wb in wellbores if wb in doc_id), None)
        colour  = wb_cols.get(wb_match, "#555") if wb_match else "#777"
        score   = r.get("score", 0)
        date    = r.get("report_date", "")
        snippet = r.get("snippet", "")[:280]
        label   = _wellbore_label(wb_match) if wb_match else doc_id[-20:]
        st.markdown(
            f"<span style='color:{colour};font-weight:bold'>{label}</span>"
            f" · {date} · score {score:.3f}",
            unsafe_allow_html=True,
        )
        st.markdown(f"> {snippet}")
        with st.expander("Full text"):
            st.code(r.get("chunk_text", ""), language=None)
        st.divider()


def _render_corpus_completeness() -> None:
    gaps = load_corpus_gaps()

    if not gaps:
        st.success("No sequence gaps detected — corpus appears complete.")
        return

    total_missing = sum(g["num_missing"] for g in gaps)
    st.warning(
        f"**{total_missing} DDR reports are missing** across {len(gaps)} gap(s) in the sequence.  "
        "Review each gap below and either supply the missing PDFs or document why they are absent."
    )

    for i, gap in enumerate(gaps, 1):
        nums    = gap["missing_nums"]
        n       = gap["num_missing"]
        d_bef   = gap["date_before"]
        d_aft   = gap["date_after"]
        ph_bef  = gap["phase_before"]
        ph_aft  = gap["phase_after"]
        cross   = gap["cross_phase"]

        if n >= 5 or cross:
            badge = "🔴 High — significant coverage gap"
        elif n >= 2:
            badge = "🟡 Medium — operations may be missing"
        else:
            badge = "🟢 Low — likely a single report gap"

        ph_bef_lbl = label_phase(ph_bef).split("/")[0].strip() if ph_bef else "—"
        ph_aft_lbl = label_phase(ph_aft).split("/")[0].strip() if ph_aft else "—"
        num_str    = (f"DDR-{nums[0]:03d}" if n == 1
                      else f"DDR-{nums[0]:03d} – DDR-{nums[-1]:03d}")

        header = f"Gap {i} · {num_str} · {n} report{'s' if n > 1 else ''} · {badge}"

        with st.expander(header, expanded=(n >= 5 or cross)):
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Last report before gap**")
                st.markdown(f"- DDR-{nums[0]-1:03d} · `{d_bef}`")
                st.markdown(f"- Phase: **{ph_bef_lbl}**")
                if gap["depth_before"]:
                    st.markdown(f"- End depth: **{gap['depth_before']} ft MD**")
                if gap["last_24hr_before"]:
                    st.markdown("**Last 24 hr summary:**")
                    st.caption(gap["last_24hr_before"][:500])

            with c2:
                st.markdown("**First report after gap**")
                st.markdown(f"- DDR-{nums[-1]+1:03d} · `{d_aft}`")
                st.markdown(f"- Phase: **{ph_aft_lbl}**")
                if gap["depth_after"]:
                    st.markdown(f"- End depth: **{gap['depth_after']} ft MD**")
                if gap["morning_after"]:
                    st.markdown("**Morning report (first day back):**")
                    st.caption(gap["morning_after"][:500])

            if cross:
                st.error(
                    f"⚠️ **Phase boundary crossed** — {ph_bef_lbl} → {ph_aft_lbl}.  "
                    "The missing reports cover an operational transition that is not recorded in this corpus.  "
                    "**Action required:** locate the missing DDRs or document what occurred during this window."
                )
            elif n >= 2:
                st.warning(
                    f"Operations before and after may be discontinuous ({n} reports missing).  "
                    "Review the last 24 hr summary and morning report above to assess whether "
                    "context has been lost.  **Action:** supply missing DDRs or confirm continuity."
                )
            else:
                st.info(
                    "Single-report gap — operations before and after appear to be part of the same run.  "
                    "Confirm by reviewing the summaries above."
                )

            st.markdown(
                f"**Missing DDR numbers:** {', '.join(str(n) for n in nums)}"
            )


def page_field_analysis() -> None:
    field_ops = load_field_ops()
    field_hdr = load_field_headers()

    if field_ops.empty:
        st.warning(
            "Multi-well data not found. "
            "Run: `python scripts/build_synthetic_well.py`"
        )
        return

    field_name = _derive_field_name(field_hdr)
    st.header(f"Field Analysis — {field_name} Field")

    if "report_date_dt" not in field_hdr.columns or field_hdr["report_date_dt"].isna().all():
        field_hdr = field_hdr.copy()
        field_hdr["report_date_dt"] = pd.to_datetime(
            field_hdr["report_date"], dayfirst=True, errors="coerce"
        )

    if "npt_category" not in field_ops.columns:
        field_ops = field_ops.copy()
        field_ops["npt_category"] = classify_ops_df(field_ops)

    wellbores = sorted(field_ops["wellbore"].unique())
    wb_cols   = {wb: _WB_COLOURS.get(wb, "#555") for wb in wellbores}

    total_h  = float(field_ops["duration_hr"].sum())
    npt_h    = float(field_ops.loc[field_ops["is_npt"], "duration_hr"].sum())
    npt_pct  = 100 * npt_h / max(total_h, 1)
    cost     = float(field_hdr["cum_cost_num"].max() or 0)
    n_days   = int(field_hdr["report_date_dt"].nunique())
    rig      = field_hdr["rig_name"].dropna().mode().iloc[0] if not field_hdr["rig_name"].dropna().empty else ""
    wb_labels = " and ".join(_wellbore_label(wb) for wb in wellbores)
    st.info(
        f"**{n_days}-day campaign** covering wellbores {wb_labels} on **{rig}**.  "
        f"Total ingested cost **£{cost/1e6:.1f} M** · Overall NPT **{npt_pct:.0f}%** "
        f"({npt_h:.0f} h of {total_h:.0f} h)."
    )

    gaps = load_corpus_gaps()
    if gaps:
        total_missing = sum(g["num_missing"] for g in gaps)
        cross_phase   = sum(1 for g in gaps if g["cross_phase"])
        st.warning(
            f"**Corpus completeness:** {total_missing} DDR report(s) missing "
            f"across {len(gaps)} gap(s)"
            + (f", including **{cross_phase} phase-boundary gap(s)**" if cross_phase else "")
            + " — see the **⚠️ Corpus Completeness** tab for details."
        )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Campaign Overview",
        "📈 Phase Analysis",
        "🔄 Recurring Challenges",
        "🔍 Cross-DDR Search",
        "⚠️ Corpus Completeness",
    ])

    with tab1:
        _render_campaign_overview(field_ops, field_hdr, wellbores, wb_cols)

    with tab2:
        _render_phase_analysis(field_ops, wellbores, wb_cols)

    with tab3:
        _render_recurring_challenges(field_ops, wellbores)

    with tab4:
        _render_cross_ddr_search(field_ops, wellbores, wb_cols)

    with tab5:
        _render_corpus_completeness()
