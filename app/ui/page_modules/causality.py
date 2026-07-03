from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from .constants import PHASE_COLOURS
    from .loaders import load_causality
    from .utils import _apply_chart_theme
except ImportError:
    from constants import PHASE_COLOURS        # type: ignore[no-redef]
    from loaders import load_causality         # type: ignore[no-redef]
    from utils import _apply_chart_theme       # type: ignore[no-redef]


def _row_style(row: pd.Series) -> list[str]:
    if row.get("Escalating") == "↑ Yes":
        return ["background-color:#FFE0B2; color:#212121"] * len(row)
    return [""] * len(row)


def _render_timeline(report: dict) -> None:
    st.subheader("Well phase timeline")
    timeline = report["timeline"]
    fig = go.Figure()
    for row in timeline:
        fig.add_trace(go.Bar(
            x=[row["npt_pct"]], y=[row["label"]],
            orientation="h",
            marker_color=PHASE_COLOURS.get(row["phase"], "#999"),
            text=f"{row['npt_pct']:.0f}% NPT  |  {row['npt_hrs']:.0f}h  |  {row['n_days']} days",
            textposition="inside",
            name=row["label"], showlegend=False,
            hovertemplate=(
                f"<b>{row['label']}</b><br>"
                f"{row['date_start']} → {row['date_end']}<br>"
                f"NPT: {row['npt_pct']:.0f}%  "
                f"({row['npt_hrs']:.0f}h / {row['total_hrs']:.0f}h)<extra></extra>"
            ),
        ))
    fig.update_layout(
        xaxis=dict(title="NPT %", range=[0, 110]),
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA",
    )
    st.plotly_chart(_apply_chart_theme(fig), use_container_width=True)


def _render_transition_findings(report: dict) -> None:
    st.subheader("Phase transition findings")
    st.caption(
        f"Each transition window covers {report['graph']['window_days']} days before and after "
        "the phase boundary. "
        "Causal terms = vocabulary that persists across the boundary with ≥40% NPT rate. "
        "Escalating = frequency increased by ≥50% from one phase to the next."
    )

    for t in report["transitions"]:
        pre_npt  = t["pre_npt_pct"]
        post_npt = t["post_npt_pct"]
        n_causal = len(t["causal_terms"])
        n_esc    = len(t["escalating_terms"])

        if pre_npt > 60 and post_npt > 30:
            badge = "🔴 High carry-over risk"
        elif n_esc >= 3:
            badge = "🟠 Escalating signals"
        else:
            badge = "🟢 Clean handover"

        label = (
            f"**{t['phase_from']} → {t['phase_to']}** — "
            f"{t['label_from']} → {t['label_to']}  {badge}"
        )

        with st.expander(label, expanded=(t["phase_from"] == "PROD1")):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pre-transition NPT",  f"{pre_npt:.0f}%")
            c2.metric("Post-transition NPT", f"{post_npt:.0f}%",
                      delta=f"{post_npt - pre_npt:+.0f}pp",
                      delta_color="inverse")
            c3.metric("Causal terms",     n_causal)
            c4.metric("Escalating terms", n_esc)

            st.markdown(f"_{t['narrative']}_")

            if t["causal_terms"]:
                st.markdown(
                    "**Terms with elevated NPT association** "
                    "(present in both phases, ≥40% NPT in next phase):"
                )
                rows = [
                    {
                        "Term":           c["term"],
                        "NPT (pre→post)": f"{c['pre_npt_ratio']:.0%} → {c['post_npt_ratio']:.0%}",
                        "Freq change":    f"×{c['freq_ratio']:.1f}",
                        "Post NPT hours": f"{c['post_npt_hrs']:.0f}h",
                        "Escalating":     "↑ Yes" if c["is_escalating"] else "—",
                    }
                    for c in t["causal_terms"][:12]
                ]
                st.dataframe(
                    pd.DataFrame(rows).style.apply(_row_style, axis=1),
                    hide_index=True, use_container_width=True,
                )


def _render_prod1_compzn_deepdive(report: dict, ops: pd.DataFrame) -> None:
    st.divider()
    st.subheader("Deep-dive: Production → Completion causality")

    prod_comp = next(
        (t for t in report["transitions"]
         if t["phase_from"] == "PROD1" and t["phase_to"] == "COMPZN"),
        None,
    )
    if not prod_comp:
        return

    st.markdown(
        "This is the well's strongest observed operational association across phase boundaries. "
        "The 65 operations in which the PROD1 magnet BHA crossed NCS frac sleeves — "
        "all at reduced speed with restrictions noted — directly preceded the sleeve "
        "location difficulties in COMPZN."
    )

    overpull_monthly = (
        ops[ops["operation_text"].str.contains("overpull", case=False, na=False)]
        .assign(month=lambda d: d["report_date_parsed"].dt.to_period("M").astype(str))
        .groupby("month")
        .agg(
            count=("duration_hr", "size"),
            npt_hrs=("duration_hr", lambda x: x[ops.loc[x.index, "is_npt"]].sum()),
        )
        .reset_index()
    )
    if not overpull_monthly.empty:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=overpull_monthly["month"],
            y=overpull_monthly["count"],
            marker_color=[
                "#F44336" if m >= "2024-08" else "#FF9800"
                for m in overpull_monthly["month"]
            ],
            name="Overpull occurrences",
            hovertemplate="<b>%{x}</b><br>%{y} overpull ops<extra></extra>",
        ))
        _pre = overpull_monthly[overpull_monthly["month"] < "2024-08"]["month"]
        _transition_x = _pre.iloc[-1] if not _pre.empty else "2024-07"
        fig2.add_shape(
            type="line",
            x0=_transition_x, x1=_transition_x,
            y0=0, y1=1, yref="paper",
            line=dict(dash="dash", color="#555", width=1.5),
        )
        fig2.add_annotation(
            x=_transition_x, y=1.02, yref="paper",
            text="← PROD1  |  COMPZN →",
            showarrow=False, xanchor="center",
            font=dict(size=9, color="#555"),
        )
        fig2.update_layout(
            title="Overpull occurrences by month — PROD1 → COMPZN escalation",
            height=280, margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Month",
            yaxis_title="Operations mentioning overpull",
            plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA",
            showlegend=False,
        )
        st.plotly_chart(_apply_chart_theme(fig2), use_container_width=True)
        st.caption(
            "Orange = PROD1 (magnet BHA crossing frac sleeves). "
            "Red = COMPZN (FIA locator tool encountering overpull at sleeve profiles). "
            "The escalation from 29 to 71 monthly occurrences across the phase boundary "
            "is the primary leading indicator."
        )

    st.info(
        "**Operational recommendation for offset wells using similar MPD casing + NCS completion programmes:** "
        "Dedicated wellbore profile assessment after the metallic debris recovery programme — "
        "specifically checking frac sleeve shoulder profiles for deformation from repeated "
        "magnet BHA passes before switching to the FIA locator. "
        "An NCS sleeve condition log from the final PROD1 magnet run should be available "
        "before COMPZN tool selection is finalised."
    )


def page_causality(ops: pd.DataFrame) -> None:
    st.header("Cross-Phase Causality")
    st.caption(
        "Operational signals that carry from one drilling phase into the next — "
        "identifying what in Phase A predicts problems in Phase B."
    )

    report = load_causality()
    if report is None:
        st.warning("Causality data not found. Run: `python scripts/build_causality.py`")
        return

    _render_timeline(report)
    st.divider()
    _render_transition_findings(report)
    _render_prod1_compzn_deepdive(report, ops)
