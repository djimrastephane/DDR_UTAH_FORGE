from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from .constants import repo_root
    from .utils import _ddr_citation_row
except ImportError:
    from constants import repo_root  # type: ignore[no-redef]
    from utils import _ddr_citation_row  # type: ignore[no-redef]

_root = Path(__file__).resolve().parents[3]
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from ddr_rag.vocab import label_phase
from ddr_rag.npt_classifier import CATEGORY_LABELS, classify_equipment_subtype

ANNOTATIONS_PATH = repo_root / "data" / "annotations" / "npt_root_cause_log.json"

PREVENTABILITY_OPTIONS = [
    "Not yet assessed", "Preventable", "Partially preventable", "Unavoidable",
]
RESPONSIBLE_DOMAIN_OPTIONS = [
    "Not yet assessed", "Rig contractor", "Operator", "Directional",
    "Fluids", "Third party", "Other",
]


def _event_id(row: pd.Series) -> str:
    """Stable key for one NPT event: the source DDR (doc_id already encodes
    report date/rig/wellbore + a content hash) plus its time window, so the
    same event keeps the same key even if the corpus is reprocessed."""
    return f"{row.get('doc_id', '')}::{row.get('start_time', '')}-{row.get('end_time', '')}"


def _load_annotations() -> dict:
    if not ANNOTATIONS_PATH.exists():
        return {}
    try:
        return json.loads(ANNOTATIONS_PATH.read_text())
    except Exception:
        return {}


def _save_annotation(event_id: str, data: dict) -> None:
    all_ann = _load_annotations()
    all_ann[event_id] = data
    ANNOTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_PATH.write_text(json.dumps(all_ann, indent=2, sort_keys=True))


def page_root_cause_tracker(ops: pd.DataFrame) -> None:
    st.header("Root Cause & Action Tracking")
    st.caption(
        "Engineer-recorded root cause, preventability, and corrective actions for each "
        "NPT event. This is judgment DDRs don't record, so nothing here is auto-filled — "
        "entries are saved locally for reuse in DWOPs, EOWRs, and lookback sessions."
    )

    npt_ops = ops[ops["is_npt"]].copy()
    if npt_ops.empty:
        st.info("No NPT events found for this well.")
        return

    npt_ops = npt_ops.sort_values("duration_hr", ascending=False)
    annotations = _load_annotations()
    event_ids = npt_ops.apply(_event_id, axis=1)
    n_annotated = sum(1 for eid in event_ids if eid in annotations)

    c1, c2 = st.columns(2)
    c1.metric("NPT events", len(npt_ops))
    c2.metric("Annotated", f"{n_annotated} / {len(npt_ops)}")

    show_filter = st.radio(
        "Show", ["All events", "Needs annotation", "Annotated"], horizontal=True,
    )

    for (_, row), event_id in zip(npt_ops.iterrows(), event_ids):
        existing = annotations.get(event_id, {})
        is_annotated = bool(existing)

        if show_filter == "Needs annotation" and is_annotated:
            continue
        if show_filter == "Annotated" and not is_annotated:
            continue

        badge = "✅" if is_annotated else "◻️"
        cat_code = str(row.get("npt_category") or "")
        cat_label = CATEGORY_LABELS.get(cat_code, "—")
        header = (
            f"{badge} {row['report_date']} · {label_phase(row['phase'])} · "
            f"{cat_label} · {row['duration_hr']:.1f}h"
        )
        with st.expander(header):
            st.markdown(f"**DDR evidence:** {_ddr_citation_row(row)}")
            excerpt = str(row.get("operation_text") or "")[:400]
            if excerpt:
                st.markdown(f"> {excerpt}")
            if cat_code == "equipment":
                subtype = classify_equipment_subtype(str(row.get("operation_text") or ""))
                if subtype != "Unspecified":
                    st.caption(f"Equipment sub-type (auto-detected): {subtype}")

            with st.form(key=f"rct_form_{event_id}"):
                col1, col2 = st.columns(2)
                with col1:
                    immediate_cause = st.text_input(
                        "Immediate cause", value=existing.get("immediate_cause", "")
                    )
                    root_cause = st.text_input(
                        "Root cause", value=existing.get("root_cause", "")
                    )
                    prev_default = existing.get("preventability", "Not yet assessed")
                    preventability = st.selectbox(
                        "Preventability", PREVENTABILITY_OPTIONS,
                        index=PREVENTABILITY_OPTIONS.index(prev_default)
                        if prev_default in PREVENTABILITY_OPTIONS else 0,
                    )
                with col2:
                    dom_default = existing.get("responsible_domain", "Not yet assessed")
                    responsible_domain = st.selectbox(
                        "Responsible domain", RESPONSIBLE_DOMAIN_OPTIONS,
                        index=RESPONSIBLE_DOMAIN_OPTIONS.index(dom_default)
                        if dom_default in RESPONSIBLE_DOMAIN_OPTIONS else 0,
                    )
                corrective_action = st.text_area(
                    "Corrective action", value=existing.get("corrective_action", "")
                )
                carry_forward_risk = st.text_area(
                    "Carry-forward risk to next well",
                    value=existing.get("carry_forward_risk", ""),
                )
                if st.form_submit_button("Save"):
                    _save_annotation(event_id, {
                        "immediate_cause": immediate_cause,
                        "root_cause": root_cause,
                        "preventability": preventability,
                        "responsible_domain": responsible_domain,
                        "corrective_action": corrective_action,
                        "carry_forward_risk": carry_forward_risk,
                        "event_date": str(row["report_date"]),
                        "event_phase": label_phase(row["phase"]),
                        "event_category": cat_label,
                        "event_duration_hr": float(row["duration_hr"]),
                    })
                    st.success("Saved.")
                    st.rerun()

    if annotations:
        st.divider()
        export_rows = []
        for (_, row), event_id in zip(npt_ops.iterrows(), event_ids):
            ann = annotations.get(event_id)
            if not ann:
                continue
            export_rows.append({
                "Date": row["report_date"],
                "Phase": label_phase(row["phase"]),
                "Category": CATEGORY_LABELS.get(str(row.get("npt_category") or ""), "—"),
                "Duration (h)": row["duration_hr"],
                "Immediate cause": ann.get("immediate_cause", ""),
                "Root cause": ann.get("root_cause", ""),
                "Preventability": ann.get("preventability", ""),
                "Responsible domain": ann.get("responsible_domain", ""),
                "Corrective action": ann.get("corrective_action", ""),
                "Carry-forward risk": ann.get("carry_forward_risk", ""),
            })
        if export_rows:
            st.download_button(
                "⬇ Download root cause log CSV",
                data=pd.DataFrame(export_rows).to_csv(index=False),
                file_name="npt_root_cause_log.csv",
                mime="text/csv",
            )
