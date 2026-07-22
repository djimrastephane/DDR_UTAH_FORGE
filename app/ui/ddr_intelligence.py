from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_repo_root = Path(__file__).resolve().parents[2]
_ui_dir    = Path(__file__).resolve().parent

for _p in (str(_repo_root / "src"), str(_ui_dir)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from loaders import (
    load_all_ops, load_all_headers,
    load_weather, load_planned_time, load_vessels,
    load_wellbore_events,
)

from page_modules.operation_sequence import page_operation_sequence
from page_modules.campaign_summary  import page_executive_summary
from page_modules.well_overview     import page_well_overview
from page_modules.npt_intelligence  import page_npt_intelligence
from page_modules.upload_ddrs       import page_upload_ddrs
from page_modules.operations_log    import page_operations_log
from page_modules.operational_graph import page_operational_graph
from page_modules.lessons_learned   import page_lessons_learned
from page_modules.root_cause_tracker import page_root_cause_tracker
from page_modules.corpus_search     import page_corpus_search


_EXEC_PAGES = ["📊 Campaign Summary"]

_ENG_PAGES = [
    "🔩 Well Overview",
    "📈 NPT Intelligence",
    "🔄 Operation Sequence",
    "🕸 Operational Graph",
    "📝 Lessons Learned",
    "🔧 Root Cause & Actions",
    "📋 Operations Log",
    "📥 Upload DDRs",
]

_INV_PAGES = [
    "🔍 Corpus Search",
]

_REQUIRES_PROCESSED_DATA = {
    "🔩 Well Overview",
    "📈 NPT Intelligence",
    "🔄 Operation Sequence",
    "📋 Operations Log",
}


def _page_title(label: str) -> str:
    return label.split(" ", 1)[1] if " " in label else label


def _show_missing_processed_data(label: str) -> None:
    st.header(_page_title(label))
    st.info(
        "No processed DDR data found yet. The raw Utah FORGE PDFs are staged in "
        "`data/raw/`; run `python scripts/batch_preprocess_raw_ddrs.py --build-index` "
        "to populate this dashboard."
    )


def main() -> None:
    st.set_page_config(
        page_title="DDR Operational Intelligence",
        page_icon="🛢",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("DDR Intelligence")
    st.sidebar.caption("Utah FORGE · FORGE-16A-78-32")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        _EXEC_PAGES + _ENG_PAGES + _INV_PAGES,
        label_visibility="collapsed",
    )
    st.sidebar.divider()

    st.sidebar.caption("v2.0")

    # ── Force clean rerun on page change ─────────────────────────────────────
    if st.session_state.get("_active_page") != page:
        st.session_state["_active_page"] = page
        st.rerun()

    if page in _REQUIRES_PROCESSED_DATA:
        _ops_probe = load_all_ops()
        _hdr_probe = load_all_headers()
        if _ops_probe.empty or _hdr_probe.empty:
            _show_missing_processed_data(page)
            return

    if page == "🔍 Corpus Search":       return page_corpus_search()
    if page == "🔄 Operation Sequence":  return page_operation_sequence()
    if page == "📥 Upload DDRs":         return page_upload_ddrs()

    with st.spinner("Loading data..."):
        ops = load_all_ops()
        hdr = load_all_headers()
        wth = load_weather()
        pt  = load_planned_time()
        vsl = load_vessels()

    _kw = dict(
        weather=wth      if not wth.empty else None,
        planned_time=pt  if not pt.empty  else None,
        vessels=vsl      if not vsl.empty else None,
    )

    if page == "📊 Campaign Summary":
        page_executive_summary(ops, hdr, load_wellbore_events())
    elif page == "🔩 Well Overview":
        page_well_overview(ops, hdr, **_kw)
    elif page == "🕸 Operational Graph":
        page_operational_graph(ops)
    elif page == "📈 NPT Intelligence":
        page_npt_intelligence(ops)
    elif page == "📝 Lessons Learned":
        page_lessons_learned(ops, load_wellbore_events())
    elif page == "🔧 Root Cause & Actions":
        page_root_cause_tracker(ops)
    elif page == "📋 Operations Log":
        page_operations_log(ops)


if __name__ == "__main__":
    main()
