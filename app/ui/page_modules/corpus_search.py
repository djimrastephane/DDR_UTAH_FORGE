from __future__ import annotations

import re

import streamlit as st

try:
    from .loaders import load_global_search, _looks_like_aggregate, _run_search, load_all_headers
except ImportError:
    from loaders import load_global_search, _looks_like_aggregate, _run_search, load_all_headers  # type: ignore[no-redef]


_SCORE_HIGH = 0.60
_SCORE_MED  = 0.40


def _highlight_keywords(text: str, keywords: list[str]) -> str:
    for kw in keywords[:4]:
        text = re.sub(
            f"({re.escape(kw)})", r"**\1**",
            text, flags=re.IGNORECASE, count=2,
        )
    return text


def _render_result_card(r: dict, keywords: list[str]) -> None:
    score      = r.get("score", 0)
    doc_id     = r.get("doc_id", "")
    date_str   = r.get("report_date", "")
    report_no  = r.get("report_no", "")
    section    = r.get("section_title", "")
    page       = r.get("page_start")
    is_table   = r.get("is_table", False)
    chunk_text = r.get("chunk_text", "")
    snippet    = r.get("snippet", chunk_text[:300])

    if score >= _SCORE_HIGH:
        badge = "🟢 Strong match"
    elif score >= _SCORE_MED:
        badge = "🟡 Moderate match"
    else:
        badge = "🔴 Weak match"

    ddr_label     = f"DDR {report_no}" if report_no else doc_id[-20:]
    page_label    = f" · page {page}"  if page else ""
    section_label = f" · {section}"    if section and section.lower() != "unknown" else ""
    table_label   = " · 📊 table"      if is_table else ""

    st.markdown(
        f"**{date_str}** — {ddr_label}{page_label}{section_label}{table_label}"
        f"  &nbsp; `{badge}` &nbsp; score: {score:.3f}",
        unsafe_allow_html=True,
    )
    st.markdown(f"> {_highlight_keywords(snippet, keywords)}")
    with st.expander("Full chunk text"):
        st.code(chunk_text, language=None)
    st.divider()


def _render_results(results: list[dict], question: str) -> None:
    keywords = [w for w in question.lower().split() if len(w) > 3]

    st.markdown(f"**{len(results)} results** — ranked by relevance")
    st.divider()

    for r in results:
        _render_result_card(r, keywords)

    unique_docs = list({r["doc_id"] for r in results})
    date_spread = sorted({r["report_date"] for r in results if r.get("report_date")})
    if date_spread:
        st.caption(
            f"Results span {len(unique_docs)} document(s) from "
            f"{date_spread[0]} to {date_spread[-1]}."
        )


def page_corpus_search() -> None:
    st.header("Corpus Search")
    n_ddrs = len(load_all_headers())
    st.caption(
        f"Search across all {n_ddrs} daily drilling reports simultaneously. "
        "Returns the most relevant operational passages ranked by semantic similarity."
    )

    if not load_global_search():
        st.error("Global index not found. Run: `python scripts/build_global_index.py`")
        return

    question = st.text_input(
        "Ask a question about any operation on this well",
        placeholder="e.g. What caused the NPT on the intermediate casing section?",
        key="global_search_q",
    )

    st.sidebar.subheader("Search filters")
    k           = st.sidebar.slider("Results to return", 5, 30, 10, 5, key="gs_k")
    show_tables = st.sidebar.checkbox(
        "Include table chunks", value=False,
        help="Table chunks contain structured data but less narrative text.",
    )

    if question and _looks_like_aggregate(question):
        st.info(
            "**Tip:** This looks like a statistics question (maximum, total, average). "
            "The Operations Log and NPT Intelligence pages compute these directly from "
            "structured data and will give a more accurate answer than text search."
        )

    if not question:
        st.markdown(
            "**Example questions:**\n"
            "- What caused the NPT on the intermediate casing section?\n"
            "- Which operations required reduced tripping speed?\n"
            "- What happened on days with overpull above 20 klbs?\n"
            "- What issues occurred while running production casing?\n"
            "- When was the surface casing cemented and how long did it take?"
        )
        return

    with st.spinner("Searching… (first search loads the embedding model, ~3 s)"):
        results, search_err = _run_search(question, k=k * 2)

    if search_err:
        st.error(f"Search failed: {search_err}")
        return

    if not show_tables:
        results = [r for r in results if not r.get("is_table", False)]
    results = results[:k]

    if not results:
        st.warning("No results found. Try a different question.")
        return

    _render_results(results, question)
