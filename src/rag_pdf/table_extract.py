from __future__ import annotations

from typing import Optional

from pathlib import Path
import re
import pandas as pd

from rag_pdf.chunking import count_tokens
from rag_pdf.ocr_table_fallback import accept_and_classify_ocr_table
from rag_pdf.schemas import build_page_list_struct, make_chunk_id_global
from rag_pdf.table_pdfplumber_region import (
    TableResult,
    TABLE_EXTRACT_CFG,
    extract_tables_for_page_region,
)
from rag_pdf.table_canonicalize import extract_table_facts_from_dataframe
from rag_pdf.table_chunking import build_table_chunk_payloads
from rag_pdf.table_markdown import (
    build_header_injected_facts,
    enrich_table_markdown,
    table_to_markdown,
)
from rag_pdf.table_ocr_handoff import (
    build_ocr_table_chunk_record,
    build_rejected_ocr_table_page,
)


def _normalize_cell(v: object) -> str:
    return str(v or "").strip()


def _known_staff_group_label(label: str) -> bool:
    return str(label or "").strip().lower() in {"clinicians", "other"}


def _split_table_candidate(table_type: Optional[str], page_text: str) -> bool:
    ttype = str(table_type or "").strip().lower()
    text = str(page_text or "").lower()
    if ttype not in {"staff_costs", "remuneration", "pay_bands"}:
        return False
    return ("clinicians" in text) and ("other" in text)


def _infer_staff_group_label(df: pd.DataFrame, default_label: str = "") -> str:
    if df is None or len(df) == 0:
        return default_label
    probe_limit = min(3, len(df))
    for ridx in range(probe_limit):
        row_vals = [_normalize_cell(x) for x in df.iloc[ridx].tolist()]
        non_empty = [x for x in row_vals if x]
        if len(non_empty) == 1 and _known_staff_group_label(non_empty[0]):
            return non_empty[0].title()
    if default_label:
        return default_label
    first_cell = _normalize_cell(df.iloc[0, 0]) if len(df.columns) > 0 else ""
    if first_cell.startswith("£"):
        return "Clinicians"
    return default_label


def _drop_group_label_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    keep_rows: list[list[str]] = []
    cols = [str(c) for c in df.columns]
    for ridx in range(len(df)):
        row_vals = [_normalize_cell(x) for x in df.iloc[ridx].tolist()]
        non_empty = [x for x in row_vals if x]
        if len(non_empty) == 1 and _known_staff_group_label(non_empty[0]):
            continue
        if any(row_vals):
            keep_rows.append(row_vals)
    return pd.DataFrame(keep_rows, columns=cols) if keep_rows else pd.DataFrame(columns=cols)


def _looks_like_staff_band_label(label: object) -> bool:
    text = _normalize_cell(label).replace(" ", "").lower()
    if not text:
        return False
    if text.startswith("£") and "to£" in text:
        return True
    if text.startswith("£") and "andabove" in text:
        return True
    return False


def _filter_staff_band_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    cols = [str(c) for c in df.columns]
    keep_rows: list[list[str]] = []
    for ridx in range(len(df)):
        row_vals = [_normalize_cell(x) for x in df.iloc[ridx].tolist()]
        if row_vals and _looks_like_staff_band_label(row_vals[0]):
            keep_rows.append(row_vals)
    return pd.DataFrame(keep_rows, columns=cols) if keep_rows else pd.DataFrame(columns=cols)


def _merge_staff_cost_table_group(frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged_frames: list[pd.DataFrame] = []
    current_label = ""
    seen_band_labels: set[str] = set()
    for idx, raw_df in enumerate(frames):
        df = clean_table_dataframe(raw_df.copy(), "staff_costs")
        if df is None or len(df) == 0:
            continue
        inferred_label = _infer_staff_group_label(df, default_label=current_label)
        current_label = inferred_label or current_label
        cleaned_df = _drop_group_label_rows(df)
        cleaned_df = _filter_staff_band_rows(cleaned_df)
        if cleaned_df is None or len(cleaned_df) == 0:
            continue
        first_band = _normalize_cell(cleaned_df.iloc[0, 0]) if len(cleaned_df) > 0 else ""
        if first_band and first_band in seen_band_labels and current_label == "Clinicians":
            current_label = "Other"
        seen_band_labels.update(_normalize_cell(v) for v in cleaned_df.iloc[:, 0].tolist() if _looks_like_staff_band_label(v))
        cleaned_df = cleaned_df.copy()
        cleaned_df.insert(len(cleaned_df.columns), "group_label", current_label or f"group_{idx+1}")
        merged_frames.append(cleaned_df)
    if not merged_frames:
        return pd.DataFrame()
    return pd.concat(merged_frames, axis=0, ignore_index=True)


def merge_split_table_results(
    table_results: list[TableResult],
    *,
    table_type: Optional[str],
    page_no: int,
    page_text: str = "",
) -> list[TableResult]:
    # Intentionally conservative: only activates for remuneration-style staff costs tables
    # whose source page clearly contains both "Clinicians" and "Other" sections.
    if len(table_results) <= 1:
        return table_results
    if not _split_table_candidate(table_type, page_text):
        return table_results

    ordered = sorted(
        table_results,
        key=lambda tr: int(float((tr.parsing_report or {}).get("order", 0.0) or 0.0)),
    )
    valid_frames = [tr.dataframe for tr in ordered if tr.dataframe is not None and len(tr.dataframe) > 0]
    if len(valid_frames) <= 1:
        return table_results

    merged_df = _merge_staff_cost_table_group(valid_frames)
    if merged_df is None or len(merged_df) == 0:
        return table_results

    base = ordered[0]
    merged_logs = list(base.logs) + [f"page {page_no}: merged {len(valid_frames)} split table blocks"]
    merged_result = TableResult(
        page_no=base.page_no,
        flavor=f"{base.flavor}_merged",
        dataframe=merged_df,
        parsing_report=base.parsing_report,
        logs=merged_logs,
    )
    return [merged_result]


def clean_table_dataframe(df: pd.DataFrame, table_type: Optional[str]) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df

    df = df.map(lambda x: str(x).strip() if pd.notna(x) else "")
    df = df.loc[~(df == "").all(axis=1)]

    if len(df) > 1:
        first_row = df.iloc[0]
        has_text = sum(
            1 for v in first_row
            if str(v).strip() and not str(v).replace(",", "").replace(".", "").replace("-", "").isdigit()
        )

        if has_text / len(first_row) > 0.5:
            df.columns = [str(v) for v in first_row]
            df = df.iloc[1:].reset_index(drop=True)

    return df


def extract_table_pdfplumber(pdf_plumber, page_no: int) -> Optional[pd.DataFrame]:
    try:
        page_idx = page_no - 1
        page = pdf_plumber.pages[page_idx]
        tables = page.extract_tables()
        if not tables:
            return None
        candidates = []
        for table in tables:
            if table and len(table) > 1:
                df = pd.DataFrame(table[1:], columns=table[0])
                rows, cols = df.shape
                if rows == 0 or cols == 0:
                    continue
                candidates.append((rows * cols, cols, df))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][2]
    except Exception:
        pass

    return None


def generate_table_summary(
    df: pd.DataFrame,
    table_type: Optional[str],
    page_no: int,
) -> str:
    rows, cols = df.shape

    summary_parts = [
        f"Financial table on page {page_no}",
    ]

    if table_type:
        type_label = table_type.replace("_", " ").title()
        summary_parts.append(f"({type_label})")

    summary_parts.append(f"Contains {rows} line items across {cols} columns.")

    header_vals = [str(c).strip() for c in df.columns]
    if header_vals and any(h and not h.isdigit() for h in header_vals):
        summary_parts.append("Column headers: " + " | ".join(header_vals[:6]) + ".")

    first_col_labels = []
    if cols > 0:
        for i in range(min(TABLE_EXTRACT_CFG.TABLE_SUMMARY_MAX_ROWS, len(df))):
            v = str(df.iloc[i, 0]).strip()
            if v:
                first_col_labels.append(v)
    if first_col_labels:
        summary_parts.append("Row labels: " + "; ".join(first_col_labels[:8]) + ".")

    key_items = []
    for i in range(min(TABLE_EXTRACT_CFG.TABLE_SUMMARY_MAX_ROWS, len(df))):
        row_values = [str(v).strip() for v in df.iloc[i] if str(v).strip()]
        if len(row_values) >= 2:
            row_text = " | ".join(row_values[:4])
            key_items.append(row_text)

    if key_items:
        summary_parts.append("Key items: " + "; ".join(key_items) + ".")

    return " ".join(summary_parts)


def _materialize_table_result(
    *,
    text_chunks: list[dict],
    structured_tables: list[dict],
    table_facts: list[dict],
    tresult: TableResult,
    raw_table: pd.DataFrame,
    table_summary: str,
    table_markdown: str,
    header_injected_facts: str,
    doc_id: str,
    corpus_id: str,
    report_year,
    report_year_source: Optional[str],
    period_end_date,
    run_date_utc: str,
    page_no: int,
    table_type: Optional[str],
    idx: int,
    table_results_count: int,
    table_chunking_strategy: str,
    chunk_size_tokens: int,
    enc,
) -> None:
    table_id = f"table_p{page_no:04d}" if table_results_count == 1 else f"table_p{page_no:04d}_{idx:02d}"
    parsing_report = tresult.parsing_report or {}
    parsing_order = int(float(parsing_report.get("order", 0.0) or 0.0))

    structured_tables.append({
        "doc_id": doc_id,
        "corpus_id": corpus_id,
        "report_year": report_year,
        "period_end_date": period_end_date,
        "run_date_utc": run_date_utc,
        "page": page_no,
        "page_no": page_no,
        "table_id": table_id,
        "table_type": table_type or "unknown",
        "rows": len(raw_table),
        "cols": len(raw_table.columns),
        "extraction_method": "pdfplumber",
        "flavor": tresult.flavor,
        "parsing_report_accuracy": float(parsing_report.get("accuracy", 0.0)),
        "parsing_report_whitespace": float(parsing_report.get("whitespace", 0.0)),
        "parsing_report_order": parsing_order,
        "parsing_report_page": str(parsing_report.get("page", "")),
        "table_summary": table_summary,
        "table_markdown": table_markdown,
        "table_header_injection": header_injected_facts,
    })
    table_facts.extend(
        extract_table_facts_from_dataframe(
            raw_table,
            doc_id=doc_id,
            corpus_id=corpus_id,
            report_year=report_year,
            period_end_date=period_end_date,
            run_date_utc=run_date_utc,
            page=page_no,
            table_id=table_id,
            table_type=table_type,
        )
    )

    chunk_payloads = build_table_chunk_payloads(
        strategy=str(table_chunking_strategy or "baseline"),
        page_no=page_no,
        table_type=table_type,
        table_summary=table_summary,
        raw_table=raw_table,
        header_injected_facts=header_injected_facts,
        table_markdown=table_markdown,
        chunk_size_tokens=int(chunk_size_tokens),
        enc=enc,
    )
    for cidx, payload in enumerate(chunk_payloads):
        summary = str(payload.get("chunk_text") or "").strip()
        if not summary:
            continue
        base_id = f"table_p{page_no:04d}" if table_results_count == 1 else f"table_p{page_no:04d}_{idx:02d}"
        chunk_id_local = base_id if len(chunk_payloads) == 1 else f"{base_id}_s{cidx:02d}"
        pages = [page_no]
        text_chunks.append({
            "doc_id": doc_id,
            "corpus_id": corpus_id,
            "report_year": report_year,
            "report_year_source": report_year_source,
            "period_end_date": period_end_date,
            "run_date_utc": run_date_utc,
            "chunk_id": chunk_id_local,
            "chunk_id_global": make_chunk_id_global(doc_id, chunk_id_local),
            "part": "Unknown",
            "section_title": "Financial Tables",
            "subsection_title": None,
            "page_start": page_no,
            "page_end": page_no,
            "pages": pages,
            "page_list": build_page_list_struct(pages),
            "chunk_text": summary,
            "chunk_tokens": count_tokens(summary, enc),
            "word_count": len(summary.split()),
            "is_table_like": True,
            "many_numbers": True,
            "is_table": True,
            "table_type": table_type,
            "table_ref": table_id,
            "table_chunk_kind": payload.get("table_chunk_kind"),
            "row_start_idx": payload.get("row_start_idx"),
            "row_end_idx": payload.get("row_end_idx"),
            "table_word_budget_target": payload.get("table_word_budget_target"),
        })


def _self_test_staff_costs() -> None:
    sample_path = (
        Path("data/processed")
        / "sample-ddr"
        / "tables_markdown"
        / "table_p0001.md"
    )
    if not sample_path.exists():
        print(f"Missing sample: {sample_path}")
        return
    raw = sample_path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    table_lines = [ln for ln in lines if ln.strip().startswith("|")]
    table_md = "\n".join(table_lines)
    enriched = enrich_table_markdown(table_md)
    out_lines = enriched.splitlines()
    header_line = next((ln for ln in out_lines if ln.startswith("Column headers: ")), "")
    row_map_lines = [ln for ln in out_lines if ln.startswith("- ")]
    print(header_line)
    for ln in row_map_lines[:3]:
        print(ln)


def process_table_pages(
    table_pages: list[dict],
    pdf_plumber,
    doc_id: str,
    corpus_id: str,
    report_year: Optional[str],
    period_end_date: Optional[str],
    report_year_source: Optional[str],
    run_date_utc: str,
    enc,
    chunk_size_tokens: int = 224,
    table_chunking_strategy: str = "baseline",
    return_all_tables: bool = False,
    enable_secondary_bottom_pass: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    text_chunks = []
    structured_tables = []
    table_facts = []
    rejected_ocr_table_pages: list[dict] = []

    for tpage in table_pages:
        page_no = tpage["page"]
        table_type = tpage["table_type"]
        page_text = str(tpage.get("text", "") or "")
        page_raw_text = str(tpage.get("raw_text", "") or "")

        # return_all_tables / enable_secondary_bottom_pass supported the old
        # camelot multi-table splitting path (only ever exercised by
        # reference-pipeline "staff_costs"-style tables, never Utah FORGE
        # DDR data); the pdfplumber-region path below returns a single best
        # table per page and does not consume them.
        table_results = extract_tables_for_page_region(
            pdf_plumber=pdf_plumber,
            page_no=page_no,
            cleaner=lambda df: clean_table_dataframe(df, table_type=None),
        )

        primary_tables_found = len(table_results)
        page_extractor = str(tpage.get("extractor", "") or "").lower()
        page_rotation = int(tpage.get("rotation", 0) or 0)

        # OCR-table fallback: scanned/ocr pages where the primary pass found no tables.
        if (
            primary_tables_found == 0
            and bool(tpage.get("is_table", False))
            and page_extractor == "ocr"
        ):
            accept, ocr_table_type, ocr_chunk_text, debug = accept_and_classify_ocr_table(
                page_raw_text or page_text,
                page_no=page_no,
                rotation_deg=page_rotation,
            )
            print(
                f"[OCR_TABLE_FALLBACK] page={page_no} rotation={page_rotation} "
                f"accept={accept} table_type={ocr_table_type} "
                f"digit_ratio={debug.get('digit_ratio', 0.0):.3f} "
                f"currency_hits={debug.get('currency_hits', 0)} "
                f"num_lines_with_2plus_nums={debug.get('num_lines_with_2plus_nums', 0)} "
                f"corrupted_flag={debug.get('corrupted_flag', False)} "
                f"matched_triggers={debug.get('matched_triggers', [])}"
            )

            if accept:
                text_chunks.append(
                    build_ocr_table_chunk_record(
                        doc_id=doc_id,
                        corpus_id=corpus_id,
                        report_year=report_year,
                        report_year_source=report_year_source,
                        period_end_date=period_end_date,
                        run_date_utc=run_date_utc,
                        page_no=page_no,
                        ocr_chunk_text=ocr_chunk_text,
                        ocr_table_type=ocr_table_type,
                        debug=debug,
                        enc=enc,
                    )
                )
                continue

            # Rejected OCR-table: send back to normal text chunking path.
            rejected_ocr_table_pages.append(
                build_rejected_ocr_table_page(page_no=page_no, page_text=page_text)
            )
            continue

        if not table_results:
            # Last-resort whole-page fallback for pages that don't match any
            # known anchor (e.g. an unrecognized report template). Naive and
            # noisier than the anchor-scoped path above, but keeps the
            # pipeline resilient rather than dropping the page entirely.
            df_fallback = extract_table_pdfplumber(pdf_plumber, page_no)
            if df_fallback is not None and len(df_fallback) > 0:
                table_results = [
                    TableResult(
                        page_no=page_no,
                        flavor="pdfplumber_whole_page",
                        dataframe=clean_table_dataframe(df_fallback, table_type),
                        parsing_report={"accuracy": 0.0, "whitespace": 0.0, "order": 0, "page": str(page_no)},
                        logs=[f"page {page_no}: whole-page pdfplumber fallback succeeded"],
                    )
                ]
                print(f"page {page_no}: anchor-scoped extraction failed; whole-page pdfplumber fallback used")

        table_results = merge_split_table_results(
            table_results,
            table_type=table_type,
            page_no=page_no,
            page_text=page_text or page_raw_text,
        )

        for idx, tresult in enumerate(table_results):
            raw_table = tresult.dataframe
            if raw_table is None or len(raw_table) == 0:
                continue

            table_summary = generate_table_summary(raw_table, table_type, page_no)
            table_markdown = enrich_table_markdown(table_to_markdown(raw_table))
            header_injected_facts = build_header_injected_facts(table_markdown)

            _materialize_table_result(
                text_chunks=text_chunks,
                structured_tables=structured_tables,
                table_facts=table_facts,
                tresult=tresult,
                raw_table=raw_table,
                table_summary=table_summary,
                table_markdown=table_markdown,
                header_injected_facts=header_injected_facts,
                doc_id=doc_id,
                corpus_id=corpus_id,
                report_year=report_year,
                report_year_source=report_year_source,
                period_end_date=period_end_date,
                run_date_utc=run_date_utc,
                page_no=page_no,
                table_type=table_type,
                idx=idx,
                table_results_count=len(table_results),
                table_chunking_strategy=str(table_chunking_strategy or "baseline"),
                chunk_size_tokens=int(chunk_size_tokens),
                enc=enc,
            )

    return (
        pd.DataFrame(text_chunks),
        pd.DataFrame(structured_tables),
        pd.DataFrame(table_facts),
        rejected_ocr_table_pages,
    )
