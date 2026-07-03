from __future__ import annotations

import logging
import os
import re
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

from ddr_rag.extractor_registry import ExtractorRegistry, ReportExtractor, import_report_extractor
from ddr_rag.npt_classifier import classify_utah_forge_npt

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    import pymupdf
    _PYMUPDF_OK = True
except ImportError:
    _PYMUPDF_OK = False

try:
    import pdfplumber as _pdfplumber
    _PDFPLUMBER_OK = True
except ImportError:
    _PDFPLUMBER_OK = False

try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:
    _PANDAS_OK = False


_DATE_RE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")
_REPORT_NO_RE = re.compile(r"Report\s+No\.?\s*(\d+)", re.I)
_JOB_START_RE = re.compile(r"Job\s+Start\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})", re.I)
_SPUD_RE = re.compile(r"Spud\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})", re.I)
_COST_RE = re.compile(r"[\d,]+\.\d{2}")
_TIME_RE = re.compile(r"^(\d{1,2}:\d{2})$")
_DUR_RE = re.compile(r"^(\d+(?:\.\d+)?)$")
_SHIFT_HDR_RE = re.compile(r"(\d{2}:\d{2})-(\d{2}:\d{2})")
_UTAH_FORGE_RE = re.compile(r"^Utah_Forge_FORGE_16A_[\[(]78[\)]?-32_", re.I)


def _infer_shift_block(start_time: str) -> str:
    try:
        hh = int(str(start_time).split(":", 1)[0])
    except (ValueError, TypeError):
        return "unknown"
    if hh < 6:
        return "00:00-06:00"
    if hh < 12:
        return "06:00-12:00"
    if hh < 18:
        return "12:00-18:00"
    return "18:00-00:00"
_WELLBORE_RE = re.compile(r"^\d+[-/]\d+[a-z][-/][A-Z][A-Za-z0-9]*$", re.I)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _label_value(block_text: str) -> tuple[str, str]:
    lines = [ln.strip() for ln in block_text.splitlines() if ln.strip()]
    if len(lines) >= 2:
        return lines[0], " ".join(lines[1:])
    if len(lines) == 1:
        return lines[0], ""
    return "", ""


def _norm_cell(cell: Any) -> str:
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", str(cell)).strip()


def _is_utah_forge_pdf(pdf_path: Path, doc_id: str = "") -> bool:
    return pdf_path.name.startswith("Utah_Forge_FORGE_16A_") or str(doc_id).startswith(
        "UtahForge-DDR-FORGE-16A-78-32-"
    )


def _parse_date_iso(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if pd.notna(parsed):
            return parsed.date().isoformat()
    try:
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.notna(parsed):
            return parsed.date().isoformat()
    except Exception:
        pass
    return ""


def _line_value(lines: list[str], label: str) -> str:
    raw_label = label.rstrip(":")
    next_label = r"(?:[A-Z][A-Z0-9 #().-]{1,24}|[0-9]+(?:\s+[A-Z][A-Z0-9 #().-]*){1,4})"
    inline_re = re.compile(
        rf"(?i:{re.escape(raw_label)})" + rf"\.?:\s*(.*?)(?=\s*{next_label}\.?:|$)",
    )
    for line in lines:
        match = inline_re.search(line)
        if match:
            return match.group(1).strip()
    return ""


def _first_number(text: str) -> str:
    match = re.search(r"\d[\d,]*(?:\.\d+)?", str(text or ""))
    return match.group(0).replace(",", "") if match else ""


def _utah_filename_fallbacks(pdf_path: Path) -> tuple[str, str]:
    try:
        from ddr_rag.filename_qc import parse_ddr_filename

        record = parse_ddr_filename(pdf_path)
        report_date = record.report_date.isoformat() if record.report_date else ""
    except Exception:
        report_date = ""

    report_no = ""
    base = re.sub(r"(?: 2)?\.pdf$", "", pdf_path.name, flags=re.I)
    parts = base.split("_")
    if len(parts) >= 2 and parts[-1].lower() in {"reporttmp", "tmp"}:
        if (
            parts[-1].lower() == "reporttmp"
            and len(parts) >= 3
            and parts[-2].isdigit()
            and parts[-3].isdigit()
            and len(parts[-2]) <= 2
        ):
            report_no = parts[-3]
        elif parts[-2].isdigit():
            report_no = parts[-2]
    return report_date, report_no


def _read_pdf_text_lines(pdf_path: Path) -> tuple[str, list[str]]:
    if not _PYMUPDF_OK:
        raise RuntimeError("PyMuPDF (pymupdf) is required for Utah FORGE extraction.")
    with pymupdf.open(str(pdf_path)) as doc:
        text = "\n".join(page.get_text("text") for page in doc)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return text, lines


_HEADER_COLS = [
    "doc_id", "corpus_id", "run_date_utc",
    "report_date", "report_no", "wellbore", "job_start", "spud_date",
    "report_type", "job_number", "api_uwi", "wellbore_api",
    "depth_progress_ft", "end_depth_md_ft", "end_depth_tvd_ft",
    "afe_no", "afe_amt", "daily_cost", "cumulative_cost",
    "rig_name", "field_name", "kb_elevation_ft", "water_depth_ft",
    "avg_rop",
    "last_casing_string",
    "head_count", "cum_personnel_hours",
    "morning_report_ops", "forecast_24hr", "last_24hr_summary",
]


def _extract_utah_forge_header_fields(pdf_path: Path) -> dict[str, str]:
    text, lines = _read_pdf_text_lines(pdf_path)
    if _PDFPLUMBER_OK:
        try:
            with _pdfplumber.open(str(pdf_path)) as pdf:
                plumber_text = "\n".join((page.extract_text() or "") for page in pdf.pages).strip()
            if plumber_text:
                text = plumber_text
                lines = [line.strip() for line in text.splitlines() if line.strip()]
        except Exception:
            pass
    fallback_date, fallback_report_no = _utah_filename_fallbacks(pdf_path)

    fields: dict[str, str] = {col: "" for col in _HEADER_COLS}
    fields["report_date"] = _parse_date_iso(_line_value(lines, "RPT DATE")) or fallback_date
    fields["report_no"] = _line_value(lines, "RPT NUM.") or fallback_report_no
    fields["wellbore"] = "FORGE-16A-78-32"
    fields["wellbore_api"] = "FORGE-16A-78-32"
    fields["api_uwi"] = "FORGE-16A-78-32"
    fields["field_name"] = "UtahForge"
    fields["rig_name"] = _line_value(lines, "RIG") or "Frontier Rig 16"
    fields["report_type"] = "COMPLETION" if "Completion-C" in pdf_path.name else "DRILLING"

    well_name = _line_value(lines, "WELL NAME")
    if well_name:
        fields["last_24hr_summary"] = f"Well name: {well_name}"

    job = _line_value(lines, "JOB")
    if job:
        fields["job_number"] = job

    spud = _parse_date_iso(_line_value(lines, "SPUD DATE"))
    if spud:
        fields["spud_date"] = spud
        fields["job_start"] = spud

    fields["kb_elevation_ft"] = _line_value(lines, "ELEVATION")
    fields["depth_progress_ft"] = _first_number(_line_value(lines, "24 HR FTG"))

    md_value = _line_value(lines, "MD/TVD")
    if "24 HR FTG" in md_value.upper():
        md_value = ""
    md_value = md_value or _line_value(lines, "MD")
    if not md_value:
        td_match = re.search(r"\bTD:\s*([0-9,]+)\s*TMD", text, re.I)
        md_value = td_match.group(1) if td_match else ""
    fields["end_depth_md_ft"] = _first_number(md_value)

    tvd_match = re.search(r"\b(?:MD/TVD|TD):[^\n]*?\b([0-9,]+)\s*TVD", text, re.I)
    if tvd_match:
        fields["end_depth_tvd_ft"] = tvd_match.group(1).replace(",", "")

    dfs = _line_value(lines, "DFS")
    dol = _line_value(lines, "DOL")
    if dfs or dol:
        fields["cum_personnel_hours"] = f"DFS={dfs}; DOL={dol}".strip("; ")

    present_ops = _line_value(lines, "PRESENT OPERATIONS") or _line_value(lines, "Present Ops")
    planned_ops = _line_value(lines, "ACTIVITY PLANNED") or _line_value(lines, "Next 24 Hours")
    fields["morning_report_ops"] = present_ops[:600]
    fields["forecast_24hr"] = planned_ops[:600]
    if present_ops:
        fields["last_24hr_summary"] = present_ops[:600]

    last_bop = re.search(r"LAST BOP TEST:\s*([0-9/]+)", text, re.I)
    if last_bop:
        fields["afe_no"] = f"Last BOP test: {_parse_date_iso(last_bop.group(1))}"

    return fields


def _extract_generic_header_fields(pdf_path: Path) -> dict[str, str]:
    if not _PYMUPDF_OK:
        raise RuntimeError("PyMuPDF (pymupdf) is required for header extraction.")

    fields: dict[str, str] = {col: "" for col in _HEADER_COLS}

    doc = pymupdf.open(str(pdf_path))
    page = doc[0]
    blocks = page.get_text("blocks", sort=True)
    doc.close()

    full_text = " ".join(b[4] for b in blocks)

    # Load coordinate bands from active profile (falls back to hardcoded defaults)
    _top_max_y   = 80.0
    _grid_min_y  = 75.0
    _grid_max_y  = 200.0
    _narr_min_y  = 180.0
    _narr_max_y  = 360.0
    _profile_field_labels: dict[str, list[str]] = {}
    try:
        from ddr_rag.ddr_profile import load_profile as _lp
        _prof = _lp("operator_alpha")
        _top_max_y   = _prof.header_top_max_y
        _grid_min_y, _grid_max_y = _prof.header_grid_band
        _narr_min_y, _narr_max_y = _prof.header_narr_band
        _profile_field_labels    = _prof.header_field_labels
    except Exception:
        pass

    top_blocks = [b for b in blocks if b[1] < _top_max_y]
    top_text = " ".join(b[4] for b in top_blocks)

    m = _DATE_RE.search(top_text)
    if m:
        fields["report_date"] = m.group(1)
    m = _REPORT_NO_RE.search(top_text)
    if m:
        fields["report_no"] = m.group(1)
    m = _JOB_START_RE.search(top_text)
    if m:
        fields["job_start"] = m.group(1)
    m = _SPUD_RE.search(full_text[:600])
    if m:
        fields["spud_date"] = m.group(1)
    for b in top_blocks:
        txt = _clean(b[4])
        if _WELLBORE_RE.match(txt):
            fields["wellbore"] = txt
            break
    if "DRILLING ORIGINAL" in full_text.upper():
        fields["report_type"] = "DRILLING ORIGINAL"
    elif "DRILLING" in full_text.upper():
        fields["report_type"] = "DRILLING"

    for b in [blk for blk in blocks if _grid_min_y <= blk[1] <= _grid_max_y]:
        txt = b[4]
        label, value = _label_value(txt)
        ll = label.lower()

        if "maxwell job" in ll or "job number" in ll:
            fields["job_number"] = value
        elif "api / uwi" in ll and "wellbore" not in ll:
            fields["api_uwi"] = value
        elif "wellbore api" in ll:
            fields["wellbore_api"] = value
        elif "depth progress" in ll:
            fields["depth_progress_ft"] = value
        elif "end depth (ftoth)" in ll and "tvd" not in ll:
            fields["end_depth_md_ft"] = value
        elif "end depth (tvd)" in ll:
            fields["end_depth_tvd_ft"] = value
        elif "afe / rfe" in ll or "afe/rfe" in ll:
            fields["afe_no"] = value
        elif "afe+supp amt" in ll or "afe+supp" in ll:
            fields["afe_amt"] = value
        elif "daily cost total" in ll:
            fields["daily_cost"] = value
        elif "cumulative cost" in ll:
            fields["cumulative_cost"] = value
        elif "rig name" in ll:
            fields["rig_name"] = value
        elif "field name" in ll:
            fields["field_name"] = value
        elif "kb/rt elevation" in ll or "original kb" in ll:
            fields["kb_elevation_ft"] = value
        elif "water depth" in ll:
            fields["water_depth_ft"] = value
        elif "avg rop" in ll:
            nums = _COST_RE.findall(txt)
            if nums:
                fields["avg_rop"] = nums[0]
        elif "last casing string" in ll:
            fields["last_casing_string"] = value
        elif "head count" in ll:
            fields["head_count"] = value
        elif "cum personnel total" in ll:
            fields["cum_personnel_hours"] = value

    for b in [blk for blk in blocks if _narr_min_y <= blk[1] <= _narr_max_y]:
        txt = b[4]
        label, value = _label_value(txt)
        ll = label.lower()
        if ("ops and depth @ morning" in ll or "ops and depth" in ll) and not fields["morning_report_ops"]:
            fields["morning_report_ops"] = value[:600]
        elif ("24hr forecast" in ll or "24 hr forecast" in ll) and not fields["forecast_24hr"]:
            fields["forecast_24hr"] = value[:600]
        elif ("last 24hr summary" in ll or "last 24 hr" in ll) and not fields["last_24hr_summary"]:
            fields["last_24hr_summary"] = value[:600]

    return fields


_OP_COLS = [
    "doc_id", "corpus_id", "run_date_utc",
    "report_date", "wellbore", "rig_name",
    "page", "shift_block",
    "start_time", "end_time", "duration_hr",
    "phase", "op_code", "activity_code", "pt_x",
    "is_npt",
    "operation_text",
    "parse_warning",
]


def _detect_col_map(raw_row: list[Any]) -> dict[str, int] | None:
    cells = [_norm_cell(c) for c in raw_row]
    joined = " ".join(cells).lower()
    if "dur" not in joined or "phase" not in joined:
        return None

    col_map: dict[str, int] = {}
    for i, c in enumerate(cells):
        cl = c.lower()
        if "start" in cl and "time" in cl and "start_time" not in col_map:
            col_map["start_time"] = i
        elif "end" in cl and "time" in cl and "end_time" not in col_map:
            col_map["end_time"] = i
        elif cl in ("dur (hr)", "dur(hr)") or (cl.startswith("dur") and "hr" in cl):
            col_map["dur_hr"] = i
        elif cl == "phase":
            col_map["phase"] = i
        elif "op code" in cl or cl == "op_code":
            col_map["op_code"] = i
        elif "activity" in cl:
            col_map["activity_code"] = i
        elif "p-t-x" in cl or "ptx" in cl:
            col_map["pt_x"] = i
        elif cl == "operation" or cl.startswith("operation"):
            col_map["operation"] = i

    if not {"start_time", "dur_hr", "operation"}.issubset(col_map):
        return None
    return col_map


def _get_cell(row: list[str], col_map: dict[str, int], key: str) -> str:
    idx = col_map.get(key)
    if idx is None or idx >= len(row):
        return ""
    return row[idx]


def _parse_op_table(
    table: list[list[Any]],
    page_num: int,
    current_shift: str | None,
) -> tuple[list[dict], str | None]:
    rows: list[dict] = []
    col_map: dict[str, int] | None = None

    for raw_row in table:
        row = [_norm_cell(c) for c in raw_row]
        first = row[0] if row else ""

        if first.upper().startswith("OPERATION SUMMARY") and sum(1 for c in row if c) <= 2:
            continue

        m_shift = _SHIFT_HDR_RE.match(first)
        if m_shift and sum(1 for c in row if c) <= 2:
            current_shift = first
            col_map = None
            continue

        candidate = _detect_col_map(raw_row)
        if candidate is not None:
            col_map = candidate
            continue

        if col_map is None:
            continue

        non_empty = [c for c in row if c]
        if len(non_empty) < 2:
            continue

        start_time  = _get_cell(row, col_map, "start_time")
        end_time    = _get_cell(row, col_map, "end_time")
        dur_raw     = _get_cell(row, col_map, "dur_hr")
        phase       = _get_cell(row, col_map, "phase")
        op_code     = _get_cell(row, col_map, "op_code")
        act_code    = _get_cell(row, col_map, "activity_code")
        pt_x        = _get_cell(row, col_map, "pt_x")
        op_text     = _get_cell(row, col_map, "operation")

        warn = ""
        dur_hr: float | None = None
        if start_time and not _TIME_RE.match(start_time):
            warn = f"unexpected start_time {start_time!r}"

        m_dur = _DUR_RE.match(dur_raw)
        if m_dur:
            try:
                dur_hr = float(m_dur.group(1))
            except ValueError:
                warn += f" bad duration {dur_raw!r}"
        elif dur_raw:
            warn += f" unparsed duration {dur_raw!r}"

        # Continuation row — no time, has text
        if not start_time and not dur_raw and op_text:
            if rows:
                rows[-1]["operation_text"] = rows[-1]["operation_text"] + " " + op_text
            continue

        if not start_time and not op_text:
            continue

        # Filter: non-time start_time with no valid duration = non-Op-Summary row
        if start_time and not _TIME_RE.match(start_time) and dur_hr is None:
            continue

        shift = current_shift if current_shift is not None else _infer_shift_block(start_time)
        rows.append({
            "page": page_num,
            "shift_block": shift,
            "start_time": start_time,
            "end_time": end_time,
            "duration_hr": dur_hr,
            "phase": phase,
            "op_code": op_code,
            "activity_code": act_code,
            "pt_x": pt_x,
            "is_npt": pt_x.upper() == "T",
            "operation_text": op_text[:800],
            "parse_warning": warn.strip(),
        })

    return rows, current_shift


def _to_float(text: str) -> float | None:
    try:
        return float(str(text).replace(",", "").strip())
    except Exception:
        return None


def _looks_like_utah_npt(text: str, code: str = "", npt_code: str = "") -> bool:
    is_npt, _category = classify_utah_forge_npt(text, code, code, npt_code)
    return is_npt


def _classify_utah_phase_from_operation(operation: str) -> str:
    text = str(operation or "").lower()
    if not text:
        return ""
    if "no activity" in text:
        return "No Activity"
    if any(
        token in text
        for token in (
            "rig up",
            "rigged up",
            "move in",
            "matting",
            "man lift",
            "subbase",
            "mud pit",
            "delivered",
            "frontier rig 16",
            "bope",
            "bop",
        )
    ):
        return "Rig Move In"
    if "drill out" in text or "drillout" in text:
        return "Drillout"
    if "production casing" in text:
        return "Production Casing"
    if "production" in text:
        return "Production Drilling"
    if "intermediate casing" in text:
        return "Intermediate Casing"
    if "intermediate" in text:
        return "Intermediate Drilling"
    if "surface casing" in text:
        return "Surface Casing"
    if "surface" in text:
        return "Surface Drilling"
    return ""


def _extract_utah_operation_payload(
    row: list[str],
) -> tuple[float | None, str, str, str, str]:
    """Return duration, phase, code, npt_code, operation text from a Utah FORGE time row."""
    duration_idx = -1
    duration_hr: float | None = None
    for idx in range(2, min(len(row), 5)):
        value = _to_float(row[idx])
        if value is not None:
            duration_idx = idx
            duration_hr = value
            break

    if duration_idx < 0:
        return None, "", "", "", ""

    # Drilling reports have stable wide-table columns.
    if len(row) > 14 and row[5] and row[8]:
        phase = row[5]
        code = row[8]
        operation = row[14] if len(row) > 14 else ""
    else:
        non_empty_after = [(idx, cell) for idx, cell in enumerate(row[duration_idx + 1 :], start=duration_idx + 1) if cell]
        phase = non_empty_after[0][1] if len(non_empty_after) > 1 else ""
        code = non_empty_after[1][1] if len(non_empty_after) > 2 else ""
        operation = non_empty_after[-1][1] if non_empty_after else ""

    npt_code = ""
    op_lines = [line.strip() for line in str(operation or "").splitlines() if line.strip()]
    if len(op_lines) > 1 and re.fullmatch(r"[A-Z]{2,8}", op_lines[0]):
        # Some pdfplumber rows merge the narrow NPT/code cell into the operations cell.
        npt_code = op_lines[0]
        operation = " ".join(op_lines[1:])
    else:
        operation = " ".join(op_lines) if op_lines else str(operation or "").strip()

    if not phase:
        phase = _classify_utah_phase_from_operation(operation)

    return duration_hr, phase.strip(), code.strip(), npt_code, operation.strip()


def _extract_utah_forge_op_summary(pdf_path: Path) -> list[dict]:
    if not _PDFPLUMBER_OK:
        raise RuntimeError("pdfplumber is required for Utah FORGE operation extraction.")

    rows: list[dict] = []
    with _pdfplumber.open(str(pdf_path)) as pdf:
        for pg_idx, page in enumerate(pdf.pages):
            tables = page.extract_tables() or []
            for table in tables:
                for raw_row in table:
                    row = [_norm_cell(cell) for cell in (raw_row or [])]
                    if len(row) < 2:
                        continue
                    start_time = row[0]
                    end_time = row[1]
                    if not (_TIME_RE.match(start_time) and _TIME_RE.match(end_time)):
                        continue

                    duration_hr, phase, code, npt_code, operation = _extract_utah_operation_payload(row)
                    if duration_hr is None or not phase or not operation:
                        continue

                    is_npt = _looks_like_utah_npt(operation, code, npt_code)
                    rows.append(
                        {
                            "page": pg_idx + 1,
                            "shift_block": _infer_shift_block(start_time),
                            "start_time": start_time,
                            "end_time": end_time,
                            "duration_hr": duration_hr,
                            "phase": phase,
                            "op_code": code,
                            "activity_code": code,
                            "pt_x": npt_code,
                            "is_npt": is_npt,
                            "operation_text": operation[:1200],
                            "parse_warning": "utah_forge_layout",
                        }
                    )

    return rows


def _extract_generic_op_summary(pdf_path: Path) -> list[dict]:
    if not _PDFPLUMBER_OK:
        raise RuntimeError("pdfplumber is required for Op Summary extraction.")

    all_rows: list[dict] = []
    current_shift = None  # set by explicit header; inferred from time until first header seen

    with _pdfplumber.open(str(pdf_path)) as pdf:
        for pg_idx, page in enumerate(pdf.pages):
            pg_num = pg_idx + 1
            page_text = page.extract_text() or ""

            if "OPERATION SUMMARY" not in page_text.upper() and \
               not re.search(r"\d{2}:\d{2}\s+\d{2}:\d{2}\s+\d+\.\d+", page_text):
                continue

            tables = page.extract_tables()
            if not tables:
                continue

            for table in tables:
                table_text = " ".join(_norm_cell(c) for row in table for c in row)
                if "OPERATION SUMMARY" not in table_text.upper() and \
                   not re.search(r"\d{1,2}:\d{2}.*?\d+\.\d+", table_text):
                    continue
                rows, current_shift = _parse_op_table(table, pg_num, current_shift)
                all_rows.extend(rows)

    return all_rows


class UtahForgeReportExtractor:
    name = "utah_forge"
    priority = 100

    def matches(self, pdf_path: Path, doc_id: str = "") -> bool:
        return _is_utah_forge_pdf(pdf_path, doc_id=doc_id)

    def extract_header_fields(self, pdf_path: Path) -> dict[str, str]:
        return _extract_utah_forge_header_fields(pdf_path)

    def extract_op_summary(self, pdf_path: Path) -> list[dict]:
        return _extract_utah_forge_op_summary(pdf_path)


class GenericDDRReportExtractor:
    name = "generic_ddr"
    priority = -100

    def matches(self, pdf_path: Path, doc_id: str = "") -> bool:
        return True

    def extract_header_fields(self, pdf_path: Path) -> dict[str, str]:
        return _extract_generic_header_fields(pdf_path)

    def extract_op_summary(self, pdf_path: Path) -> list[dict]:
        return _extract_generic_op_summary(pdf_path)


def _configured_extra_extractor_paths() -> list[str]:
    paths: list[str] = []
    raw_env = os.getenv("DDR_RAG_EXTRA_EXTRACTORS", "")
    paths.extend(item.strip() for item in raw_env.split(",") if item.strip())

    config_path = _REPO_ROOT / "configs" / "ddr_rag.yaml"
    if not config_path.exists():
        return paths

    try:
        import yaml

        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Could not read extractor config from %s: %s", config_path, exc)
        return paths

    extraction_cfg = data.get("extraction", {}) if isinstance(data, dict) else {}
    configured = extraction_cfg.get("extra_extractor_classes", []) if isinstance(extraction_cfg, dict) else []
    if isinstance(configured, str):
        configured = [configured]
    paths.extend(str(item).strip() for item in configured if str(item).strip())
    return paths


@lru_cache(maxsize=1)
def get_extractor_registry() -> ExtractorRegistry:
    registry = ExtractorRegistry([UtahForgeReportExtractor(), GenericDDRReportExtractor()])
    for import_path in _configured_extra_extractor_paths():
        try:
            registry.register(import_report_extractor(import_path))
        except Exception as exc:
            logger.warning("Could not register DDR extractor %s: %s", import_path, exc)
    return registry


def register_report_extractor(extractor: ReportExtractor) -> None:
    get_extractor_registry().register(extractor)


def reset_extractor_registry_cache() -> None:
    get_extractor_registry.cache_clear()


def select_report_extractor(pdf_path: Path, doc_id: str = "") -> ReportExtractor:
    return get_extractor_registry().select(pdf_path, doc_id=doc_id)


def extract_header_fields(pdf_path: Path, doc_id: str = "") -> dict[str, str]:
    extractor = select_report_extractor(pdf_path, doc_id=doc_id)
    return extractor.extract_header_fields(pdf_path)


def extract_op_summary(pdf_path: Path, doc_id: str = "") -> list[dict]:
    extractor = select_report_extractor(pdf_path, doc_id=doc_id)
    return extractor.extract_op_summary(pdf_path)


def run_ddr_extraction(
    pdf_path: Path,
    doc_id: str,
    corpus_id: str,
    run_date_utc: str,
) -> tuple[Any, Any]:
    if not _PANDAS_OK:
        raise RuntimeError("pandas is required for DDR extraction.")

    header_record: dict[str, Any] = {col: "" for col in _HEADER_COLS}
    header_record.update({"doc_id": doc_id, "corpus_id": corpus_id, "run_date_utc": run_date_utc})

    op_rows: list[dict] = []
    extraction_ok = True

    try:
        fields = extract_header_fields(pdf_path, doc_id=doc_id)
        header_record.update(
            {
                key: value
                for key, value in fields.items()
                if key not in {"doc_id", "corpus_id", "run_date_utc"}
            }
        )
    except Exception as exc:
        logger.warning("DDR header extraction failed for %s: %s", doc_id, exc)
        extraction_ok = False

    try:
        raw_rows = extract_op_summary(pdf_path, doc_id=doc_id)
        report_date = header_record.get("report_date", "")
        wellbore = header_record.get("wellbore", "")
        rig_name = header_record.get("rig_name", "")
        for r in raw_rows:
            r.update({
                "doc_id": doc_id,
                "corpus_id": corpus_id,
                "run_date_utc": run_date_utc,
                "report_date": report_date,
                "wellbore": wellbore,
                "rig_name": rig_name,
            })
        op_rows = raw_rows
    except Exception as exc:
        logger.warning("DDR op summary extraction failed for %s: %s", doc_id, exc)
        extraction_ok = False

    header_df = pd.DataFrame([header_record], columns=_HEADER_COLS)

    if op_rows:
        ops_df = pd.DataFrame(op_rows, columns=_OP_COLS)
        ops_df["duration_hr"] = pd.to_numeric(ops_df["duration_hr"], errors="coerce")
        ops_df["is_npt"] = ops_df["is_npt"].astype(bool)
        ops_df["page"] = pd.to_numeric(ops_df["page"], errors="coerce").fillna(0).astype(int)
    else:
        ops_df = pd.DataFrame(columns=_OP_COLS)

    if not extraction_ok:
        logger.info("DDR extraction completed with warnings for %s", doc_id)

    return header_df, ops_df
