from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

RAW_DIR = repo_root / "data" / "raw"
OUT_DIR = repo_root / "data" / "processed" / "qc"

DDR_DATE_RE = re.compile(r"(\d{2})\.(\d{2})\.(\d{4})\.+pdf$", re.IGNORECASE)


def _to_float(s: str) -> float | None:
    if s is None:
        return None
    clean = str(s).strip().rstrip(".")
    # Remove units suffix (ppg, psi, ft, etc.)
    clean = re.sub(r"\s*(ppg|psi|ft|m|kips|klbs|lb/ft|in|bbl)\b.*", "", clean, flags=re.I)
    clean = clean.strip()
    if not clean:
        return None
    # Remove comma thousands separators
    clean = clean.replace(",", "")
    # Period-as-thousands: "14.950" with 3 digits after = 14950
    if re.match(r"^\d+\.\d{3}$", clean):
        parts = clean.split(".")
        clean = parts[0] + parts[1]
    try:
        return float(clean)
    except ValueError:
        return None


def _cell_label_value(cell: str) -> tuple[str, str]:
    parts = str(cell or "").split("\n")
    label = parts[0].strip().lower()
    value = " ".join(p.strip() for p in parts[1:] if p.strip())
    return label, value


def _keyword_match(label: str, keywords: list[str]) -> bool:
    # Keywords prefixed with ^ and suffixed with $ require exact match; others are substring.
    label_l = label.lower().strip()
    for kw in keywords:
        kw_l = kw.lower().strip()
        if kw_l.startswith("^") and kw_l.endswith("$"):
            if label_l == kw_l[1:-1]:
                return True
        elif kw_l in label_l:
            return True
    return False


def _find_section_start(table: list[list], section_keywords: list[str],
                        strict: bool = False) -> int | None:
    # strict=True: keyword must be dominant text in the cell, not a substring of a longer
    # description — prevents false matches from Op Summary text containing keyword phrases
    for i, row in enumerate(table):
        non_empty = [str(c or "").strip() for c in row if str(c or "").strip()]
        if not non_empty:
            continue
        row_text = " ".join(non_empty).upper()

        for kw in section_keywords:
            kw_up = kw.upper()
            if kw_up not in row_text:
                continue
            if strict:
                # In strict mode: the keyword must fill most of the first non-empty cell
                first = non_empty[0].upper()
                if kw_up == first or first.startswith(kw_up) or len(first) < len(kw_up) + 20:
                    return i
            else:
                return i
    return None


def _next_section_end(table: list[list], start: int,
                      all_section_keywords: list[str]) -> int:
    # Uses strict first-cell matching to avoid false termination on data rows
    # whose cell text happens to contain a section-header substring
    for i in range(start + 1, len(table)):
        non_empty = [str(c or "").strip() for c in table[i] if str(c or "").strip()]
        if not non_empty:
            continue
        first = non_empty[0].upper()
        if any(first == kw.upper() or first.startswith(kw.upper())
               for kw in all_section_keywords):
            return i
    return len(table)


def _detect_header_row(rows: list[list], col_keywords: dict[str, list[str]],
                        min_matches: int = 2) -> tuple[int, dict[str, int]] | tuple[None, None]:
    # Try to find a dedicated header row (has multiple keyword matches, mostly no values)
    for i, row in enumerate(rows):
        col_map: dict[str, int] = {}
        for j, cell in enumerate(row):
            if cell is None:
                continue
            cell_str = str(cell).strip()
            # In Label\nValue cells, check the label part (before \n)
            label = cell_str.split("\n")[0].lower()
            for col_name, keywords in col_keywords.items():
                if _keyword_match(label, keywords) and col_name not in col_map:
                    col_map[col_name] = j
                    break
        if len(col_map) >= min_matches:
            return i, col_map
    return None, None


def _get_cell(row: list, col_map: dict[str, int], col_name: str) -> str:
    idx = col_map.get(col_name)
    if idx is None or idx >= len(row):
        return ""
    cell = str(row[idx] or "").strip()
    if "\n" in cell:
        # Label\nValue format — return the value part
        parts = cell.split("\n")
        return " ".join(p.strip() for p in parts[1:] if p.strip())
    return cell


def _load_profile_keywords() -> dict[str, list[str]]:
    defaults = {
        "casing_summary":    ["CASING SUMMARY", "CASING PROGRAMME", "CASING RECORD",
                               "CASING TALLY", "STRING SUMMARY"],
        "pressure_tests":    ["EQUIPMENT PRESSURE TEST", "PRESSURE TEST", "LEAK OFF TEST",
                               "CASING TEST", "BOP TEST"],
        "mud_data":          ["MUD DATA", "MUD REPORT", "FLUID DATA", "MUD PROPERTIES",
                               "DRILLING FLUID"],
        "personnel_data":    ["PERSONNEL DATA", "PERSONNEL ON BOARD", "POB",
                               "CREW MANIFEST", "PERSONNEL ONBOARD"],
        "all_sections":      [],  # populated below
    }
    try:
        from ddr_rag.ddr_profile import load_profile
        prof = load_profile("operator_alpha")
        gn   = prof.general_notes
        # Override headers if profile specifies them
        for key in ("casing_summary", "pressure_tests", "mud_data", "personnel_data"):
            prof_headers = gn.get(key, {}).get("section_headers")
            if prof_headers:
                defaults[key] = list(prof_headers) + defaults[key]
    except Exception:
        pass

    all_kw = []
    for k, v in defaults.items():
        if k != "all_sections":
            all_kw.extend(v)
    defaults["all_sections"] = list(set(all_kw))
    return defaults


_SECTION_KW = _load_profile_keywords()

# Column keyword maps — order matters (first match wins)
_CASING_COL_KW: dict[str, list[str]] = {
    "description": ["casing description", "string description", "description"],
    "od_in":       ["od (in)", "od(in)", " od ", "outer diameter", "nominal od"],
    "set_depth_ft":["set depth", "shoe depth", "setting depth"],
    "top_depth_ft":["top depth", "top of string", "lap depth"],
    "run_date":    ["run date", "date run", "installed"],
    "min_drift_in":["min drift", "drift", "id"],
    "weight_lb_ft":["weight/length", "weight", "lb/ft"],
}

_PRESSURE_COL_KW: dict[str, list[str]] = {
    "test_type": ["test type", "type", "test"],
    "test_date": ["date"],
    "comment":   ["com", "comment", "description", "result"],
}

_MUD_COL_KW: dict[str, list[str]] = {
    "mud_type":      ["^type$"],           # exact match: avoids "test type", "casing type"
    "density_ppg":   ["density (lb/gal)", "density(lb/gal)", "mud weight (ppg)",
                      "mw (ppg)", "fluid weight"],
    "temp_f":        ["density temperature", "temperature (°f)", "mud temp"],
    "depth_ft":      ["depth (ftoth)", "depth (m)", "sample depth"],
    "pv_cp":         ["pv calc", "plastic viscosity (cp)", "pv (cp)"],
    "yp_lbf":        ["yp calc", "yield point (lbf", "yp (lbf"],
    "gel_10s":       ["gel 10 sec", "gel 10s", "10 sec gel"],
    "gel_10min":     ["gel 10 min", "gel 10m", "10 min gel"],
    "oil_water":     ["oil water ratio", "o/w ratio", "oil/water"],
    "lgs_pct":       ["low gravity solids"],
    "hgs_pct":       ["high gravity solids"],
}

_PERSONNEL_COL_KW: dict[str, list[str]] = {
    "company":  ["company", "contractor", "operator"],
    "function": ["function", "role", "service", "description"],
    "count":    ["count", "number", "pob", "quantity", "no."],
}


# ---------------------------------------------------------------------------
# 1. Casing Programme
# ---------------------------------------------------------------------------

def extract_casing(pdf_path: Path) -> tuple[list[dict], list[dict]]:
    import pdfplumber

    casing_rows: list[dict] = []
    pressure_rows: list[dict] = []

    all_kw    = _SECTION_KW["all_sections"]
    cas_kw    = _SECTION_KW["casing_summary"]
    press_kw  = _SECTION_KW["pressure_tests"]

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                flat = " ".join(str(c or "") for row in table for c in row).upper()

                # ── Casing Summary ──────────────────────────────────────────
                if any(k.upper() in flat for k in cas_kw):
                    s = _find_section_start(table, cas_kw)
                    if s is not None:
                        e = _next_section_end(table, s, all_kw)
                        block = table[s:e]
                        _, col_map = _detect_header_row(block, _CASING_COL_KW,
                                                         min_matches=2)
                        if col_map:
                            for row in block:
                                desc_raw = _get_cell(row, col_map, "description")
                                if not desc_raw or desc_raw.upper() in ("CASING DESCRIPTION",):
                                    continue
                                od_raw   = _get_cell(row, col_map, "od_in")
                                sd_raw   = _get_cell(row, col_map, "set_depth_ft")
                                td_raw   = _get_cell(row, col_map, "top_depth_ft")
                                rd_raw   = _get_cell(row, col_map, "run_date")
                                wt_raw   = _get_cell(row, col_map, "weight_lb_ft")
                                # Skip empty rows or section header rows
                                if not sd_raw and not od_raw:
                                    continue
                                # Infer casing type from description
                                desc_l   = desc_raw.lower()
                                if "conductor" in desc_l:
                                    casing_type = "Conductor"
                                elif "intermediate" in desc_l:
                                    casing_type = "Intermediate"
                                elif "production" in desc_l and "liner" not in desc_l:
                                    casing_type = "Production casing"
                                elif "liner" in desc_l:
                                    casing_type = "Liner"
                                elif "surface" in desc_l:
                                    casing_type = "Surface"
                                else:
                                    casing_type = "Unknown"

                                casing_rows.append({
                                    "casing_description": desc_raw,
                                    "casing_type":        casing_type,
                                    "od_in":              od_raw,
                                    "set_depth_ft":       _to_float(sd_raw),
                                    "top_depth_ft":       _to_float(td_raw),
                                    "run_date":           rd_raw,
                                    "weight_lb_per_ft":   _to_float(wt_raw),
                                })

                # ── Equipment Pressure Tests (only in the same table as Casing Summary) ──
                if any(k.upper() in flat for k in cas_kw) and any(k.upper() in flat for k in press_kw):
                    s = _find_section_start(table, press_kw, strict=True)
                    if s is not None:
                        e = _next_section_end(table, s, all_kw)
                        block = table[s:e]
                        # Skip block[0] (section header row) when detecting column headers
                        _, col_map = _detect_header_row(block[1:], _PRESSURE_COL_KW,
                                                         min_matches=1)
                        if col_map:
                            for row in block[1:]:
                                # Use raw cell access — multi-line cells like "Casing\nBOP\nBOP"
                                # contain multiple test types, NOT Label\nValue pairs
                                t_idx = col_map.get("test_type")
                                d_idx = col_map.get("test_date")
                                c_idx = col_map.get("comment")
                                test_type_raw = str(row[t_idx] or "") if t_idx is not None and t_idx < len(row) else ""
                                test_date_raw = str(row[d_idx] or "") if d_idx is not None and d_idx < len(row) else ""
                                comment_raw   = str(row[c_idx] or "") if c_idx is not None and c_idx < len(row) else ""
                                if not test_type_raw.strip() and not comment_raw.strip():
                                    continue

                                # Split multi-line cells (multiple tests per row)
                                types   = [t.strip() for t in test_type_raw.split("\n") if t.strip()]
                                dates   = [d.strip() for d in test_date_raw.split("\n") if d.strip()]
                                comments_split = comment_raw.split("\n")

                                if not types:
                                    continue

                                for i, ttype in enumerate(types):
                                    tdate   = dates[i] if i < len(dates) else ""
                                    tcomm   = comments_split[i].strip() if i < len(comments_split) else ""
                                    # Extract pressures from comment: "300/7,500psi"
                                    lo_psi = hi_psi = None
                                    p_match = re.search(r"([\d,]+)\s*/\s*([\d,]+)\s*psi", tcomm, re.I)
                                    if p_match:
                                        lo_psi = _to_float(p_match.group(1))
                                        hi_psi = _to_float(p_match.group(2))

                                    # Only keep rows that look like genuine equipment pressure tests
                                    # Must either have a pressure value OR match known test-type keywords
                                    _test_keywords = r"(bop|casing|conductor|liner|annular|ram|blind|shear|" \
                                                     r"pipe|riser|wellhead|christmas|xtree|x-tree|fit|lot|" \
                                                     r"formation integrity|surface line|hydraulic)"
                                    has_pressure = lo_psi is not None or hi_psi is not None
                                    is_test_type = bool(re.search(_test_keywords, ttype.lower()))
                                    if not has_pressure and not is_test_type:
                                        continue
                                    # Skip rows that are clearly Op Summary or narrative bleeds
                                    skip_patterns = r"^(\d{2}:\d{2}|no accidents|vessel status|held jsa|" \
                                                    r"ran |rigged|recover|nipple|pre-tour|weekly|type\s+last)"
                                    if re.match(skip_patterns, ttype.lower().strip()):
                                        continue
                                    if not ttype or len(ttype) < 3:
                                        continue

                                    pressure_rows.append({
                                        "test_type":        ttype,
                                        "test_date":        tdate,
                                        "low_pressure_psi": lo_psi,
                                        "high_pressure_psi":hi_psi,
                                        "comment":          tcomm,
                                    })

    # Deduplicate casing rows (casing summary is cumulative — same rows repeat)
    seen = set()
    unique_casing = []
    for r in casing_rows:
        key = (r.get("casing_description",""), r.get("set_depth_ft"))
        if key not in seen and r.get("set_depth_ft") is not None:
            seen.add(key)
            unique_casing.append(r)

    return unique_casing, pressure_rows


# ---------------------------------------------------------------------------
# 2. Mud Data
# ---------------------------------------------------------------------------

def extract_mud_data(pdf_path: Path) -> dict | None:
    import pdfplumber

    mud_kw  = _SECTION_KW["mud_data"]
    all_kw  = _SECTION_KW["all_sections"]

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                flat = " ".join(str(c or "") for row in table for c in row).upper()
                if not any(k.upper() in flat for k in mud_kw):
                    continue
                s = _find_section_start(table, mud_kw)
                if s is None:
                    continue
                e = _next_section_end(table, s, all_kw)
                block = table[s:e]

                # Build field dict from all Label\nValue cells in the block
                fields: dict[str, str] = {}
                for row in block:
                    for cell in row:
                        if not cell:
                            continue
                        label, value = _cell_label_value(str(cell))
                        if not label or not value:
                            continue
                        for col_name, keywords in _MUD_COL_KW.items():
                            if _keyword_match(label, keywords) and col_name not in fields:
                                fields[col_name] = value
                                break

                if not fields.get("density_ppg") and not fields.get("mud_type"):
                    continue   # no useful data found

                # Parse Oil/Water ratio: "78.9/21.1" or "78.9 / 21.1"
                oil_pct = water_pct = None
                ow_raw = fields.get("oil_water", "")
                ow_m = re.search(r"([\d.]+)\s*/\s*([\d.]+)", ow_raw)
                if ow_m:
                    oil_pct   = _to_float(ow_m.group(1))
                    water_pct = _to_float(ow_m.group(2))

                return {
                    "mud_type":      fields.get("mud_type", ""),
                    "density_ppg":   _to_float(fields.get("density_ppg")),
                    "density_temp_f":_to_float(fields.get("temp_f")),
                    "sample_depth_ft":_to_float(fields.get("depth_ft")),
                    "pv_cp":         _to_float(fields.get("pv_cp")),
                    "yp_lbf_100ft2": _to_float(fields.get("yp_lbf")),
                    "gel_10s":       _to_float(fields.get("gel_10s")),
                    "gel_10min":     _to_float(fields.get("gel_10min")),
                    "oil_pct":       oil_pct,
                    "water_pct":     water_pct,
                    "lgs_pct":       _to_float(fields.get("lgs_pct")),
                    "hgs_pct":       _to_float(fields.get("hgs_pct")),
                }
    return None


# ---------------------------------------------------------------------------
# 3. Personnel Data (POB)
# ---------------------------------------------------------------------------

def extract_personnel(pdf_path: Path) -> list[dict]:
    import pdfplumber

    pers_kw = _SECTION_KW["personnel_data"]
    all_kw  = _SECTION_KW["all_sections"]
    rows_out: list[dict] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                flat = " ".join(str(c or "") for row in table for c in row).upper()
                if not any(k.upper() in flat for k in pers_kw):
                    continue
                s = _find_section_start(table, pers_kw)
                if s is None:
                    continue
                e = _next_section_end(table, s, all_kw)
                block = table[s:e]

                # Find the Company / Function / Count header row
                _, col_map = _detect_header_row(block, _PERSONNEL_COL_KW, min_matches=2)
                if col_map is None:
                    continue

                for row in block:
                    # Access raw cells directly — do NOT use _get_cell here because
                    # the personnel count cell "51\n6\n..." is NOT Label\nValue format;
                    # applying Label\nValue splitting would discard the first count value.
                    c_idx = col_map.get("company")
                    f_idx = col_map.get("function")
                    n_idx = col_map.get("count")
                    comp_raw = str(row[c_idx] or "") if c_idx is not None and c_idx < len(row) else ""
                    func_raw = str(row[f_idx] or "") if f_idx is not None and f_idx < len(row) else ""
                    cnt_raw  = str(row[n_idx] or "") if n_idx is not None and n_idx < len(row) else ""

                    if not comp_raw.strip():
                        continue

                    # Multi-company rows: split on newline and zip
                    companies  = [c.strip() for c in comp_raw.split("\n") if c.strip()]
                    functions  = [f.strip() for f in func_raw.split("\n") if f.strip()]
                    counts_str = [c.strip() for c in cnt_raw.split("\n") if c.strip()]

                    for i, company in enumerate(companies):
                        # Filter header cells, garbled concatenations, and non-company strings
                        if company.upper() in ("COMPANY", "CONTRACTOR", "OPERATOR"):
                            continue
                        if len(company) > 60 and "\n" not in company:
                            continue   # garbled multi-company concat without newlines
                        if re.match(r"^(\d|\(|Depth|OD|ftOTH)", company):
                            continue   # depth/label bleed from adjacent table
                        function = functions[i] if i < len(functions) else ""
                        count_s  = counts_str[i] if i < len(counts_str) else ""
                        count_v  = int(float(count_s)) if re.match(r"^\d+\.?\d*$", count_s) else None

                        rows_out.append({
                            "company":  company,
                            "function": function,
                            "count":    count_v,
                        })

                if rows_out:
                    return rows_out   # stop after first valid table

    return rows_out


def run_batch(raw_dir: Path, sample: int | None = None) -> dict[str, pd.DataFrame]:
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if sample:
        step = max(1, len(pdfs) // sample)
        pdfs = pdfs[::step][:sample]

    print(f"Processing {len(pdfs)} PDFs...")

    casing_records: list[dict] = []
    pressure_records: list[dict] = []
    mud_records: list[dict] = []
    personnel_records: list[dict] = []

    for i, pdf_path in enumerate(pdfs, 1):
        m = DDR_DATE_RE.search(pdf_path.name)
        if not m:
            continue
        report_date = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
        meta = {"source_filename": pdf_path.name, "report_date": report_date}

        try:
            # Casing + pressure tests
            cas_rows, press_rows = extract_casing(pdf_path)
            for r in cas_rows:
                casing_records.append({**meta, **r})
            for r in press_rows:
                pressure_records.append({**meta, **r})

            # Mud data
            mud = extract_mud_data(pdf_path)
            if mud:
                mud_records.append({**meta, **mud})

            # Personnel
            pers = extract_personnel(pdf_path)
            for r in pers:
                personnel_records.append({**meta, **r})

        except Exception as exc:
            print(f"  WARNING {pdf_path.name}: {exc}")

        if i % 20 == 0:
            print(f"  {i}/{len(pdfs)}  casing={len(casing_records)}  "
                  f"mud={len(mud_records)}  pob_rows={len(personnel_records)}")

    def _make_df(rows, date_col="report_date"):
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df[date_col] = pd.to_datetime(df[date_col])
        return df.sort_values([date_col]).reset_index(drop=True)

    return {
        "casing":    _make_df(casing_records),
        "pressure":  _make_df(pressure_records),
        "mud":       _make_df(mud_records),
        "personnel": _make_df(personnel_records),
    }


def print_summary(results: dict[str, pd.DataFrame]) -> None:
    print()
    print("=== Extraction Summary ===")

    cas = results["casing"]
    if not cas.empty:
        print(f"\nCasing programme ({len(cas)} string records):")
        for _, r in cas.drop_duplicates("casing_description").iterrows():
            print(f"  {r.casing_type:<22} {r.casing_description[:40]:<42}"
                  f"  OD={r.od_in:<8} SD={r.set_depth_ft:<10.0f}ft  Run={r.run_date}")

    pres = results["pressure"]
    if not pres.empty:
        print(f"\nPressure tests ({len(pres)} records):")
        for _, r in pres.drop_duplicates(["test_type","test_date"]).head(8).iterrows():
            hi = f"{r.high_pressure_psi:.0f}psi" if pd.notna(r.high_pressure_psi) else ""
            print(f"  {r.test_type:<15} {r.test_date:<12} {hi:<12}  {r.comment[:60]}")

    mud = results["mud"]
    if not mud.empty:
        print(f"\nMud data ({len(mud)} daily records):")
        types = mud.groupby("mud_type")["density_ppg"].agg(["min","max","count"])
        for mud_type, row in types.iterrows():
            print(f"  {mud_type:<20} density={row['min']:.2f}–{row['max']:.2f} ppg  "
                  f"({int(row['count'])} days)")

    pers = results["personnel"]
    if not pers.empty:
        print(f"\nPersonnel ({len(pers)} company-day records):")
        top_companies = (pers.groupby("company")["count"]
                         .sum().sort_values(ascending=False).head(8))
        for company, total in top_companies.items():
            print(f"  {company:<35} cumulative POB = {total:,.0f}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--sample",  type=int, default=0,
                        help="Process only a sample of N DDRs (0 = all)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_batch(raw_dir, sample=args.sample or None)

    output_map = {
        "casing":   "ddr_casing.parquet",
        "pressure": "ddr_pressure_tests.parquet",
        "mud":      "ddr_mud_data.parquet",
        "personnel":"ddr_personnel.parquet",
    }
    for key, filename in output_map.items():
        df = results[key]
        if not df.empty:
            out_path = out_dir / filename
            df.to_parquet(out_path, index=False)
            df.to_csv(out_path.with_suffix(".csv"), index=False)
            print(f"  {filename}: {len(df)} rows")

    print_summary(results)


if __name__ == "__main__":
    main()
