from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

RAW_DIR = repo_root / "data" / "raw"
OUT_DIR = repo_root / "data" / "processed" / "qc"

DDR_DATE_RE = re.compile(r"(\d{2})\.(\d{2})\.(\d{4})\.?pdf$", re.IGNORECASE)


def _report_date(pdf_path: Path) -> str | None:
    m = DDR_DATE_RE.search(pdf_path.name)
    if not m:
        return None
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"


def _parse_float(s: str) -> float | None:
    try:
        return float(str(s).replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def extract_planned_time(pdf_path: Path) -> dict | None:
    import pdfplumber

    record: dict[str, float | None] = {
        "p_pct": None, "p_hrs": None,
        "t_pct": None, "t_hrs": None,
        "x_pct": None, "x_hrs": None,
        "cumulative_hrs": None, "cumulative_npt_pct": None,
    }

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    flat = " ".join(str(c or "") for row in table for c in row)
                    if "PLANNED TIME SUMMARY" not in flat:
                        continue

                    found_header = False
                    for row in table:
                        cells = [str(c or "").strip() for c in row if c is not None]
                        joined = " ".join(cells)

                        if "Time P-T-X" in joined:
                            found_header = True
                            continue

                        if not found_header:
                            continue

                        # Data row: code_cell | pct_cell | hrs_cell
                        if len(cells) < 2:
                            continue
                        # Skip if first meaningful cell looks like a section header
                        if any(k in cells[0].upper() for k in
                               ["MATERIAL", "SUPPORT", "FORMATION", "SURVEY", "PLANNED"]):
                            break

                        # Extract multi-line values
                        code_lines = cells[0].replace("\n", " ").split()
                        pct_lines  = [_parse_float(v) for v in cells[1].split("\n") if v.strip()]
                        hrs_lines  = [_parse_float(v) for v in cells[2].split("\n") if v.strip()] \
                                     if len(cells) > 2 else []

                        if not pct_lines:
                            continue

                        # Map codes to values
                        code_map: dict[str, tuple[float | None, float | None]] = {}
                        for i, code in enumerate(code_lines):
                            pct = pct_lines[i] if i < len(pct_lines) else None
                            hrs = hrs_lines[i] if i < len(hrs_lines) else None
                            code_map[code.upper()] = (pct, hrs)

                        record["p_pct"] = code_map.get("P", (None, None))[0]
                        record["p_hrs"] = code_map.get("P", (None, None))[1]
                        record["t_pct"] = code_map.get("T", (None, None))[0]
                        record["t_hrs"] = code_map.get("T", (None, None))[1]
                        record["x_pct"] = code_map.get("X", (None, None))[0]
                        record["x_hrs"] = code_map.get("X", (None, None))[1]

                        # Cumulative totals
                        all_hrs = [v for v in [record["p_hrs"], record["t_hrs"],
                                                record["x_hrs"]] if v is not None]
                        if all_hrs:
                            record["cumulative_hrs"] = round(sum(all_hrs), 2)
                            if record["t_hrs"] is not None:
                                record["cumulative_npt_pct"] = round(
                                    100 * record["t_hrs"] / sum(all_hrs), 2
                                )
                        return record
    except Exception:
        pass
    return record if any(v is not None for v in record.values()) else None


def extract_support_vessels(pdf_path: Path) -> list[dict]:
    import pdfplumber

    records: list[dict] = []

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    flat = " ".join(str(c or "") for row in table for c in row)
                    if "Vessel Type" not in flat and "Support Vessel" not in flat.replace("s",""):
                        continue

                    found_header = False
                    for row in table:
                        cells = [str(c or "").strip() for c in row if c is not None]
                        joined = " ".join(cells)

                        if "Vessel Type" in joined and "Vessel Name" in joined:
                            found_header = True
                            continue

                        if not found_header:
                            continue
                        if not cells or not cells[0]:
                            break
                        # Stop at next section
                        if any(k in cells[0].upper() for k in
                               ["FORMATION", "SURVEY", "CASING", "EQUIPMENT", "GENERAL"]):
                            break

                        # Multi-line cells: split on \n
                        types    = [t.strip() for t in cells[0].split("\n") if t.strip()]
                        names    = [n.strip() for n in cells[1].split("\n") if n.strip()] \
                                   if len(cells) > 1 else []
                        arrivals = [a.strip() for a in cells[2].split("\n") if a.strip()] \
                                   if len(cells) > 2 else []
                        departs  = [d.strip() for d in cells[3].split("\n") if d.strip()] \
                                   if len(cells) > 3 else []

                        for i, vtype in enumerate(types):
                            records.append({
                                "vessel_type":      vtype,
                                "vessel_name":      names[i] if i < len(names) else "",
                                "arrival_time":     arrivals[i] if i < len(arrivals) else "",
                                "departure_time":   departs[i] if i < len(departs) else "",
                            })

                        if records:
                            return records
    except Exception:
        pass
    return records


def _wind_cardinal(deg: float | None) -> str:
    if deg is None or (isinstance(deg, float) and deg != deg):
        return ""
    directions = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
                  "S","SSW","SW","WSW","W","WNW","NW","NNW"]
    idx = round(float(deg) / 22.5) % 16
    return directions[idx]


def _beaufort(kn: float | None) -> tuple[int, str]:
    if kn is None or (isinstance(kn, float) and kn != kn):
        return (0, "Calm")
    kn = float(kn)
    scale = [
        (1,  "Calm"),
        (3,  "Light air"),
        (6,  "Light breeze"),
        (10, "Gentle breeze"),
        (16, "Moderate breeze"),
        (21, "Fresh breeze"),
        (27, "Strong breeze"),
        (33, "Near gale"),
        (40, "Gale"),
        (47, "Strong gale"),
        (55, "Storm"),
        (63, "Violent storm"),
        (999,"Hurricane"),
    ]
    for i, (limit, label) in enumerate(scale):
        if kn <= limit:
            return (i, label)
    return (12, "Hurricane")


def _head_to_wind(vessel_hdg: float | None, wind_dir: float | None) -> float | None:
    if vessel_hdg is None or wind_dir is None:
        return None
    if isinstance(vessel_hdg, float) and vessel_hdg != vessel_hdg:
        return None
    if isinstance(wind_dir, float) and wind_dir != wind_dir:
        return None
    angle = (float(wind_dir) - float(vessel_hdg)) % 360
    if angle > 180:
        angle -= 360
    return round(angle, 1)


def _wave_steepness(wave_ft: float | None, period_s: float | None) -> float | None:
    # H / L where L = g*T²/(2π)
    import math
    if wave_ft is None or period_s is None or period_s <= 0:
        return None
    if isinstance(wave_ft, float) and wave_ft != wave_ft:
        return None
    wave_m = float(wave_ft) * 0.3048
    wavelength = 9.81 * float(period_s) ** 2 / (2 * math.pi)
    return round(wave_m / wavelength, 4)


def add_derived_indicators(weather_df: pd.DataFrame) -> pd.DataFrame:
    df = weather_df.copy()

    df["wind_cardinal"] = df["wind_direction_deg"].apply(_wind_cardinal)
    df[["beaufort_num", "beaufort_label"]] = df["wind_speed_kn"].apply(
        lambda v: pd.Series(_beaufort(v))
    )
    df["head_to_wind_angle"] = df.apply(
        lambda r: _head_to_wind(r.get("vessel_heading_deg"), r.get("wind_direction_deg")),
        axis=1,
    )
    df["wave_steepness"] = df.apply(
        lambda r: _wave_steepness(r.get("wave_height_ft"), r.get("wave_period_s")),
        axis=1,
    )
    df["swell_dominance"] = (
        df["swell_height_ft"] / df["wave_height_ft"].replace(0, float("nan"))
    ).round(3)

    # Day-on-day deltas
    df = df.sort_values("report_date").reset_index(drop=True)
    df["wind_delta_kn"]   = df["wind_speed_kn"].diff().round(1)
    df["wave_delta_ft"]   = df["wave_height_ft"].diff().round(2)

    return df


def main() -> None:
    import pymupdf

    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    planned_rows: list[dict] = []
    vessel_rows:  list[dict] = []

    print(f"Processing {len(pdfs)} PDFs...")

    for i, pdf_path in enumerate(pdfs, 1):
        report_date = _report_date(pdf_path)
        if not report_date:
            continue

        meta = {"source_filename": pdf_path.name, "report_date": report_date}

        pt = extract_planned_time(pdf_path)
        if pt:
            planned_rows.append({**meta, **pt})

        vessels = extract_support_vessels(pdf_path)
        for v in vessels:
            vessel_rows.append({**meta, **v})

        if i % 20 == 0:
            print(f"  {i}/{len(pdfs)}  planned={len(planned_rows)}  vessels={len(vessel_rows)}")

    pt_df = pd.DataFrame(planned_rows)
    pt_df["report_date"] = pd.to_datetime(pt_df["report_date"])
    pt_df = pt_df.sort_values("report_date").reset_index(drop=True)
    pt_path = OUT_DIR / "ddr_planned_time.parquet"
    pt_df.to_parquet(pt_path, index=False)
    print(f"\nPlanned Time: {len(pt_df)} rows → {pt_path}")
    print(f"  Coverage: {pt_df['t_hrs'].notna().sum()}/{len(pt_df)} DDRs with T hours")
    if len(pt_df) > 0:
        final = pt_df.iloc[-1]
        print(f"  Final entry ({final.report_date.date()}): "
              f"P={final.p_hrs:.0f}h ({final.p_pct:.1f}%), "
              f"T={final.t_hrs:.0f}h ({final.t_pct:.1f}%)")

    v_df = pd.DataFrame(vessel_rows)
    v_df["report_date"] = pd.to_datetime(v_df["report_date"])
    v_df = v_df.sort_values("report_date").reset_index(drop=True)
    v_path = OUT_DIR / "ddr_vessels.parquet"
    v_df.to_parquet(v_path, index=False)
    print(f"\nSupport Vessels: {len(v_df)} vessel-day records → {v_path}")
    print(f"  Unique vessel names: {sorted(v_df['vessel_name'].dropna().unique())[:10]}")
    print(f"  Vessel type counts: {v_df['vessel_type'].value_counts().to_dict()}")

    wx_path = OUT_DIR / "ddr_weather.parquet"
    if wx_path.exists():
        wx_df = pd.read_parquet(wx_path)
        wx_df["report_date"] = pd.to_datetime(wx_df["report_date"])
        wx_df = add_derived_indicators(wx_df)
        wx_df.to_parquet(wx_path, index=False)
        print(f"\nWeather enriched → {wx_path}")
        print(f"  New columns: wind_cardinal, beaufort_num/label, head_to_wind_angle,")
        print(f"               wave_steepness, swell_dominance, wind_delta_kn, wave_delta_ft")
        # Sample
        sample = wx_df[["report_date","wind_speed_kn","wind_cardinal","beaufort_label",
                        "head_to_wind_angle","wave_steepness","swell_dominance"]].head(5)
        print(f"\n  Sample:\n{sample.to_string(index=False)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
