from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

RAW_DIR       = repo_root / "data" / "raw"
PROCESSED_DIR = repo_root / "data" / "processed"
OUT_DIR       = repo_root / "data" / "processed" / "qc"

DDR_DATE_RE = re.compile(
    r"(\d{2})\.(\d{2})\.(\d{4})\.?pdf$", re.IGNORECASE
)

# Fields to extract from the weather table
FIELD_MAP = {
    "Temperature - High":  "temperature_high_f",
    "Wind Speed":          "wind_speed_kn",
    "Wind Direction":      "wind_direction_deg",
    "Vessel Heading":      "vessel_heading_deg",
    "Vessel Offset":       "vessel_offset_ft",
    "Swell Height":        "swell_height_ft",
    "Current Speed":       "current_speed_kn",
    "Current Direction":   "current_direction_deg",
    "Wave Height":         "wave_height_ft",
    "Wave Direction":      "wave_direction_deg",
    "Wave Period":         "wave_period_s",
    "Heave":               "heave_ft",
    "Pitch":               "pitch_deg",
    "Roll":                "roll_deg",
    "Ceiling":             "ceiling_ft",
    "Visibility":          "visibility_miles",
}


def _parse_float(s: str) -> float | None:
    try:
        return float(str(s).replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def extract_weather_from_pdf(pdf_path: Path) -> dict | None:
    import pdfplumber

    record: dict[str, float | None] = {v: None for v in FIELD_MAP.values()}

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if "Wind Speed" not in page_text:
                    continue

                tables = page.extract_tables()
                for table in tables:
                    flat = " ".join(str(c or "") for row in table for c in row)
                    if "Wind Speed" not in flat and "Wave Height" not in flat:
                        continue

                    for row in table:
                        for cell in row:
                            if cell is None:
                                continue
                            cell_str = str(cell).strip()
                            if "\n" not in cell_str:
                                continue
                            lines = cell_str.split("\n")
                            label = lines[0].strip()
                            value_str = " ".join(lines[1:]).strip() if len(lines) > 1 else ""
                            value = _parse_float(value_str)

                            for field_label, field_name in FIELD_MAP.items():
                                if field_label.lower() in label.lower() and value is not None:
                                    record[field_name] = value
                                    break

                    # If we found at least wind speed, we're done
                    if record["wind_speed_kn"] is not None or record["wave_height_ft"] is not None:
                        return record

    except Exception as exc:
        print(f"  WARNING: {pdf_path.name}: {exc}")

    has_any = any(v is not None for v in record.values())
    return record if has_any else None


def build_weather_corpus() -> pd.DataFrame:
    import pymupdf

    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    rows = []

    print(f"Scanning {len(pdfs)} PDFs for weather data...")

    for i, pdf_path in enumerate(pdfs, 1):
        # Quick pre-check with pymupdf (faster than pdfplumber)
        try:
            doc = pymupdf.open(str(pdf_path))
            has_wind = any("Wind Speed" in pg.get_text() for pg in doc.pages())
            doc.close()
        except Exception:
            continue

        if not has_wind:
            continue

        # Parse date from filename
        m = DDR_DATE_RE.search(pdf_path.name)
        if not m:
            continue
        report_date = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"

        weather = extract_weather_from_pdf(pdf_path)
        if weather is None:
            continue

        weather["source_filename"] = pdf_path.name
        weather["report_date"]     = report_date
        rows.append(weather)

        if i % 10 == 0:
            print(f"  {len(rows)} weather records extracted so far...")

    df = pd.DataFrame(rows)
    df["report_date"] = pd.to_datetime(df["report_date"])
    df = df.sort_values("report_date").reset_index(drop=True)
    print(f"\nExtracted weather data from {len(df)} DDRs")
    return df


def correlate_weather_npt(
    weather_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for d in PROCESSED_DIR.iterdir():
        f = d / "ddr_facts.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
    ops = pd.concat(frames, ignore_index=True)
    ops["report_date_dt"] = pd.to_datetime(
        ops["report_date"], dayfirst=True, errors="coerce"
    )

    daily = (
        ops.groupby("report_date_dt")
        .apply(lambda g: pd.Series({
            "total_hrs":    g["duration_hr"].sum(),
            "npt_hrs":      g.loc[g["is_npt"], "duration_hr"].sum(),
            "npt_pct":      100 * g.loc[g["is_npt"], "duration_hr"].sum()
                            / max(g["duration_hr"].sum(), 1),
            "n_ops":        len(g),
            "n_npt_ops":    g["is_npt"].sum(),
            "phase":        g["phase"].mode()[0] if not g["phase"].empty else "",
        }))
        .reset_index()
        .rename(columns={"report_date_dt": "report_date"})
    )

    merged = weather_df.merge(daily, on="report_date", how="inner")

    # Beaufort classification
    def beaufort(spd):
        if spd is None or pd.isna(spd): return "Unknown"
        if spd < 7:   return "Light (0-6kn)"
        if spd < 14:  return "Moderate (7-13kn)"
        if spd < 22:  return "Fresh (14-21kn)"
        if spd < 28:  return "Strong (22-27kn)"
        return "Gale+ (28kn+)"

    if "wind_speed_kn" in merged.columns:
        merged["beaufort_class"] = merged["wind_speed_kn"].apply(beaufort)

    # Wave sea state
    def sea_state(h):
        if h is None or pd.isna(h): return "Unknown"
        if h < 2:  return "Calm (<2ft)"
        if h < 5:  return "Slight (2-4ft)"
        if h < 8:  return "Moderate (5-7ft)"
        if h < 13: return "Rough (8-12ft)"
        return "Very Rough (13ft+)"

    if "wave_height_ft" in merged.columns:
        merged["sea_state"] = merged["wave_height_ft"].apply(sea_state)

    return merged, daily


def print_findings(merged: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("WEATHER–NPT CORRELATION FINDINGS")
    print("=" * 60)

    # Coverage
    print(f"\nWeather data: {len(merged)} days matched with NPT records")
    print(f"Date range:   {merged['report_date'].min().date()} → "
          f"{merged['report_date'].max().date()}")
    print(f"Phases:       {sorted(merged['phase'].unique())}")

    # Wind speed vs NPT
    if "wind_speed_kn" in merged.columns and merged["wind_speed_kn"].notna().sum() > 10:
        print("\n-- Wind Speed vs NPT --")
        by_beaufort = (
            merged.groupby("beaufort_class")
            .agg(n_days=("npt_pct", "count"),
                 avg_npt_pct=("npt_pct", "mean"),
                 avg_npt_hrs=("npt_hrs", "mean"))
            .sort_values("avg_npt_pct", ascending=False)
        )
        print(by_beaufort.to_string())

        corr = merged[["wind_speed_kn", "npt_pct"]].dropna().corr().iloc[0, 1]
        print(f"\n  Pearson correlation (wind speed vs NPT%): {corr:.3f}")

    # Wave height vs NPT
    if "wave_height_ft" in merged.columns and merged["wave_height_ft"].notna().sum() > 10:
        print("\n-- Wave Height vs NPT --")
        by_sea = (
            merged.groupby("sea_state")
            .agg(n_days=("npt_pct", "count"),
                 avg_npt_pct=("npt_pct", "mean"),
                 avg_npt_hrs=("npt_hrs", "mean"))
            .sort_values("avg_npt_pct", ascending=False)
        )
        print(by_sea.to_string())

        corr = merged[["wave_height_ft", "npt_pct"]].dropna().corr().iloc[0, 1]
        print(f"\n  Pearson correlation (wave height vs NPT%): {corr:.3f}")

    # High-weather days vs normal days
    if "wind_speed_kn" in merged.columns:
        p75_wind  = merged["wind_speed_kn"].quantile(0.75)
        p75_wave  = merged["wave_height_ft"].quantile(0.75) if "wave_height_ft" in merged.columns else None
        high_wind = merged[merged["wind_speed_kn"] >= p75_wind]
        low_wind  = merged[merged["wind_speed_kn"] <  p75_wind]

        print(f"\n-- High wind days (≥{p75_wind:.0f}kn, top 25%) vs normal days --")
        print(f"  High wind: avg NPT = {high_wind['npt_pct'].mean():.1f}%  "
              f"(n={len(high_wind)} days, avg wind={high_wind['wind_speed_kn'].mean():.1f}kn)")
        print(f"  Normal:    avg NPT = {low_wind['npt_pct'].mean():.1f}%  "
              f"(n={len(low_wind)} days, avg wind={low_wind['wind_speed_kn'].mean():.1f}kn)")

    # Phase-level summary
    print("\n-- NPT by phase (weather period only) --")
    phase_summary = (
        merged.groupby("phase")
        .agg(n_days=("npt_pct", "count"),
             avg_npt_pct=("npt_pct", "mean"),
             avg_wind_kn=("wind_speed_kn", "mean"),
             avg_wave_ft=("wave_height_ft", "mean"))
        .sort_values("avg_npt_pct", ascending=False)
    )
    print(phase_summary.to_string())

    # Weather-driven NPT ops from ddr_facts
    frames = []
    for d in PROCESSED_DIR.iterdir():
        f = d / "ddr_facts.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
    ops = pd.concat(frames, ignore_index=True)
    ops["report_date_dt"] = pd.to_datetime(ops["report_date"], dayfirst=True, errors="coerce")

    weather_npt_ops = ops[
        ops["operation_text"].str.contains(
            r"sea state|weather|wind|wave|swell|crane.*rest|unable.*vessel|vessel.*unable",
            case=False, na=False
        ) & ops["is_npt"]
    ]
    print(f"\n-- Weather-related NPT operations (text mentions) --")
    print(f"  Count: {len(weather_npt_ops)} ops  |  "
          f"Hours: {weather_npt_ops['duration_hr'].sum():.1f}h")
    print(f"  Phases: {weather_npt_ops['phase'].value_counts().to_dict()}")

    # Top weather-affected operational days
    if len(merged) > 0 and "wind_speed_kn" in merged.columns:
        worst_days = merged.nlargest(8, "wind_speed_kn")[
            ["report_date", "phase", "wind_speed_kn", "wave_height_ft",
             "swell_height_ft", "npt_pct", "npt_hrs"]
        ]
        print("\n-- Top 8 windiest days --")
        print(worst_days.to_string(index=False))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    weather_df = build_weather_corpus()

    out_weather = OUT_DIR / "ddr_weather.parquet"
    weather_df.to_parquet(out_weather, index=False)
    print(f"Weather data written to: {out_weather}")
    print(f"Columns: {list(weather_df.columns)}")
    print(f"\nWeather field coverage (non-null):")
    for col in FIELD_MAP.values():
        if col in weather_df.columns:
            n = weather_df[col].notna().sum()
            print(f"  {col:<28} {n:4d} / {len(weather_df)}")

    merged, _ = correlate_weather_npt(weather_df)

    out_corr = OUT_DIR / "weather_npt_corr.csv"
    merged.to_csv(out_corr, index=False)
    print(f"\nCorrelation table written to: {out_corr}")

    print_findings(merged)


if __name__ == "__main__":
    main()