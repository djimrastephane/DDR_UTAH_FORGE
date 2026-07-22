from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from .constants import PHASE_ORDER
    from .loaders import load_weather, _parse_report_dates
    from .utils import _apply_chart_theme, _sea_state
except ImportError:
    from constants import PHASE_ORDER                   # type: ignore[no-redef]
    from loaders import load_weather, _parse_report_dates  # type: ignore[no-redef]
    from utils import _apply_chart_theme, _sea_state   # type: ignore[no-redef]

_root = Path(__file__).resolve().parents[3]
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from ddr_rag.vocab import label_phase, label_op_code
from ddr_rag.npt_classifier import apply_corpus_npt_rules, classify_ops_df, CATEGORY_LABELS


def _prepare_ops(ops: pd.DataFrame) -> pd.DataFrame:
    ops = apply_corpus_npt_rules(ops)
    ops["report_date_parsed"] = _parse_report_dates(ops["report_date"])
    ops["npt_category"]  = classify_ops_df(ops)
    ops["phase_label"]   = ops["phase"].map(label_phase)
    ops["op_code_label"] = ops["op_code"].map(label_op_code)
    ops["npt_cat_label"] = ops["npt_category"].map(
        lambda x: CATEGORY_LABELS.get(x, "") if pd.notna(x) else ""
    )
    ops.loc[~ops["is_npt"], "npt_cat_label"] = ""
    return ops


def _merge_weather(ops: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    if weather.empty:
        return ops
    ops = ops.merge(
        weather[["report_date", "wind_speed_kn", "wave_height_ft",
                 "swell_height_ft", "wind_direction_deg"]],
        left_on="report_date_parsed",
        right_on="report_date",
        how="left",
        suffixes=("", "_wx"),
    )
    ops["sea_state"] = ops["wave_height_ft"].apply(_sea_state)
    return ops


def _sidebar_filters(ops: pd.DataFrame, has_weather: bool) -> dict:
    st.sidebar.subheader("Filters")

    phases   = ["All"] + sorted(
        ops["phase"].dropna().unique(),
        key=lambda x: PHASE_ORDER.index(x) if x in PHASE_ORDER else 99,
    )
    sel_phase = st.sidebar.selectbox(
        "Phase", phases,
        format_func=lambda x: "All" if x == "All" else label_phase(x),
    )

    sel_class = st.sidebar.selectbox(
        "Classification",
        ["All", "Flagged NPT", "Normal operation"],
    )

    op_codes = ["All"] + sorted(ops["op_code"].dropna().unique())
    sel_op = st.sidebar.selectbox(
        "Op code", op_codes,
        format_func=lambda x: x if x == "All" else f"{x} — {label_op_code(x)}",
    )

    npt_cats = ["All"] + sorted(ops["npt_category"].dropna().unique())
    sel_cat = st.sidebar.selectbox(
        "NPT type", npt_cats,
        format_func=lambda x: x if x == "All" else CATEGORY_LABELS.get(x, x),
    )

    sel_sea = "All"
    if has_weather:
        sel_sea = st.sidebar.selectbox(
            "Sea state",
            ["All", "Calm", "Slight", "Moderate", "Rough", "Very Rough"],
            help="Filter by wave height category (available Apr–Jun 2024 only)",
        )

    search = st.sidebar.text_input("Search operation text", "")

    min_d = ops["report_date_parsed"].dropna().min()
    max_d = ops["report_date_parsed"].dropna().max()
    dr = None
    if pd.notna(min_d) and pd.notna(max_d):
        dr = st.sidebar.date_input(
            "Date range", (min_d.date(), max_d.date()),
            min_value=min_d.date(), max_value=max_d.date(),
        )

    return dict(
        phase=sel_phase, classification=sel_class, op=sel_op,
        cat=sel_cat, sea=sel_sea, search=search, date_range=dr,
    )


def _apply_filters(ops: pd.DataFrame, f: dict) -> pd.DataFrame:
    out = ops
    if f["phase"] != "All":
        out = out[out["phase"] == f["phase"]]
    if f["classification"] == "Flagged NPT":
        out = out[out["is_npt"]]
    elif f["classification"] == "Normal operation":
        out = out[~out["is_npt"]]
    if f["op"] != "All":
        out = out[out["op_code"] == f["op"]]
    if f["cat"] != "All":
        out = out[out["npt_category"] == f["cat"]]
    if f["sea"] != "All" and "sea_state" in out.columns:
        out = out[out["sea_state"] == f["sea"]]
    if f["search"]:
        out = out[out["operation_text"].str.contains(f["search"], case=False, na=False)]
    dr = f["date_range"]
    if dr and len(dr) == 2:
        out = out[
            (out["report_date_parsed"] >= pd.Timestamp(dr[0])) &
            (out["report_date_parsed"] <= pd.Timestamp(dr[1]))
        ]
    return out


def _render_kpis(filtered: pd.DataFrame) -> None:
    t_h  = filtered["duration_hr"].sum()
    nt_h = filtered.loc[filtered["is_npt"], "duration_hr"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows shown",  f"{len(filtered):,}")
    c2.metric("Total hours", f"{t_h:.1f} h")
    c3.metric("NPT hours",
              f"{nt_h:.1f} h ({100*nt_h/t_h:.0f}%)" if t_h else "—")


def _render_log_table(filtered: pd.DataFrame, has_weather: bool) -> None:
    cols = {
        "report_date":    "Date",
        "start_time":     "Start",
        "end_time":       "End",
        "duration_hr":    "Dur (h)",
        "phase_label":    "Phase",
        "op_code_label":  "Op type",
        "is_npt":         "Classification",
        "npt_cat_label":  "NPT type",
        "operation_text": "Operation",
    }
    if has_weather and "wind_speed_kn" in filtered.columns:
        if filtered["wind_speed_kn"].notna().any():
            cols["wind_speed_kn"]  = "Wind (kn)"
            cols["wave_height_ft"] = "Wave (ft)"

    avail = {k: v for k, v in cols.items() if k in filtered.columns}
    disp  = filtered[list(avail)].rename(columns=avail).copy()
    # column_config's NumberColumn(format=...) is silently ignored once the
    # dataframe is wrapped in a pandas Styler (needed below for NPT row
    # colouring), so format numeric columns to strings up front instead of
    # relying on column_config for them.
    disp["Dur (h)"] = disp["Dur (h)"].map(lambda v: f"{v:.2f}")
    if "Classification" in disp.columns:
        disp["Classification"] = disp["Classification"].map(
            {True: "Flagged NPT", False: "Normal operation"}
        )

    if "NPT type" in disp.columns:
        disp["NPT type"] = disp["NPT type"].apply(
            lambda x: (str(x)[:28] + "…") if isinstance(x, str) and len(x) > 28 else x
        )

    def _row_colour(row: pd.Series) -> list[str]:
        if row.get("Classification") == "Flagged NPT":
            return ["background-color:#FFCDD2; color:#212121"] * len(row)
        return [""] * len(row)

    col_cfg: dict = {
        "Date":      st.column_config.TextColumn(width="small"),
        "Start":     st.column_config.TextColumn(width="small"),
        "End":       st.column_config.TextColumn(width="small"),
        "Dur (h)":   st.column_config.TextColumn(width="small"),
        "Phase":     st.column_config.TextColumn(width="small"),
        "Op type":   st.column_config.TextColumn(width="medium"),
        "Classification": st.column_config.TextColumn(width="medium"),
        "NPT type":  st.column_config.TextColumn(width="medium"),
        "Operation": st.column_config.TextColumn(width="large"),
    }
    if "Wind (kn)" in disp.columns:
        disp["Wind (kn)"] = disp["Wind (kn)"].map(lambda v: f"{v:.0f}" if pd.notna(v) else "")
        disp["Wave (ft)"] = disp["Wave (ft)"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "")
        col_cfg["Wind (kn)"] = st.column_config.TextColumn(width="small")
        col_cfg["Wave (ft)"] = st.column_config.TextColumn(width="small")

    st.dataframe(
        disp.style.apply(_row_colour, axis=1),
        hide_index=True,
        use_container_width=True,
        height=520,
        column_config=col_cfg,
    )


def _render_export(filtered: pd.DataFrame, sel_phase: str) -> None:
    disp_cols = ["report_date", "start_time", "end_time", "duration_hr",
                 "phase_label", "op_code_label", "is_npt", "npt_cat_label", "operation_text"]
    export = filtered[[c for c in disp_cols if c in filtered.columns]].copy()
    if "is_npt" in export.columns:
        export["is_npt"] = export["is_npt"].map({True: "Flagged NPT", False: "Normal operation"})
    export.columns = [c.replace("_label", "").replace("_", " ").title()
                      for c in export.columns]
    csv = export.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        csv,
        file_name=f"ddr_ops_{sel_phase}.csv",
        mime="text/csv",
    )


def page_operations_log(ops: pd.DataFrame) -> None:
    st.header("Operations Log")

    ops     = _prepare_ops(ops)
    weather = load_weather()
    ops     = _merge_weather(ops, weather)

    has_weather = "wind_speed_kn" in ops.columns and ops["wind_speed_kn"].notna().any()

    filters  = _sidebar_filters(ops, has_weather)
    filtered = _apply_filters(ops, filters)

    _render_kpis(filtered)
    _render_log_table(filtered, has_weather)
    _render_export(filtered, filters["phase"])
