from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from .constants import repo_root, PROCESSED_DIR, GRAPHS_DIR, FIELD_DIR
except ImportError:
    # Fallback for environments where app/ui is inserted directly into sys.path
    # (e.g. the verification command: python3 -c "import sys; sys.path.insert(0,'app/ui'); ...")
    from constants import repo_root, PROCESSED_DIR, GRAPHS_DIR, FIELD_DIR  # type: ignore[no-redef]

# Ensure src/ is on the path (constants already inserts it, but this module
# may be imported independently in test environments).
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from ddr_rag.vocab import label_phase, label_op_code
from ddr_rag.npt_classifier import (
    apply_corpus_npt_rules,
    classify_ops_df,
    CATEGORY_LABELS,
    CATEGORY_COLOURS,
)


def _parse_num(s: object) -> float | None:
    try:
        return float(re.sub(r"[^\d.]", "", str(s).split()[0]))
    except Exception:
        return None


def _parse_report_dates(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    iso_mask = text.str.match(r"^\d{4}-\d{1,2}-\d{1,2}$", na=False)
    if iso_mask.any():
        parsed.loc[iso_mask] = pd.to_datetime(
            text.loc[iso_mask],
            format="%Y-%m-%d",
            errors="coerce",
        )

    legacy_mask = ~iso_mask
    if legacy_mask.any():
        parsed.loc[legacy_mask] = pd.to_datetime(
            text.loc[legacy_mask],
            dayfirst=True,
            errors="coerce",
        )

    missing = parsed.isna() & text.notna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text.loc[missing], errors="coerce")

    return parsed


@st.cache_data(show_spinner=False)
def load_all_ops() -> pd.DataFrame:
    frames = []
    for doc_dir in sorted(PROCESSED_DIR.iterdir()):
        f = doc_dir / "ddr_facts.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = apply_corpus_npt_rules(df)
    df["report_date_parsed"] = _parse_report_dates(df["report_date"])
    df["phase_label"]        = df["phase"].map(label_phase).fillna(df["phase"])
    df["op_code_label"]      = df["op_code"].apply(label_op_code)
    df["npt_category"]       = classify_ops_df(df)
    df["npt_cat_label"]      = df["npt_category"].map(CATEGORY_LABELS).fillna("")
    df.loc[~df["is_npt"], "npt_cat_label"] = ""
    return df.sort_values(["report_date_parsed", "start_time"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_all_headers() -> pd.DataFrame:
    frames = []
    for doc_dir in sorted(PROCESSED_DIR.iterdir()):
        f = doc_dir / "ddr_header.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
    if not frames:
        return pd.DataFrame()
    hdr = pd.concat(frames, ignore_index=True)
    hdr["report_date_parsed"] = _parse_report_dates(hdr["report_date"])
    hdr["daily_cost_num"] = hdr["daily_cost"].apply(_parse_num)
    hdr["cum_cost_num"]   = hdr["cumulative_cost"].apply(_parse_num)
    hdr["end_depth_num"]  = hdr["end_depth_md_ft"].apply(_parse_num)
    return hdr.sort_values("report_date_parsed").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def generate_well_narrative(
    ops: pd.DataFrame,
    hdr: pd.DataFrame,
    planned_time: pd.DataFrame | None,
) -> str:
    def _num(s: object) -> float:
        try:
            return float(re.sub(r"[^\d.]", "", str(s).split()[0]))
        except Exception:
            return 0.0

    _rig  = hdr["rig_name"].dropna().mode()
    _well = hdr["wellbore"].dropna().mode()
    rig_name  = _rig.iloc[0].title() if not _rig.empty else "Rig"
    well_name = _well.iloc[0] if not _well.empty else "Well"

    total_h    = float(ops["duration_hr"].sum())
    npt_h      = float(ops.loc[ops["is_npt"], "duration_hr"].sum())
    npt_pct    = 100 * npt_h / total_h if total_h else 0
    cost_values = pd.to_numeric(hdr.get("cum_cost_num", pd.Series(dtype=float)), errors="coerce").dropna()
    total_cost = float(cost_values.max()) if not cost_values.empty else None
    n_days     = int(ops["report_date_parsed"].dropna().nunique())
    max_depth  = float(hdr["end_depth_num"].dropna().max() or 0)

    phase_stats = ops.groupby("phase").apply(
        lambda g: pd.Series({
            "npt_h":  g.loc[g["is_npt"], "duration_hr"].sum(),
            "tot_h":  g["duration_hr"].sum(),
        }), include_groups=False,
    )
    phase_stats["npt_pct"] = 100 * phase_stats["npt_h"] / phase_stats["tot_h"].replace(0, 1)
    worst_phase    = phase_stats["npt_h"].idxmax()
    worst_pct      = phase_stats.loc[worst_phase, "npt_pct"]
    worst_h        = phase_stats.loc[worst_phase, "npt_h"]
    eff_phases     = phase_stats[phase_stats["tot_h"] > 100]
    best_eff_phase = eff_phases["npt_pct"].idxmin()
    best_eff_pct   = eff_phases.loc[best_eff_phase, "npt_pct"]

    _programme = {"mpd_csg_programme", "stimulation_programme", "formation_testing",
                  "completion_monitoring", "other_npt"}
    worst_unplanned = (
        ops[(ops["phase"] == worst_phase) & ops["is_npt"] &
            ~ops["npt_category"].isin(_programme)]
        .groupby("npt_category")["duration_hr"].sum()
        .sort_values(ascending=False)
    )
    primary_cause_label = (
        CATEGORY_LABELS.get(worst_unplanned.index[0], worst_unplanned.index[0])
        if not worst_unplanned.empty else "non-productive operations"
    )
    primary_cause_h = float(worst_unplanned.iloc[0]) if not worst_unplanned.empty else 0

    _daily_npt = (
        ops.groupby("report_date_parsed")["is_npt"]
        .mean()
        .reset_index(name="npt_rate")
    )
    _rolling = _daily_npt["npt_rate"].rolling(7, min_periods=3).mean()
    _best_idx = _rolling.idxmin() if not _rolling.empty else 0
    _best_dt_fallback = _daily_npt.loc[_best_idx, "report_date_parsed"] if _best_idx else None
    best_date_str = _best_dt_fallback.strftime("%-d %b %Y") if _best_dt_fallback is not None else "unknown"
    best_cum_pct  = round(float(_rolling.min()) * 100, 1) if not _rolling.empty else npt_pct

    if planned_time is not None and not planned_time.empty:
        pt_clean = planned_time[
            planned_time["cumulative_npt_pct"].notna() &
            ~((planned_time["cumulative_npt_pct"] > 95) &
              (planned_time["cumulative_hrs"].fillna(0) > 500))
        ]
        if not pt_clean.empty:
            br = pt_clean.loc[pt_clean["cumulative_npt_pct"].idxmin()]
            best_date_str = br["report_date"].strftime("%-d %b %Y")
            best_cum_pct  = float(br["cumulative_npt_pct"])

    if planned_time is not None and not planned_time.empty:
        pt_clean2 = planned_time[
            planned_time["cumulative_npt_pct"].notna() &
            ~((planned_time["cumulative_npt_pct"] > 95) &
              (planned_time["cumulative_hrs"].fillna(0) > 500))
        ]
        if not pt_clean2.empty:
            best_date_dt = pt_clean2.loc[pt_clean2["cumulative_npt_pct"].idxmin(), "report_date"]
            post_best_ops = ops[ops["report_date_parsed"] > best_date_dt]
            post_unplanned = (
                post_best_ops[post_best_ops["is_npt"] &
                              ~post_best_ops["npt_category"].isin(_programme)]
                .groupby("npt_category")["duration_hr"].sum()
                .sort_values(ascending=False)
            )
            deterioration_cause = (
                CATEGORY_LABELS.get(post_unplanned.index[0], post_unplanned.index[0])
                if not post_unplanned.empty else "operational complexity"
            )
        else:
            deterioration_cause = "operational complexity"
    else:
        deterioration_cause = "operational complexity"

    start_date = ops["report_date_parsed"].dropna().min().strftime("%-d %b %Y")
    end_date   = ops["report_date_parsed"].dropna().max().strftime("%-d %b %Y")

    cost_clause = (
        f" at a total cost of **£{total_cost/1e6:.1f}M**"
        if total_cost is not None and total_cost > 0
        else ""
    )
    dominant_phase = phase_stats["tot_h"].idxmax()
    dominant_h = float(phase_stats.loc[dominant_phase, "tot_h"])
    npt_clause = (
        f"The extracted operation rows flag **{npt_h:.0f}h of NPT** "
        f"({npt_pct:.1f}% of reported time); the largest contribution is in "
        f"**{label_phase(worst_phase)}** ({worst_h:.0f}h), led by "
        f"**{primary_cause_label}** ({primary_cause_h:.0f}h)."
        if npt_h > 0
        else "The extracted operation rows do not flag material NPT."
    )

    narrative = (
        f"The **{rig_name}** reported **{well_name}** across **{n_days} DDRs** "
        f"({start_date} – {end_date}), with **{total_h:.0f} operation hours** captured. "
        f"The maximum extracted measured depth is **{max_depth:,.0f} ft MD**{cost_clause}. "
        f"Most reported time is in **{label_phase(dominant_phase)}** "
        f"({dominant_h:.0f}h). {npt_clause}"
    )
    return narrative


@st.cache_data(show_spinner=False)
def load_graph(phase: str, window: str) -> dict | None:
    fname = "graph.json" if window == "Same row" else "w2_graph.json"
    path  = GRAPHS_DIR / phase / fname
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_weather() -> pd.DataFrame:
    path = repo_root / "data" / "processed" / "qc" / "ddr_weather.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df.sort_values("report_date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_wellbore_events() -> pd.DataFrame:
    path = repo_root / "data" / "processed" / "qc" / "ddr_wellbore_events.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["report_date_dt"]    = pd.to_datetime(df["report_date_dt"], errors="coerce")
    df["force_klbs"]        = pd.to_numeric(df["force_klbs"], errors="coerce")
    df["event_depth_ft_md"] = pd.to_numeric(df["event_depth_ft_md"], errors="coerce")
    df["ecd_ppge"]          = pd.to_numeric(
        df.get("ecd_ppge", pd.Series(dtype=float)), errors="coerce"
    )
    df["loss_rate_bbl_hr"]  = pd.to_numeric(
        df.get("loss_rate_bbl_hr", pd.Series(dtype=float)), errors="coerce"
    )
    return df.sort_values("report_date_dt").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_fit_lot() -> pd.DataFrame:
    path = repo_root / "data" / "processed" / "qc" / "ddr_fit_lot_results.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["report_date_dt"] = pd.to_datetime(df["report_date_dt"], errors="coerce")
    df["limit_ppge"]     = pd.to_numeric(df["limit_ppge"], errors="coerce")
    return df.sort_values("report_date_dt").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_casing() -> pd.DataFrame:
    path = repo_root / "data" / "processed" / "qc" / "ddr_casing.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_completion_string() -> pd.DataFrame:
    path = repo_root / "data" / "processed" / "qc" / "ddr_completion_string.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_frac_sleeve_status() -> pd.DataFrame:
    path = repo_root / "data" / "processed" / "qc" / "ddr_frac_sleeve_status.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_personnel() -> pd.DataFrame:
    path = repo_root / "data" / "processed" / "qc" / "ddr_personnel.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["report_date_dt"] = pd.to_datetime(df["report_date"], errors="coerce")
    df["count"]          = pd.to_numeric(df["count"], errors="coerce")
    df = df[df["count"].notna() & (df["count"] > 0)].copy()
    return df.sort_values("report_date_dt").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_field_ops() -> pd.DataFrame:
    path = FIELD_DIR / "analysis" / "combined_facts.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df = apply_corpus_npt_rules(df)
    df["report_date_parsed"] = _parse_report_dates(df["report_date"])
    df["phase_label"]        = df["phase"].map(label_phase).fillna(df["phase"])
    df["op_code_label"]      = df["op_code"].apply(label_op_code)
    df["npt_category"]       = classify_ops_df(df)
    df["npt_cat_label"]      = df["npt_category"].map(CATEGORY_LABELS).fillna("")
    df.loc[~df["is_npt"], "npt_cat_label"] = ""
    return df.sort_values(["well_id", "report_date_parsed", "start_time"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_field_headers() -> pd.DataFrame:
    path = FIELD_DIR / "analysis" / "combined_headers.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["report_date_parsed"] = _parse_report_dates(df["report_date"])
    df["daily_cost_num"]     = df["daily_cost"].apply(_parse_num)
    df["cum_cost_num"]       = df["cumulative_cost"].apply(_parse_num)
    df["end_depth_num"]      = df["end_depth_md_ft"].apply(_parse_num)
    return df.sort_values(["well_id", "report_date_parsed"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_well_metadata() -> dict:
    meta: dict = {}
    wells_dir = FIELD_DIR / "wells"
    if not wells_dir.exists():
        return meta
    for well_dir in sorted(wells_dir.iterdir()):
        mp = well_dir / "metadata.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            meta[m["well_id"]] = m
    return meta


@st.cache_data(show_spinner=False)
def load_ditch_magnets() -> pd.DataFrame:
    path = repo_root / "data" / "processed" / "qc" / "ddr_ditch_magnets.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df.sort_values("report_date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_drilling_metrics() -> pd.DataFrame:
    path = repo_root / "data" / "processed" / "qc" / "ddr_drilling_metrics.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["report_date_dt"] = pd.to_datetime(df["report_date_dt"], errors="coerce")
    return df.sort_values("report_date_dt").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_planned_time() -> pd.DataFrame:
    path = repo_root / "data" / "processed" / "qc" / "ddr_planned_time.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df.sort_values("report_date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_vessels() -> pd.DataFrame:
    path = repo_root / "data" / "processed" / "qc" / "ddr_vessels.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df.sort_values("report_date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_causality() -> dict | None:
    path = repo_root / "data" / "graphs" / "causality.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_resource(show_spinner=False)
def _load_search_index():
    idx_dir = repo_root / "data" / "global_index"
    if not idx_dir.exists():
        return None, None, None, f"Index directory not found: {idx_dir}"
    try:
        import faiss  # type: ignore[import]

        index  = faiss.read_index(str(idx_dir / "faiss.index"))
        meta   = pd.read_parquet(idx_dir / "chunk_meta.parquet")
        chunks = pd.read_parquet(idx_dir / "chunks.parquet")
        if "chunk_id_global" in chunks.columns:
            chunks = chunks.set_index("chunk_id_global")
        return index, meta, chunks, None
    except Exception as exc:
        return None, None, None, str(exc)


@st.cache_resource(show_spinner=False)
def _load_embed_model():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        local = repo_root / "models" / "all-MiniLM-L6-v2"
        name  = str(local) if local.exists() else "sentence-transformers/all-MiniLM-L6-v2"
        return SentenceTransformer(name, device="cpu"), None
    except Exception as exc:
        return None, str(exc)


def _run_search(
    question: str,
    k: int = 10,
    doc_filter: str | None = None,
) -> tuple[list[dict], str]:
    import numpy as np  # noqa: F401 — required by faiss

    index, meta, chunks, idx_err = _load_search_index()
    if index is None:
        return [], f"Index load failed: {idx_err}"

    model, mdl_err = _load_embed_model()
    if model is None:
        return [], f"Embedding model load failed: {mdl_err}"

    q_emb = model.encode(
        [question], normalize_embeddings=True, show_progress_bar=False
    ).astype("float32")
    fetch_k        = min(k * 4, index.ntotal)
    scores, indices = index.search(q_emb, fetch_k)
    scores   = scores[0].tolist()
    indices  = indices[0].tolist()

    results: list[dict] = []
    for score, row_idx in zip(scores, indices):
        if len(results) >= k:
            break
        if row_idx < 0 or row_idx >= len(meta):
            continue
        m_row  = meta.iloc[row_idx]
        doc_id = str(m_row.get("doc_id", ""))
        if doc_filter and doc_filter not in doc_id:
            continue
        cid        = str(m_row.get("chunk_id_global", ""))
        chunk_text = ""
        if cid in chunks.index:
            chunk_text = str(chunks.loc[cid].get("chunk_text", ""))
        elif "chunk_text" in m_row:
            chunk_text = str(m_row["chunk_text"])
        results.append({
            "rank":          len(results) + 1,
            "score":         round(float(score), 4),
            "doc_id":        doc_id,
            "report_date":   str(m_row.get("report_date", "")),
            "wellbore":      str(m_row.get("wellbore", "")),
            "section_title": str(m_row.get("section_title", "")),
            "chunk_text":    chunk_text,
            "snippet":       chunk_text[:300],
        })
    return results, ""


def load_global_search() -> bool:
    index, _, _, _ = _load_search_index()
    return index is not None


@st.cache_data(show_spinner=False)
def load_corpus_gaps() -> list[dict]:
    legacy_pattern = re.compile(r"Ensco120-DDR-(\d+)-JRP-\w+-(\d{4}-\d{2}-\d{2})$")
    utah_pattern = re.compile(
        r"UtahForge-DDR-FORGE-16A-78-32-\w+-(\d{4}-\d{2}-\d{2})-R\d+-[a-f0-9]{8}$"
    )
    entries: list[tuple[int, str, Path]] = []
    for d in sorted(PROCESSED_DIR.iterdir()):
        m = legacy_pattern.match(d.name)
        if m:
            entries.append((int(m.group(1)), m.group(2), d))
            continue
        m = utah_pattern.match(d.name)
        if m:
            report_date = pd.to_datetime(m.group(1), errors="coerce")
            if pd.notna(report_date):
                day_num = int((report_date.date() - pd.Timestamp("2020-10-21").date()).days + 1)
                entries.append((day_num, m.group(1), d))

    if not entries:
        return []

    by_num = {num: (date, path) for num, date, path in entries}
    all_nums = sorted(by_num)
    missing = sorted(set(range(all_nums[0], all_nums[-1] + 1)) - set(all_nums))
    if not missing:
        return []

    blocks: list[list[int]] = []
    cur: list[int] = [missing[0]]
    for n in missing[1:]:
        if n == cur[-1] + 1:
            cur.append(n)
        else:
            blocks.append(cur)
            cur = [n]
    blocks.append(cur)

    def _hdr(path: Path, field: str) -> str:
        try:
            df = pd.read_parquet(path / "ddr_header.parquet", columns=[field])
            val = df[field].iloc[0]
            return str(val) if pd.notna(val) else ""
        except Exception:
            return ""

    def _phases(path: Path) -> list[str]:
        try:
            df = pd.read_parquet(path / "ddr_facts.parquet", columns=["phase"])
            return df["phase"].dropna().unique().tolist()
        except Exception:
            return []

    gaps = []
    for block in blocks:
        b_entry = by_num.get(block[0] - 1)
        a_entry = by_num.get(block[-1] + 1)
        pb = _phases(b_entry[1])[0] if b_entry else ""
        pa = _phases(a_entry[1])[0] if a_entry else ""
        gaps.append({
            "missing_nums":     block,
            "num_missing":      len(block),
            "date_before":      b_entry[0] if b_entry else "",
            "date_after":       a_entry[0]  if a_entry else "",
            "phase_before":     pb,
            "phase_after":      pa,
            "cross_phase":      pb != pa and bool(pb) and bool(pa),
            "depth_before":     _hdr(b_entry[1], "end_depth_md_ft") if b_entry else "",
            "depth_after":      _hdr(a_entry[1], "end_depth_md_ft") if a_entry else "",
            "last_24hr_before": _hdr(b_entry[1], "last_24hr_summary") if b_entry else "",
            "morning_after":    _hdr(a_entry[1], "morning_report_ops") if a_entry else "",
        })
    return gaps


_AGGREGATE_KEYWORDS: frozenset[str] = frozenset({
    "maximum", "minimum", "highest", "lowest", "total", "average",
    "how many", "most", "least", "all phases", "across all", "entire well",
    "cumulative", "sum of", "count of",
})


def _looks_like_aggregate(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _AGGREGATE_KEYWORDS)
