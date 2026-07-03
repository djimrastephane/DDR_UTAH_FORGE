from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

PROCESSED_DIR = repo_root / "data" / "processed"
OUT_DIR = PROCESSED_DIR / "qc"


# Depth in ft — plausible MD range 50–30 000 ft
_RE_DEPTH = re.compile(r"([\d,]+)\s*ft(?:OTH|MD|TVD)?(?:\b|,)", re.I)

# Force: "30klbs", "190Klbs", "7-15k", "5Klbs"
_RE_KLBS_VAL = re.compile(
    r"([\d]+(?:[–\-][\d]+)?(?:\.\d+)?)\s*[Kk]lbs?\b", re.I
)

_RE_RESTRICT_FD = re.compile(
    r"([\d]+(?:[–\-][\d]+)?)\s*[Kk]lbs?\s+(?:restriction|tight|packoff)\s+(?:at|@)\s*([\d,]+)\s*ft",
    re.I,
)
_RE_RESTRICT_D = re.compile(
    r"(?:restriction|packoff|tight\s+spot)\s+(?:at|@)\s*([\d,]+)\s*ft",
    re.I,
)
_RE_TAG_DEPTH = re.compile(
    r"tagged\s+\w+\s+at\s+([\d,]+)\s*ft\s+with\s+([\d]+(?:\.\d+)?)\s*[Kk]lbs?",
    re.I,
)

_RE_OP_FORCE = re.compile(
    r"([\d]+(?:[–\-][\d]+)?(?:\.\d+)?)\s*[Kk](?:lbs?)?\s+overpull|"
    r"overpull\s+(?:of\s+)?([\d]+(?:\.\d+)?)\s*[Kk](?:lbs?)?",
    re.I,
)
_RE_OP_DEPTH = re.compile(
    r"overpull\s+(?:at|to|from)\s+([\d,]+)\s*ft|"
    r"([\d,]+)\s*ft[^.]{0,40}?overpull",
    re.I,
)
# Exclude: WOB context, kft.lbs (torque), "confirmed latched" minor checks
_RE_WOB_CTX = re.compile(r"\bWOB\b|\bweight\s+on\s+bit\b", re.I)
_RE_TORQUE = re.compile(r"\d+\s*[Kk]ft\.?lbs?\b", re.I)
_RE_LATCH_CHECK = re.compile(
    r"confirm(?:ed)?\s+(?:latched|engaged|landed)|lockdown\s+dogs", re.I
)

_RE_LOSS_RATE = re.compile(
    r"loss\s+rate\s+of\s+([\d]+(?:[–\-][\d]+)?(?:\.\d+)?)\s*bbls?(?:/hr)?|"
    r"([\d]+(?:[–\-][\d]+)?)\s*bbls?\s+loss(?:es)?|"
    r"observed\s+([\d]+(?:[–\-][\d]+)?)\s*bbls?\s+loss",
    re.I,
)
_RE_FULL_LOSS = re.compile(r"(?:observed\s+)?full\s+loss(?:es)?|no\s+returns\b", re.I)
_RE_STATIC_LOSS = re.compile(
    r"static\s+loss\s+rate\s+of\s+([\d.]+)\s*(?:bbls?/hr|gpm)", re.I
)
_RE_LCM = re.compile(r"\bLCM\b|lost\s+circulation\s+material", re.I)
_RE_SEEPAGE = re.compile(r"seepage\s+loss|minor\s+loss(?:es)?|partial\s+loss(?:es)?", re.I)
# True negatives to skip
_RE_NO_LOSS = re.compile(
    r"no\s+loss(?:es)?\s+observed|none\s+observed|"
    r"loss\s+of\s+(?:signal|communication|contact|near.bit|inclination|"
    r"string\s+weight|comms|power|visibility)|"
    r"monitoring\s+for\s+loss.*none\s+observed",
    re.I,
)

_RE_HIGH_ECD = re.compile(
    r"ECD[^.]{0,30}?([\d]{2}\.[\d]{2,3})\s*ppge?|"
    r"([\d]{2}\.[\d]{2,3})\s*ppge?\s+(?:at|observed|ECD|at\s+PWD)",
    re.I,
)
_RE_GAS_INFLUX = re.compile(
    r"gas\s+(?:cut|influx|in\s+mud|on\s+shakers)|"
    r"evacuate\s+gas\s+from\s+well|"
    r"gas\s+(?:kick|bubble)|"
    r"(?:high|elevated)\s+background\s+gas",
    re.I,
)
_RE_INSTABILITY = re.compile(
    r"wellbore\s+instab|caving|sloughing|spalling|heaving\s+(?:shale|formation)|"
    r"unconsolidated\s+formation",
    re.I,
)
_RE_BALLOONING = re.compile(
    r"ballooning|breathing\s+formation|wellbore\s+breathing", re.I
)
_RE_DIFF_STICK = re.compile(
    r"differential\s+st(?:ick|uck)|wall\s+stick|diff\s+st(?:ick|uck)", re.I
)

_RE_FIT_TRIGGER = re.compile(
    r"\b(?:dynamic\s+)?(?:FIT|DFIT|LOT|leak[\s\-]?off\s+test|"
    r"formation\s+integrity\s+test)\b",
    re.I,
)
# Only ppge (not ppg) in plausible frac-gradient range 12–16 ppge
# \d{1,3} covers both "15.0ppge" (1 decimal) and "14.43ppge" (2 decimals)
_RE_FIT_PPGE = re.compile(r"(1[2-6]\.\d{1,3})\s*ppge\b", re.I)
_RE_FIT_SHOE = re.compile(
    r'(20"\s*(?:shoe|conductor)|13[\-\s]3/8"\s*(?:casing|shoe)|'
    r'9[\-\s]7/8"\s*(?:casing|liner|shoe)|7"\s*(?:liner|shoe))',
    re.I,
)
_RE_FIT_PASS = re.compile(r"good\s+(?:FIT|DFIT|LOT)|no\s+losses\s+observed|converge\s+to\s+stable", re.I)
# "until divergence" = LOT fracture initiation; plain "diverge, but converge" is NOT initiation
_RE_FIT_INIT = re.compile(r"until\s+divergen|diversion|leak[\s\-]?off|initiat", re.I)


def _depths_in_text(text: str) -> list[float]:
    out = []
    for m in _RE_DEPTH.finditer(text):
        raw = m.group(1).replace(",", "").strip()
        if not raw:
            continue
        try:
            v = float(raw)
        except ValueError:
            continue
        if 50 <= v <= 30_000:
            out.append(v)
    return out


def _parse_klbs(raw: str) -> float | None:
    raw = raw.replace(",", "").strip()
    if re.match(r"[\d]+[–\-][\d]+", raw):
        parts = re.split(r"[–\-]", raw)
        try:
            return float(parts[-1])
        except ValueError:
            return None
    try:
        return float(raw)
    except ValueError:
        return None


def _snippet(text: str, pattern: re.Pattern, window: int = 70) -> str:
    m = pattern.search(text)
    if not m:
        return text[:window]
    s = max(0, m.start() - 20)
    e = min(len(text), m.end() + 50)
    snippet = text[s:e].replace("\n", " ").strip()
    return f"...{snippet}..." if s > 0 else snippet


def _hole_type(phase: str, text: str) -> str:
    text_l = text.lower()
    cased = [
        "inside casing", "inside liner", "inside 9-7/8", 'inside 7"',
        "inside the liner", "cased hole", "through tubing",
        "inside completion", "frac sleeve", "ncs sleeve",
    ]
    open_h = [
        "open hole", "below shoe", "below casing shoe",
        "in formation", "rat hole", "below 9-7/8",
        "open-hole", "below 13-3/8",
    ]
    if any(k in text_l for k in cased):
        return "cased"
    if any(k in text_l for k in open_h):
        return "open"
    defaults = {
        "MIRU": "surface",
        "COND1": "open",
        "INTRM1": "open",
        "INTRM2": "open",
        "COMPZN": "cased",
    }
    if phase in defaults:
        return defaults[phase]
    if phase == "PROD1":
        if any(k in text_l for k in ["rih liner", "set packer", "liner hanger",
                                       "running liner", "liner deployment"]):
            return "cased"
        return "open"
    return "unknown"


def _extract_restrictions(text: str, meta: dict) -> list[dict]:
    events = []
    base = {**meta, "event_type": "restriction"}
    hole = _hole_type(meta["phase"], text)
    depths = _depths_in_text(text)

    for m in _RE_RESTRICT_FD.finditer(text):
        force = _parse_klbs(m.group(1))
        depth = float(m.group(2).replace(",", ""))
        events.append({
            **base, "hole_type": hole,
            "event_depth_ft_md": depth,
            "force_klbs": force,
            "sub_type": "restriction",
            "raw_snippet": _snippet(text, _RE_RESTRICT_FD),
        })

    for m in _RE_RESTRICT_D.finditer(text):
        depth = float(m.group(1).replace(",", ""))
        if any(abs(depth - e["event_depth_ft_md"]) < 50
               for e in events if e.get("event_depth_ft_md")):
            continue  # already captured above
        # Try to find a nearby klbs value
        window = text[max(0, m.start() - 80):m.end() + 80]
        force_m = _RE_KLBS_VAL.search(window)
        force = _parse_klbs(force_m.group(1)) if force_m else None
        events.append({
            **base, "hole_type": hole,
            "event_depth_ft_md": depth,
            "force_klbs": force,
            "sub_type": "restriction",
            "raw_snippet": _snippet(text, _RE_RESTRICT_D),
        })

    for m in _RE_TAG_DEPTH.finditer(text):
        depth = float(m.group(1).replace(",", ""))
        force = _parse_klbs(m.group(2))
        if force and force < 20:   # minor confirmation overpull, skip
            continue
        events.append({
            **base, "hole_type": hole,
            "event_depth_ft_md": depth,
            "force_klbs": force,
            "sub_type": "tag",
            "raw_snippet": _snippet(text, _RE_TAG_DEPTH),
        })

    if not events and re.search(r"\bpackoff\b", text, re.I):
        events.append({
            **base, "hole_type": hole,
            "event_depth_ft_md": depths[0] if depths else None,
            "force_klbs": None,
            "sub_type": "packoff",
            "raw_snippet": text[:80],
        })

    return events


def _extract_overpull(text: str, meta: dict) -> list[dict]:
    # Skip if this is clearly a WOB / torque / minor latch-check context
    if _RE_WOB_CTX.search(text) and not re.search(r"overpull", text, re.I):
        return []
    if _RE_LATCH_CHECK.search(text) and re.search(
        r"\b[1-9]\s*[Kk]lbs?\s+overpull", text, re.I
    ):
        return []  # "confirmed latched with 5klbs overpull" — skip minor check

    events = []
    base = {**meta, "event_type": "overpull"}
    hole = _hole_type(meta["phase"], text)

    # Find all overpull force mentions
    for m in _RE_OP_FORCE.finditer(text):
        raw_force = m.group(1) or m.group(2) or ""
        force = _parse_klbs(raw_force) if raw_force else None
        if force is None:
            continue
        # Minor latching confirmations: ≤ 5klbs single value — skip
        if force <= 5 and not re.search(r"overpull|restriction", text[max(0,m.start()-30):m.end()+30], re.I):
            continue

        # Find closest depth to this match
        depth = None
        # First, look for "overpull at XXXX ft" or "XXXX ft...overpull" nearby
        window_text = text[max(0, m.start() - 120):m.end() + 120]
        dm = _RE_OP_DEPTH.search(window_text)
        if dm:
            raw_d = dm.group(1) or dm.group(2) or ""
            d = float(raw_d.replace(",", "")) if raw_d else None
            if d and 50 <= d <= 30_000:
                depth = d
        # Fallback: first plausible depth in sentence
        if depth is None:
            sentence = re.split(r"[.•\n]", text[max(0, m.start() - 200):m.end() + 200])
            for seg in sentence:
                ds = _depths_in_text(seg)
                if ds:
                    depth = ds[0]
                    break

        snippet = text[max(0, m.start() - 20):min(len(text), m.end() + 60)]
        events.append({
            **base, "hole_type": hole,
            "event_depth_ft_md": depth,
            "force_klbs": force,
            "sub_type": "overpull",
            "raw_snippet": snippet.replace("\n", " ").strip(),
        })

    return events


def _extract_losses(text: str, meta: dict) -> list[dict]:
    if _RE_NO_LOSS.search(text):
        if not _RE_FULL_LOSS.search(text) and not _RE_LOSS_RATE.search(text):
            return []

    events = []
    base = {**meta, "event_type": "mud_loss"}
    hole = _hole_type(meta["phase"], text)
    depths = _depths_in_text(text)

    if _RE_FULL_LOSS.search(text):
        events.append({
            **base, "hole_type": hole,
            "event_depth_ft_md": depths[0] if depths else None,
            "loss_volume_bbl": None,
            "loss_rate_bbl_hr": None,
            "severity": "total",
            "lcm_flag": bool(_RE_LCM.search(text)),
            "sub_type": "full_losses",
            "raw_snippet": _snippet(text, _RE_FULL_LOSS),
        })

    for m in _RE_LOSS_RATE.finditer(text):
        raw = next((g for g in m.groups() if g), "")
        vol = _parse_klbs(raw)  # reuse range parser — same logic
        if vol is None:
            continue
        is_rate = bool(re.search(r"loss\s+rate", text[max(0, m.start()-5):m.start()+15], re.I)
                       or re.search(r"bbls?/hr", text[m.start():m.end()+10], re.I))
        if vol >= 50:
            sev = "total"
        elif vol >= 10:
            sev = "partial"
        else:
            sev = "seepage"

        events.append({
            **base, "hole_type": hole,
            "event_depth_ft_md": depths[0] if depths else None,
            "loss_volume_bbl": None if is_rate else vol,
            "loss_rate_bbl_hr": vol if is_rate else None,
            "severity": sev,
            "lcm_flag": bool(_RE_LCM.search(text)),
            "sub_type": "rate" if is_rate else "volume",
            "raw_snippet": _snippet(text, _RE_LOSS_RATE),
        })

    # Static loss (completion phase bleed-off)
    m = _RE_STATIC_LOSS.search(text)
    if m and not events:
        rate_val = float(m.group(1))
        events.append({
            **base, "hole_type": hole,
            "event_depth_ft_md": depths[0] if depths else None,
            "loss_volume_bbl": None,
            "loss_rate_bbl_hr": rate_val,
            "severity": "seepage" if rate_val < 5 else "partial",
            "lcm_flag": False,
            "sub_type": "static",
            "raw_snippet": _snippet(text, _RE_STATIC_LOSS),
        })

    if _RE_LCM.search(text) and not events:
        events.append({
            **base, "hole_type": hole,
            "event_depth_ft_md": depths[0] if depths else None,
            "loss_volume_bbl": None,
            "loss_rate_bbl_hr": None,
            "severity": "lcm_treatment",
            "lcm_flag": True,
            "sub_type": "lcm",
            "raw_snippet": _snippet(text, _RE_LCM),
        })

    return events


def _extract_formation(text: str, meta: dict) -> list[dict]:
    events = []
    base = {**meta, "event_type": "formation"}
    hole = _hole_type(meta["phase"], text)
    depths = _depths_in_text(text)

    def _add(sub: str, pattern: re.Pattern, extra: dict | None = None):
        events.append({
            **base, "hole_type": hole,
            "event_depth_ft_md": depths[0] if depths else None,
            "sub_type": sub,
            "raw_snippet": _snippet(text, pattern),
            **(extra or {}),
        })

    for m in _RE_HIGH_ECD.finditer(text):
        ecd_val = float(m.group(1) or m.group(2))
        if ecd_val > 11.0:   # above background mud weight — genuine ECD signal
            events.append({
                **base, "hole_type": hole,
                "event_depth_ft_md": depths[0] if depths else None,
                "sub_type": "high_ecd",
                "ecd_ppge": ecd_val,
                "raw_snippet": _snippet(text, _RE_HIGH_ECD),
            })

    if _RE_GAS_INFLUX.search(text):
        _add("gas_influx", _RE_GAS_INFLUX)

    if _RE_INSTABILITY.search(text):
        _add("instability", _RE_INSTABILITY)

    if _RE_BALLOONING.search(text):
        _add("ballooning", _RE_BALLOONING)

    if _RE_DIFF_STICK.search(text):
        _add("diff_sticking", _RE_DIFF_STICK)

    return events


def _extract_fit_lot(text: str, meta: dict) -> list[dict]:
    if not _RE_FIT_TRIGGER.search(text):
        return []

    # Search the full operation text — FIT/LOT results often appear in trailing
    # "Note:" bullets well beyond a fixed window around the trigger keyword.
    all_ppge = list(_RE_FIT_PPGE.finditer(text))
    if not all_ppge:
        return []

    events: list[dict] = []
    seen_ppge: set[float] = set()
    shoe_m = _RE_FIT_SHOE.search(text)

    for m in _RE_FIT_TRIGGER.finditer(text):
        # Pick the ppge value closest to this trigger (handles multiple triggers)
        trig_mid = (m.start() + m.end()) // 2
        best     = min(all_ppge, key=lambda pm: abs(pm.start() - trig_mid))
        ppge_val = float(best.group(1))

        if ppge_val in seen_ppge:
            continue
        seen_ppge.add(ppge_val)

        raw_kw    = m.group(0).upper()
        test_type = "LOT" if "LOT" in raw_kw else "DFIT" if "DFIT" in raw_kw else "FIT"
        result    = ("initiation" if _RE_FIT_INIT.search(text)
                     else "pass" if _RE_FIT_PASS.search(text)
                     else "unknown")

        events.append({
            "report_date":    meta["report_date"],
            "report_date_dt": meta["report_date_dt"],
            "phase":          meta["phase"],
            "test_type":      test_type,
            "limit_ppge":     ppge_val,
            "result":         result,
            "casing_shoe":    shoe_m.group(1) if shoe_m else None,
            "doc_id":         meta.get("doc_id", ""),
            "page":           meta.get("page"),
            "shift_block":    meta.get("shift_block", ""),
            "start_time":     meta.get("start_time", ""),
            "end_time":       meta.get("end_time", ""),
            "raw_snippet":    _snippet(text, _RE_FIT_TRIGGER),
        })

    return events


def _load_corpus() -> pd.DataFrame:
    facts_frames, header_frames = [], []
    for doc_dir in sorted(PROCESSED_DIR.iterdir()):
        f = doc_dir / "ddr_facts.parquet"
        h = doc_dir / "ddr_header.parquet"
        if f.exists():
            facts_frames.append(pd.read_parquet(f))
        if h.exists():
            header_frames.append(pd.read_parquet(h))

    if not facts_frames:
        raise FileNotFoundError("No ddr_facts.parquet found")

    facts = pd.concat(facts_frames, ignore_index=True)
    facts["report_date_dt"] = pd.to_datetime(facts["report_date"], dayfirst=True, errors="coerce")

    if header_frames:
        headers = pd.concat(header_frames, ignore_index=True)
        headers["report_date_dt"] = pd.to_datetime(
            headers["report_date"], dayfirst=True, errors="coerce"
        )
        depth_by_date = (
            headers.dropna(subset=["end_depth_md_ft"])
            .sort_values("report_date_dt")
            .groupby("report_date_dt")["end_depth_md_ft"]
            .last()
        )
        facts["header_depth_ft_md"] = facts["report_date_dt"].map(depth_by_date)
    else:
        facts["header_depth_ft_md"] = None

    return facts


def run_extraction() -> pd.DataFrame:
    corpus = _load_corpus()
    print(f"Scanning {len(corpus):,} operation rows across "
          f"{corpus['report_date_dt'].nunique()} report dates...")

    all_events: list[dict] = []

    for _, row in corpus.iterrows():
        text = str(row.get("operation_text") or "")
        if not text.strip():
            continue

        meta = {
            "report_date":       row.get("report_date", ""),
            "report_date_dt":    row.get("report_date_dt"),
            "phase":             row.get("phase", ""),
            "is_npt":            bool(row.get("is_npt", False)),
            "duration_hr":       float(row.get("duration_hr") or 0),
            "header_depth_ft_md": row.get("header_depth_ft_md"),
            "full_op_text":      text,
            "doc_id":            row.get("doc_id", ""),
            "page":              row.get("page"),
            "shift_block":       row.get("shift_block", ""),
            "start_time":        row.get("start_time", ""),
            "end_time":          row.get("end_time", ""),
        }

        all_events.extend(_extract_restrictions(text, meta))
        all_events.extend(_extract_overpull(text, meta))
        all_events.extend(_extract_losses(text, meta))
        all_events.extend(_extract_formation(text, meta))

    if not all_events:
        print("No events extracted.")
        return pd.DataFrame()

    df = pd.DataFrame(all_events)
    df["report_date_dt"] = pd.to_datetime(df["report_date_dt"])
    df = df.sort_values(["report_date_dt", "event_type"]).reset_index(drop=True)

    # Coerce depth to numeric before dedup
    df["event_depth_ft_md"] = pd.to_numeric(df["event_depth_ft_md"], errors="coerce")
    df["header_depth_ft_md"] = pd.to_numeric(df["header_depth_ft_md"], errors="coerce")
    df["event_depth_ft_md"] = df["event_depth_ft_md"].combine_first(df["header_depth_ft_md"])

    # Deduplicate: same date + phase + type + sub_type + depth (within 20 ft)
    df["_depth_bucket"] = (df["event_depth_ft_md"].fillna(-1) / 20).round(0)
    df = df.drop_duplicates(
        subset=["report_date", "phase", "event_type", "sub_type", "_depth_bucket", "force_klbs"]
    ).drop(columns=["_depth_bucket"])

    return df.reset_index(drop=True)


def run_fit_lot_extraction() -> pd.DataFrame:
    corpus = _load_corpus()
    all_results: list[dict] = []

    for _, row in corpus.iterrows():
        text = str(row.get("operation_text") or "")
        if not text.strip():
            continue
        meta = {
            "report_date":    row.get("report_date", ""),
            "report_date_dt": row.get("report_date_dt"),
            "phase":          row.get("phase", ""),
            "doc_id":         row.get("doc_id", ""),
            "page":           row.get("page"),
            "shift_block":    row.get("shift_block", ""),
            "start_time":     row.get("start_time", ""),
            "end_time":       row.get("end_time", ""),
        }
        all_results.extend(_extract_fit_lot(text, meta))

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    df["report_date_dt"] = pd.to_datetime(df["report_date_dt"])
    df = df.sort_values(["report_date_dt", "phase"]).reset_index(drop=True)

    # Deduplicate: daily reports repeat prior-day ops; same date+phase+ppge = same test
    df = df.drop_duplicates(
        subset=["report_date", "phase", "limit_ppge"]
    ).reset_index(drop=True)
    return df


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No events found.")
        return

    print(f"\n{'='*60}")
    print(f"WELLBORE EVENTS SUMMARY  —  {len(df)} total events")
    print(f"{'='*60}")

    for etype in ["restriction", "overpull", "mud_loss", "formation"]:
        sub = df[df["event_type"] == etype]
        if sub.empty:
            continue

        headers = {
            "restriction": f"TIGHT SPOTS / RESTRICTIONS  ({len(sub)} events)",
            "overpull":    f"OVERPULL EVENTS  ({len(sub)} events)",
            "mud_loss":    f"MUD LOSSES  ({len(sub)} events)",
            "formation":   f"FORMATION CHALLENGES  ({len(sub)} events)",
        }
        print(f"\n── {headers[etype]}")

        if etype == "restriction":
            by_phase = sub.groupby("phase")
            for phase, grp in by_phase:
                depths = grp["event_depth_ft_md"].dropna().sort_values()
                forces = grp["force_klbs"].dropna()
                d_range = (f"{depths.min():.0f}–{depths.max():.0f} ft"
                           if not depths.empty else "depth unknown")
                f_range = (f"{forces.min():.0f}–{forces.max():.0f} klbs"
                           if not forces.empty else "force not recorded")
                holes = grp["hole_type"].value_counts().to_dict()
                print(f"   {phase:<8} {len(grp):>3} events  depth {d_range:<28} "
                      f"force {f_range:<20} {holes}")

        elif etype == "overpull":
            by_phase = sub.groupby("phase")
            for phase, grp in by_phase:
                depths = grp["event_depth_ft_md"].dropna().sort_values()
                forces = grp["force_klbs"].dropna()
                d_range = (f"{depths.min():.0f}–{depths.max():.0f} ft"
                           if not depths.empty else "depth unknown")
                f_range = (f"{forces.min():.0f}–{forces.max():.0f} klbs"
                           if not forces.empty else "force unknown")
                print(f"   {phase:<8} {len(grp):>3} events  "
                      f"depth {d_range:<28} force {f_range}")
            # Top 5 highest overpull events
            top5 = sub.nlargest(5, "force_klbs")[
                ["report_date", "phase", "event_depth_ft_md", "force_klbs", "raw_snippet"]
            ]
            print("\n   Top 5 highest overpull events:")
            for _, r in top5.iterrows():
                depth_s = f"{r.event_depth_ft_md:.0f} ft" if pd.notna(r.event_depth_ft_md) else "depth n/a"
                print(f"     [{r.report_date}] {r.phase:<8} {depth_s:<14} "
                      f"{r.force_klbs:.0f} klbs — {str(r.raw_snippet)[:70]}")

        elif etype == "mud_loss":
            by_sev = sub.groupby("severity").size().sort_values(ascending=False)
            for sev, cnt in by_sev.items():
                grp = sub[sub["severity"] == sev]
                rates = grp["loss_rate_bbl_hr"].dropna()
                rate_s = (f"rate {rates.min():.0f}–{rates.max():.0f} bbl/hr"
                          if not rates.empty else "")
                phases = grp["phase"].value_counts().to_dict()
                print(f"   {sev:<18} {cnt:>3} events  {rate_s:<28} {phases}")

        elif etype == "formation":
            by_sub = sub.groupby("sub_type").size().sort_values(ascending=False)
            for sub_t, cnt in by_sub.items():
                grp = sub[sub["sub_type"] == sub_t]
                phases = grp["phase"].value_counts().to_dict()
                extra = ""
                if sub_t == "high_ecd":
                    ecds = grp["ecd_ppge"].dropna()
                    if not ecds.empty:
                        extra = f"ECD {ecds.min():.2f}–{ecds.max():.2f} ppge  "
                print(f"   {sub_t:<20} {cnt:>3} occurrences  {extra}{phases}")

    print()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true",
                        help="Print detailed summary only, do not overwrite parquet")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = run_extraction()
    if df.empty:
        return

    fit_lot = run_fit_lot_extraction()

    if not args.summary:
        out_path = OUT_DIR / "ddr_wellbore_events.parquet"
        df.to_parquet(out_path, index=False)
        df.to_csv(out_path.with_suffix(".csv"), index=False)
        print(f"\nSaved: {out_path}  ({len(df)} rows)")

        if not fit_lot.empty:
            fl_path = OUT_DIR / "ddr_fit_lot_results.parquet"
            fit_lot.to_parquet(fl_path, index=False)
            fit_lot.to_csv(fl_path.with_suffix(".csv"), index=False)
            print(f"Saved: {fl_path}  ({len(fit_lot)} rows)")

    print_summary(df)

    if not fit_lot.empty:
        print(f"── FIT / LOT RESULTS  ({len(fit_lot)} tests)")
        for _, r in fit_lot.iterrows():
            m = re.search(r"DDR-?(\d+)", str(r["doc_id"]), re.I)
            cite  = f"DDR-{m.group(1)}" if m else str(r["doc_id"])[:20]
            shoe  = f"  [{r['casing_shoe']}]" if r.get("casing_shoe") else ""
            print(f"   [{r['report_date']}] {r['phase']:<8} {r['test_type']:<5} "
                  f"{r['limit_ppge']:.2f} ppge  {r['result']:<12}{shoe}  {cite}")
        print()


if __name__ == "__main__":
    main()
