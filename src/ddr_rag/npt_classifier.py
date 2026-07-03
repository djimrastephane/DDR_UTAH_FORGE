from __future__ import annotations

import re

CATEGORY_LABELS: dict[str, str] = {
    "fishing":              "Fishing & Recovery",
    "well_control":         "Well Control",
    "wellbore_condition":   "Wellbore Condition",
    "bop_equipment":        "BOP / Wellhead",
    "mpd":                  "MPD System Issue",
    "equipment":            "Equipment Failure",
    "downhole_tools":       "Downhole Tool / BHA Issue",
    "completion":           "Completion Failure",
    "formation_testing":    "Formation Testing / Well Eval",
    "completion_monitoring":"Completion Monitoring",
    "weather":              "Weather / Sea State",
    "waiting_logistics":    "Waiting / Logistics",
    "procedural":           "Procedural / Safety Standby",
    "other_npt":            "Other NPT",
}

CATEGORY_COLOURS: dict[str, str] = {
    "fishing":              "#795548",   # brown
    "well_control":         "#D32F2F",   # deep red
    "wellbore_condition":   "#C62828",   # red
    "bop_equipment":        "#E64A19",   # deep orange
    "mpd":                  "#7B1FA2",   # purple
    "equipment":            "#F57C00",   # orange
    "downhole_tools":       "#6D4C41",   # brown-grey
    "completion":           "#1976D2",   # blue
    "formation_testing":    "#00897B",   # teal
    "completion_monitoring":"#26A69A",   # light teal
    "weather":              "#0288D1",   # light blue
    "waiting_logistics":    "#455A64",   # blue-grey
    "procedural":           "#689F38",   # light green
    "other_npt":            "#9E9E9E",   # grey
}

_RULES: list[tuple[str, list[str]]] = [
    ("fishing", [
        r"\bfishing\b", r"\bwashover\b", r"\bjunk mill\b", r"\bmill\b",
        r"\bmagnet\b", r"\bcatcher sub\b", r"\bstacey\b", r"\bmega magnet\b",
        r"\bhtpac\b", r"\binterwell\b", r"\brecovery tool\b",
        r"\bovershot\b", r"\bspear\b",
    ]),
    ("well_control", [
        r"\bkick\b", r"\binflux\b", r"\bshut.?in\b", r"\bwell control\b",
        r"\bshut in drill pipe pressure\b", r"\bsidpp\b", r"\bsicp\b",
        r"\bpit gain\b", r"\bflow check\b", r"\bbarrier\b",
        r"\blost circulation\b", r"\btight hole\b",
    ]),
    # Weather is checked BEFORE bop_equipment: weather ops often mention BOP
    # in a concurrent section ("Concurrently: • Completed BOP test"), causing
    # \bbop\b to fire incorrectly.  Explicit "waited on weather" / WOW patterns
    # anchor the primary activity reliably.
    ("weather", [
        r"\bwaited on weather\b", r"\bwow\b", r"\bwaiting on weather\b",
        r"\bweather window\b", r"\bweather\b", r"\bsea state\b",
        r"\bwind speed\b", r"\bwave\b.*\bheight", r"\bswell\b",
        r"\bfog\b", r"\bvisibility\b", r"\bcrane\b.*\brest",
        r"\bsupply vessel\b.*\bunabl", r"\bunable to work.*vessel",
        r"\boperational limit\b",
    ]),
    ("bop_equipment", [
        # \bbop\b is kept but weather now takes priority for ops that mention
        # BOP only incidentally.  \bannular\b tightened to failure/issue
        # context — routine jetting/testing the annular is not BOP NPT.
        r"\bbop\b.*\b(?:fail|stuck|leak|problem|issue|trip|test)\b",
        r"\b(?:fail|stuck|leak|problem|issue)\b.*\bbop\b",
        r"\bannular\b.*\b(?:fail|stuck|leak|bleed|blew|burst|damage)\b",
        r"\b(?:fail|stuck|leak|blew|burst)\b.*\bannular\b",
        r"\bpipe ram\b", r"\bblind ram\b", r"\bshear ram\b",
        r"\briser\b.*\bfail", r"\bwellhead\b.*\bfail",
        r"\bhp riser\b", r"\besd\b", r"\bpower.?outage\b", r"\bblackout\b",
    ]),
    ("mpd", [
        r"\bmpd\b.*\bfail", r"\bmpd\b.*\btrip", r"\brcd\b.*\bfail",
        r"\bventuri\b.*\bfail", r"\bchoke\b.*\bfail",
        r"\bmpd\b.*\bunable", r"\bpressure\b.*\bunable",
        r"\bsbp\b.*\bunable",
    ]),
    # Formation testing / well evaluation from legacy datasets. Utah FORGE
    # rows bypass these rules and use the corpus-specific classifier below.
    ("formation_testing", [
        r"\brtts\b", r"\bswab\b", r"\badt\b",
        r"\bformation test\b", r"\bwell test\b", r"\bfil test\b",
        r"\bpressure build.?up\b", r"\bflow period\b", r"\bshut.?in period\b",
        r"\bdsm\b", r"\bdsv\b.*\btest", r"\bpressure transient\b",
        r"\bwellbore\b.*\btest\b",
    ]),
    # Completion monitoring — routine FIA/NCS wash pipe operations, flow
    # checks, and stimulation programme monitoring coded T.
    ("completion_monitoring", [
        r"\bfia\b", r"\bfia tool\b", r"\bfia packer\b",
        r"\bwash.*in hole\b", r"\bwash.?pipe\b.*\brih\b",
        r"\blocate.*sleeve\b", r"\bsleeve.*locat",
        r"\bmuster\b", r"\bfire alarm\b", r"\bfalse alarm\b",
        r"\bmonitor.*well\b", r"\bmonitoring well\b",
        r"\bstaged.*pump\b", r"\bpump.*stage",
        r"\bcycled\b",
    ]),
    ("equipment", [
        r"\bprs\b.*\bfail", r"\bprs\b.*\bfault", r"\bprofibus\b",
        r"\bcable\b.*\bfail", r"\bcable\b.*\bfault", r"\bcable\b.*\bdamag",
        r"\bmotor\b.*\bfail", r"\belect.*\bfail", r"\bhydraul.*\bfail",
        r"\bequipment\b.*\bfail", r"\brepair\b", r"\bbreakdown\b",
        r"\bpump\b.*\bfail", r"\bvalve\b.*\bfail", r"\btop.?drive\b.*\bfail",
        r"\bcrane\b.*\bfail",
    ]),
    ("completion", [
        r"\bunable to open\b", r"\bunable to locate\b", r"\bunable to close\b",
        r"\bsleeve\b.*\bunable", r"\bunable.*\bsleeve",
        r"\bpacker\b.*\bleak", r"\bleak\b.*\bpacker",
        r"\bncs\b.*\bunable", r"\bncs.*\bfail",
        r"\bcoil\b.*\bstuck", r"\bstuck\b.*\bcoil",
        r"\bunable to increase pressure\b", r"\bleak\b.*\btest\b.*\bfail",
    ]),
    # weather rule moved before bop_equipment — see comment above
    ("waiting_logistics", [
        r"\bwaiting on\b", r"\bwait.*vessel\b", r"\bwait.*helicopter\b",
        r"\bwait.*parts\b", r"\bwait.*equipment\b", r"\bwait.*service\b",
        r"\bwos\b", r"\bwoc\b", r"\bdelayed\b", r"\bdelay\b.*\bsupply",
        r"\bcustoms\b", r"\bbackload\b.*\bdelay", r"\bno.*available\b",
    ]),
    ("procedural", [
        r"\bpermit\b.*\bdelay", r"\bapproval\b.*\bwait",
        r"\bstandby\b.*\bdecision", r"\bstandby\b.*\bmanagement",
        r"\bconfer.*onshore", r"\bwaiting.*approval",
        r"\bstandby\b.*\bonshore",
    ]),
]

_OP_CODE_FALLBACK: dict[str, str] = {
    "MPDCSG":      "mpd_csg_programme",
    "MPDDRLG":     "mpd_csg_programme",
    "MPDCMT":      "mpd_csg_programme",
    "STIM":        "stimulation_programme",
    "RPCOMPU PP":  "stimulation_programme",
    "XTREEOPS":    "stimulation_programme",
    "DRILLOUT":    "stimulation_programme",
    "TDTRIP":      "formation_testing",
}

CATEGORY_LABELS["mpd_csg_programme"]    = "MPD / Casing Programme"
CATEGORY_LABELS["stimulation_programme"] = "Stimulation Programme"

CATEGORY_COLOURS["mpd_csg_programme"]    = "#6A1B9A"   # dark purple
CATEGORY_COLOURS["stimulation_programme"] = "#1565C0"   # dark blue

# Compile all regex patterns once
_COMPILED: list[tuple[str, list[re.Pattern]]] = [
    (category, [re.compile(pat, re.IGNORECASE) for pat in patterns])
    for category, patterns in _RULES
]

_UTAH_DOC_RE = re.compile(r"^UtahForge-DDR-FORGE-16A-78-32-", re.I)
_UTAH_WELL_RE = re.compile(r"\bFORGE-16A-78-32\b", re.I)

_UTAH_EXPLICIT_NPT_RE = re.compile(
    r"\b(?:npt|non[- ]productive|down\s*time|downtime|on\s+npt)\b",
    re.I,
)
_UTAH_WAITING_RE = re.compile(
    r"\bwait(?:ing)?\s+on\s+(?:parts|equipment|water|crew|orders|service|vendor)\b",
    re.I,
)
_UTAH_WAIT_REPAIRS_RE = re.compile(
    r"\bwait(?:ing)?\s+on\s+(?:pump\s+)?repairs?\b",
    re.I,
)
_UTAH_REPAIR_CODE_RE = re.compile(r"\brepair\s+rig\b", re.I)
_UTAH_EQUIPMENT_BREAKDOWN_RE = re.compile(
    r"\b(?:hydraulic\s+line\s+failed|line\s+failed|went\s+down|"
    r"bearing\s+broke|repair\s+time|module\s+on\s+pump|"
    r"replacing?\s+pump|replace\s+hydraulic\s+hose|top\s+drive\s+brake|"
    r"damages?\s+beyond\s+repair)\b",
    re.I,
)
_UTAH_WELLBORE_CONDITION_RE = re.compile(
    r"\b(?:stuck|tight\s+hole|lost\s+tool\s+face|over\s*pull|"
    r"lost\s+circulation|no\s+returns|total\s+losses|partial\s+returns)\b",
    re.I,
)
_UTAH_FISHING_RE = re.compile(
    r"\b(?:fishing|fish(?:ing)?\s+bha|milled?\s+up|lost\s+pieces?\s+of\s+bit|"
    r"junk|mill(?:ed)?\s+(?:up\s+)?(?:junk|lost|bit))\b",
    re.I,
)
_UTAH_DOWNHOLE_TOOL_RE = re.compile(
    r"\b(?:did\s+not\s+seat|did\s+not\s+set|would\s+not\s+set|"
    r"lower\s+port\s+did\s+not\s+open|lost\s+two\s+packer\s+elements|"
    r"stator\s+failure|core\s+barrel\s+was\s+jammed)\b",
    re.I,
)


def classify_npt_row(
    text: str,
    op_code: str = "",
    activity_code: str = "",
) -> str:
    combined = f"{text or ''} {op_code or ''} {activity_code or ''}".strip()
    if not combined:
        return "other_npt"

    for category, patterns in _COMPILED:
        if any(pat.search(combined) for pat in patterns):
            return category

    # Op-code fallback: planned programme work coded T by convention
    fallback = _OP_CODE_FALLBACK.get(str(op_code).strip().upper())
    if fallback:
        return fallback

    return "other_npt"


def is_utah_forge_record(row: object) -> bool:
    def _value(key: str) -> str:
        try:
            value = row.get(key, "")  # type: ignore[attr-defined]
        except AttributeError:
            value = ""
        return "" if value is None else str(value)

    return bool(
        _UTAH_DOC_RE.search(_value("doc_id"))
        or _UTAH_WELL_RE.search(_value("wellbore"))
        or _value("field_name").replace(" ", "").lower() == "utahforge"
    )


def classify_utah_forge_npt(
    text: str,
    op_code: str = "",
    activity_code: str = "",
    npt_code: str = "",
) -> tuple[bool, str]:
    """Classify Utah FORGE rows from report text, without legacy P-T-X assumptions."""
    combined = f"{text or ''} {op_code or ''} {activity_code or ''} {npt_code or ''}".strip()
    if not combined:
        return False, "productive"

    code_text = f"{op_code or ''} {activity_code or ''}".strip()
    explicit_code = str(npt_code or "").strip().upper()
    if explicit_code in {"T", "NPT", "DOWN", "DOWNTIME"}:
        return True, classify_npt_row(text, op_code, activity_code)

    if _UTAH_REPAIR_CODE_RE.search(code_text):
        return True, "equipment"

    if _UTAH_WAIT_REPAIRS_RE.search(combined):
        return True, "equipment"

    if _UTAH_EQUIPMENT_BREAKDOWN_RE.search(combined):
        return True, "equipment"

    if _UTAH_DOWNHOLE_TOOL_RE.search(combined):
        return True, "downhole_tools"

    if _UTAH_WELLBORE_CONDITION_RE.search(combined):
        return True, "wellbore_condition"

    if _UTAH_FISHING_RE.search(combined):
        return True, "fishing"

    if _UTAH_WAITING_RE.search(combined):
        return True, "waiting_logistics"

    if _UTAH_EXPLICIT_NPT_RE.search(combined):
        return True, "other_npt"

    return False, "productive"


def classify_utah_forge_npt_row(row: object) -> tuple[bool, str]:
    def _value(key: str) -> str:
        try:
            value = row.get(key, "")  # type: ignore[attr-defined]
        except AttributeError:
            value = ""
        return "" if value is None else str(value)

    return classify_utah_forge_npt(
        _value("operation_text"),
        _value("op_code"),
        _value("activity_code"),
        _value("pt_x"),
    )


def apply_corpus_npt_rules(df: "pd.DataFrame") -> "pd.DataFrame":
    import pandas as pd

    if df.empty:
        return df.copy()

    out = df.copy()
    if "is_npt" not in out.columns:
        out["is_npt"] = False

    utah_mask = out.apply(is_utah_forge_record, axis=1)
    if utah_mask.any():
        classified = out.loc[utah_mask].apply(classify_utah_forge_npt_row, axis=1)
        out.loc[utah_mask, "is_npt"] = classified.map(lambda item: bool(item[0])).astype(bool)
        out.loc[utah_mask, "npt_category"] = classified.map(lambda item: item[1])

    out["is_npt"] = out["is_npt"].fillna(False).astype(bool)
    return out


def classify_ops_df(df: "pd.DataFrame") -> "pd.Series":
    import pandas as pd

    def _classify_row(row: "pd.Series") -> str:
        if is_utah_forge_record(row):
            _is_npt, category = classify_utah_forge_npt_row(row)
            return category
        if not row.get("is_npt", False):
            return "productive"
        return classify_npt_row(
            str(row.get("operation_text") or ""),
            str(row.get("op_code") or ""),
            str(row.get("activity_code") or ""),
        )

    return df.apply(_classify_row, axis=1)
