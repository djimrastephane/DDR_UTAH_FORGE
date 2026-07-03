from __future__ import annotations

import re


DDR_MARKERS = (
    "daily well operation",
    "drilling original",
    "operation summary",
)

DDR_SECTION_LABELS: dict[str, tuple[str, ...]] = {
    "operation_summary": ("operation summary",),
    "personnel_data": ("personnel data",),
    "weather": ("weather", "current weather"),
    "planned_time_summary": ("planned time summary",),
    "materials": ("material", "materials"),
    "support_vessels": ("support vessels",),
    "formation_data": ("formation data",),
    "general_notes": ("general notes", "rig / equipment info", "rig equipment info"),
    "safety_checks": ("safety meeting", "safety observations", "permit observations"),
}


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" :\t\r\n")


def looks_like_daily_drilling_report(text: str) -> bool:
    lower = str(text or "").lower()
    hits = sum(1 for marker in DDR_MARKERS if marker in lower)
    return hits >= 2


def classify_ddr_section(text: str) -> str | None:
    lower = str(text or "").lower()
    for label, markers in DDR_SECTION_LABELS.items():
        if any(marker in lower for marker in markers):
            return label
    return None


def _first_match(pattern: str, text: str, flags: int = re.IGNORECASE) -> str | None:
    match = re.search(pattern, text, flags)
    if not match:
        return None
    if "value" in match.groupdict():
        return _clean(match.group("value"))
    return _clean(match.group(1))


def _well_name_from_header(lines: list[str]) -> str | None:
    for idx, line in enumerate(lines):
        if line.lower() == "daily well operation":
            for candidate in lines[idx + 1 : idx + 5]:
                if candidate and not re.search(r"report|job start|spud|daily well operation", candidate, re.I):
                    return _clean(candidate)
    return None


def extract_ddr_header_fields(text: str) -> dict[str, str]:
    header = str(text or "")[:2500]
    lines = [_clean(line) for line in header.splitlines() if _clean(line)]

    fields: dict[str, str] = {}
    well_name = _well_name_from_header(lines)
    if well_name:
        fields["well_name"] = well_name

    report_date = _first_match(r"\b(?P<value>\d{1,2}/\d{1,2}/\d{4})\b", header)
    if report_date:
        fields["report_date"] = report_date

    report_no = _first_match(r"\bReport\s+No\.?\s*:?\s*(?P<value>[A-Za-z0-9._/-]+)", header)
    if report_no:
        fields["report_no"] = report_no

    job_start = _first_match(r"\bJob\s+Start\s*:?\s*(?P<value>\d{1,2}/\d{1,2}/\d{4})", header)
    if job_start:
        fields["job_start"] = job_start

    status = _first_match(r"\b(?P<value>DRILLING\s+[A-Z][A-Z ]{2,})\b", header, flags=0)
    if status:
        fields["report_status"] = status

    contractor = _first_match(r"\bContractor\s*:?\s*(?P<value>[A-Za-z0-9 &./-]+)", header)
    if contractor:
        fields["contractor"] = contractor

    rig = _first_match(r"\bRig\s*:?\s*(?P<value>[A-Za-z0-9 &./-]+)", header)
    if rig:
        fields["rig"] = rig

    water_depth = _first_match(r"\bWater\s+Depth\s*\(ft\)\s*:?\s*(?P<value>[\d,.]+)", header)
    if water_depth:
        fields["water_depth_ft"] = water_depth

    return fields

