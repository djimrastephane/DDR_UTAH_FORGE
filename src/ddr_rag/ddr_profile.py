from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import yaml as _yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

_REPO_ROOT    = Path(__file__).resolve().parents[2]
_PROFILES_DIR = _REPO_ROOT / "configs" / "ddr_profiles"


@dataclass
class DDRProfile:
    name:        str = "Default"
    description: str = ""
    source_file: str = ""

    filename_re:      re.Pattern | None = None
    date_format:      str = "%d.%m.%Y"
    doc_id_template:  str = "{rig}-DDR-{ddr_number:03d}-{date_iso}"

    wellbore_block_field:  str        = "asset_or_project"
    wellbore_suffix_labels: dict[str, str] = field(default_factory=dict)

    header_top_max_y:    float = 80.0
    header_grid_band:    tuple[float, float] = (75.0, 200.0)
    header_narr_band:    tuple[float, float] = (180.0, 360.0)
    header_field_labels: dict[str, list[str]] = field(default_factory=dict)

    op_section_headers:    list[str]        = field(default_factory=lambda: ["OPERATION SUMMARY"])
    op_columns:            dict[str, list[str]] = field(default_factory=dict)
    op_required_cols:      list[str]        = field(default_factory=lambda: ["start_time", "dur_hr", "operation"])
    op_shift_sep_pattern:  re.Pattern | None = None
    op_default_shift:      str              = "00:00-12:00"

    phase_order:  list[str]        = field(default_factory=list)
    phase_labels: dict[str, str]   = field(default_factory=dict)
    high_npt_phases:     list[str] = field(default_factory=list)
    benchmark_phases:    list[str] = field(default_factory=list)

    general_notes: dict[str, Any] = field(default_factory=dict)

    _raw: dict = field(default_factory=dict, repr=False)

    def label_phase(self, code: str) -> str:
        return self.phase_labels.get(code, code)

    def phase_sort_key(self, code: str) -> int:
        try:
            return self.phase_order.index(code)
        except ValueError:
            return 999

    def header_field_keywords(self, field_name: str) -> list[str]:
        return self.header_field_labels.get(field_name, [])

    def match_filename(self, filename: str) -> re.Match | None:
        if self.filename_re is None:
            return None
        return self.filename_re.search(filename)

    def op_column_match(self, header_cell: str) -> str | None:
        cell_lower = header_cell.lower().replace("\n", " ").strip()
        for col_name, keywords in self.op_columns.items():
            if any(kw.lower() in cell_lower for kw in keywords):
                return col_name
        return None


def _parse_profile(data: dict, source_file: str) -> DDRProfile:
    p = DDRProfile(_raw=data, source_file=source_file)
    meta = data.get("profile", {})
    p.name        = meta.get("name", "Unknown")
    p.description = meta.get("description", "")

    fn = data.get("filename", {})
    raw_pattern = fn.get("pattern", "")
    # Collapse multi-line YAML block scalars (strip whitespace between lines)
    raw_pattern = re.sub(r"\s+", "", raw_pattern) if raw_pattern else ""
    if raw_pattern:
        flags_str = fn.get("flags", "IGNORECASE")
        flags = 0
        for flag_name in flags_str.split("|"):
            flags |= getattr(re, flag_name.strip(), 0)
        try:
            p.filename_re = re.compile(raw_pattern, flags)
        except re.error as e:
            print(f"[DDRProfile] Warning: bad filename pattern in {source_file}: {e}",
                  file=sys.stderr)
    p.date_format     = fn.get("date_format", "%d.%m.%Y")
    p.doc_id_template = fn.get("doc_id_template", "{rig}-DDR-{ddr_number:03d}-{date_iso}")

    wb = data.get("wellbore", {})
    p.wellbore_block_field   = wb.get("block_field", "asset_or_project")
    p.wellbore_suffix_labels = wb.get("suffix_labels", {})

    hdr = data.get("header", {})
    p.header_top_max_y  = float(hdr.get("top_band_max_y", 80))
    grid = hdr.get("grid_band", [75, 200])
    narr = hdr.get("narrative_band", [180, 360])
    p.header_grid_band  = (float(grid[0]), float(grid[1]))
    p.header_narr_band  = (float(narr[0]), float(narr[1]))
    p.header_field_labels = {
        k: ([v] if isinstance(v, str) else list(v))
        for k, v in hdr.get("field_labels", {}).items()
    }

    ops = data.get("op_summary", {})
    p.op_section_headers   = ops.get("section_headers", ["OPERATION SUMMARY"])
    p.op_required_cols     = ops.get("required_columns", ["start_time", "dur_hr", "operation"])
    p.op_default_shift     = ops.get("default_shift_label", "00:00-12:00")
    raw_sep = ops.get("shift_separator_pattern", r"\d{2}:\d{2}-\d{2}:\d{2}")
    try:
        p.op_shift_sep_pattern = re.compile(raw_sep)
    except re.error:
        p.op_shift_sep_pattern = re.compile(r"\d{2}:\d{2}-\d{2}:\d{2}")
    p.op_columns = {
        k: ([v] if isinstance(v, str) else list(v))
        for k, v in ops.get("columns", {}).items()
    }

    ph = data.get("phases", {})
    p.phase_order        = list(ph.get("order", []))
    p.phase_labels       = dict(ph.get("labels", {}))
    p.high_npt_phases    = list(ph.get("typically_high_npt", []))
    p.benchmark_phases   = list(ph.get("benchmark_phases", []))

    p.general_notes = data.get("general_notes", {})

    return p


def _read_yaml(path: Path) -> dict:
    if not _YAML_OK:
        raise ImportError("PyYAML is required for DDR profiles. Install with: pip install pyyaml")
    with open(path, encoding="utf-8") as f:
        return _yaml.safe_load(f) or {}


@lru_cache(maxsize=32)
def load_profile(name: str) -> DDRProfile:
    path = _PROFILES_DIR / f"{name}.yaml"
    if not path.exists():
        if name != "default":
            return load_profile("default")
        # No default either — return a bare DDRProfile
        return DDRProfile(name="Bare default", source_file="<builtin>")
    return _parse_profile(_read_yaml(path), str(path))


def detect_profile(filename: str) -> DDRProfile:
    if not _PROFILES_DIR.exists():
        return load_profile("default")

    for yaml_path in sorted(_PROFILES_DIR.glob("*.yaml")):
        if yaml_path.stem == "default":
            continue
        profile = load_profile(yaml_path.stem)
        if profile.match_filename(filename) is not None:
            return profile

    return load_profile("default")


def list_profiles() -> list[str]:
    if not _PROFILES_DIR.exists():
        return []
    return [p.stem for p in sorted(_PROFILES_DIR.glob("*.yaml"))]
