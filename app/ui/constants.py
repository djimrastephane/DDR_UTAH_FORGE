from __future__ import annotations

import sys
from pathlib import Path

repo_root: Path = Path(__file__).resolve().parents[2]

# adds src/ to sys.path so ddr_rag is importable throughout the app
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

PROCESSED_DIR: Path = repo_root / "data" / "processed"
GRAPHS_DIR: Path = repo_root / "data" / "graphs"
FIELD_DIR: Path = repo_root / "data" / "fields" / "UtahForge"

PHASE_ORDER: list[str] = [
    "No Activity",
    "Rig Move In",
    "Surface Drilling",
    "Surface Casing",
    "Intermediate Drilling",
    "Intermediate Casing",
    "Production Drilling",
    "Production Casing",
    "Drillout",
    "DRILLING",
    "COMPLETION",
    "MIRU",
    "COND1",
    "INTRM1",
    "INTRM2",
    "PROD1",
    "COMPZN",
]

PHASE_COLOURS: dict[str, str] = {
    "No Activity": "#9E9E9E",
    "Rig Move In": "#546E7A",
    "Surface Drilling": "#1976D2",
    "Surface Casing": "#5C6BC0",
    "Intermediate Drilling": "#388E3C",
    "Intermediate Casing": "#689F38",
    "Production Drilling": "#E65100",
    "Production Casing": "#F57C00",
    "Drillout": "#00838F",
    "DRILLING": "#1976D2",
    "COMPLETION": "#00838F",
    "MIRU":   "#546E7A",
    "COND1":  "#1976D2",
    "INTRM1": "#388E3C",
    "INTRM2": "#7B1FA2",
    "PROD1":  "#E65100",
    "COMPZN": "#00838F",
}

WELL_COLOURS: dict[str, str] = {
    "FORGE-16A-78-32": "#1976D2",
    "Block-A-W1": "#1565C0",
    "Block-A-W2": "#2E7D32",
}

_COMP_COLOURS: dict[str, str] = {
    "tubing_hanger":       "#1565C0",
    "tubing":              "#90CAF9",
    "dhsv":                "#D32F2F",
    "crossover":           "#7B1FA2",
    "gauge_mandrel":       "#00838F",
    "production_packer":   "#E65100",
    "liner_hanger_packer": "#F57F17",
    "liner":               "#B0BEC5",
    "frac_sleeve":         "#FF8F00",
    "float_shoe":          "#455A64",
}

_COMP_LABELS: dict[str, str] = {
    "tubing_hanger":       "Tubing Hanger",
    "tubing":              "Tubing",
    "dhsv":                "DHSV",
    "crossover":           "Crossover",
    "gauge_mandrel":       "Gauge Mandrel",
    "production_packer":   "Production Packer",
    "liner_hanger_packer": "Liner Hanger Packer",
    "liner":               "Liner",
    "frac_sleeve":         "Frac Sleeve",
    "float_shoe":          "Float Shoe",
}
