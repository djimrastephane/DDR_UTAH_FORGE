from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

OUT_DIR       = repo_root / "data" / "processed" / "qc"
PROCESSED_DIR = repo_root / "data" / "processed"

_SLEEVE_NUM_RE  = re.compile(r"sleeve\s*#?\s*(\d+)", re.I)
_OPEN_RE        = re.compile(r"sleeve\s*#?\s*\d+\s+open\s+at\s+([\d,]+)\s*psi|"
                              r"observed\s+sleeve\s*#?\s*\d+\s+open\s+at\s+([\d,]+)\s*psi", re.I)
_BREAKOVER_RE   = re.compile(r"breakover\s+at\s+([\d,]+)\s*psi", re.I)
_NCS_OPEN_RE    = re.compile(r"NCS\s+confirm(?:ed)?\s+sleeve\s+open|sleeve\s+confirm(?:ed)?\s+open", re.I)
_INJECT_RE      = re.compile(r"inject(?:ed)?\s+([\d,]+\.?\d*)\s*bbls?", re.I)
_PUMP_STAGE_RE  = re.compile(r"staged?\s+(?:up\s+)?(?:rate|flowrate)\s+to\s+([\d.]+)\s*bpm[,\s]+([\d,]+)\s*psi", re.I)
_NO_IND_RE      = re.compile(r"no\s+positive\s+indication|unable\s+to\s+(?:locate|open|find)|"
                              r"no\s+indication\s+of\s+(?:the\s+)?sleeve", re.I)
_PACKER_FAIL_RE = re.compile(r"packer\s+not\s+seal(?:ing)?|packer\s+did\s+not\s+set|"
                              r"packer\s+not\s+set|integrity\s+(?:not\s+)?(?:confirmed|achieved)", re.I)
_ISIP_RE        = re.compile(r"ISIP\D{0,20}?([\d,]+)\s*psi", re.I)
# Proppant / fluid volumes
_PROP_LBS_RE    = re.compile(r"([\d,]+)\s*lbs?\s+(?:of\s+)?(?:2/40|100[\s-]*mesh|proppant|sand\b|resin)", re.I)
_SLURRY_RE      = re.compile(r"([\d,]+)\s*(?:gals?|bbls?)\s+(?:slurry|total\s+clean\s+vol|clean\s+vol|pad\s+vol)", re.I)
_FLUID_BBL_RE   = re.compile(r"([\d,]+)\s*bbls?\s+(?:of\s+)?(?:clean\s+pad|linear\s+gel|crosslink|gel\s+pad|pad)", re.I)
_MAIN_FRAC_RE   = re.compile(r"main\s+frac|carried\s+out\s+(?:main\s+)?frac", re.I)


def _load_compzn_ops() -> pd.DataFrame:
    frames = []
    for doc_dir in sorted(PROCESSED_DIR.iterdir()):
        f = doc_dir / "ddr_facts.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            if "COMPZN" in df.get("phase", pd.Series()).values:
                frames.append(df[df["phase"] == "COMPZN"])
    if not frames:
        raise FileNotFoundError("No COMPZN ddr_facts.parquet found")
    ops = pd.concat(frames, ignore_index=True)
    ops["dt"] = pd.to_datetime(ops["report_date"], dayfirst=True, errors="coerce")
    return ops.sort_values("dt")


def _get_report_no(date: str) -> str:
    try:
        hdr_frames = []
        for doc_dir in PROCESSED_DIR.iterdir():
            h = doc_dir / "ddr_header.parquet"
            if h.exists():
                hdr_frames.append(pd.read_parquet(h, columns=["report_date", "report_no"]))
        if hdr_frames:
            hdr = pd.concat(hdr_frames)
            row = hdr[hdr["report_date"] == date]
            if not row.empty:
                return str(int(row.iloc[0]["report_no"]))
    except Exception:
        pass
    return "?"


def extract_sleeve_status() -> pd.DataFrame:
    ops = _load_compzn_ops()

    # Per-sleeve accumulators
    acc: dict[int, dict] = {n: {
        "status": "NOT FRACKED",
        "open_psi": None,
        "breakover_psi": None,
        "isip_psi": None,
        "max_treat_psi": None,
        "max_rate_bpm": None,
        "vol_injected_bbl": None,
        "proppant_lbs_total": None,
        "fluid_vol_bbl": None,
        "first_date": None,
        "last_date": None,
        "ddr_citation": "",
        "notes": [],
    } for n in range(1, 19)}

    for _, row in ops.iterrows():
        text = str(row.get("operation_text") or "")
        date = str(row.get("report_date", ""))
        page = row.get("page", "?")

        nums = set(
            int(m.group(1)) for m in _SLEEVE_NUM_RE.finditer(text)
            if 1 <= int(m.group(1)) <= 18
        )
        if not nums:
            continue

        for num in nums:
            d = acc[num]

            dt = row.get("dt")
            if dt is not None and pd.notna(dt):
                if d["first_date"] is None or dt < d["first_date"]:
                    d["first_date"] = dt
                    rno = _get_report_no(date)
                    d["ddr_citation"] = f"DDR-{rno} · p.{page} · {date}"
                if d["last_date"] is None or dt > d["last_date"]:
                    d["last_date"] = dt

            m = _OPEN_RE.search(text)
            if m:
                psi = float((m.group(1) or m.group(2)).replace(",", ""))
                if d["open_psi"] is None or psi < d["open_psi"]:
                    d["open_psi"] = psi
                if d["status"] not in ("OPENED",):
                    d["status"] = "OPENED"

            if _NCS_OPEN_RE.search(text):
                d["status"] = "OPENED"

            m = _BREAKOVER_RE.search(text)
            if m:
                psi = float(m.group(1).replace(",", ""))
                if d["breakover_psi"] is None or psi > d["breakover_psi"]:
                    d["breakover_psi"] = psi

            m = _ISIP_RE.search(text)
            if m:
                psi = float(m.group(1).replace(",", ""))
                if d["isip_psi"] is None or psi > d["isip_psi"]:
                    d["isip_psi"] = psi

            for m in _INJECT_RE.finditer(text):
                vol = float(m.group(1).replace(",", ""))
                d["vol_injected_bbl"] = (d["vol_injected_bbl"] or 0) + vol

            # Proppant mass — only count rows that are explicitly a main frac op
            # to avoid double-counting from summary/planning rows
            if _MAIN_FRAC_RE.search(text) or re.search(r"carried\s+out\s+frac|main\s+frac\s+at", text, re.I):
                for m in _PROP_LBS_RE.finditer(text):
                    lbs = float(m.group(1).replace(",", ""))
                    if lbs > 1000:   # filter out minor fluid/mesh slugs
                        d["proppant_lbs_total"] = (d["proppant_lbs_total"] or 0) + lbs
                # Fluid / slurry volumes
                for m in _FLUID_BBL_RE.finditer(text):
                    vol = float(m.group(1).replace(",", ""))
                    d["fluid_vol_bbl"] = (d["fluid_vol_bbl"] or 0) + vol

            for m in _PUMP_STAGE_RE.finditer(text):
                rate = float(m.group(1))
                psi  = float(m.group(2).replace(",", ""))
                if d["max_rate_bpm"] is None or rate > d["max_rate_bpm"]:
                    d["max_rate_bpm"] = rate
                if d["max_treat_psi"] is None or psi > d["max_treat_psi"]:
                    d["max_treat_psi"] = psi

            # only downgrade if not already opened
            if _NO_IND_RE.search(text) and d["status"] == "NOT FRACKED":
                d["status"] = "NO INDICATION"
            if _PACKER_FAIL_RE.search(text) and d["status"] not in ("OPENED",):
                if d["status"] == "NOT FRACKED":
                    d["status"] = "LOCATED"
                d["notes"].append("packer sealing issue")

            if re.search(r"(?:packer set|packer confirmed|located into sleeve|sleeve located)", text, re.I):
                if d["status"] == "NOT FRACKED":
                    d["status"] = "LOCATED"

    for n, d in acc.items():
        if d["proppant_lbs_total"] and d["proppant_lbs_total"] > 0:
            if d["status"] in ("NOT FRACKED", "LOCATED", "NO INDICATION"):
                d["status"] = "OPENED"
                if "proppant injected — sleeve confirmed open" not in d["notes"]:
                    d["notes"].append("proppant injected — sleeve confirmed open")

    for n, d in acc.items():
        reasons = []

        if (d["proppant_lbs_total"] and d["proppant_lbs_total"] > 0
                and d["open_psi"] is None and d["status"] == "OPENED"):
            reasons.append("OPENED inferred from proppant injection only — no opening pressure in DDR text")

        if d["status"] == "OPENED" and d["open_psi"] is not None and not d["proppant_lbs_total"]:
            reasons.append("Opening pressure recorded but no proppant volume extracted — check DDR for frac design")

        if (d["vol_injected_bbl"] and d["vol_injected_bbl"] > 20
                and d["status"] in ("NO INDICATION", "LOCATED")):
            reasons.append(f"Fluid injected ({d['vol_injected_bbl']:.0f} bbl) but sleeve not confirmed open")

        if d["status"] == "LOCATED" and not d["open_psi"] and not d["vol_injected_bbl"]:
            reasons.append("Packer located but no opening/failure outcome recorded — incomplete DDR coverage")

        d["review_flag"] = len(reasons) > 0
        d["review_reason"] = " | ".join(reasons)

    rows = []
    for num in range(1, 19):
        d = acc[num]
        rows.append({
            "sleeve_no":            num,
            "status":               d["status"],
            "open_psi":             d["open_psi"],
            "breakover_psi":        d["breakover_psi"],
            "isip_psi":             d["isip_psi"],
            "max_treat_psi":        d["max_treat_psi"],
            "max_rate_bpm":         d["max_rate_bpm"],
            "vol_injected_bbl":     d["vol_injected_bbl"],
            "proppant_lbs_total":   d["proppant_lbs_total"],
            "fluid_vol_bbl":        d["fluid_vol_bbl"],
            "first_date":           d["first_date"].strftime("%d/%m/%Y") if d["first_date"] else None,
            "last_date":            d["last_date"].strftime("%d/%m/%Y") if d["last_date"] else None,
            "ddr_citation":         d["ddr_citation"],
            "notes":                "; ".join(d["notes"]) if d["notes"] else "",
            "review_flag":          d.get("review_flag", False),
            "review_reason":        d.get("review_reason", ""),
        })

    return pd.DataFrame(rows)


def main() -> None:
    print("Extracting frac sleeve status...")
    df = extract_sleeve_status()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "ddr_frac_sleeve_status.parquet", index=False)
    print(f"Saved {len(df)} sleeves → {OUT_DIR / 'ddr_frac_sleeve_status.parquet'}")
    print()
    print(f"{'#':<4} {'Status':<16} {'Open psi':>9} {'Brkover':>8} {'ISIP':>7} {'Vol bbl':>8} {'Max bpm':>8} {'First date'}")
    print("-" * 80)
    for _, r in df.iterrows():
        def fmt(v): return f"{v:,.0f}" if pd.notna(v) and v else "—"
        print(f"#{r['sleeve_no']:<3} {r['status']:<16} {fmt(r['open_psi']):>9} "
              f"{fmt(r['breakover_psi']):>8} {fmt(r['isip_psi']):>7} "
              f"{fmt(r['vol_injected_bbl']):>8} {fmt(r['max_rate_bpm']):>8}  "
              f"{r['first_date'] or '—'}")


if __name__ == "__main__":
    main()
