from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from ddr_rag.causality_analyzer import run_causality_analysis, PHASE_LABELS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(repo_root / "data" / "processed"))
    parser.add_argument("--out",      default=str(repo_root / "data" / "graphs" / "causality.json"))
    args = parser.parse_args()

    print("Running cross-phase causality analysis...")
    report = run_causality_analysis(Path(args.data_dir))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))

    print()
    print("=== Phase timeline ===")
    for row in report["timeline"]:
        print(f"  {row['phase']:<10} {row['date_start']} → {row['date_end']}  "
              f"NPT={row['npt_pct']:.0f}%  {row['npt_hrs']:.0f}h/{row['total_hrs']:.0f}h")

    print()
    print("=== Transition causality ===")
    for t in report["transitions"]:
        arrow = f"{t['phase_from']} → {t['phase_to']}"
        npt_arrow = f"{t['pre_npt_pct']:.0f}% → {t['post_npt_pct']:.0f}%"
        print(f"\n  {arrow}  ({npt_arrow} NPT,  {len(t['causal_terms'])} causal terms,  {len(t['escalating_terms'])} escalating)")
        for c in t["causal_terms"][:6]:
            esc = " ↑ESCALATING" if c["is_escalating"] else ""
            print(f"    {c['term']:<25} npt {c['pre_npt_ratio']:.0%}→{c['post_npt_ratio']:.0%}  "
                  f"freq×{c['freq_ratio']:.1f}  {c['post_npt_hrs']:.0f}h NPT{esc}")

    print()
    print("=== Key operational narratives ===")
    for t in report["transitions"]:
        print(f"\n  {t['phase_from']} → {t['phase_to']}:")
        print(f"    {t['narrative']}")

    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
