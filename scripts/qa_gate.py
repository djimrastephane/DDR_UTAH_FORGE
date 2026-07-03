from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import time
import types
import unittest.mock
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_SRC  = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_PROCESSED_DIR  = _REPO / "data" / "processed"
_QC_DIR         = _REPO / "data" / "processed" / "qc"
_GLOBAL_IDX_DIR = _REPO / "data" / "global_index"
_UI_DIR         = _REPO / "app" / "ui"

_REQUIRED_ARTIFACTS = [
    "chunks.parquet",
    "chunk_meta.parquet",
    "embeddings.npy",
    "faiss.index",
    "ddr_facts.parquet",
    "ddr_header.parquet",
]

# thresholds set at ~85% of known-good counts
_QC_THRESHOLDS: dict[str, int] = {
    "ddr_casing.parquet":            450,  # known good: 533
    "ddr_drilling_metrics.parquet":  780,  # known good: 920
    "ddr_personnel.parquet":        4100,  # known good: 4852
    "ddr_pressure_tests.parquet":    460,  # known good: 540
    "ddr_wellbore_events.parquet":   185,  # known good: 221
    "ddr_weather.parquet":            57,  # known good: 68
    "ddr_vessels.parquet":           250,  # known good: 296
    "ddr_planned_time.parquet":       88,  # known good: 104
    "ddr_completion_string.parquet":  24,  # known good: 29
    "ddr_frac_sleeve_status.parquet": 15,  # known good: 18
    "ddr_mud_data.parquet":           87,  # known good: 103
    "ddr_ditch_magnets.parquet":     145,  # known good: 171
    "ddr_fit_lot_results.parquet":     4,  # known good: 6
}

_PAGE_MODULES = [
    "campaign_summary",
    "field_analysis",
    "well_overview",
    "well_schematic",
    "npt_intelligence",
    "wellbore_events",
    "completion_string",
    "cost_analysis",
    "operation_sequence",
    "drilling_metrics",
    "eowr",
    "upload_ddrs",
    "operations_log",
    "operational_graph",
    "causality",
    "lessons_learned",
    "corpus_search",
]


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: list[str] = field(default_factory=list)
    skipped: bool = False


def check_filename_qc(skip_pdf_open: bool) -> CheckResult:
    from ddr_rag.filename_qc import audit_raw_pdfs

    raw_dir = _REPO / "data" / "raw"
    if not raw_dir.exists():
        return CheckResult("Filename QC", False, [f"raw dir not found: {raw_dir}"])

    audit = audit_raw_pdfs(
        str(raw_dir),
        block_id="FORGE",
        check_pdf_readability=not skip_pdf_open,
    )
    s = audit.summary()
    detail = [
        f"{s['pdf_count']} PDFs, {s['parsed_count']} parsed, "
        f"{s['blocking_issue_count']} blocking, {s['warning_count']} warnings",
    ]
    if s["missing_report_count"]:
        detail.append(f"{s['missing_report_count']} missing DDR numbers (gaps in sequence)")

    passed = s["blocking_issue_count"] == 0
    if not passed:
        for issue in getattr(audit, "issues", []):
            if getattr(issue, "is_blocking", False):
                detail.append(f"  BLOCKING: {issue}")
    return CheckResult("Filename QC", passed, detail)


def check_artifact_coverage() -> CheckResult:
    doc_dirs = [
        d for d in sorted(_PROCESSED_DIR.iterdir())
        if d.is_dir() and d.name != "qc"
    ]
    missing: list[str] = []
    for doc_dir in doc_dirs:
        absent = [f for f in _REQUIRED_ARTIFACTS if not (doc_dir / f).exists()]
        if absent:
            missing.append(f"{doc_dir.name}: missing {', '.join(absent)}")

    detail = [f"{len(doc_dirs)} doc dirs checked, {len(missing)} incomplete"]
    detail += missing[:10]
    if len(missing) > 10:
        detail.append(f"  … and {len(missing) - 10} more")

    return CheckResult("Artifact coverage", passed=len(missing) == 0, detail=detail)


def check_qc_parquets() -> CheckResult:
    failures: list[str] = []
    summary_lines: list[str] = []

    for fname, threshold in _QC_THRESHOLDS.items():
        path = _QC_DIR / fname
        if not path.exists():
            failures.append(f"{fname}: file missing")
            continue
        n = len(pd.read_parquet(path))
        status = "ok" if n >= threshold else f"BELOW THRESHOLD ({n} < {threshold})"
        summary_lines.append(f"{fname}: {n} rows [{status}]")
        if n < threshold:
            failures.append(summary_lines[-1])

    detail = [f"{len(_QC_THRESHOLDS)} tables checked, {len(failures)} below threshold"]
    detail += failures if failures else [f"all ≥ thresholds"]
    return CheckResult("QC parquets", passed=len(failures) == 0, detail=detail)


def check_global_index() -> CheckResult:
    issues: list[str] = []
    for fname in ("faiss.index", "embeddings.npy", "chunk_meta.parquet", "chunks.parquet"):
        if not (_GLOBAL_IDX_DIR / fname).exists():
            issues.append(f"{fname} missing")

    if issues:
        return CheckResult("Global index", False, issues)

    emb   = np.load(_GLOBAL_IDX_DIR / "embeddings.npy", mmap_mode="r")
    meta  = pd.read_parquet(_GLOBAL_IDX_DIR / "chunk_meta.parquet")
    n_emb, dim = emb.shape
    n_meta = len(meta)

    detail = [f"{n_emb:,} chunks, embedding shape ({n_emb}, {dim})"]

    if n_emb != n_meta:
        issues.append(f"shape mismatch: embeddings {n_emb} vs chunk_meta {n_meta}")
    if dim != 384:
        issues.append(f"unexpected embedding dim {dim} (expected 384)")
    if n_emb < 13_000:
        issues.append(f"chunk count suspiciously low: {n_emb} (expected ≥ 13,000)")

    detail += issues
    return CheckResult("Global index", passed=len(issues) == 0, detail=detail)


def check_tests() -> CheckResult:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=str(_REPO),
    )
    lines = (result.stdout + result.stderr).strip().splitlines()
    summary = next((l for l in reversed(lines) if "passed" in l or "failed" in l), "")
    detail = [summary] if summary else lines[-3:]

    failed_lines = [l for l in lines if l.startswith("FAILED")]
    detail += failed_lines[:10]

    return CheckResult("Test suite", passed=result.returncode == 0, detail=detail)


def check_ui_imports() -> CheckResult:
    # Build a minimal set of stubs so imports don't require a running server
    # or optional heavy libraries (faiss, sentence_transformers, plotly).
    # Use MagicMock-backed modules so any attribute access (e.g.
    # `from plotly.subplots import make_subplots`) resolves without error.
    _stub_names = [
        "streamlit", "streamlit.components", "streamlit.components.v1",
        "plotly", "plotly.graph_objects", "plotly.subplots", "plotly.express",
        "faiss", "sentence_transformers",
        "networkx",
    ]
    saved: dict[str, object] = {}
    for mod_name in _stub_names:
        saved[mod_name] = sys.modules.get(mod_name)
        stub = types.ModuleType(mod_name)
        # Delegate all attribute lookups to a MagicMock so `from x import y` works
        stub.__class__ = type(
            "_MockModule",
            (types.ModuleType,),
            {"__getattr__": lambda self, name: unittest.mock.MagicMock()},
        )
        sys.modules[mod_name] = stub

    if str(_UI_DIR) not in sys.path:
        sys.path.insert(0, str(_UI_DIR))

    failures: list[str] = []
    for mod_name in _PAGE_MODULES:
        full = f"page_modules.{mod_name}"
        # Remove any cached version so we get a fresh import
        sys.modules.pop(full, None)
        try:
            importlib.import_module(full)
        except Exception as exc:
            failures.append(f"{mod_name}: {type(exc).__name__}: {exc}")

    for mod_name, original in saved.items():
        if original is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = original  # type: ignore[assignment]

    total = len(_PAGE_MODULES)
    detail = [f"{total - len(failures)}/{total} page modules imported cleanly"]
    detail += failures
    return CheckResult("UI imports", passed=len(failures) == 0, detail=detail)


def check_smoke(port: int) -> CheckResult:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import]
    except ImportError:
        return CheckResult(
            "Smoke test", passed=True, skipped=True,
            detail=["playwright not installed — run: pip install playwright && playwright install chromium"],
        )

    launch_cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(_REPO / "app" / "ui" / "ddr_intelligence.py"),
        "--server.port", str(port),
        "--server.headless", "true",
        "--server.runOnSave", "false",
    ]
    proc = subprocess.Popen(
        launch_cmd, cwd=str(_REPO),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base_url = f"http://localhost:{port}"
    failures: list[str] = []

    try:
        import urllib.request, urllib.error
        for _ in range(30):
            try:
                urllib.request.urlopen(base_url, timeout=1)
                break
            except Exception:
                time.sleep(1)
        else:
            return CheckResult("Smoke test", False, ["Streamlit server did not start within 30 s"])

        pages_to_check = [
            "📊 Campaign Summary",
            "🔩 Well Overview",
            "📈 NPT Intelligence",
            "⚡ Wellbore Events",
            "🧵 Completion String",
            "🔄 Operation Sequence",
            "📊 Drilling Metrics",
            "🕸 Operational Graph",
            "🔗 Cross-Phase Causality",
            "🔍 Corpus Search",
        ]

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(base_url, timeout=20_000)
            page.wait_for_load_state("networkidle", timeout=20_000)

            for label in pages_to_check:
                try:
                    page.get_by_label(label).or_(
                        page.get_by_text(label, exact=True)
                    ).first.click(timeout=5_000)
                    page.wait_for_load_state("networkidle", timeout=15_000)
                    # Fail if Streamlit renders an uncaught exception traceback
                    if page.locator("[data-testid='stException']").count() > 0:
                        failures.append(f"{label}: uncaught exception on page")
                except Exception as exc:
                    failures.append(f"{label}: {exc}")

            browser.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    detail = [f"{len(pages_to_check)} pages navigated, {len(failures)} errors"]
    detail += failures
    return CheckResult("Smoke test", passed=len(failures) == 0, detail=detail)


_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _icon(result: CheckResult) -> str:
    if result.skipped:
        return f"{_YELLOW}─{_RESET}"
    return f"{_GREEN}✓{_RESET}" if result.passed else f"{_RED}✗{_RESET}"


def print_results(results: list[CheckResult]) -> None:
    width = 56
    print(f"\n{_BOLD}{'═' * width}{_RESET}")
    print(f"{_BOLD}  DDR Pipeline QA Gate{_RESET}")
    print(f"{_BOLD}{'═' * width}{_RESET}")

    for r in results:
        icon   = _icon(r)
        label  = r.name.ljust(20)
        first  = r.detail[0] if r.detail else ("skipped" if r.skipped else "")
        prefix = "skipped" if r.skipped else ""
        print(f"  {icon} {label}  {prefix or first}")
        for line in r.detail[1:]:
            print(f"       {' ' * 20}  {line}")

    print(f"{_BOLD}{'─' * width}{_RESET}")
    failed = [r for r in results if not r.passed and not r.skipped]
    if failed:
        print(f"  {_RED}{_BOLD}❌  NOT READY — {len(failed)} check(s) failed{_RESET}")
    else:
        print(f"  {_GREEN}{_BOLD}✅  READY TO DEMO{_RESET}")
    print(f"{_BOLD}{'═' * width}{_RESET}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DDR pipeline QA gate — verifies pipeline health and app readiness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Run Playwright browser navigation smoke test (requires playwright).",
    )
    p.add_argument(
        "--skip-pdf-open", action="store_true",
        help="Skip PyMuPDF readability check in filename QC (faster for CI).",
    )
    p.add_argument(
        "--smoke-port", type=int, default=8599,
        help="Port to start Streamlit on for the smoke test (default: 8599).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    checks = [
        ("Filename QC",       lambda: check_filename_qc(args.skip_pdf_open)),
        ("Artifact coverage", check_artifact_coverage),
        ("QC parquets",       check_qc_parquets),
        ("Global index",      check_global_index),
        ("Test suite",        check_tests),
        ("UI imports",        check_ui_imports),
    ]
    if args.smoke:
        checks.append(("Smoke test", lambda: check_smoke(args.smoke_port)))

    results: list[CheckResult] = []
    for name, fn in checks:
        print(f"  running: {name} …", end="\r", flush=True)
        try:
            result = fn()
        except Exception as exc:
            result = CheckResult(name, False, [f"unexpected error: {exc}"])
        results.append(result)

    print_results(results)
    return 0 if all(r.passed or r.skipped for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
