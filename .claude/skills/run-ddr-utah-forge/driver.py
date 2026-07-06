from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PORT = 8502
BASE_URL = f"http://localhost:{PORT}"
PID_FILE = Path("/tmp/ddr_utah_forge_streamlit.pid")
LOG_FILE = Path("/tmp/ddr_utah_forge_streamlit.log")

SIDEBAR_PLACEHOLDER = "e.g. What caused the overpull at frac sleeve #1?"


def _wait_for_server(timeout_s: int = 30) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(BASE_URL, timeout=2)
            return True
        except (urllib.error.URLError, ConnectionError):
            time.sleep(1)
    return False


def cmd_serve(_args: argparse.Namespace) -> int:
    if PID_FILE.exists():
        print(f"Server may already be running (pid file exists at {PID_FILE}). Run 'stop' first if stale.")
        return 1
    log = LOG_FILE.open("w")
    proc = subprocess.Popen(
        [
            "streamlit", "run", "app/ui/ddr_intelligence.py",
            "--server.port", str(PORT),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=REPO_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    PID_FILE.write_text(str(proc.pid))
    if _wait_for_server():
        print(f"Server up at {BASE_URL} (pid {proc.pid}). Log: {LOG_FILE}")
        return 0
    print(f"Server did not respond within timeout. Check {LOG_FILE}.")
    return 1


def cmd_stop(_args: argparse.Namespace) -> int:
    if not PID_FILE.exists():
        print("No pid file found; nothing to stop.")
        return 0
    pid = int(PID_FILE.read_text().strip())
    subprocess.run(["kill", str(pid)], check=False)
    subprocess.run(["pkill", "-f", "streamlit run app/ui/ddr_intelligence.py"], check=False)
    PID_FILE.unlink(missing_ok=True)
    print(f"Stopped server (pid {pid}).")
    return 0


def _sidebar(page):
    # Scoped to avoid strict-mode ambiguity: a sidebar label like "Campaign
    # Summary" also appears as that page's own <h2> heading once selected
    # (it's the default landing page), so an unscoped get_by_text matches both.
    return page.locator("section[data-testid='stSidebar']")


def _new_page(playwright):
    browser = playwright.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 1200})
    errors: list[str] = []
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    return browser, page, errors


def cmd_shot(args: argparse.Namespace) -> int:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser, page, errors = _new_page(p)
        page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(2000)
        if args.page and args.page.lower() != "default":
            _sidebar(page).get_by_text(args.page, exact=False).click()
            page.wait_for_timeout(2500)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(out), full_page=True)
        browser.close()
        print(f"Screenshot saved to {out}")
        if errors:
            print("Console errors:", errors)
            return 1
        return 0


def cmd_search(args: argparse.Namespace) -> int:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser, page, errors = _new_page(p)
        page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(2000)
        _sidebar(page).get_by_text("Corpus Search", exact=False).click()
        page.wait_for_timeout(2000)
        page.get_by_placeholder(SIDEBAR_PLACEHOLDER).fill(args.query)
        page.keyboard.press("Enter")
        page.wait_for_timeout(15000)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(out), full_page=True)
        browser.close()
        print(f"Screenshot saved to {out}")
        if errors:
            print("Console errors:", errors)
            return 1
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Driver for the DDR Operational Intelligence Streamlit app.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("serve", help="Start the Streamlit server in the background.").set_defaults(func=cmd_serve)
    sub.add_parser("stop", help="Stop the background Streamlit server.").set_defaults(func=cmd_stop)

    p_shot = sub.add_parser("shot", help="Navigate to a sidebar page and screenshot it.")
    p_shot.add_argument("page", help="Sidebar label substring (e.g. 'Campaign Summary', 'Corpus Search'), or 'default' for the initial page.")
    p_shot.add_argument("out", help="Output PNG path.")
    p_shot.set_defaults(func=cmd_shot)

    p_search = sub.add_parser("search", help="Run a Corpus Search query and screenshot the results.")
    p_search.add_argument("query", help="Search query text.")
    p_search.add_argument("out", help="Output PNG path.")
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
