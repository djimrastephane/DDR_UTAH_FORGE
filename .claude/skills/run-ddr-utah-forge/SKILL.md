---
name: run-ddr-utah-forge
description: Build, run, and drive the DDR_UTAH_FORGE Streamlit dashboard (app/ui/ddr_intelligence.py). Use when asked to start the app, launch the dashboard, take a screenshot of a page, run a Corpus Search query, or confirm a change works in the real running app (not just tests).
---

This is a Streamlit app (`app/ui/ddr_intelligence.py`), a server rendering
a JS client, not a static page - `curl` only proves the process is up, not
that a page rendered. Drive it with the Python Playwright script at
`.claude/skills/run-ddr-utah-forge/driver.py`, which launches headless
Chromium, navigates the sidebar, and screenshots the result. All paths
below are relative to the repo root.

## Prerequisites

The app's own dependencies come from `requirements.txt` (already covers
`streamlit`, `pandas`, etc. - see repo README for setup). Driving/
screenshotting it additionally needs Playwright, which is *not* a project
dependency - it's agent tooling for this skill only:

```bash
pip show playwright   # if this fails:
pip install playwright
python3 -m playwright install chromium
```

Verified in this container with Playwright 1.60.0; Chromium was already
cached at `~/Library/Caches/ms-playwright/` from a prior install, so
`playwright install chromium` wasn't needed here - run it if `driver.py`
fails with an executable-not-found error.

## Run (agent path)

```bash
python3 .claude/skills/run-ddr-utah-forge/driver.py serve
python3 .claude/skills/run-ddr-utah-forge/driver.py shot "Campaign Summary" /tmp/shots/campaign_summary.png
python3 .claude/skills/run-ddr-utah-forge/driver.py search "What caused the NPT on the conductor section?" /tmp/shots/search.png
python3 .claude/skills/run-ddr-utah-forge/driver.py stop
```

`serve` starts Streamlit on port 8502 in the background (pid file at
`/tmp/ddr_utah_forge_streamlit.pid`, log at
`/tmp/ddr_utah_forge_streamlit.log`) and blocks until it responds to
HTTP, so there's no fixed sleep to tune. Each of `shot`/`search` opens
its own fresh browser context against the already-running server - `shot`
takes a sidebar label substring (or `default` for the initial page,
which is Campaign Summary) and screenshots it; `search` goes to Corpus
Search specifically, fills the query box, presses Enter, waits ~15s for
the embedding model to load on first query, and screenshots the results.
Both print any browser console errors to stdout and exit 1 if there were
any - check for that, not just that a PNG exists.

| command | what it does |
|---|---|
| `serve` | Start Streamlit in the background; blocks until it's responding |
| `stop` | Kill the background server and remove the pid file |
| `shot <sidebar-label\|default> <out.png>` | Click a sidebar item, screenshot the page |
| `search <query> <out.png>` | Run a Corpus Search query, screenshot the results |

Sidebar labels seen in this app: `Campaign Summary`, `Well Overview`,
`NPT Intelligence`, `Operation Sequence`, `Drilling Metrics`,
`Operations Log`, `Upload DDRs`, `Corpus Search`.

## Run (human path)

```bash
bash scripts/run_ddr_intelligence.sh   # → opens http://localhost:8502. Ctrl-C to stop.
```

## Test

```bash
python -m pytest -q
```

74 tests pass (as of 2026-07-06); raw-PDF-fixture-dependent tests skip
if `data/raw/` isn't populated (it's gitignored - see README).

---

## Gotchas

- **`--help` on some project scripts actually runs the script.** Not
  this driver, but a real trap hit in this repo:
  `scripts/build_global_index.py` has no argparse, so passing `--help`
  to it just executes normally instead of showing usage. Don't assume
  `--help` is safe on scripts here without checking first.
- **First Corpus Search query is slow (~3-15s)** - it loads the
  sentence-transformers embedding model on first use per Streamlit
  session. The driver's `search` command already waits 15s to cover
  this; a fresh custom script that waits less will screenshot a
  "Searching..." spinner instead of results.
- **Corpus Search's placeholder text is the fill target**, not a label
  or test-id - `driver.py` uses
  `page.get_by_placeholder("e.g. What caused the overpull at frac sleeve #1?")`.
  If that placeholder copy changes, the `search` command breaks
  silently (fill() will raise a timeout, not a clear error).

## Troubleshooting

- **`serve` prints "Server may already be running"**: a stale pid file
  exists at `/tmp/ddr_utah_forge_streamlit.pid` from a previous session
  that didn't call `stop`. Run `stop` first (safe even if the process
  is already dead), or `rm /tmp/ddr_utah_forge_streamlit.pid` directly.
- **`shot`/`search` hang or time out with no server running**: `serve`
  wasn't run first, or died - check `/tmp/ddr_utah_forge_streamlit.log`.
