# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and version numbers follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [2.0.0] - 2026-07-22

First tracked release. No versions were tagged before this point, so this
entry captures the project's full history to date, from the initial import
through the current README/dashboard state.

### Added

- Initial Utah FORGE DDR intelligence pipeline: filename QC, hybrid PDF
  preprocessing with OCR fallback, domain fact extraction, NPT
  classification, and the Streamlit dashboard.
- CI: test suite, import checks, API startup check, and a scheduled
  `pip-audit` dependency scan.
- Anchor-cropped `pdfplumber` table extraction, replacing `camelot-py`
  (fixes silent numeric corruption in extracted table facts and a
  transitive `pypdf` CVE chain).
- `run-ddr-utah-forge` agent skill for driving and screenshotting the
  Streamlit dashboard.
- Operational Graph: real op_code hours/NPT breakdown.
- Campaign Summary: Superintendent Takeaways, equipment sub-cause
  classification, and Flat Time Reconciliation.
- Root Cause & Action Tracking annotation feature.
- MIT license.

### Changed

- Operation Sequence page: UX rework and visual consistency pass.
- Well Performance Chart: taller layout, phase labels moved to a separate
  timeline row, "Theoretical (no NPT)" renamed to "Linear reference
  (no-delay estimate)", stronger >50%-NPT-day markers, callouts on the
  largest flat sections, and "Days needing review" expanded by default.
- Drilling Metrics: elegant coverage summary shown when data is sparse.
- Removed dead/broken UI pages; surfaced Operational Graph and Lessons
  Learned in the sidebar.
- README rewritten with a 30-second overview, dashboard screenshots, a
  pipeline architecture diagram, and a grounded Limitations section.
- Demo-only assets moved out of `app/ui/` into `demos/`.
- Dependency bumps: GitHub Actions to node24-native versions, `torch`
  2.12.1 → 2.13.0.

### Fixed

- Lessons Learned phase filter and "Show source DDR excerpts".
- Negative-width bars in Well Overview's Daily Drill-Down gantt chart.
- Operation ordering and gantt/graph row ordering for the 06:00→06:00
  DDR reporting day.
- Windowed graph outputs colliding on filename (`w2_graph.json`).
- NPT row highlighting on the Programme Steps table.
- Phase Performance chart color/legend mismatch.
- DDR citation truncation and Operations Log duration formatting.
- Corpus Search: `report_no`/page/table metadata not reaching the UI.
- Duplicate daily-NPT calculation on the Operation Sequence page.
- An f-string syntax error incompatible with Python 3.11.

### Security

- Patched Dependabot-flagged CVEs across `torch`, `transformers`,
  `pyarrow`, `streamlit`, `requests`, `pytest`.
- Removed `camelot-py`/`ghostscript` entirely, closing the transitive
  `pypdf` CVE chain they pulled in.
- Sanitized `well_id`/`rig_name` inputs on the Upload DDRs page.

### Removed

- The LinkedIn demo feature and its real-entity redaction map.
- Residual references to the original reference engagement (client/rig
  identifiers, vocabulary entries, example queries) from docs, tests,
  and loaders.
- Orphaned `data/graphs_w2/` output directory.
