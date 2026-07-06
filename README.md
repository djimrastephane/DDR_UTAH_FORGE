# DDR_UTAH_FORGE

Utah FORGE DDR intelligence project scaffolded from the `DDR_RAG_Pipeline` architecture.

This repository keeps the same separation of concerns as the reference pipeline:

```text
app/          Streamlit UI and FastAPI API
configs/      DDR profiles, vocabulary, and project defaults
data/raw/     source Utah FORGE PDF reports
data/processed/
              generated extraction outputs
data/fields/  field/well manifests and combined analysis artefacts
data/graphs/  operational graph and causality outputs (scripts/build_graphs.py, scripts/build_causality.py)
demos/        LinkedIn/marketing demo assets, not part of the production dashboard
docs/         implementation notes
scripts/      command-line pipeline entry points
src/          reusable extraction, retrieval, and analytics code
tests/        focused regression tests
```

## Project Defaults

- Field/project: `UtahForge`
- Wellbore: `FORGE-16A-78-32`
- Raw PDFs: `data/raw/`
- DDR profile: `configs/ddr_profiles/utah_forge.yaml`
- Well manifest: `data/fields/UtahForge/wells/FORGE-16A-78-32/ddr_ids.txt`

The filename QA/QC layer supports the existing Utah FORGE filenames, including compact date pairs such as `11920201192020` and duplicate copy markers such as `reporttmp 2.pdf`. Generated document IDs are stable and unique per source filename.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Validate Raw PDFs

```bash
python scripts/qc_raw_ddr_filenames.py --skip-pdf-open-check
```

Outputs are written to `data/processed/qc/`:

- `raw_pdf_manifest.csv`
- `raw_pdf_issues.csv`
- `raw_pdf_missing_reports.csv`
- `raw_pdf_qc_summary.json`

## Run The Pipeline

```bash
python scripts/batch_preprocess_raw_ddrs.py --build-index
python scripts/extract_wellbore_events.py
python scripts/extract_weather.py
python scripts/extract_completion_string.py
python scripts/extract_frac_sleeve_status.py
python scripts/rebuild_field_analysis.py
```

`data/raw/` and `data/processed/` are gitignored - neither the source PDFs nor
the generated outputs are distributed via this repo. Anyone working with this
project needs the raw PDFs placed in `data/raw/` first, then regenerates
`data/processed/` locally with the command above (fast: ~1s/document). To
force a full reprocessing of documents that already have output (e.g. after a
pipeline change like the extraction backend swap in #1), add `--no-resume`.

## Launch The Dashboard

```bash
bash scripts/run_ddr_intelligence.sh
```

The Streamlit app runs on `http://localhost:8502` by default.

## `src/rag_pdf/` vs `src/ddr_rag/`

Two layers, kept separate on purpose:

- `src/rag_pdf/` is the generic, report-agnostic PDF pipeline: page extraction, OCR fallback, region/table detection, boilerplate stripping, chunking, and the hybrid BM25 + dense search service. It has no notion of "DDR" and would work on any PDF corpus. `scripts/preprocess_hybrid.py` and `scripts/ask_query.py` / `scripts/run_eval.py` drive it directly.
- `src/ddr_rag/` is the Utah FORGE / DDR-specific domain layer built on top of that output: filename QC, DDR header/section parsing, extractor registry, NPT classification, causality analysis, and graph building. `scripts/batch_preprocess_raw_ddrs.py`, `scripts/extract_*.py`, `scripts/build_graphs.py`, and `scripts/build_causality.py` drive it.

They are not merged because `rag_pdf` is meant to stay reusable for a future non-Utah-FORGE corpus; `ddr_rag` is where anything specific to this well's report format belongs.

## Notes

Some reference/demo scripts and docs from `DDR_RAG_Pipeline` are retained because they are useful architecture examples. Treat anything mentioning the original North Sea corpus, `Ensco120`, `JRP`, `ThetisField`, or `Block-A` as reference material unless it has been explicitly adapted for Utah FORGE.

## License

[MIT](LICENSE).
