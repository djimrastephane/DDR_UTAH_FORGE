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

## Launch The Dashboard

```bash
bash scripts/run_ddr_intelligence.sh
```

The Streamlit app runs on `http://localhost:8502` by default.

## Notes

Some reference/demo scripts and docs from `DDR_RAG_Pipeline` are retained because they are useful architecture examples. Treat anything mentioning the original North Sea corpus, `Ensco120`, `JRP`, `ThetisField`, or `Block-A` as reference material unless it has been explicitly adapted for Utah FORGE.
