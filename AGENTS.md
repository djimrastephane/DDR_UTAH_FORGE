# Project Instructions

## Objective

Build and maintain the Utah FORGE DDR intelligence pipeline using the architecture inherited from `DDR_RAG_Pipeline`.

The immediate corpus is the Utah FORGE `FORGE-16A-78-32` drilling/completion report set under `data/raw/`.

## Architecture

Keep this repository structure:

```text
DDR_UTAH_FORGE/
├── app/          # Streamlit UI and FastAPI API
├── configs/      # DDR profiles, vocabulary, defaults
├── data/         # raw PDFs and generated artefacts
├── dashboards/   # dashboard exports/assets
├── docs/         # implementation notes
├── notebooks/    # exploratory work only
├── scripts/      # command-line workflows
├── src/          # reusable production logic
└── tests/        # regression tests
```

Rules:

- Keep production logic in `src/`.
- Keep CLI orchestration in `scripts/`.
- Keep UI/API code in `app/`.
- Keep generated outputs under `data/processed/`, `data/global_index/`, `data/graphs/`, or `data/fields/`.
- Do not mutate source PDFs in `data/raw/`.
- Preserve traceability from every extracted record back to source filename, document ID, page, section, and row/chunk where available.

## Utah FORGE Filename Handling

Current filenames look like:

```text
Utah_Forge_FORGE_16A_(78)-32_Drilling-C_01032021_01032021_15_1_reporttmp 2.pdf
Utah_Forge_FORGE_16A_[78]-32_Drilling-C_11920201192020_11_reporttmp.pdf
Utah_Forge_FORGE_16A_(78)-32_Completion-C_01062021_01062021_1_tmp.pdf
```

Interpretation:

- Project/field: `UtahForge`
- Wellbore: `FORGE-16A-78-32`
- Phase source token: `Drilling-C` or `Completion-C`
- Date tokens may be padded (`01032021_01032021`) or compact repeated pairs (`11920201192020` = `2020-11-09`)
- Copy markers such as ` 2` and source revision markers such as `_1` are source-file variants, not separate wells

The filename parser in `src/ddr_rag/filename_qc.py` creates unique stable document IDs with this shape:

```text
UtahForge-DDR-FORGE-16A-78-32-Drilling-2021-01-03-R015-<hash>
```

Duplicate same-day files are warnings, not blockers, because the raw folder currently contains multiple variants/copies for several dates.

## Defaults

- Main profile: `configs/ddr_profiles/utah_forge.yaml`
- Project config: `configs/ddr_rag.yaml`
- Field directory: `data/fields/UtahForge`
- Well manifest: `data/fields/UtahForge/wells/FORGE-16A-78-32/ddr_ids.txt`
- Dashboard entry point: `app/ui/ddr_intelligence.py`

## Development Standards

- Use Python 3.11+.
- Prefer `pathlib` for file handling.
- Add focused tests for parsing, extraction, and transformation changes.
- Keep broad architectural examples from the reference project only when they remain useful; do not treat old `Ensco120`, `JRP`, `ThetisField`, or `Block-A` references as Utah FORGE defaults.
