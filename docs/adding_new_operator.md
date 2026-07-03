# Adding a New Operator or Report Format

This guide explains how to onboard a new operator, a new well from an existing operator, or a completely different DDR format. It is written for a drilling data engineer with basic Python and YAML familiarity.

---

## Architecture overview

The pipeline separates **format-specific extraction** from **format-agnostic analysis**. All format assumptions live in a single YAML profile file. The NLP, graph, retrieval, and comparison layers are fully portable and require no changes when onboarding a new operator.

```
configs/ddr_profiles/
  operator_alpha.yaml   ← current operator (North Sea, MaxWell DDR format)
  default.yaml          ← generic fallback for unknown formats
  operator_beta.yaml    ← add a new file here for a new operator
```

---

## Scenario 1 — New well, same operator, same DDR software

**Example:** A second well from Operator Alpha on the same asset, drilled by a different rig.

**Effort:** 30 minutes.

No profile changes are needed if the filename follows the same pattern (`{Rig} DDR {Number} {Asset} {Wellbore} {DD.MM.YYYY}.pdf`). The pipeline auto-detects this as Operator Alpha.

**Steps:**

1. Copy the new PDF files into `data/raw/`
2. Run QC to validate filenames:
   ```bash
   python scripts/qc_raw_ddr_filenames.py
   ```
3. Run batch preprocessing:
   ```bash
   python scripts/batch_preprocess_raw_ddrs.py
   ```
4. Build the graph and index:
   ```bash
   python scripts/build_graphs.py
   python scripts/build_global_index.py
   ```

**If the new rig uses a different filename format**, see Scenario 2.

---

## Scenario 2 — New operator, same DDR software family (MaxWell or similar)

**Example:** A different operator using the same MaxWell Drilling reporting system but with a different naming convention.

**Effort:** 1–3 hours.

**Steps:**

### Step 1 — Create the profile file

Copy the template and rename it:
```bash
cp configs/ddr_profiles/operator_alpha.yaml configs/ddr_profiles/operator_beta.yaml
```

### Step 2 — Update the profile header

```yaml
profile:
  name: "Operator Beta"
  description: "Brief description of this operator and DDR format"
  version: "1.0"
  ddr_software: "MaxWell Drilling"
```

### Step 3 — Update the filename pattern

This is the most critical change. The pattern uses Python `re` named groups.

**Required groups:**
| Group | Description | Example value |
|-------|-------------|---------------|
| `rig` | Rig or source identifier | `Transocean1`, `Nobel`, `Valaris1` |
| `report_type` | Always `DDR` | `DDR` |
| `ddr_number` | Sequential DDR number | `007`, `85` |
| `asset_or_project` | Asset or project code | `TAL`, `NOR`, `WA1` |
| `wellbore_suffix` | Wellbore designator | `W1`, `A`, `R2`, `T1` |
| `report_date` | Report date in the file | `18.04.2024` |

**Example — new operator uses underscores and different order:**

Filename: `TAL_DDR_007_W2_18.04.2024.pdf`

```yaml
filename:
  pattern: >-
    ^(?P<asset_or_project>[A-Za-z0-9]+)_
    (?P<report_type>DDR)_
    (?P<ddr_number>\d+)_
    (?P<wellbore_suffix>[A-Za-z0-9]+)_
    (?P<report_date>\d{2}\.\d{2}\.\d{4})
    (?P<rig>)
    .*\.pdf$
  flags: IGNORECASE
  date_format: "%d.%m.%Y"
  doc_id_template: "{asset_or_project}-DDR-{ddr_number:03d}-{wellbore_suffix}-{date_iso}"
```

> **Note on `doc_id_template`:** Use only the groups that are reliably present in the filename. The `{date_iso}` placeholder is always available (converted automatically from `report_date`). The `{ddr_number:03d}` formats the number as three digits (`007`).

**Test your pattern before running:**
```python
import re
pattern = r"^(?P<asset_or_project>[A-Za-z0-9]+)_DDR_(?P<ddr_number>\d+)_(?P<wellbore_suffix>[A-Za-z0-9]+)_(?P<report_date>\d{2}\.\d{2}\.\d{4}).*\.pdf$"
m = re.match(pattern, "TAL_DDR_007_W2_18.04.2024.pdf", re.I)
print(m.groupdict())
```

### Step 4 — Update wellbore suffix labels

Map the wellbore suffix codes in the filename to human-readable descriptions:

```yaml
wellbore:
  block_field: "asset_or_project"
  suffix_labels:
    W1: "Well 1 (parent)"
    W2: "Well 2 (first sidetrack)"
    A:  "Section A"
    T1: "Target 1"
```

If your filename does not have a wellbore suffix, leave this section empty (`suffix_labels: {}`).

### Step 5 — Update phase codes and labels

Phase codes come from the `Phase` column in the Op Summary table of the DDR itself. To find what codes are used, inspect a sample DDR:

```bash
python3 -c "
import pdfplumber, sys
with pdfplumber.open(sys.argv[1]) as pdf:
    for page in pdf.pages:
        for t in page.extract_tables():
            flat = ' '.join(str(c or '') for row in t for c in row)
            if 'OPERATION SUMMARY' in flat.upper():
                print(flat[:500])
                break
" your_sample.pdf
```

Then update the profile:

```yaml
phases:
  order: [MIRU, DRILL, CASE, CEMENT, COMP, PROD, TEST]
  labels:
    MIRU:   "Move In / Rig Up"
    DRILL:  "Drilling"
    CASE:   "Casing"
    CEMENT: "Cementing"
    COMP:   "Completion"
    PROD:   "Production"
    TEST:   "Well Testing"
```

> **If you don't know the phase codes yet**, leave `order: []` and `labels: {}`. The pipeline will display raw codes and sort them alphabetically. You can add labels after reviewing the first extracted dataset.

### Step 6 — Verify the profile detects correctly

```python
import sys; sys.path.insert(0, 'src')
from ddr_rag.ddr_profile import detect_profile

profile = detect_profile("your_sample_filename.pdf")
print(profile.name)                        # Should print "Operator Beta"
m = profile.match_filename("your_sample_filename.pdf")
print(m.groupdict())                       # Should show all named groups
```

### Step 7 — Run a diagnostic extraction on 5 sample PDFs

```bash
python scripts/diagnose_extraction.py \
  --raw-dir path/to/new/pdfs \
  --sample 5
```

Review the output:
- `hdr=0` means header extraction found nothing — check coordinate bands (Step 8)
- `ops=0` means Op Summary not found — check column headers (Step 9)
- `hrs=0.0` means durations are not parsing — check `op_summary.columns.duration_hr` keywords

---

## Scenario 3 — Different DDR software or layout

**Example:** A different software system (Halliburton InSite, Baker Hughes' WellPlan export, or custom Word/PDF format) where the page layout differs significantly.

**Effort:** 1–3 days.

In addition to Steps 1–6 above, you will need to calibrate the header extraction coordinate bands and possibly the Op Summary column keywords.

### Calibrating header coordinate bands

The header extractor uses y-coordinates (distance from the top of the page in PDF points) to locate different sections of the first page.

**To find the correct bands for a new format:**

```python
import pymupdf, sys

doc = pymupdf.open(sys.argv[1])
page = doc[0]
blocks = page.get_text("blocks", sort=True)

for i, b in enumerate(blocks[:40]):
    x0, y0, x1, y1, text, *_ = b
    print(f"y={y0:.0f}-{y1:.0f}  x={x0:.0f}-{x1:.0f}  {text[:60]!r}")
```

Run this on your sample PDF and look for:

| What you see | Profile setting |
|-------------|----------------|
| Date, report number, wellbore near the top | `header.top_band_max_y` — set to just below these blocks |
| Property grid (depths, costs, rig name) | `header.grid_band: [min_y, max_y]` |
| Morning report, narrative text | `header.narrative_band: [min_y, max_y]` |

Update the profile:
```yaml
header:
  top_band_max_y: 95         # adjusted from 80 for this format
  grid_band:  [90, 240]      # adjusted from [75, 200]
  narrative_band: [220, 400] # adjusted from [180, 360]
```

### Calibrating Op Summary column keywords

If the Op Summary table uses different column header text, update `op_summary.columns`:

```yaml
op_summary:
  section_headers: ["DAILY OPERATIONS", "OPERATION SUMMARY"]
  columns:
    start_time:    ["start", "from"]
    end_time:      ["end", "to"]
    duration_hr:   ["hours", "hrs", "duration"]
    phase:         ["phase", "section", "interval"]
    op_code:       ["code", "operation code", "op code"]
    activity_code: ["activity", "sub-code"]
    pt_x:          ["p/t/x", "type", "p-t-x"]
    operation:     ["description", "operation", "activity description"]
  required_columns: ["start_time", "dur_hr", "operation"]
```

### Updating field label keywords

If the property grid uses different label text (e.g., "Well Depth" instead of "Water Depth"), update `header.field_labels`:

```yaml
header:
  field_labels:
    water_depth_ft:  ["water depth", "wd", "rated water depth"]
    daily_cost:      ["daily cost", "cost today", "rig cost"]
    rig_name:        ["rig name", "drilling unit", "vessel"]
```

The extractor does a case-insensitive substring match, so partial keywords work.

---

## Scenario 4 — Completely different format (Excel, Word, custom PDF)

**Effort:** 1–2 weeks for a new extraction module.

If the DDR is not a structured PDF (e.g., Word document converted to PDF, scanned image, Excel export), the extraction layer needs a new module. The profile system can still define the output schema (phase labels, filename pattern, field names), but the extraction code itself requires a new implementation.

**Architecture guidance:**

1. Create `src/ddr_rag/extractors/operator_gamma_extractor.py`
2. Implement `extract_header_fields(pdf_path)` → `dict`
3. Implement `extract_op_summary(pdf_path)` → `list[dict]`
4. Both functions must return data conforming to the schemas in `ddr_extractor.py` (`_HEADER_COLS`, `_OP_COLS`)
5. Register the extractor class name in the profile:

```yaml
profile:
  name: "Operator Gamma"
  extractor_module: "ddr_rag.extractors.operator_gamma_extractor"
```

The `run_ddr_extraction()` function in `ddr_extractor.py` will need to be updated to dispatch to the registered extractor.

---

## Quick-reference checklist

Use this checklist when onboarding any new operator.

### 1. Profile file

- [ ] Created `configs/ddr_profiles/{operator_name}.yaml`
- [ ] `profile.name` and `profile.description` filled in
- [ ] `filename.pattern` matches sample filenames
- [ ] `filename.pattern` tested with `re.match()` on 5+ sample filenames
- [ ] `filename.doc_id_template` produces unique, stable IDs
- [ ] `wellbore.suffix_labels` populated (or left empty if no suffix)

### 2. Phase configuration

- [ ] Phase codes identified from sample DDR Op Summary table
- [ ] `phases.order` set (or left empty for alphabetical)
- [ ] `phases.labels` added for all known phase codes

### 3. Extraction validation

- [ ] `scripts/diagnose_extraction.py --sample 10` run on new PDFs
- [ ] `hdr_fld` count ≥ 15 per DDR (header extraction working)
- [ ] `ops` count > 0 per DDR (Op Summary extraction working)
- [ ] `hrs` total between 20–31h per DDR (time accounting correct)
- [ ] No issues with phase code `—` or `None` appearing in results

### 4. Pipeline run

- [ ] `python scripts/qc_raw_ddr_filenames.py` passes
- [ ] `python scripts/batch_preprocess_raw_ddrs.py --dry-run` shows correct DDR count
- [ ] `python scripts/batch_preprocess_raw_ddrs.py` completes with 0 failures
- [ ] `python scripts/build_graphs.py` completes
- [ ] `python scripts/build_global_index.py` completes
- [ ] App loads and shows data for new well on Field Analysis page

### 5. Quality check

- [ ] Sample 5 dates from the Operations Log and verify operations match source PDF
- [ ] Verify daily cost figures match the DDR header for 3+ dates
- [ ] Confirm phase labels display correctly in all charts
- [ ] Run `python scripts/run_eval.py` if evaluation questions exist for the new well

---

## Common errors and fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `qc_raw_ddr_filenames.py` shows 0 parsed files | Filename pattern doesn't match | Re-test `filename.pattern` with `re.match()` |
| `hdr_fld = 0` in diagnostics | `top_band_max_y` or `grid_band` wrong | Run the y-coordinate inspection script, update bands |
| `ops = 0` in diagnostics | Op Summary section header not found | Add variants to `op_summary.section_headers` |
| `hrs = 0.0` in diagnostics | Duration column not detected | Add variants to `op_summary.columns.duration_hr` |
| Phase labels show raw codes (e.g., `PROD1`) | `phases.labels` not populated | Add the phase codes to `phases.labels` in the profile |
| Phase order is wrong in charts | `phases.order` missing or incomplete | Add all phase codes to `phases.order` in the correct operational sequence |
| `detect_profile()` returns `Default` | No profile matches the filename | Check `filename.pattern` regex — test it standalone in Python |
| Costs appear as `None` | Label keyword not matching the PDF | Add the exact label text (lowercased) to `header.field_labels.daily_cost` |
| Wrong cumulative costs | Multiple `cumulative cost` matches | Make the keyword more specific: `"cumulative cost (cost)"` |

---

## File locations reference

```
DDR_RAG_Pipeline/
├── configs/
│   └── ddr_profiles/
│       ├── default.yaml           ← fallback profile (edit with caution)
│       ├── operator_alpha.yaml    ← current operator (reference implementation)
│       └── operator_beta.yaml    ← new operator (create here)
├── src/ddr_rag/
│   ├── ddr_profile.py             ← profile loader (load_profile, detect_profile)
│   ├── ddr_extractor.py           ← extraction code (reads coordinate bands)
│   ├── filename_qc.py             ← filename parsing (reads filename regex)
│   ├── vocab.py                   ← phase labels (reads from profile)
│   └── causality_analyzer.py     ← phase order (reads from profile)
└── scripts/
    ├── diagnose_extraction.py     ← validate extraction on sample PDFs
    ├── qc_raw_ddr_filenames.py    ← validate filename parsing
    └── batch_preprocess_raw_ddrs.py ← run full preprocessing
```

---

## Contact and support

If you encounter a DDR format that cannot be handled by the YAML profile configuration, the extraction layer will need a custom module. Capture the following information and share it with the development team:

1. Sample PDF (3–5 representative DDRs)
2. Which fields are missing from `diagnose_extraction.py` output
3. Screenshot of the PDF page 1 header and the Op Summary table
4. The software name shown in the DDR footer or cover page
