# Data Contract

## General Rules

- All timestamps must use ISO-8601 format.
- Units must always be specified.
- Missing values must be explicit.
- Column names should use snake_case.
- No duplicate column names allowed.

---

# Drilling Data

| Column | Unit | Description |
|--------|------|-------------|
| well_name | text | Well identifier |
| rig_name | text | Rig identifier |
| date_time | ISO-8601 | Timestamp |
| rop | m/hr | Rate of penetration |
| wob | klbs | Weight on bit |
| rpm | rpm | Rotary speed |
| spp | psi | Standpipe pressure |
| hookload | klbs | Surface hookload |
| flow_rate | bpm | Pump flow rate |
| operation_phase | text | Current operation |
| npt_flag | boolean | Non-productive time indicator |

---

# Completion Data

| Column | Unit | Description |
|--------|------|-------------|
| stage_number | integer | Fracture stage |
| treating_pressure | psi | Surface pressure |
| slurry_rate | bpm | Slurry pump rate |
| proppant_concentration | ppa | Proppant concentration |
| fluid_volume | bbl | Total fluid volume |
| proppant_mass | lbm | Total proppant mass |

---

# Gravel Pack Failure Images

| Field | Description |
|------|-------------|
| image_id | Unique image identifier |
| capture_date | Inspection date |
| well_name | Related well |
| erosion_percentage | Estimated erosion |
| failure_type | Failure classification |
| image_resolution | Resolution in pixels |

---

# File Validation Rules

## Allowed File Types

- PDF
- CSV
- XLSX
- PNG
- JPG
- JPEG

## Rejected Files

- Executables
- Unsupported archives
- Corrupted documents

---

# Missing Data Rules

- Null numeric values must use NaN.
- Missing categorical values must use NULL or explicit "unknown".
- Invalid timestamps must be rejected during ingestion.

---

# Naming Conventions

## Preferred Naming

- well_name
- stage_number
- treating_pressure

## Avoid

- WellName
- Stage#
- TreatPress

---

# Data Quality Checks

Validate:

- duplicate rows
- invalid units
- timestamp ordering
- impossible values
- missing critical fields
- OCR extraction quality

---

# Engineering Assumptions

- Units must remain consistent across all datasets.
- Converted units must be logged.
- Source documents remain the authoritative reference.
