# Modular Report Extractors

DDR post-processing is routed through a small extractor registry. The public
entry points stay the same:

- `ddr_rag.ddr_extractor.extract_header_fields`
- `ddr_rag.ddr_extractor.extract_op_summary`
- `ddr_rag.ddr_extractor.run_ddr_extraction`

The registry selects the first extractor whose `matches(pdf_path, doc_id)`
method returns `True`. Built-ins are ordered as:

1. `UtahForgeReportExtractor`
2. `GenericDDRReportExtractor`

The generic extractor is always the fallback.

## Add A New Report Type

Create a class that implements the report extractor interface:

```python
from pathlib import Path


class NewReportExtractor:
    name = "new_report_type"
    priority = 90

    def matches(self, pdf_path: Path, doc_id: str = "") -> bool:
        return "New_Report_Token" in pdf_path.name or doc_id.startswith("NewField-DDR-")

    def extract_header_fields(self, pdf_path: Path) -> dict[str, str]:
        return {
            "report_date": "",
            "report_no": "",
            "wellbore": "",
        }

    def extract_op_summary(self, pdf_path: Path) -> list[dict]:
        return []
```

Register it in `configs/ddr_rag.yaml`:

```yaml
extraction:
  extra_extractor_classes:
    - ddr_rag.extractors.new_report:NewReportExtractor
```

Extractor rows should use the canonical column names in
`src/ddr_rag/ddr_extractor.py` (`_HEADER_COLS` and `_OP_COLS`). Missing fields
can be left blank; `run_ddr_extraction` fills the stable `doc_id`,
`corpus_id`, and `run_date_utc` values.

## Priority

Higher `priority` values are checked first. Keep specialized report layouts
above the generic fallback. Use a narrow `matches` method so one extractor does
not accidentally claim another report family.
