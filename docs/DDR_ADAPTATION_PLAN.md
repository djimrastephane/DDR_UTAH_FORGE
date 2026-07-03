# DDR Adaptation Plan

## Current Baseline

The project starts with the existing generic PDF RAG flow:

1. Extract page text and layout.
2. Detect and extract tables.
3. Build text/table chunks.
4. Embed chunks and build FAISS indexes.
5. Search with hybrid retrieval and optional grounded generation.

## DDR Changes To Implement Next

1. **Header metadata enrichment**
   - Use `ddr_rag.report_profile.extract_ddr_header_fields`.
   - Add extracted fields to `pages.parquet`, `chunks.parquet`, and
     `chunk_meta.parquet`.

2. **Operation summary parser**
   - Parse rows by start time, end time, duration, phase, operation code,
     activity code, time code, and operation text.
   - Emit one row-preserving chunk per operation block.

3. **Structured facts output**
   - Add `ddr_facts.parquet`.
   - Keep fact types such as `operation_event`, `personnel_count`,
     `material_balance`, `weather_observation`, and `support_vessel`.

4. **Retrieval filters**
   - Add filters for `report_date`, `well_name`, `rig`, `phase`,
     `operation_code`, and `section`.

5. **Evaluation set**
   - Start with 30-50 questions across 10-20 representative reports.
   - Include time-window, personnel, material, vessel, weather, and notes
     questions.

## Suggested First Milestone

Build a parser that can answer these reliably:

- What was the well name and report date?
- What operations happened during a specified time range?
- How many personnel were onboard by company or function?
- What materials were consumed, received, and on location?
- Which support vessels were listed?
- What were the general notes or rig/equipment notes?

