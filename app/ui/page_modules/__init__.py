"""DDR Intelligence UI page modules — one module per dashboard page.

Each sub-module in this package implements a single Streamlit page function
that accepts pre-loaded DataFrames and renders the page content.  Page modules
import shared helpers from the sibling packages::

    from ..constants import PHASE_ORDER, WELL_COLOURS
    from ..loaders import load_all_ops
    from ..utils import _apply_chart_theme, _phase_date_ranges

Available page modules (added as the refactor progresses):

- ``page_modules.well_overview``      — cost charts, NPT phase summary, narrative
- ``page_modules.operational_graph``  — interactive co-occurrence network
- ``page_modules.npt_intelligence``   — NPT breakdown, drill-down, Gantt
- ``page_modules.operations_log``     — filterable operations log with CSV export
- ``page_modules.campaign_summary``   — cross-well comparison and field analytics
- ``page_modules.lessons_learned``    — auto-generated NPT summary and recommendations
"""
