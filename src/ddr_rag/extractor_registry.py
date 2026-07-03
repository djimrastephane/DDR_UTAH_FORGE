from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ReportExtractor(Protocol):
    """Interface implemented by report-layout-specific DDR extractors."""

    name: str
    priority: int

    def matches(self, pdf_path: Path, doc_id: str = "") -> bool:
        """Return true when this extractor should handle the source document."""

    def extract_header_fields(self, pdf_path: Path) -> dict[str, str]:
        """Extract one header record keyed by canonical DDR header column names."""

    def extract_op_summary(self, pdf_path: Path) -> list[dict]:
        """Extract operation-summary rows keyed by canonical DDR operation columns."""


class ExtractorRegistry:
    """Priority-ordered extractor registry with first-match selection."""

    def __init__(self, extractors: list[ReportExtractor] | None = None) -> None:
        self._extractors: list[ReportExtractor] = []
        for extractor in extractors or []:
            self.register(extractor)

    def register(self, extractor: ReportExtractor) -> None:
        if not isinstance(getattr(extractor, "name", None), str):
            raise TypeError("Report extractors must expose a string 'name'.")
        if not hasattr(extractor, "matches"):
            raise TypeError(f"Report extractor {extractor!r} is missing matches().")
        if not hasattr(extractor, "extract_header_fields"):
            raise TypeError(f"Report extractor {extractor!r} is missing extract_header_fields().")
        if not hasattr(extractor, "extract_op_summary"):
            raise TypeError(f"Report extractor {extractor!r} is missing extract_op_summary().")
        self._extractors.append(extractor)
        self._extractors.sort(key=lambda item: int(getattr(item, "priority", 0)), reverse=True)

    @property
    def extractors(self) -> tuple[ReportExtractor, ...]:
        return tuple(self._extractors)

    def names(self) -> list[str]:
        return [extractor.name for extractor in self._extractors]

    def select(self, pdf_path: Path, doc_id: str = "") -> ReportExtractor:
        for extractor in self._extractors:
            if extractor.matches(pdf_path, doc_id=doc_id):
                return extractor
        raise LookupError(f"No DDR report extractor matched {pdf_path}")


def import_report_extractor(import_path: str) -> ReportExtractor:
    """Load an extractor instance from 'package.module:ClassName'."""
    if ":" not in import_path:
        raise ValueError(f"Extractor import path must use 'module:ClassName': {import_path!r}")
    module_name, class_name = import_path.split(":", 1)
    module = import_module(module_name)
    cls = getattr(module, class_name)
    extractor = cls()
    if not isinstance(extractor, ReportExtractor):
        raise TypeError(f"{import_path!r} does not implement the ReportExtractor protocol.")
    return extractor
