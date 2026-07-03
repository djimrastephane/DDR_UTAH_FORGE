from __future__ import annotations

from pathlib import Path

from ddr_rag.ddr_extractor import (
    GenericDDRReportExtractor,
    UtahForgeReportExtractor,
    get_extractor_registry,
    select_report_extractor,
)
from ddr_rag.extractor_registry import ExtractorRegistry


class _LowPriorityExtractor:
    name = "low"
    priority = 1

    def matches(self, pdf_path: Path, doc_id: str = "") -> bool:
        return True

    def extract_header_fields(self, pdf_path: Path) -> dict[str, str]:
        return {}

    def extract_op_summary(self, pdf_path: Path) -> list[dict]:
        return []


class _HighPriorityExtractor(_LowPriorityExtractor):
    name = "high"
    priority = 10


def test_extractor_registry_selects_highest_priority_match() -> None:
    registry = ExtractorRegistry([_LowPriorityExtractor(), _HighPriorityExtractor()])

    assert registry.names() == ["high", "low"]
    assert registry.select(Path("anything.pdf")).name == "high"


def test_default_registry_orders_specialized_before_generic() -> None:
    names = get_extractor_registry().names()

    assert names[0] == "utah_forge"
    assert names[-1] == "generic_ddr"


def test_selects_utah_forge_extractor_by_filename() -> None:
    extractor = select_report_extractor(
        Path("Utah_Forge_FORGE_16A_(78)-32_Drilling-C_12012020_12012020_18_reporttmp.pdf")
    )

    assert isinstance(extractor, UtahForgeReportExtractor)


def test_selects_utah_forge_extractor_by_doc_id() -> None:
    extractor = select_report_extractor(
        Path("renamed.pdf"),
        doc_id="UtahForge-DDR-FORGE-16A-78-32-Drilling-2020-12-01-R018-test",
    )

    assert isinstance(extractor, UtahForgeReportExtractor)


def test_generic_extractor_is_fallback() -> None:
    extractor = select_report_extractor(Path("Other_Operator_DDR_001.pdf"))

    assert isinstance(extractor, GenericDDRReportExtractor)
