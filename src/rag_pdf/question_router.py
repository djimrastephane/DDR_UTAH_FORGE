from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Optional


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _question_quarter(q_lower: str) -> Optional[str]:
    for q in ("q1", "q2", "q3", "q4"):
        if q in q_lower:
            return q
    return None


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(n in text for n in needles)


@dataclass(frozen=True)
class QueryRoute:
    intent: str
    confidence: float
    slots: dict[str, Any] = field(default_factory=dict)


def route_question(question: str) -> QueryRoute:
    q_lower = _normalize_text(question)
    quarter = _question_quarter(q_lower)

    if "significant" in q_lower and "delay" in q_lower:
        return QueryRoute(
            intent="table_metric_significant_delay",
            confidence=0.95,
            slots={"row_terms": ["significant_delay"], "quarter": quarter, "prefer_percent": False},
        )
    if "on track" in q_lower:
        return QueryRoute(
            intent="table_metric_on_track",
            confidence=0.95,
            slots={"row_terms": ["on_track"], "quarter": quarter, "prefer_percent": False},
        )
    if "proportion" in q_lower and "complete" in q_lower:
        return QueryRoute(
            intent="table_metric_complete_ratio",
            confidence=0.9,
            slots={"row_terms": ["complete"], "quarter": quarter, "prefer_percent": True},
        )

    if _contains_any(q_lower, ["staff cost", "staff costs", "employee benefit", "remuneration", "pension cost"]):
        row_terms = []
        if "pension" in q_lower:
            row_terms = ["pension_cost", "pension"]
        elif "remuneration" in q_lower:
            row_terms = ["total_remuneration", "remuneration"]
        elif "total" in q_lower and "staff" in q_lower and "cost" in q_lower:
            row_terms = ["total_staff_costs", "staff_costs"]
        else:
            row_terms = ["staff_costs", "employee_benefit", "remuneration"]
        return QueryRoute(
            intent="table_metric_staff_costs",
            confidence=0.85,
            slots={
                "row_terms": row_terms,
                "quarter": quarter,
                "prefer_percent": False,
                "table_type_hint": "staff_costs",
            },
        )

    if _contains_any(q_lower, ["emission", "emissions", "greenhouse gas", "co2", "carbon"]):
        row_terms = []
        if "building energy" in q_lower:
            row_terms = ["building_energy"]
        elif "waste" in q_lower:
            row_terms = ["waste"]
        elif "water" in q_lower:
            row_terms = ["water"]
        elif "fleet" in q_lower:
            row_terms = ["nhs_fleet_travel", "fleet_travel"]
        elif "business travel" in q_lower:
            row_terms = ["business_travel"]
        elif "medical gas" in q_lower:
            row_terms = ["medical_gases"]
        elif "total emission" in q_lower:
            row_terms = ["total_emissions"]
        else:
            row_terms = ["total_emissions", "emissions"]

        column_terms = []
        if "target" in q_lower:
            column_terms.append("target emissions")
        if "percentage change" in q_lower or "change" in q_lower:
            column_terms.append("percentage change")
        if "difference" in q_lower:
            column_terms.append("percentage difference")

        return QueryRoute(
            intent="table_metric_emissions",
            confidence=0.8,
            slots={
                "row_terms": row_terms,
                "column_terms": column_terms,
                "quarter": quarter,
                "prefer_percent": ("percent" in q_lower or "percentage" in q_lower),
                "table_type_hint": "unknown",
            },
        )
    if "board committee" in q_lower and "strategic risk register" in q_lower:
        return QueryRoute(intent="governance_board_committee", confidence=0.85)
    if "endorse" in q_lower and "risk appetite" in q_lower and "strategic risk profile" in q_lower:
        return QueryRoute(intent="governance_endorsements", confidence=0.9)
    if "endorse" in q_lower and ("risk appetite" in q_lower or "strategic risk profile" in q_lower):
        return QueryRoute(intent="governance_endorsements", confidence=0.8)
    if "significant issue" in q_lower and "accountable officer" in q_lower:
        return QueryRoute(intent="governance_significant_issue", confidence=0.85)
    return QueryRoute(intent="unknown", confidence=0.5)
