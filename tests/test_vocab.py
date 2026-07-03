"""Tests for ddr_rag.vocab normalisation and tokenisation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddr_rag.vocab import (
    tokenize_for_graph,
    normalize_text,
    label_op_code,
    label_activity_code,
    label_phase,
    STOP_WORDS,
    UNIT_TOKENS,
    ABBREVIATION_MAP,
)


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

def test_normalize_lowercase():
    assert normalize_text("RIH WITH BHA") == normalize_text("rih with bha")


def test_normalize_compound_rih():
    result = normalize_text("Run in hole with BHA")
    assert "rih" in result


def test_normalize_compound_pooh():
    result = normalize_text("Pull out of hole from 5000ft")
    assert "pooh" in result


def test_normalize_compound_tight_hole():
    result = normalize_text("Tight hole observed on connections")
    assert "tight-hole" in result


def test_normalize_compound_lost_circulation():
    result = normalize_text("Lost circulation observed at 14200ft")
    assert "lost-circulation" in result


def test_normalize_compound_wiper_trip():
    assert "wiper-trip" in normalize_text("Wiper trip from TD")


def test_normalize_compound_pressure_test():
    assert "pressure-test" in normalize_text("Pressure tested cement unit lines")


def test_normalize_compound_broke_circulation():
    assert "break-circulation" in normalize_text("Broke circulation at 13354ft")


def test_normalize_abbreviation_rih():
    result = normalize_text("RIH to 5000ft")
    assert "rih" in result


def test_normalize_abbreviation_pooh():
    result = normalize_text("POOH from 13000ft")
    assert "pooh" in result


def test_normalize_abbreviation_ncs():
    result = normalize_text("Followed NCS instructions for sleeve location")
    assert "ncs-multistage" in result


def test_normalize_abbreviation_sbp():
    result = normalize_text("SBP held at 500psi")
    assert "surface-back-pressure" in result


def test_normalize_abbreviation_tdx():
    result = normalize_text("TDX torque increased to 15klbs")
    assert "top-drive" in result


def test_normalize_abbreviation_prs():
    result = normalize_text("PRS drag chain removed")
    assert "pipe-racking-system" in result


# ---------------------------------------------------------------------------
# tokenize_for_graph
# ---------------------------------------------------------------------------

def test_tokenize_removes_stop_words():
    tokens = tokenize_for_graph("RIH with BHA and pipe")
    assert "and" not in tokens
    assert "with" not in tokens


def test_tokenize_removes_units():
    tokens = tokenize_for_graph("Ran to 13000ft at 84gpm and 1800psi")
    assert "ft" not in tokens
    assert "gpm" not in tokens
    assert "psi" not in tokens


def test_tokenize_removes_measurements():
    # Embedded measurements like "14.0ppg" or "13354ft" must not become nodes
    tokens = tokenize_for_graph("Mud weight 14.0ppg OBM. Depth 13354ft.")
    assert "14.0ppg" not in tokens
    assert "13354ft" not in tokens


def test_tokenize_keeps_canonical_compounds():
    tokens = tokenize_for_graph("RIH with 2-7/8 mule shoe magnet BHA")
    assert "rih" in tokens
    assert "mule-shoe" in tokens
    assert "bha" in tokens


def test_tokenize_keeps_risk_signals():
    tokens = tokenize_for_graph("Observed 10klbs overpull. Drag increasing through restriction.")
    assert "overpull" in tokens
    assert "drag" in tokens
    assert "restriction" in tokens


def test_tokenize_keeps_vendor_names():
    tokens = tokenize_for_graph("Ran NCS FIA BHA to locate frac sleeve.")
    assert "ncs-multistage" in tokens
    assert "frac-sleeve" in tokens


def test_tokenize_drops_held():
    tokens = tokenize_for_graph("Held JSA for running in hole with BHA")
    assert "held" not in tokens


def test_tokenize_drops_running():
    tokens = tokenize_for_graph("Running in hole with drill pipe")
    # "running" is a stop word; compound "run in hole" → "rih"
    assert "running" not in tokens


def test_tokenize_drops_filler_verbs():
    tokens = tokenize_for_graph("Confirmed lined up. Monitored well. Observed returns.")
    assert "confirmed" not in tokens
    assert "monitored" not in tokens
    assert "observed" not in tokens


def test_tokenize_minimum_length():
    tokens = tokenize_for_graph("RIH to TD on DP")
    # "to", "on" are stop words; "TD" → "total-depth"; "DP" → "drill-pipe"
    for tok in tokens:
        assert len(tok) >= 3 or "-" in tok


def test_tokenize_returns_list():
    result = tokenize_for_graph("Pressure tested BHA at surface")
    assert isinstance(result, list)


def test_tokenize_empty_string():
    assert tokenize_for_graph("") == []


def test_tokenize_none_handled():
    # None should not raise
    assert tokenize_for_graph(None) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def test_label_phase_known():
    assert label_phase("PROD1") == "Production / Reservoir Section"
    assert label_phase("COMPZN") == "Completion / Zonal"
    assert label_phase("COND1") == "Conductor Section"
    assert label_phase("MIRU") == "Move In / Rig Up"


def test_label_phase_unknown():
    result = label_phase("UNKNOWN_PHASE")
    assert result == "UNKNOWN_PHASE"


def test_label_op_code_known():
    assert label_op_code("MPDCSG") == "MPD Casing"
    assert label_op_code("STIM") == "Stimulation"
    assert label_op_code("CEMENT") == "Cementing"


def test_label_op_code_unknown():
    assert label_op_code("XYZ") == "XYZ"


def test_label_activity_code_known():
    assert label_activity_code("TRIP") == "Tripping"
    assert label_activity_code("CIRC") == "Circulation"
    assert label_activity_code("FISH") == "Fishing"


# ---------------------------------------------------------------------------
# Vocabulary consistency
# ---------------------------------------------------------------------------

def test_stop_words_are_lowercase():
    for word in STOP_WORDS:
        assert word == word.lower(), f"Stop word not lowercase: {word!r}"


def test_unit_tokens_are_lowercase():
    for tok in UNIT_TOKENS:
        assert tok == tok.lower(), f"Unit token not lowercase: {tok!r}"


def test_abbreviation_map_keys_are_lowercase():
    for key in ABBREVIATION_MAP:
        assert key == key.lower(), f"Abbrev key not lowercase: {key!r}"
