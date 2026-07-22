from __future__ import annotations

from ddr_rag.npt_classifier import classify_equipment_subtype, classify_utah_forge_npt


def test_utah_forge_no_losses_is_not_npt() -> None:
    is_npt, category = classify_utah_forge_npt(
        "Good returns throughout job. No losses.",
        "Run Csg & Cement",
        "Run Csg & Cement",
    )

    assert is_npt is False
    assert category == "productive"


def test_utah_forge_minor_seepage_losses_is_not_npt() -> None:
    is_npt, category = classify_utah_forge_npt(
        "Drill ahead 92 ft. Mud fluids seepage losses in six hours 9 bbls.",
        "Drilling",
        "Drilling",
    )

    assert is_npt is False
    assert category == "productive"


def test_utah_forge_dsm_text_does_not_trigger_formation_testing() -> None:
    is_npt, category = classify_utah_forge_npt(
        "PJSM with FORGE DSM's. Mud pump number one went down; "
        "bearing broke. Frontier Rig on NPT.",
        "Trips",
        "Trips",
    )

    assert is_npt is True
    assert category == "equipment"


def test_utah_forge_failed_packer_setting_is_downhole_tool_npt() -> None:
    is_npt, category = classify_utah_forge_npt(
        "Attempted multiple times to set packers. Ball did not seat and packers did not set.",
        "Other",
        "Other",
    )

    assert is_npt is True
    assert category == "downhole_tools"


def test_equipment_subtype_identifies_pump() -> None:
    assert classify_equipment_subtype(
        "Mud pump number one went down; bearing broke."
    ) == "Pump"


def test_equipment_subtype_identifies_hydraulic() -> None:
    assert classify_equipment_subtype(
        "Hydraulic line failed on grabber box. Replace line."
    ) == "Hydraulic System"


def test_equipment_subtype_falls_back_to_unspecified() -> None:
    assert classify_equipment_subtype(
        "Equipment failure caused delay. Repair completed."
    ) == "Unspecified"

