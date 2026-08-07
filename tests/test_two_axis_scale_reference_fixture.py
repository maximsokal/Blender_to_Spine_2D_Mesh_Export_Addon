"""Machine-readable regression contract for the two-axis reference fixture."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "two_axis_scale_reference.json"


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_reference_fixture_is_valid_current_spine42_json_shape() -> None:
    fixture = _fixture()

    assert fixture["skeleton"]["spine"] == "4.2.43"
    assert isinstance(fixture["bones"], list)
    assert isinstance(fixture["ik"], list)
    assert isinstance(fixture["transform"], list)


def test_reference_preserves_scale_and_five_phase_schedule() -> None:
    fixture = _fixture()
    bones = {bone["name"]: bone for bone in fixture["bones"]}
    transforms = fixture["transform"]

    assert bones["ROTATE_X_CTRL"]["rotation"] == -134.67
    assert bones["ROTATE_Y_CTRL"]["rotation"] == -17.43
    assert bones["scale"]["parent"] == "root"
    assert fixture["ik"][0]["order"] == 1
    assert transforms[0].get("order", 0) == 0
    assert transforms[1]["order"] == 4
    assert transforms[2]["order"] == 2
    assert transforms[3]["order"] == 3
    assert transforms[2]["bones"] == [
        "ROTATE_X",
        "TOP_ROTATION",
        "BOTTOM_ROTATION",
    ]
