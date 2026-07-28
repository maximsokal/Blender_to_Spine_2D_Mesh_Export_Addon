"""Keep the documented two-axis reference identical to the machine fixture."""

from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "docs" / "rig-profiles.md"
FIXTURE = ROOT / "tests" / "fixtures" / "two_axis_scale_reference.json"
JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def _documented_reference() -> dict[str, object]:
    source = DOCUMENT.read_text(encoding="utf-8")
    matches = JSON_BLOCK.findall(source)
    assert len(matches) == 1, "rig-profiles.md must contain one complete JSON reference"
    return json.loads(matches[0])


def test_documented_reference_matches_machine_fixture_exactly():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert _documented_reference() == fixture


def test_reference_preserves_scale_and_five_phase_schedule():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    bones = {bone["name"]: bone for bone in fixture["bones"]}
    transforms = fixture["transform"]

    assert fixture["skeleton"]["spine"] == "4.2.43"
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
