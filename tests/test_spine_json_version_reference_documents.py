"""Integrity contracts for the user-provided multi-object Spine JSON references."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = ROOT / "docs" / "spine-json-versioning" / "reference"

REFERENCE_CASES = (
    (
        "Cone_plus_2_objects_spine_3.8.json",
        "3.8-from-4.0-from-4.1-from-4.2.43",
        3,
        12,
    ),
    (
        "Cone_plus_2_objects_spine_4.0.json",
        "4.0-from-4.1-from-4.2.43",
        3,
        12,
    ),
    (
        "Cone_plus_2_objects_spine_4.1.json",
        "4.1-from-4.2.43",
        3,
        12,
    ),
    ("Cone_plus_2_objects_spine_4.2.json", "4.2.43", 3, 12),
    ("Cone_plus_2_objects_spine_4.3.json", "4.3.23", 0, 0),
)


@pytest.mark.parametrize(
    "filename,expected_version,expected_ik,expected_transform",
    REFERENCE_CASES,
)
def test_reference_document_is_complete_multi_object_json(
    filename: str,
    expected_version: str,
    expected_ik: int,
    expected_transform: int,
) -> None:
    path = REFERENCE_ROOT / filename
    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["skeleton"]["spine"] == expected_version
    assert len(document["bones"]) == 58
    assert len(document["slots"]) == 24
    assert len(document.get("ik", ())) == expected_ik
    assert len(document.get("transform", ())) == expected_transform
    assert len(document.get("constraints", ())) == 0
    assert len(document["skins"]) == 1
    assert document["skins"][0]["name"] == "default"
    assert len(document["animations"]) == 6

    attachment_groups = document["skins"][0]["attachments"]
    assert any(name.startswith("Cone_Segment_") for name in attachment_groups)
    assert any(name.startswith("Cone.001_Segment_") for name in attachment_groups)
    assert any(name.startswith("Cone.002_Segment_") for name in attachment_groups)
