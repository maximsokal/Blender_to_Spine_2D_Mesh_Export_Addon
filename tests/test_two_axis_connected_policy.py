"""Document the deliberate connected-composition boundary for the new rig."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPOSITION = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
    / "a1_multi_object_composition.py"
)
READINESS = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
    / "a1_export_readiness.py"
)
SETTINGS = ROOT / "docs" / "settings-reference.md"


def test_connected_two_axis_policy_is_an_explicit_error_not_a_fake_constraint():
    source = COMPOSITION.read_text(encoding="utf-8")

    assert "A1RigProfile.TWO_AXIS_ROTATION_SCALE" in source
    assert "CONNECTED mode does not yet support TWO_AXIS_ROTATION_SCALE" in source
    assert "five-phase constraint schedule" in source
    assert "fake" not in source.lower()


def test_readiness_executes_the_same_composition_boundary():
    source = READINESS.read_text(encoding="utf-8")

    assert "compose_a1_multi_object_document(" in source
    assert "compose_a1_mixed_document(" in source
    assert "Run the production preparation/composition pipeline" in source


def test_public_settings_explain_current_connected_boundary():
    source = SETTINGS.read_text(encoding="utf-8")

    assert "TWO_AXIS_ROTATION_SCALE" in source
    assert "Connected composition remains blocked" in source
    assert "never substitutes a fake sixth constraint" in source
