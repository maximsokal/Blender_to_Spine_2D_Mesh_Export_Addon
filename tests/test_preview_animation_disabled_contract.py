"""Contracts for the setup-pose-only target-version export scope."""

from pathlib import Path

from Blender_to_Spine2D_Mesh_Exporter import ui


def test_preview_animation_rna_is_retained_but_not_drawn() -> None:
    """Old .blend files keep their RNA value without exposing an active UI toggle."""

    source = Path(ui.__file__).read_text(encoding="utf-8")
    property_names = tuple(registration.name for registration in ui.RNA_PROPERTIES)

    assert "spine2d_export_preview_animation" in property_names
    assert 'row.label(text="Preview animation")' not in source
    assert (
        'row.prop(scene, "spine2d_export_preview_animation", text="")'
        not in source
    )
    assert "Preview animation is intentionally hidden" in source
