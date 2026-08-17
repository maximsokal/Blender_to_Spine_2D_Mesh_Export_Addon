"""Regression coverage for Texture size ownership after UI simplification."""

from __future__ import annotations

import inspect

from Blender_to_Spine2D_Mesh_Exporter import ui, ui_layout


def test_main_export_settings_remain_the_texture_size_owner() -> None:
    source = inspect.getsource(ui.OBJECT_PT_Spine2DMeshPanel._draw_export_settings)

    assert '"spine2d_texture_size"' in source
    assert '"spine2d_target_spine_version"' in source
    assert '"spine2d_json_path"' in source
    assert '"spine2d_images_path"' in source


def test_child_panels_do_not_duplicate_texture_size() -> None:
    source = inspect.getsource(ui_layout)

    assert "spine2d_texture_size" not in source


def test_texture_size_remains_scene_owned_and_not_per_object() -> None:
    source = inspect.getsource(ui.OBJECT_PT_Spine2DMeshPanel._draw_export_settings)

    assert "context.scene" in source
    assert "spine2d_bake_settings" not in source
