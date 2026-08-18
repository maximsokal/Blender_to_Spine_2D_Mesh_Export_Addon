"""Regression coverage for Scene Texture size ownership in the Bake foldout."""

from __future__ import annotations

import inspect

from Blender_to_Spine2D_Mesh_Exporter import ui, ui_layout


def test_paths_and_version_settings_do_not_draw_texture_size() -> None:
    source = inspect.getsource(ui.OBJECT_PT_Spine2DMeshPanel._draw_export_settings)

    assert '"spine2d_target_spine_version"' in source
    assert '"spine2d_json_path"' in source
    assert '"spine2d_images_path"' in source
    assert '"spine2d_texture_size"' not in source


def test_bake_draws_texture_size_before_frame_controls() -> None:
    source = inspect.getsource(ui.OBJECT_PT_Spine2DMeshPanel._draw_bake_settings)

    texture = source.index('"spine2d_texture_size"')
    single_frames = source.index('"spine2d_frames_for_render"')
    per_object_frames = source.index('"frames_for_render"')

    assert texture < single_frames
    assert texture < per_object_frames
    assert 'text="Texture size"' in source


def test_child_panels_do_not_duplicate_texture_size() -> None:
    source = inspect.getsource(ui_layout)

    assert "spine2d_texture_size" not in source


def test_texture_size_remains_scene_owned_and_not_per_object() -> None:
    source = inspect.getsource(ui.OBJECT_PT_Spine2DMeshPanel._draw_bake_settings)

    assert "scene = context.scene" in source
    assert 'column.prop(scene, "spine2d_texture_size"' in source
    assert 'settings, "spine2d_texture_size"' not in source
