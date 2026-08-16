"""Regression coverage for Texture size ownership in the ordered Blender UI."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

from Blender_to_Spine2D_Mesh_Exporter import ui, ui_layout


def test_ordered_paths_foldout_does_not_draw_texture_size() -> None:
    source = inspect.getsource(
        ui_layout.OBJECT_PT_Spine2DOrderedMeshPanel._draw_export_settings
    )

    assert "spine2d_texture_size" not in source
    assert '"spine2d_target_spine_version"' in source
    assert '"spine2d_json_path"' in source
    assert '"spine2d_images_path"' in source


def test_ordered_paths_foldout_draws_the_preference_exact_version(monkeypatch) -> None:
    column = MagicMock()
    scene = SimpleNamespace(
        spine2d_target_spine_version="SPINE_4_2",
        spine2d_json_path="output",
        spine2d_images_path="images",
    )
    context = SimpleNamespace(scene=scene)
    monkeypatch.setattr(
        ui_layout,
        "resolve_spine_project_exact_version",
        lambda target, *, context: "4.2.20",
    )

    ui_layout.OBJECT_PT_Spine2DOrderedMeshPanel._draw_export_settings(
        None,
        column,
        context,
    )

    labels = tuple(
        call.kwargs.get("text")
        for call in column.label.call_args_list
    )
    assert "Exact JSON version: 4.2.20" in labels
    assert "Exact JSON version: 4.2.43" not in labels


def test_ordered_bake_foldout_draws_texture_size_before_shared_bake_controls(
    monkeypatch,
) -> None:
    column = MagicMock()
    scene = SimpleNamespace(spine2d_texture_size=1024)
    context = SimpleNamespace(scene=scene)
    delegated = MagicMock()
    monkeypatch.setattr(
        ui.OBJECT_PT_Spine2DMeshPanel,
        "_draw_bake_settings",
        delegated,
    )

    ui_layout.OBJECT_PT_Spine2DOrderedMeshPanel._draw_bake_settings(
        column,
        context,
    )

    column.prop.assert_called_once_with(
        scene,
        "spine2d_texture_size",
        text="Texture size",
    )
    column.separator.assert_called_once_with()
    delegated.assert_called_once_with(column, context)


def test_texture_size_is_still_scene_owned_and_not_per_object() -> None:
    source = inspect.getsource(ui_layout.OBJECT_PT_Spine2DOrderedMeshPanel._draw_bake_settings)

    assert "context.scene" in source
    assert "spine2d_bake_settings" not in source
