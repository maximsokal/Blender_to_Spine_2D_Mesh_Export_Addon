"""Contracts for exact Spine Editor version tokens in final JSON filenames."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1GeometryPreparationSettings,
    resolve_a1_output_paths,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import a1_ui_export_plan
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_scene_capture import (
    _SceneExportProfile,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_selection import (
    _ObjectExportProfile,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_settings import (
    _settings_from_profiles,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import BakeExecutionSettings
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import A1RigProfile
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import (
    SpineJsonTarget,
    spine_json_version_filename_token,
)


def _scene_profile(
    root: Path,
    target: SpineJsonTarget,
) -> _SceneExportProfile:
    return _SceneExportProfile(
        output_directory=root,
        images_relative_path="images",
        texture_size=128,
        seam_mode="AUTO",
        angle_limit_degrees=30.0,
        geometry=A1GeometryPreparationSettings(),
        bake_execution=BakeExecutionSettings(),
        include_control_icons=True,
        include_preview_animation=False,
        spine_target=target,
        rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE,
    )


@pytest.mark.parametrize(
    "target,expected",
    (
        (SpineJsonTarget.SPINE_4_0, "spine_4.0.64"),
        (SpineJsonTarget.SPINE_4_1, "spine_4.1.24"),
        (SpineJsonTarget.SPINE_4_2, "spine_4.2.43"),
    ),
)
def test_filename_token_uses_the_exact_registered_patch_version(
    target: SpineJsonTarget,
    expected: str,
) -> None:
    assert spine_json_version_filename_token(target) == expected
    assert spine_json_version_filename_token(target.exact_version) == expected


@pytest.mark.parametrize(
    "target,expected_name",
    (
        (SpineJsonTarget.SPINE_4_0, "Hero_merged_spine_4.0.64.json"),
        (SpineJsonTarget.SPINE_4_1, "Hero_merged_spine_4.1.24.json"),
        (SpineJsonTarget.SPINE_4_2, "Hero_merged_spine_4.2.43.json"),
    ),
)
def test_single_ui_settings_append_exact_spine_version_once(
    tmp_path: Path,
    target: SpineJsonTarget,
    expected_name: str,
) -> None:
    source = SimpleNamespace(name="Hero")
    profile = _ObjectExportProfile(source, "Hero", 0, 0, False)
    scene = _scene_profile(tmp_path, target)

    settings = _settings_from_profiles(
        profile,
        scene,
        json_output_stem="Hero_merged",
    )
    repeated = _settings_from_profiles(
        profile,
        scene,
        json_output_stem=settings.json_output_stem,
    )

    assert resolve_a1_output_paths("Hero", settings).json_path.name == expected_name
    assert repeated.json_output_stem == settings.json_output_stem


@pytest.mark.parametrize(
    "target,expected_token",
    (
        (SpineJsonTarget.SPINE_4_0, "4.0.64"),
        (SpineJsonTarget.SPINE_4_1, "4.1.24"),
    ),
)
def test_multi_ui_plan_appends_exact_spine_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target: SpineJsonTarget,
    expected_token: str,
) -> None:
    objects = (
        SimpleNamespace(name="Cone"),
        SimpleNamespace(name="Cone.001"),
        SimpleNamespace(name="Cone.002"),
    )
    profiles = tuple(
        _ObjectExportProfile(
            source_object=obj,
            object_name=obj.name,
            sequence_start_frame=0,
            sequence_frame_count=0,
            connect_enabled=True,
        )
        for obj in objects
    )
    scene = _scene_profile(tmp_path, target)

    monkeypatch.setattr(
        a1_ui_export_plan,
        "_ordered_selected_meshes",
        lambda _context: objects,
    )
    monkeypatch.setattr(
        a1_ui_export_plan,
        "_capture_scene_profile",
        lambda _scene: scene,
    )
    monkeypatch.setattr(
        a1_ui_export_plan,
        "_capture_selected_profiles",
        lambda _objects: profiles,
    )

    plan = a1_ui_export_plan.build_selected_ui_export_plan(
        SimpleNamespace(scene=object())
    )

    assert plan.settings.output_stem == (
        f"Cone_plus_2_objects_spine_{expected_token}"
    )
    assert plan.settings.json_path.name == (
        f"Cone_plus_2_objects_spine_{expected_token}.json"
    )
