"""Tests for the shared Analyze/Export UI request-plan owner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1SingleObjectExportSettings,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import a1_ui_export_plan
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_multi_object_contracts import (
    A1MultiObjectSource,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_selection import (
    _ObjectExportProfile,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import (
    A1RigSetupPoseMode,
)


def _settings(name: str) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=128,
            texture_height=128,
            output_directory=Path("ui-plan-output"),
        ),
        prefix=name,
        output_stem=name,
    )


def _source(index: int, name: str) -> A1MultiObjectSource:
    return A1MultiObjectSource(
        source_object=SimpleNamespace(name=name),
        component_id=f"object_{index}:{name}",
        animation_namespace=f"object_{index}",
        settings=_settings(name),
    )


def test_active_ui_plan_explicitly_requests_normalized_single_setup(monkeypatch):
    source_object = SimpleNamespace(name="Cone")
    object_profile = _ObjectExportProfile(source_object, "Cone", 0, 0, False)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        a1_ui_export_plan,
        "_active_mesh",
        lambda _context: source_object,
    )
    monkeypatch.setattr(
        a1_ui_export_plan,
        "_capture_scene_profile",
        lambda _scene: object(),
    )
    monkeypatch.setattr(
        a1_ui_export_plan,
        "_capture_object_profile",
        lambda *_args, **_kwargs: object_profile,
    )

    def _capture_settings(_object_profile, _scene_profile, **kwargs):
        captured.update(kwargs)
        return _settings("Cone")

    monkeypatch.setattr(
        a1_ui_export_plan,
        "_settings_from_profiles",
        _capture_settings,
    )

    plan = a1_ui_export_plan.build_active_ui_export_plan(
        SimpleNamespace(scene=SimpleNamespace())
    )

    assert plan.source_object is source_object
    assert captured["rig_setup_pose_mode"] is A1RigSetupPoseMode.NORMALIZED_SINGLE


def test_multi_ui_plan_preserves_explicit_mixed_subgroup_order():
    """Internal callers may still build an explicit developer-only mixed plan."""

    first = _source(1, "First")
    second = _source(2, "Second")
    third = _source(3, "Third")
    plan = a1_ui_export_plan.A1UiMultiExportPlan(
        connected_sources=(first, second),
        standalone_sources=(third,),
        settings=A1MultiObjectExportSettings(
            output_directory=Path("ui-plan-output"),
            output_stem="mixed",
            mode=A1MultiObjectMode.MIXED,
            anchor_component_id=first.component_id,
        ),
    )

    assert plan.all_sources == (first, second, third)
    assert plan.settings.mode is A1MultiObjectMode.MIXED


def test_multi_ui_plan_rejects_mode_and_partition_disagreement():
    first = _source(1, "First")
    second = _source(2, "Second")

    with pytest.raises(ValueError, match="CONNECTED UI plan"):
        a1_ui_export_plan.A1UiMultiExportPlan(
            connected_sources=(),
            standalone_sources=(first, second),
            settings=A1MultiObjectExportSettings(
                output_directory=Path("ui-plan-output"),
                output_stem="wrong",
                mode=A1MultiObjectMode.CONNECTED,
            ),
        )


def test_capture_selected_profiles_ignores_persisted_connect_values(monkeypatch):
    objects = (
        SimpleNamespace(
            name="First",
            spine2d_bake_settings=SimpleNamespace(
                bake_frame_start=3,
                frames_for_render=4,
            ),
            spine2d_connect_settings=SimpleNamespace(enabled=True),
        ),
        SimpleNamespace(
            name="Second",
            spine2d_bake_settings=SimpleNamespace(
                bake_frame_start=5,
                frames_for_render=6,
            ),
            spine2d_connect_settings=SimpleNamespace(enabled=True),
        ),
    )
    captured: list[dict[str, object]] = []

    def _capture(obj, **kwargs):
        captured.append(dict(kwargs))
        return _ObjectExportProfile(
            obj,
            obj.name,
            int(kwargs["sequence_start_frame"]),
            int(kwargs["sequence_frame_count"]),
            bool(kwargs["connect_enabled"]),
        )

    monkeypatch.setattr(a1_ui_export_plan, "_capture_object_profile", _capture)

    profiles = a1_ui_export_plan._capture_selected_profiles(objects)

    assert tuple(profile.connect_enabled for profile in profiles) == (False, False)
    assert captured == [
        {
            "sequence_start_frame": 3,
            "sequence_frame_count": 4,
            "connect_enabled": False,
        },
        {
            "sequence_start_frame": 5,
            "sequence_frame_count": 6,
            "connect_enabled": False,
        },
    ]


@pytest.mark.parametrize(
    "profiles",
    (
        (
            _ObjectExportProfile(SimpleNamespace(name="First"), "First", 0, 0, True),
            _ObjectExportProfile(SimpleNamespace(name="Second"), "Second", 0, 0, False),
        ),
        (
            _ObjectExportProfile(SimpleNamespace(name="First"), "First", 0, 0, True),
            _ObjectExportProfile(SimpleNamespace(name="Second"), "Second", 0, 0, True),
            _ObjectExportProfile(SimpleNamespace(name="Third"), "Third", 0, 0, False),
        ),
    ),
)
def test_public_selected_ui_plan_is_always_standalone(monkeypatch, profiles):
    objects = tuple(profile.source_object for profile in profiles)
    sources = tuple(
        _source(index, profile.object_name)
        for index, profile in enumerate(profiles, start=1)
    )
    monkeypatch.setattr(
        a1_ui_export_plan,
        "_ordered_selected_meshes",
        lambda _context: objects,
    )
    monkeypatch.setattr(
        a1_ui_export_plan,
        "_capture_scene_profile",
        lambda _scene: SimpleNamespace(output_directory=Path("ui-plan-output")),
    )
    monkeypatch.setattr(
        a1_ui_export_plan,
        "_capture_selected_profiles",
        lambda _objects: profiles,
    )
    monkeypatch.setattr(
        a1_ui_export_plan,
        "_build_sources_from_profiles",
        lambda _profiles, _scene: sources,
    )

    plan = a1_ui_export_plan.build_selected_ui_export_plan(
        SimpleNamespace(scene=object())
    )

    assert plan.settings.mode is A1MultiObjectMode.STANDALONE
    assert plan.settings.anchor_component_id is None
    assert plan.connected_sources == ()
    assert plan.standalone_sources == sources
    assert plan.issues == ()
