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
    IssueSeverity,
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


def _persisted_connect_object(name: str, *, enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        name_full=name,
        spine2d_connect_settings=SimpleNamespace(enabled=enabled),
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


def test_profile_capture_ignores_persisted_connect_flags_in_production():
    objects = (
        _persisted_connect_object("First", enabled=True),
        _persisted_connect_object("Second", enabled=True),
        _persisted_connect_object("Third", enabled=False),
    )

    production = a1_ui_export_plan._capture_selected_profiles(
        objects,
        allow_connected=False,
    )
    development = a1_ui_export_plan._capture_selected_profiles(
        objects,
        allow_connected=True,
    )

    assert tuple(profile.connect_enabled for profile in production) == (
        False,
        False,
        False,
    )
    assert tuple(profile.connect_enabled for profile in development) == (
        True,
        True,
        False,
    )


def test_production_selected_plan_is_standalone_with_stale_connected_properties(
    monkeypatch,
):
    objects = (
        _persisted_connect_object("First", enabled=True),
        _persisted_connect_object("Second", enabled=True),
    )
    sources = (_source(1, "First"), _source(2, "Second"))
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
        "_build_sources_from_profiles",
        lambda _profiles, _scene: sources,
    )

    plan = a1_ui_export_plan.build_selected_ui_export_plan(
        SimpleNamespace(scene=object())
    )

    assert plan.settings.mode is A1MultiObjectMode.STANDALONE
    assert plan.connected_sources == ()
    assert plan.standalone_sources == sources
    assert plan.settings.anchor_component_id is None
    assert plan.issues == ()


def test_single_connect_selection_falls_back_to_standalone_once(monkeypatch):
    objects = (SimpleNamespace(name="First"), SimpleNamespace(name="Second"))
    profiles = (
        _ObjectExportProfile(objects[0], "First", 0, 0, True),
        _ObjectExportProfile(objects[1], "Second", 0, 0, False),
    )
    sources = (_source(1, "First"), _source(2, "Second"))
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

    def _profiles(_objects, *, allow_connected):
        assert allow_connected is True
        return profiles

    monkeypatch.setattr(
        a1_ui_export_plan,
        "_capture_selected_profiles",
        _profiles,
    )
    monkeypatch.setattr(
        a1_ui_export_plan,
        "_build_sources_from_profiles",
        lambda _profiles, _scene: sources,
    )

    plan = a1_ui_export_plan.build_development_connected_ui_export_plan(
        SimpleNamespace(scene=object())
    )

    assert plan.settings.mode is A1MultiObjectMode.STANDALONE
    assert plan.connected_sources == ()
    assert plan.standalone_sources == sources
    assert len(plan.issues) == 1
    assert plan.issues[0].severity is IssueSeverity.WARNING
    assert plan.issues[0].code == "A1_SINGLE_CONNECT_FALLBACK"


def test_two_connected_and_one_standalone_build_development_mixed_plan(monkeypatch):
    objects = (
        SimpleNamespace(name="First"),
        SimpleNamespace(name="Second"),
        SimpleNamespace(name="Third"),
    )
    profiles = (
        _ObjectExportProfile(objects[0], "First", 0, 0, True),
        _ObjectExportProfile(objects[1], "Second", 0, 0, True),
        _ObjectExportProfile(objects[2], "Third", 0, 0, False),
    )
    sources = (
        _source(1, "First"),
        _source(2, "Second"),
        _source(3, "Third"),
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

    def _profiles(_objects, *, allow_connected):
        assert allow_connected is True
        return profiles

    monkeypatch.setattr(
        a1_ui_export_plan,
        "_capture_selected_profiles",
        _profiles,
    )
    monkeypatch.setattr(
        a1_ui_export_plan,
        "_build_sources_from_profiles",
        lambda _profiles, _scene: sources,
    )

    plan = a1_ui_export_plan.build_development_connected_ui_export_plan(
        SimpleNamespace(scene=object())
    )

    assert plan.settings.mode is A1MultiObjectMode.MIXED
    assert plan.connected_sources == sources[:2]
    assert plan.standalone_sources == sources[2:]
    assert plan.settings.anchor_component_id == sources[0].component_id
    assert plan.issues == ()
