from pathlib import Path
from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1GeometryPreparationSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_bridge import (
    _build_sources,
    _common_object_settings,
    _resolve_geometry_settings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import A1AngularMode


def _scene(**overrides):
    values = {
        "spine2d_seam_maker_mode": "AUTO",
        "spine2d_angle_limit": 30.0,
        "spine2d_control_icons": True,
        "spine2d_export_preview_animation": True,
        "render": SimpleNamespace(engine="CYCLES"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _object(name, *, frame_start=0, frame_count=0):
    return SimpleNamespace(
        name=name,
        name_full=name,
        spine2d_bake_settings=SimpleNamespace(
            bake_frame_start=frame_start,
            frames_for_render=frame_count,
        ),
    )


def _settings(obj, scene, tmp_path: Path):
    return _common_object_settings(
        obj,
        scene,
        output_directory=tmp_path,
        texture_size=64,
        images_relative_path="images",
        sequence_start_frame=0,
        sequence_frame_count=0,
    )


def test_missing_scene_angular_properties_preserve_legacy_default(tmp_path):
    settings = _settings(_object("Legacy"), _scene(), tmp_path)

    assert settings.geometry == A1GeometryPreparationSettings()


def test_retired_legacy_scene_mode_fails_closed():
    with pytest.raises(ValueError, match="Unsupported Spine2D angular mode"):
        _resolve_geometry_settings(
            _scene(
                spine2d_angular_mode="LEGACY_SEED_CONE",
                spine2d_local_angle_limit=87.0,
            )
        )


def test_hybrid_scene_properties_reach_single_object_geometry(tmp_path):
    settings = _settings(
        _object("Folded"),
        _scene(
            spine2d_angular_mode="SEED_CONE_AND_LOCAL_DIHEDRAL",
            spine2d_local_angle_limit=17.5,
        ),
        tmp_path,
    )

    assert (
        settings.geometry.angular_mode
        is A1AngularMode.SEED_CONE_AND_LOCAL_DIHEDRAL
    )
    assert settings.geometry.local_angle_limit_degrees == 17.5
    assert settings.resolved_geometry_settings().segmentation.angle_limit_degrees == 30.0


def test_hybrid_scene_properties_are_shared_by_every_multi_source(tmp_path):
    scene = _scene(
        spine2d_angular_mode="SEED_CONE_AND_LOCAL_DIHEDRAL",
        spine2d_local_angle_limit=22.0,
    )
    sources = _build_sources(
        (_object("A"), _object("B", frame_start=4, frame_count=3)),
        scene,
        output_directory=tmp_path,
        texture_size=128,
        images_relative_path="images",
    )

    assert len(sources) == 2
    assert all(
        source.settings.geometry.angular_mode
        is A1AngularMode.SEED_CONE_AND_LOCAL_DIHEDRAL
        for source in sources
    )
    assert all(
        source.settings.geometry.local_angle_limit_degrees == 22.0
        for source in sources
    )
    assert sources[1].settings.export.sequence_start_frame == 4
    assert sources[1].settings.export.sequence_frame_count == 3


def test_bridge_rejects_unknown_angular_mode_before_export():
    with pytest.raises(ValueError, match="Unsupported Spine2D angular mode"):
        _resolve_geometry_settings(
            _scene(spine2d_angular_mode="PAIRWISE_DRIFT")
        )


@pytest.mark.parametrize("value", (-1.0, 181.0, float("inf"), float("nan")))
def test_bridge_rejects_invalid_local_dihedral_limit(value):
    with pytest.raises(ValueError):
        _resolve_geometry_settings(
            _scene(
                spine2d_angular_mode="SEED_CONE_AND_LOCAL_DIHEDRAL",
                spine2d_local_angle_limit=value,
            )
        )
