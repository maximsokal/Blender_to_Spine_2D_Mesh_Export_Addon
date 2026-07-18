from pathlib import Path
from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_bridge import (
    _build_sources,
    _common_object_settings,
)


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


def test_missing_scene_threshold_preserves_legacy_default(tmp_path):
    settings = _settings(_object("Legacy"), _scene(), tmp_path)

    assert settings.bake_execution.projection_alpha_threshold == 1.0 / 255.0


def test_scene_threshold_reaches_single_object_execution_settings(tmp_path):
    settings = _settings(
        _object("TightCrop"),
        _scene(spine2d_projection_alpha_threshold=0.125),
        tmp_path,
    )

    assert settings.bake_execution.projection_alpha_threshold == 0.125


def test_scene_threshold_is_shared_by_every_multi_source(tmp_path):
    sources = _build_sources(
        (_object("A"), _object("B", frame_start=3, frame_count=2)),
        _scene(spine2d_projection_alpha_threshold=0.25),
        output_directory=tmp_path,
        texture_size=128,
        images_relative_path="images",
    )

    assert len(sources) == 2
    assert all(
        source.settings.bake_execution.projection_alpha_threshold == 0.25
        for source in sources
    )


@pytest.mark.parametrize("value", (-0.1, 1.1, float("inf"), float("nan"), True))
def test_bridge_rejects_invalid_scene_threshold(value, tmp_path):
    with pytest.raises(ValueError, match="projection_alpha_threshold"):
        _settings(
            _object("Invalid"),
            _scene(spine2d_projection_alpha_threshold=value),
            tmp_path,
        )
