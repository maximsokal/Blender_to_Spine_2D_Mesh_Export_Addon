"""Scope regressions for Normal / UV Segments Active Camera preparation."""

from __future__ import annotations

from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1MultiObjectMode,
    A1SingleObjectExportSettings,
    ExportSettings,
    resolve_a1_multi_object_preparation_settings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import A1ProjectionDirection
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import A1RigProfile
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import SpineJsonTarget


def _settings(tmp_path: Path) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=128,
            texture_height=96,
            output_directory=tmp_path,
            spine_version=SpineJsonTarget.SPINE_4_2.exact_version,
            rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE.value,
        ),
        projection_direction=A1ProjectionDirection.ACTIVE_CAMERA,
    )


def test_standalone_active_camera_settings_are_preserved(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    resolved = resolve_a1_multi_object_preparation_settings(
        settings,
        A1MultiObjectMode.STANDALONE,
    )

    assert resolved is settings
    assert resolved.use_world_location_for_main_bone is True
    assert resolved.projection_direction is A1ProjectionDirection.ACTIVE_CAMERA


def test_connected_active_camera_uses_anchor_relative_projected_origin(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)

    resolved = resolve_a1_multi_object_preparation_settings(
        settings,
        A1MultiObjectMode.CONNECTED,
    )

    assert resolved is not settings
    assert settings.use_world_location_for_main_bone is True
    assert resolved.use_world_location_for_main_bone is False
    assert resolved.projection_direction is A1ProjectionDirection.ACTIVE_CAMERA
    assert resolved.export is settings.export


def test_connected_active_camera_preserves_already_local_main_setting(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    local_settings = A1SingleObjectExportSettings(
        **{
            **settings.__dict__,
            "use_world_location_for_main_bone": False,
        }
    )

    resolved = resolve_a1_multi_object_preparation_settings(
        local_settings,
        A1MultiObjectMode.CONNECTED,
    )

    assert resolved is local_settings
    assert resolved.projection_direction is A1ProjectionDirection.ACTIVE_CAMERA


def test_mixed_must_still_resolve_to_explicit_subgroups(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="MIXED mode"):
        resolve_a1_multi_object_preparation_settings(
            _settings(tmp_path),
            A1MultiObjectMode.MIXED,
        )
