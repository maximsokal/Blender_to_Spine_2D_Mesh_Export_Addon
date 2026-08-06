from __future__ import annotations

from pathlib import Path

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1SingleObjectExportSettings,
    ExportSettings,
    calculate_a1_object_bake_main_position_pixels,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    project_a1_mesh_snapshot_camera,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import (
    A1ProjectionDirection,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import calculate_uniform_scale

from test_active_camera_projection import _perspective_frame, _snapshot


def _settings(output_directory: Path) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=200,
            texture_height=100,
            output_directory=output_directory,
        ),
        projection_direction=A1ProjectionDirection.ACTIVE_CAMERA,
        use_world_location_for_main_bone=True,
    )


def test_active_camera_normal_main_position_is_projected_object_origin(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    uniform_scale = calculate_uniform_scale(
        settings.export.texture_width,
        settings.export.texture_height,
        settings.rig_scale_mode,
    )
    projected = project_a1_mesh_snapshot_camera(
        _snapshot(translation=(0.5, -0.25, -5.0)),
        _perspective_frame(width=200, height=100),
        uniform_scale=uniform_scale,
    )

    main_position = calculate_a1_object_bake_main_position_pixels(
        projected.snapshot,
        settings,
    )

    assert main_position == (
        projected.projected_origin.u,
        projected.projected_origin.v,
    )
    assert main_position != (0.0, 0.0)
    assert projected.snapshot.world_matrix[3] * uniform_scale == main_position[0]
    assert projected.snapshot.world_matrix[7] * uniform_scale == main_position[1]


def test_connected_active_camera_normal_keeps_local_object_origin(
    tmp_path: Path,
) -> None:
    settings = A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=200,
            texture_height=100,
            output_directory=tmp_path,
        ),
        projection_direction=A1ProjectionDirection.ACTIVE_CAMERA,
        use_world_location_for_main_bone=False,
    )
    uniform_scale = calculate_uniform_scale(
        settings.export.texture_width,
        settings.export.texture_height,
        settings.rig_scale_mode,
    )
    projected = project_a1_mesh_snapshot_camera(
        _snapshot(translation=(0.5, -0.25, -5.0)),
        _perspective_frame(width=200, height=100),
        uniform_scale=uniform_scale,
    )

    assert calculate_a1_object_bake_main_position_pixels(
        projected.snapshot,
        settings,
    ) == (0.0, 0.0)
    assert projected.projected_origin.u != 0.0
    assert projected.projected_origin.v != 0.0
