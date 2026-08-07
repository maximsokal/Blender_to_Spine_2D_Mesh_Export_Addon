"""Pure contracts for applying signed-axis projection to normalized MeshSnapshot data."""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1SingleObjectExportSettings,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    A1MeshAxisProjectionError,
    EdgeId,
    FaceId,
    LoopId,
    MeshEdge,
    MeshFace,
    MeshLoop,
    MeshSnapshot,
    MeshVertex,
    SourceEdgeId,
    SourceFaceId,
    SourceLoopId,
    SourceVertexId,
    VertexId,
    project_a1_mesh_snapshot_axis,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import (
    A1ProjectionDirection,
    A1ProjectionError,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import (
    A1RigProfile,
    A1RigSetupPoseMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import SpineJsonTarget


_OBJECT_ID = "AxisFixture"
_NORMAL = (-1.0 / sqrt(6.0), -2.0 / sqrt(6.0), 1.0 / sqrt(6.0))


def _translation_matrix(x: float, y: float, z: float) -> tuple[float, ...]:
    return (
        1.0,
        0.0,
        0.0,
        x,
        0.0,
        1.0,
        0.0,
        y,
        0.0,
        0.0,
        1.0,
        z,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def _snapshot() -> MeshSnapshot:
    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(_OBJECT_ID, index),
            position=position,
            normal=_NORMAL,
        )
        for index, position in enumerate(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 1.0),
                (0.0, 1.0, 2.0),
            )
        )
    )
    edges = (
        MeshEdge(
            id=EdgeId(0),
            source_id=SourceEdgeId(_OBJECT_ID, 0),
            vertex_ids=(VertexId(0), VertexId(1)),
        ),
        MeshEdge(
            id=EdgeId(1),
            source_id=SourceEdgeId(_OBJECT_ID, 1),
            vertex_ids=(VertexId(1), VertexId(2)),
        ),
        MeshEdge(
            id=EdgeId(2),
            source_id=SourceEdgeId(_OBJECT_ID, 2),
            vertex_ids=(VertexId(2), VertexId(0)),
        ),
    )
    loops = tuple(
        MeshLoop(
            id=LoopId(index),
            source_id=SourceLoopId(_OBJECT_ID, 0, index),
            vertex_id=VertexId(index),
            edge_id=EdgeId(index),
        )
        for index in range(3)
    )
    face = MeshFace(
        id=FaceId(0),
        source_id=SourceFaceId(_OBJECT_ID, 0),
        loop_ids=(LoopId(0), LoopId(1), LoopId(2)),
        material_index=0,
        normal=_NORMAL,
    )
    return MeshSnapshot(
        snapshot_id="axis-fixture",
        source_object_id=_OBJECT_ID,
        object_name=_OBJECT_ID,
        vertices=vertices,
        edges=edges,
        loops=loops,
        faces=(face,),
        world_matrix=_translation_matrix(5.0, -7.0, 11.0),
    )


def _expected(
    value: tuple[float, float, float],
    direction: A1ProjectionDirection,
) -> tuple[float, float, float]:
    x, y, z = value
    return {
        A1ProjectionDirection.POSITIVE_X: (y, z, x),
        A1ProjectionDirection.NEGATIVE_X: (-y, z, -x),
        A1ProjectionDirection.POSITIVE_Y: (-x, z, y),
        A1ProjectionDirection.NEGATIVE_Y: (x, z, -y),
        A1ProjectionDirection.POSITIVE_Z: (x, y, z),
        A1ProjectionDirection.NEGATIVE_Z: (-x, y, -z),
    }[direction]


@pytest.mark.parametrize(
    "direction",
    tuple(item for item in A1ProjectionDirection if item.axis_aligned),
)
def test_axis_projection_transforms_geometry_normals_and_origin(
    direction: A1ProjectionDirection,
) -> None:
    source = _snapshot()
    original_positions = tuple(vertex.position for vertex in source.vertices)

    result = project_a1_mesh_snapshot_axis(source, direction)

    expected_origin = _expected((5.0, -7.0, 11.0), direction)
    assert result.projected_origin.canonical_position == expected_origin
    assert result.snapshot.world_matrix == _translation_matrix(*expected_origin)
    assert tuple(vertex.position for vertex in result.snapshot.vertices) == tuple(
        _expected(position, direction) for position in original_positions
    )
    assert tuple(vertex.normal for vertex in result.snapshot.vertices) == tuple(
        _expected(_NORMAL, direction) for _ in source.vertices
    )
    assert result.snapshot.faces[0].normal == _expected(_NORMAL, direction)

    assert result.snapshot.edges == source.edges
    assert result.snapshot.loops == source.loops
    assert result.snapshot.uv_layer_names == source.uv_layer_names
    assert tuple(vertex.position for vertex in source.vertices) == original_positions


def test_positive_z_is_exact_compatibility_path() -> None:
    source = _snapshot()

    result = project_a1_mesh_snapshot_axis(
        source,
        A1ProjectionDirection.POSITIVE_Z,
    )

    assert result.snapshot is source
    assert result.changed is False
    assert result.projected_origin.canonical_position == (5.0, -7.0, 11.0)


@pytest.mark.parametrize(
    "direction",
    (
        A1ProjectionDirection.POSITIVE_X,
        A1ProjectionDirection.NEGATIVE_X,
        A1ProjectionDirection.POSITIVE_Y,
        A1ProjectionDirection.NEGATIVE_Y,
        A1ProjectionDirection.NEGATIVE_Z,
    ),
)
def test_non_default_axis_creates_valid_replacement(
    direction: A1ProjectionDirection,
) -> None:
    source = _snapshot()

    result = project_a1_mesh_snapshot_axis(source, direction)

    assert result.snapshot is not source
    assert result.changed is True
    assert result.snapshot.snapshot_id == source.snapshot_id
    assert result.snapshot.source_object_id == source.source_object_id
    assert result.snapshot.object_name == source.object_name


@pytest.mark.parametrize(
    "direction",
    (
        A1ProjectionDirection.ACTIVE_CAMERA,
        A1ProjectionDirection.ACTIVE_CAMERA_CAMERA_ROOT,
    ),
)
def test_axis_projection_rejects_active_camera_without_camera_frame(
    direction: A1ProjectionDirection,
) -> None:
    with pytest.raises(
        A1ProjectionError,
        match="Active Camera projection requires an evaluated Blender camera frame",
    ):
        project_a1_mesh_snapshot_axis(_snapshot(), direction)


def test_axis_projection_rejects_non_normalized_world_matrix() -> None:
    source = _snapshot()
    invalid = MeshSnapshot(
        snapshot_id=source.snapshot_id,
        source_object_id=source.source_object_id,
        object_name=source.object_name,
        vertices=source.vertices,
        edges=source.edges,
        loops=source.loops,
        faces=source.faces,
        world_matrix=(
            2.0,
            0.0,
            0.0,
            5.0,
            0.0,
            1.0,
            0.0,
            -7.0,
            0.0,
            0.0,
            1.0,
            11.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    )

    with pytest.raises(A1MeshAxisProjectionError, match="translation only"):
        project_a1_mesh_snapshot_axis(
            invalid,
            A1ProjectionDirection.POSITIVE_Z,
        )


def _export_settings() -> ExportSettings:
    return ExportSettings(
        texture_width=32,
        texture_height=32,
        output_directory=Path("axis-projection-output"),
        spine_version=SpineJsonTarget.SPINE_4_2.exact_version,
        rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE.value,
    )


def test_single_object_settings_default_to_positive_z() -> None:
    settings = A1SingleObjectExportSettings(export=_export_settings())

    assert settings.projection_direction is A1ProjectionDirection.POSITIVE_Z


def test_single_object_settings_preserve_axes_and_object_root_camera() -> None:
    directions = tuple(
        direction
        for direction in A1ProjectionDirection
        if not direction.camera_root
    )
    for direction in directions:
        settings = A1SingleObjectExportSettings(
            export=_export_settings(),
            projection_direction=direction,
        )
        assert settings.projection_direction is direction


def test_single_object_settings_normalize_camera_root_to_shared_camera_geometry() -> None:
    settings = A1SingleObjectExportSettings(
        export=_export_settings(),
        projection_direction=A1ProjectionDirection.ACTIVE_CAMERA_CAMERA_ROOT,
    )

    assert settings.projection_direction is A1ProjectionDirection.ACTIVE_CAMERA
    assert settings.rig_setup_pose_mode is A1RigSetupPoseMode.PREPROJECTED_SCREEN


def test_single_object_settings_reject_untyped_projection_direction() -> None:
    with pytest.raises(TypeError, match="A1ProjectionDirection"):
        A1SingleObjectExportSettings(
            export=_export_settings(),
            projection_direction="POSITIVE_X",  # type: ignore[arg-type]
        )
