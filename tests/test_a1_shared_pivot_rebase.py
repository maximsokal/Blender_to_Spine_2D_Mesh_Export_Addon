from __future__ import annotations

from dataclasses import replace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.axis_projection import (
    calculate_a1_projected_snapshot_depth_range,
    project_a1_mesh_snapshot_axis,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.ids import (
    SourceVertexId,
    VertexId,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.model import (
    MeshSnapshot,
    MeshVertex,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.shared_pivot import (
    A1SharedPivotRebaseError,
    rebase_a1_projected_snapshot_origin,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import (
    A1ProjectedPoint,
    A1ProjectionDirection,
    resolve_a1_axis_projection_basis,
)


SIGNED_AXES = (
    A1ProjectionDirection.POSITIVE_X,
    A1ProjectionDirection.NEGATIVE_X,
    A1ProjectionDirection.POSITIVE_Y,
    A1ProjectionDirection.NEGATIVE_Y,
    A1ProjectionDirection.POSITIVE_Z,
    A1ProjectionDirection.NEGATIVE_Z,
)


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
    object_id = "shared_pivot_source"
    positions = (
        (-2.0, 1.0, -3.0),
        (4.0, -5.0, 6.0),
        (0.25, 2.5, 1.75),
    )
    return MeshSnapshot(
        snapshot_id="shared-pivot-source",
        source_object_id=object_id,
        object_name="Shared Pivot Source",
        vertices=tuple(
            MeshVertex(
                id=VertexId(index),
                source_id=SourceVertexId(object_id, index),
                position=position,
                normal=(0.0, 0.0, 1.0),
            )
            for index, position in enumerate(positions)
        ),
        edges=(),
        loops=(),
        faces=(),
        world_matrix=_translation_matrix(10.0, -7.0, 2.0),
    )


def _projected_world_vertices(snapshot: MeshSnapshot) -> tuple[tuple[float, ...], ...]:
    origin = (
        float(snapshot.world_matrix[3]),
        float(snapshot.world_matrix[7]),
        float(snapshot.world_matrix[11]),
    )
    return tuple(
        tuple(origin[axis] + float(vertex.position[axis]) for axis in range(3))
        for vertex in snapshot.vertices
    )


@pytest.mark.parametrize("direction", SIGNED_AXES)
def test_shared_pivot_rebase_preserves_world_geometry_for_every_signed_axis(
    direction: A1ProjectionDirection,
) -> None:
    projected = project_a1_mesh_snapshot_axis(_snapshot(), direction)
    basis = resolve_a1_axis_projection_basis(direction)
    target = basis.project_point((3.5, 4.25, -8.0))

    before_world = _projected_world_vertices(projected.snapshot)
    before_depth = calculate_a1_projected_snapshot_depth_range(projected.snapshot)
    rebased = rebase_a1_projected_snapshot_origin(
        projected.snapshot,
        projected.projected_origin,
        target,
    )
    after_world = _projected_world_vertices(rebased)
    after_depth = calculate_a1_projected_snapshot_depth_range(rebased)

    assert rebased.world_matrix[3] == pytest.approx(target.u)
    assert rebased.world_matrix[7] == pytest.approx(target.v)
    assert rebased.world_matrix[11] == pytest.approx(target.depth)
    for expected, actual in zip(before_world, after_world, strict=True):
        assert actual == pytest.approx(expected)
    assert after_depth.minimum_depth == pytest.approx(before_depth.minimum_depth)
    assert after_depth.maximum_depth == pytest.approx(before_depth.maximum_depth)
    assert tuple(vertex.normal for vertex in rebased.vertices) == tuple(
        vertex.normal for vertex in projected.snapshot.vertices
    )
    assert rebased.edges is projected.snapshot.edges
    assert rebased.loops is projected.snapshot.loops
    assert rebased.faces is projected.snapshot.faces
    assert tuple(vertex.source_id for vertex in rebased.vertices) == tuple(
        vertex.source_id for vertex in projected.snapshot.vertices
    )


@pytest.mark.parametrize("direction", SIGNED_AXES)
def test_shared_pivot_rebase_uses_projected_world_delta_not_axis_specific_hacks(
    direction: A1ProjectionDirection,
) -> None:
    projected = project_a1_mesh_snapshot_axis(_snapshot(), direction)
    basis = resolve_a1_axis_projection_basis(direction)
    target = basis.project_point((-11.0, 9.0, 5.0))
    delta = (
        projected.projected_origin.u - target.u,
        projected.projected_origin.v - target.v,
        projected.projected_origin.depth - target.depth,
    )

    rebased = rebase_a1_projected_snapshot_origin(
        projected.snapshot,
        projected.projected_origin,
        target,
    )

    for before, after in zip(projected.snapshot.vertices, rebased.vertices, strict=True):
        assert after.position == pytest.approx(
            tuple(before.position[axis] + delta[axis] for axis in range(3))
        )


def test_shared_pivot_rebase_is_identity_when_target_is_current_origin() -> None:
    projected = project_a1_mesh_snapshot_axis(
        _snapshot(),
        A1ProjectionDirection.POSITIVE_Z,
    )

    rebased = rebase_a1_projected_snapshot_origin(
        projected.snapshot,
        projected.projected_origin,
        projected.projected_origin,
    )

    assert rebased is projected.snapshot


def test_shared_pivot_rebase_rejects_stale_current_origin() -> None:
    projected = project_a1_mesh_snapshot_axis(
        _snapshot(),
        A1ProjectionDirection.POSITIVE_Z,
    )

    with pytest.raises(A1SharedPivotRebaseError, match="does not match"):
        rebase_a1_projected_snapshot_origin(
            projected.snapshot,
            A1ProjectedPoint(u=999.0, v=0.0, depth=0.0),
            A1ProjectedPoint(u=0.0, v=0.0, depth=0.0),
        )


def test_shared_pivot_rebase_rejects_non_translation_projected_matrix() -> None:
    snapshot = replace(
        _snapshot(),
        world_matrix=(
            2.0,
            0.0,
            0.0,
            10.0,
            0.0,
            1.0,
            0.0,
            -7.0,
            0.0,
            0.0,
            1.0,
            2.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    )

    with pytest.raises(A1SharedPivotRebaseError, match="translation only"):
        rebase_a1_projected_snapshot_origin(
            snapshot,
            A1ProjectedPoint(u=10.0, v=-7.0, depth=2.0),
            A1ProjectedPoint(u=0.0, v=0.0, depth=0.0),
        )
