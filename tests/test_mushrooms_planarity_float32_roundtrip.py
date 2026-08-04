"""Regress Blender float32 round-trip drift in captured mushrooms fixtures.

The Blender headless runner writes Python ``float`` coordinates into ``Mesh`` datablocks,
which store vertex coordinates as 32-bit floats. Re-reading those values changes both the
Newell-plane distance and the bounding-box diagonal by a few ULPs. Ratio checks must use
a tolerance propagated from the accepted distance/scale tolerances instead of imposing a
stricter unrelated constant.
"""

from __future__ import annotations

from math import sqrt
from struct import pack, unpack

from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    EdgeId,
    FaceId,
    LoopId,
    MeshEdge,
    MeshFace,
    MeshLoop,
    MeshSnapshot,
    MeshSnapshotValidator,
    MeshVertex,
    SourceEdgeId,
    SourceFaceId,
    SourceLoopId,
    SourceVertexId,
    VertexId,
    triangulate_snapshot,
)


_DISTANCE_ABS_TOLERANCE = 1.0e-11
_SCALE_ABS_TOLERANCE = 1.0e-8
_RETIRED_RATIO_ABS_TOLERANCE = 1.0e-10

_FIXTURES = (
    (
        "Cube.012",
        0.09277534946637461,
        0.00030260673225328884,
        7.565148185365435e-05,
        0.13120450643194492,
    ),
    (
        "Plane.008",
        0.0164835268501562,
        0.000379143881919382,
        9.477343601658832e-05,
        0.023314310303391664,
    ),
)


def _float32(value: float) -> float:
    """Round one Python float exactly as a Blender Mesh coordinate is stored."""

    return unpack("<f", pack("<f", float(value)))[0]


def _snapshot_after_blender_float32_roundtrip(
    object_id: str,
    *,
    side_length: float,
    warp_height: float,
) -> MeshSnapshot:
    """Build the captured quad from values quantized to Blender-style float32."""

    half = _float32(float(side_length) / 2.0)
    warp = _float32(warp_height)
    positions = (
        (-half, -half, 0.0),
        (half, -half, 0.0),
        (half, half, warp),
        (-half, half, 0.0),
    )
    edge_pairs = (
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 3),
    )
    edge_id_by_pair = {
        pair: EdgeId(index)
        for index, pair in enumerate(edge_pairs)
    }

    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(object_id, index),
            position=position,
            normal=(0.0, 0.0, 1.0),
        )
        for index, position in enumerate(positions)
    )
    edges = tuple(
        MeshEdge(
            id=edge_id_by_pair[pair],
            source_id=SourceEdgeId(
                object_id,
                edge_id_by_pair[pair].index,
            ),
            vertex_ids=(VertexId(pair[0]), VertexId(pair[1])),
        )
        for pair in edge_pairs
    )

    face_vertices = (0, 1, 2, 3)
    loops = tuple(
        MeshLoop(
            id=LoopId(corner_index),
            source_id=SourceLoopId(object_id, 0, corner_index),
            vertex_id=VertexId(vertex_index),
            edge_id=edge_id_by_pair[
                tuple(
                    sorted(
                        (
                            vertex_index,
                            face_vertices[
                                (corner_index + 1) % len(face_vertices)
                            ],
                        )
                    )
                )
            ],
        )
        for corner_index, vertex_index in enumerate(face_vertices)
    )
    face = MeshFace(
        id=FaceId(0),
        source_id=SourceFaceId(object_id, 0),
        loop_ids=tuple(loop.id for loop in loops),
        material_index=0,
        normal=(0.0, 0.0, 1.0),
        smooth=True,
    )
    snapshot = MeshSnapshot(
        snapshot_id=f"{object_id}:float32-roundtrip",
        source_object_id=object_id,
        object_name=object_id,
        vertices=vertices,
        edges=edges,
        loops=loops,
        faces=(face,),
    )
    MeshSnapshotValidator().validate_or_raise(snapshot)
    return snapshot


def _planarity_metrics(snapshot: MeshSnapshot) -> tuple[float, float]:
    face = snapshot.faces[0]
    loop_map = snapshot.loop_by_id()
    vertex_map = snapshot.vertex_by_id()
    points = tuple(
        vertex_map[loop_map[loop_id].vertex_id].position
        for loop_id in face.loop_ids
    )

    newell = [0.0, 0.0, 0.0]
    for index, current in enumerate(points):
        following = points[(index + 1) % len(points)]
        newell[0] += (
            (current[1] - following[1])
            * (current[2] + following[2])
        )
        newell[1] += (
            (current[2] - following[2])
            * (current[0] + following[0])
        )
        newell[2] += (
            (current[0] - following[0])
            * (current[1] + following[1])
        )

    magnitude = sqrt(sum(component * component for component in newell))
    normal = tuple(component / magnitude for component in newell)
    centroid = tuple(
        sum(point[axis] for point in points) / float(len(points))
        for axis in range(3)
    )
    maximum_distance = max(
        abs(
            sum(
                (point[axis] - centroid[axis]) * normal[axis]
                for axis in range(3)
            )
        )
        for point in points
    )
    extents = tuple(
        max(point[axis] for point in points)
        - min(point[axis] for point in points)
        for axis in range(3)
    )
    polygon_scale = sqrt(sum(extent * extent for extent in extents))
    return maximum_distance, polygon_scale


def _propagated_ratio_tolerance(
    expected_distance: float,
    expected_scale: float,
) -> float:
    """Bound ratio drift implied by accepted distance and scale errors."""

    lower_scale = expected_scale - _SCALE_ABS_TOLERANCE
    if lower_scale <= 0.0:
        raise ValueError("expected_scale is too small for the tolerance budget")

    distance_term = _DISTANCE_ABS_TOLERANCE / lower_scale
    scale_term = (
        abs(expected_distance)
        * _SCALE_ABS_TOLERANCE
        / (expected_scale * lower_scale)
    )
    return distance_term + scale_term


def test_captured_mushrooms_float32_roundtrip_preserves_valid_triangulation() -> None:
    plane_ratio_delta = None

    for (
        object_id,
        side_length,
        warp_height,
        expected_distance,
        expected_scale,
    ) in _FIXTURES:
        snapshot = _snapshot_after_blender_float32_roundtrip(
            object_id,
            side_length=side_length,
            warp_height=warp_height,
        )
        actual_distance, actual_scale = _planarity_metrics(snapshot)
        expected_ratio = expected_distance / expected_scale
        actual_ratio = actual_distance / actual_scale
        ratio_delta = abs(actual_ratio - expected_ratio)
        propagated_tolerance = _propagated_ratio_tolerance(
            expected_distance,
            expected_scale,
        )

        assert abs(actual_distance - expected_distance) <= (
            _DISTANCE_ABS_TOLERANCE
        )
        assert abs(actual_scale - expected_scale) <= _SCALE_ABS_TOLERANCE
        assert ratio_delta <= propagated_tolerance
        assert propagated_tolerance < 1.0e-8

        triangulated = triangulate_snapshot(snapshot)
        assert len(triangulated.snapshot.faces) == 2
        assert len(triangulated.generated_edge_ids) == 1

        if object_id == "Plane.008":
            plane_ratio_delta = ratio_delta

    # This is the exact runner regression: Blender float32 quantization exceeds the
    # retired unrelated 1e-10 ratio threshold while remaining inside both source metric
    # tolerances and their mathematically propagated ratio budget.
    assert plane_ratio_delta is not None
    assert plane_ratio_delta > _RETIRED_RATIO_ABS_TOLERANCE
