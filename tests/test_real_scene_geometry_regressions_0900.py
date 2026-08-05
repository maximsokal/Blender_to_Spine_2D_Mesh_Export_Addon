"""Regressions captured from real 0.90.0 mushrooms and flower-shop scenes."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from math import isfinite, sqrt

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application.a1_geometry_preparation import (
    A1GeometryPreparationError,
    _validate_prepared_coverage,
    prepare_a1_geometry_regions,
)
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
    TriangulationError,
    TriangulationSettings,
    VertexId,
    triangulate_snapshot,
)


Vector3 = tuple[float, float, float]

_MUSHROOMS_OBJECT_ID = "Plane.008"
_MUSHROOMS_CUBE_OBJECT_ID = "Cube.012"
_FLOWER_SHOP_OBJECT_ID = "banco"

# Earlier Plane.008 capture: one lifted corner on a unit quad yields the logged
# centroid-plane residue approximately 0.00018954694198463555.
_MUSHROOMS_WARP_HEIGHT = 0.0007581877679385422
_CAPTURED_MAXIMUM_PLANE_DISTANCE = 0.00018954694198463555

# Earlier small Plane.008 traceback.
_PLANE008_SMALL_SIDE_LENGTH = 0.0164835268501562
_PLANE008_SMALL_WARP_HEIGHT = 0.000379143881919382
_PLANE008_SMALL_CAPTURED_MAXIMUM_PLANE_DISTANCE = 9.477343601658832e-05
_PLANE008_SMALL_CAPTURED_POLYGON_SCALE = 0.023314310303391664
_PLANE008_SMALL_CAPTURED_NORMALIZED_WARP = (
    _PLANE008_SMALL_CAPTURED_MAXIMUM_PLANE_DISTANCE
    / _PLANE008_SMALL_CAPTURED_POLYGON_SCALE
)

# Exact metrics from the real E:\test_BtSe\mushrooms\mushrooms.blend failure on
# Plane.008 evaluated source face 15. The square fixture reproduces the reported Newell
# plane distance and bounding-box diagonal without object-name-specific production logic.
_PLANE008_FACE15_SIDE_LENGTH = 0.01813398508349017
_PLANE008_FACE15_WARP_HEIGHT = 0.00060171214277301
_PLANE008_FACE15_CAPTURED_MAXIMUM_PLANE_DISTANCE = 0.0001503866471090267
_PLANE008_FACE15_CAPTURED_POLYGON_SCALE = 0.025652385610684413
_PLANE008_FACE15_CAPTURED_NORMALIZED_WARP = (
    _PLANE008_FACE15_CAPTURED_MAXIMUM_PLANE_DISTANCE
    / _PLANE008_FACE15_CAPTURED_POLYGON_SCALE
)

# Exact synthetic square dimensions reconstructed from the Cube.012 traceback. With
# Newell plane + centroid measurement they reproduce both logged values to float error.
_CUBE012_SIDE_LENGTH = 0.09277534946637461
_CUBE012_WARP_HEIGHT = 0.00030260673225328884
_CUBE012_CAPTURED_MAXIMUM_PLANE_DISTANCE = 7.565148185365435e-05
_CUBE012_CAPTURED_POLYGON_SCALE = 0.13120450643194492
_CUBE012_CAPTURED_NORMALIZED_WARP = (
    _CUBE012_CAPTURED_MAXIMUM_PLANE_DISTANCE
    / _CUBE012_CAPTURED_POLYGON_SCALE
)

_MATERIAL_WARP_HEIGHT = 0.01
_DEFAULT_ABSOLUTE_PLANARITY_TOLERANCE = 2.0e-4
_DEFAULT_RELATIVE_PLANARITY_TOLERANCE = 1.0e-3
_DEFAULT_MAXIMUM_RELATIVE_PLANARITY_WARP = 1.0e-2
_RETIRED_TOO_NARROW_RELATIVE_TOLERANCE = 2.5e-4


def _newell_unit_normal(points: tuple[Vector3, ...]) -> Vector3:
    """Return the geometric unit normal Blender-style n-gon fixtures should declare."""

    if not isinstance(points, tuple) or len(points) < 3:
        raise ValueError("points must contain at least three ordered vertices")
    if any(
        not isinstance(point, tuple)
        or len(point) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            for value in point
        )
        for point in points
    ):
        raise TypeError("points must contain finite numeric Vector3 tuples")

    newell = [0.0, 0.0, 0.0]
    for index, current in enumerate(points):
        following = points[(index + 1) % len(points)]
        newell[0] += (
            (float(current[1]) - float(following[1]))
            * (float(current[2]) + float(following[2]))
        )
        newell[1] += (
            (float(current[2]) - float(following[2]))
            * (float(current[0]) + float(following[0]))
        )
        newell[2] += (
            (float(current[0]) - float(following[0]))
            * (float(current[1]) + float(following[1]))
        )

    magnitude = sqrt(sum(component * component for component in newell))
    if not isfinite(magnitude) or magnitude <= 0.0:
        raise ValueError("polygon Newell normal is zero or non-finite")
    return tuple(component / magnitude for component in newell)


def _resolved_face_normal(
    positions: tuple[Vector3, ...],
    face_normal: Vector3 | None,
) -> Vector3:
    """Use Blender-like geometric normal unless a test explicitly injects stale data."""

    if face_normal is None:
        return _newell_unit_normal(positions)
    if (
        not isinstance(face_normal, tuple)
        or len(face_normal) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            for value in face_normal
        )
    ):
        raise TypeError("face_normal must be None or a finite numeric Vector3")

    magnitude = sqrt(sum(float(value) ** 2 for value in face_normal))
    if magnitude <= 0.0:
        raise ValueError("face_normal cannot be zero")
    return tuple(float(value) / magnitude for value in face_normal)


def _quad_snapshot(
    object_id: str,
    *,
    warp_height: float = 0.0,
    side_length: float = 1.0,
    face_normal: Vector3 | None = None,
) -> MeshSnapshot:
    """Build one ordered Blender-style n-gon with complete boundary lineage.

    Blender calculates ``MeshPolygon.normal`` from the polygon vertices. Captured
    planarity fixtures therefore declare their Newell normal by default. Tests that need
    to reproduce stale or projection-inherited normal data must request it explicitly via
    ``face_normal``.
    """

    if not isinstance(object_id, str) or not object_id.strip():
        raise ValueError("object_id must be a non-empty string")
    if isinstance(warp_height, bool) or not isinstance(
        warp_height,
        (int, float),
    ):
        raise TypeError("warp_height must be numeric")
    if isinstance(side_length, bool) or not isinstance(
        side_length,
        (int, float),
    ):
        raise TypeError("side_length must be numeric")

    resolved_side = float(side_length)
    resolved_warp = float(warp_height)
    if not isfinite(resolved_side) or resolved_side <= 0.0:
        raise ValueError("side_length must be finite and positive")
    if not isfinite(resolved_warp):
        raise ValueError("warp_height must be finite")

    half = resolved_side / 2.0
    positions: tuple[Vector3, ...] = (
        (-half, -half, 0.0),
        (half, -half, 0.0),
        (half, half, resolved_warp),
        (-half, half, 0.0),
    )
    declared_normal = _resolved_face_normal(positions, face_normal)

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
            normal=declared_normal,
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
        normal=declared_normal,
        smooth=True,
    )
    snapshot = MeshSnapshot(
        snapshot_id=f"{object_id}:captured-regression",
        source_object_id=object_id,
        object_name=object_id,
        vertices=vertices,
        edges=edges,
        loops=loops,
        faces=(face,),
    )
    MeshSnapshotValidator().validate_or_raise(snapshot)
    return snapshot


def _captured_planarity_metrics(
    snapshot: MeshSnapshot,
) -> tuple[float, float]:
    """Measure one fixture independently using the production geometric definition."""

    MeshSnapshotValidator().validate_or_raise(snapshot)
    if len(snapshot.faces) != 1:
        raise ValueError("captured planarity fixture must contain one polygon")

    face = snapshot.faces[0]
    loop_map = snapshot.loop_by_id()
    vertex_map = snapshot.vertex_by_id()
    points = tuple(
        vertex_map[loop_map[loop_id].vertex_id].position
        for loop_id in face.loop_ids
    )
    normal = _newell_unit_normal(points)
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
    polygon_scale = sqrt(
        sum(extent * extent for extent in extents)
    )
    return maximum_distance, polygon_scale


def test_mushrooms_plane_008_relative_warp_triangulates_deterministically() -> None:
    source = _quad_snapshot(
        _MUSHROOMS_OBJECT_ID,
        warp_height=_MUSHROOMS_WARP_HEIGHT,
    )

    first = triangulate_snapshot(source)
    second = triangulate_snapshot(source)

    assert first == second
    assert len(first.snapshot.faces) == 2
    assert len(first.generated_edge_ids) == 1
    assert tuple(face.source_id for face in first.snapshot.faces) == (
        SourceFaceId(_MUSHROOMS_OBJECT_ID, 0),
        SourceFaceId(_MUSHROOMS_OBJECT_ID, 0),
    )


def test_mushrooms_plane_008_small_traceback_uses_bounded_absolute_floor() -> None:
    source = _quad_snapshot(
        _MUSHROOMS_OBJECT_ID,
        warp_height=_PLANE008_SMALL_WARP_HEIGHT,
        side_length=_PLANE008_SMALL_SIDE_LENGTH,
    )
    maximum_distance, polygon_scale = _captured_planarity_metrics(source)
    settings = TriangulationSettings()

    assert maximum_distance == pytest.approx(
        _PLANE008_SMALL_CAPTURED_MAXIMUM_PLANE_DISTANCE,
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert polygon_scale == pytest.approx(
        _PLANE008_SMALL_CAPTURED_POLYGON_SCALE,
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert maximum_distance > (
        settings.relative_planarity_tolerance * polygon_scale
    )
    assert maximum_distance < settings.planarity_tolerance
    assert _PLANE008_SMALL_CAPTURED_NORMALIZED_WARP == pytest.approx(
        0.004065032796736908,
        rel=1.0e-12,
    )
    assert (
        _PLANE008_SMALL_CAPTURED_NORMALIZED_WARP
        < settings.maximum_relative_planarity_warp
    )

    first = triangulate_snapshot(source)
    second = triangulate_snapshot(source)
    assert first == second
    assert len(first.snapshot.faces) == 2
    assert len(first.generated_edge_ids) == 1


def test_real_mushrooms_plane008_face15_uses_bounded_absolute_floor() -> None:
    source = _quad_snapshot(
        _MUSHROOMS_OBJECT_ID,
        warp_height=_PLANE008_FACE15_WARP_HEIGHT,
        side_length=_PLANE008_FACE15_SIDE_LENGTH,
    )
    maximum_distance, polygon_scale = _captured_planarity_metrics(source)
    settings = TriangulationSettings()

    assert maximum_distance == pytest.approx(
        _PLANE008_FACE15_CAPTURED_MAXIMUM_PLANE_DISTANCE,
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert polygon_scale == pytest.approx(
        _PLANE008_FACE15_CAPTURED_POLYGON_SCALE,
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert _PLANE008_FACE15_CAPTURED_NORMALIZED_WARP == pytest.approx(
        0.005862481930194809,
        rel=1.0e-12,
    )
    assert maximum_distance > 1.0e-4
    assert maximum_distance < settings.planarity_tolerance
    assert maximum_distance > (
        settings.relative_planarity_tolerance * polygon_scale
    )
    assert (
        _PLANE008_FACE15_CAPTURED_NORMALIZED_WARP
        < settings.maximum_relative_planarity_warp
    )

    first = triangulate_snapshot(source)
    second = triangulate_snapshot(source)
    assert first == second
    assert len(first.snapshot.faces) == 2
    assert len(first.generated_edge_ids) == 1


def test_face15_with_explicit_stale_flat_normal_remains_rejected() -> None:
    source = _quad_snapshot(
        _MUSHROOMS_OBJECT_ID,
        warp_height=_PLANE008_FACE15_WARP_HEIGHT,
        side_length=_PLANE008_FACE15_SIDE_LENGTH,
        face_normal=(0.0, 0.0, 1.0),
    )

    with pytest.raises(
        TriangulationError,
        match="declared face normal",
    ) as captured:
        triangulate_snapshot(source)

    assert "normal deviation" in str(captured.value)
    assert "exceeds tolerance 1.0 degrees" in str(captured.value)


def test_mushrooms_cube_012_traceback_warp_triangulates_deterministically() -> None:
    source = _quad_snapshot(
        _MUSHROOMS_CUBE_OBJECT_ID,
        warp_height=_CUBE012_WARP_HEIGHT,
        side_length=_CUBE012_SIDE_LENGTH,
    )
    maximum_distance, polygon_scale = _captured_planarity_metrics(source)

    assert maximum_distance == pytest.approx(
        _CUBE012_CAPTURED_MAXIMUM_PLANE_DISTANCE,
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert polygon_scale == pytest.approx(
        _CUBE012_CAPTURED_POLYGON_SCALE,
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert maximum_distance > (
        _RETIRED_TOO_NARROW_RELATIVE_TOLERANCE * polygon_scale
    )
    assert maximum_distance < (
        _DEFAULT_RELATIVE_PLANARITY_TOLERANCE * polygon_scale
    )
    assert _CUBE012_CAPTURED_NORMALIZED_WARP == pytest.approx(
        0.0005765920996996728,
        rel=1.0e-12,
    )

    first = triangulate_snapshot(source)
    second = triangulate_snapshot(source)
    assert first == second
    assert len(first.snapshot.faces) == 2
    assert len(first.generated_edge_ids) == 1
    assert tuple(face.source_id for face in first.snapshot.faces) == (
        SourceFaceId(_MUSHROOMS_CUBE_OBJECT_ID, 0),
        SourceFaceId(_MUSHROOMS_CUBE_OBJECT_ID, 0),
    )


def test_default_planarity_window_rejects_percent_level_warp() -> None:
    settings = TriangulationSettings()
    assert settings.planarity_tolerance == pytest.approx(
        _DEFAULT_ABSOLUTE_PLANARITY_TOLERANCE
    )
    assert settings.relative_planarity_tolerance == pytest.approx(
        _DEFAULT_RELATIVE_PLANARITY_TOLERANCE
    )
    assert settings.maximum_relative_planarity_warp == pytest.approx(
        _DEFAULT_MAXIMUM_RELATIVE_PLANARITY_WARP
    )
    assert settings.normal_alignment_tolerance_degrees == pytest.approx(1.0)

    source = _quad_snapshot(
        _MUSHROOMS_OBJECT_ID,
        warp_height=_MATERIAL_WARP_HEIGHT,
    )
    with pytest.raises(
        TriangulationError,
        match="Polygon is not planar",
    ):
        triangulate_snapshot(source, settings)


def test_absolute_floor_cannot_accept_grossly_folded_tiny_polygon() -> None:
    source = _quad_snapshot(
        "TinyGrossFold",
        side_length=1.0e-3,
        warp_height=2.0e-4,
    )
    maximum_distance, polygon_scale = _captured_planarity_metrics(source)
    settings = TriangulationSettings()

    assert maximum_distance < settings.planarity_tolerance
    assert (
        maximum_distance / polygon_scale
        > settings.maximum_relative_planarity_warp
    )

    with pytest.raises(
        TriangulationError,
        match="hard ceiling",
    ):
        triangulate_snapshot(source, settings)


def test_mushrooms_plane_008_warp_remains_rejectable_under_explicit_strict_policy() -> None:
    source = _quad_snapshot(
        _MUSHROOMS_OBJECT_ID,
        warp_height=_MUSHROOMS_WARP_HEIGHT,
    )
    strict = TriangulationSettings(
        planarity_tolerance=1.0e-6,
        relative_planarity_tolerance=1.0e-5,
    )

    with pytest.raises(TriangulationError) as captured:
        triangulate_snapshot(source, strict)

    message = str(captured.value)
    assert "maximum plane distance" in message
    assert "effective tolerance" in message
    assert "relative=1e-05" in message
    assert _CAPTURED_MAXIMUM_PLANE_DISTANCE > strict.planarity_tolerance


def test_flower_shop_banco_repeated_source_face_lineage_is_valid_coverage() -> None:
    source = _quad_snapshot(_FLOWER_SHOP_OBJECT_ID)
    triangulated = triangulate_snapshot(source).snapshot

    assert len(triangulated.faces) == 2
    assert len({face.id for face in triangulated.faces}) == 2
    assert len({face.source_id for face in triangulated.faces}) == 1

    prepared = prepare_a1_geometry_regions(triangulated)

    local_faces = tuple(
        face.id
        for region in prepared.regions
        for face in region.snapshot.faces
    )
    assert len(local_faces) == 2
    expected_lineage = Counter(
        face.source_id
        for face in triangulated.faces
    )
    prepared_lineage = Counter(
        source_face_id
        for region in prepared.regions
        for source_face_id in region.source_face_ids
    )
    assert prepared_lineage == expected_lineage
    assert (
        prepared_lineage[
            SourceFaceId(_FLOWER_SHOP_OBJECT_ID, 0)
        ]
        == 2
    )


def test_flower_shop_coverage_still_rejects_lost_lineage_occurrence() -> None:
    triangulated = triangulate_snapshot(
        _quad_snapshot(_FLOWER_SHOP_OBJECT_ID)
    ).snapshot
    prepared = prepare_a1_geometry_regions(triangulated)
    first = prepared.regions[0]
    assert len(first.source_face_ids) >= 2
    damaged_first = replace(
        first,
        source_face_ids=first.source_face_ids[:-1],
    )
    damaged_regions = (
        damaged_first,
        *prepared.regions[1:],
    )

    with pytest.raises(
        A1GeometryPreparationError,
        match="SourceFaceId lineage differs from decomposition plan",
    ):
        _validate_prepared_coverage(
            triangulated,
            prepared.decomposition,
            damaged_regions,
        )
