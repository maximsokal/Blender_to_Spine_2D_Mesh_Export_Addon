"""Deterministic ear-clipping triangulation for immutable mesh snapshots.

Each generated triangle retains the original face and corner lineage. Existing
boundary edges keep their source IDs and flags; generated diagonals explicitly use
``source_id=None``. Blender n-gons may contain small evaluated warp, so planarity is
validated against bounded absolute and scale-relative tolerances. Declared face normals
must also remain coherent with the geometric polygon plane. Self-intersecting, strongly
folded, projection-distorted, and degenerate polygons remain hard errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, isfinite, pi, sqrt
from typing import Iterable, Tuple

from .ids import EdgeId, FaceId, LoopId, VertexId
from .model import MeshEdge, MeshFace, MeshLoop, MeshSnapshot, Vector2, Vector3
from .validator import MeshSnapshotValidator


class TriangulationError(ValueError):
    """Raised when a polygon cannot be triangulated without ambiguity."""


@dataclass(frozen=True, slots=True)
class TriangulationSettings:
    epsilon: float = 1e-10
    planarity_tolerance: float = 1e-6
    relative_planarity_tolerance: float = 2.5e-4
    normal_alignment_tolerance_degrees: float = 1.0

    def __post_init__(self) -> None:
        for field_name in (
            "epsilon",
            "planarity_tolerance",
            "relative_planarity_tolerance",
            "normal_alignment_tolerance_degrees",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be numeric")
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError(f"{field_name} must be finite")
            if numeric <= 0.0:
                raise ValueError(f"{field_name} must be positive")
            object.__setattr__(self, field_name, numeric)
        if self.relative_planarity_tolerance > 1.0:
            raise ValueError("relative_planarity_tolerance cannot exceed 1.0")
        if self.normal_alignment_tolerance_degrees > 90.0:
            raise ValueError(
                "normal_alignment_tolerance_degrees cannot exceed 90 degrees"
            )


@dataclass(frozen=True, slots=True)
class TriangulatedFaceInfo:
    original_face_id: FaceId
    output_face_ids: Tuple[FaceId, ...]
    original_corner_count: int


@dataclass(frozen=True, slots=True)
class TriangulationResult:
    source_snapshot_id: str
    snapshot: MeshSnapshot
    faces: Tuple[TriangulatedFaceInfo, ...]
    generated_edge_ids: Tuple[EdgeId, ...]


@dataclass(frozen=True, slots=True)
class _FaceTriangle:
    source_face: MeshFace
    corner_indices: Tuple[int, int, int]


def _subtract(first: Vector3, second: Vector3) -> Vector3:
    return (
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    )


def _dot(first: Vector3, second: Vector3) -> float:
    return sum(a * b for a, b in zip(first, second, strict=True))


def _cross_3d(first: Vector3, second: Vector3) -> Vector3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _length(value: Vector3) -> float:
    return sqrt(_dot(value, value))


def _newell_normal(points: Tuple[Vector3, ...]) -> Vector3:
    x = y = z = 0.0
    for index, current in enumerate(points):
        following = points[(index + 1) % len(points)]
        x += (current[1] - following[1]) * (current[2] + following[2])
        y += (current[2] - following[2]) * (current[0] + following[0])
        z += (current[0] - following[0]) * (current[1] + following[1])
    return (x, y, z)


def _normalized(value: Vector3, epsilon: float) -> Vector3:
    magnitude = _length(value)
    if magnitude <= epsilon:
        raise TriangulationError("Polygon normal is zero or numerically unstable")
    return (
        value[0] / magnitude,
        value[1] / magnitude,
        value[2] / magnitude,
    )


def _centroid(points: Tuple[Vector3, ...]) -> Vector3:
    if not points:
        raise TriangulationError("Polygon cannot be empty")
    divisor = float(len(points))
    return (
        sum(point[0] for point in points) / divisor,
        sum(point[1] for point in points) / divisor,
        sum(point[2] for point in points) / divisor,
    )


def _polygon_scale(points: Tuple[Vector3, ...]) -> float:
    """Return the deterministic bounding-box diagonal used for relative tolerance."""

    extents = tuple(
        max(point[axis] for point in points)
        - min(point[axis] for point in points)
        for axis in range(3)
    )
    return sqrt(sum(extent * extent for extent in extents))


def _validate_planarity(
    points: Tuple[Vector3, ...],
    normal: Vector3,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    epsilon: float,
) -> None:
    """Reject warp that exceeds a narrow scale-aware tolerance window.

    Blender stores n-gons as ordered boundaries and may evaluate them with small
    non-planarity. Measuring from the centroid avoids making the result depend on the
    first corner. The absolute floor protects tiny geometry, while the bounded relative
    term accepts the captured ``Plane.008`` residue without allowing percent-level fold.
    """

    origin = _centroid(points)
    scale = _polygon_scale(points)
    if not isfinite(scale) or scale <= epsilon:
        raise TriangulationError("Polygon extent is zero or numerically unstable")
    distances = tuple(
        abs(_dot(_subtract(point, origin), normal))
        for point in points
    )
    maximum = max(distances)
    effective_tolerance = max(
        float(absolute_tolerance),
        float(relative_tolerance) * scale,
    )
    if maximum > effective_tolerance:
        raise TriangulationError(
            "Polygon is not planar within deterministic tolerance: "
            f"maximum plane distance {maximum} exceeds effective tolerance "
            f"{effective_tolerance} (absolute={absolute_tolerance}, "
            f"relative={relative_tolerance}, polygon_scale={scale})"
        )


def _validate_declared_normal_alignment(
    declared_normal: Vector3,
    geometric_normal: Vector3,
    *,
    tolerance_degrees: float,
    epsilon: float,
) -> None:
    """Reject polygons whose stored face normal no longer describes their plane.

    Perspective projection is not an affine transform in XYZ. Projecting an n-gon before
    triangulation can therefore leave a face normal inherited from source space while the
    generated points describe another plane. Small Blender evaluation residue is accepted;
    a material orientation mismatch remains a hard error so callers must triangulate in
    source/world space before non-affine projection.
    """

    declared_length = _length(declared_normal)
    if not isfinite(declared_length):
        raise TriangulationError("Declared face normal contains non-finite values")
    if declared_length <= epsilon:
        return

    normalized_declared = (
        declared_normal[0] / declared_length,
        declared_normal[1] / declared_length,
        declared_normal[2] / declared_length,
    )
    cosine = abs(_dot(normalized_declared, geometric_normal))
    if not isfinite(cosine):
        raise TriangulationError("Declared face-normal alignment became non-finite")
    clamped = min(1.0, max(0.0, cosine))
    deviation_degrees = acos(clamped) * 180.0 / pi
    if deviation_degrees > tolerance_degrees:
        raise TriangulationError(
            "Polygon is not planar relative to its declared face normal: "
            f"normal deviation {deviation_degrees} degrees exceeds tolerance "
            f"{tolerance_degrees} degrees"
        )


def _validate_triangle_orientation(
    points: Tuple[Vector3, ...],
    triangles: Tuple[Tuple[int, int, int], ...],
    reference_normal: Vector3,
    *,
    epsilon: float,
) -> None:
    """Require every generated 3D triangle to follow one coherent winding."""

    scale = _polygon_scale(points)
    area_tolerance = max(epsilon, scale * scale * epsilon)
    for triangle_index, triangle in enumerate(triangles):
        first, second, third = (points[index] for index in triangle)
        cross = _cross_3d(
            _subtract(second, first),
            _subtract(third, first),
        )
        signed_area = _dot(cross, reference_normal)
        if not isfinite(signed_area) or signed_area <= area_tolerance:
            raise TriangulationError(
                "Warped polygon produced a collapsed or reversed 3D triangle; "
                f"triangle={triangle_index}, corners={triangle}, "
                f"oriented_area={signed_area}, tolerance={area_tolerance}"
            )


def _projection_axis(normal: Vector3) -> int:
    absolute = tuple(abs(component) for component in normal)
    return max(range(3), key=lambda index: (absolute[index], -index))


def _project(point: Vector3, dropped_axis: int) -> Vector2:
    if dropped_axis == 0:
        return (point[1], point[2])
    if dropped_axis == 1:
        return (point[0], point[2])
    return (point[0], point[1])


def _cross_2d(first: Vector2, second: Vector2, third: Vector2) -> float:
    return (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _signed_area(points: Tuple[Vector2, ...]) -> float:
    return 0.5 * sum(
        current[0] * following[1] - following[0] * current[1]
        for current, following in zip(points, points[1:] + points[:1], strict=True)
    )


def _orientation(
    first: Vector2,
    second: Vector2,
    third: Vector2,
    epsilon: float,
) -> int:
    value = _cross_2d(first, second, third)
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def _point_on_segment(
    point: Vector2,
    first: Vector2,
    second: Vector2,
    epsilon: float,
) -> bool:
    if _orientation(first, second, point, epsilon) != 0:
        return False
    return (
        min(first[0], second[0]) - epsilon
        <= point[0]
        <= max(first[0], second[0]) + epsilon
        and min(first[1], second[1]) - epsilon
        <= point[1]
        <= max(first[1], second[1]) + epsilon
    )


def _segments_intersect(
    a1: Vector2,
    a2: Vector2,
    b1: Vector2,
    b2: Vector2,
    epsilon: float,
) -> bool:
    o1 = _orientation(a1, a2, b1, epsilon)
    o2 = _orientation(a1, a2, b2, epsilon)
    o3 = _orientation(b1, b2, a1, epsilon)
    o4 = _orientation(b1, b2, a2, epsilon)
    if o1 != o2 and o3 != o4:
        return True
    return any(
        (
            orientation == 0
            and _point_on_segment(point, segment_start, segment_end, epsilon)
        )
        for orientation, point, segment_start, segment_end in (
            (o1, b1, a1, a2),
            (o2, b2, a1, a2),
            (o3, a1, b1, b2),
            (o4, a2, b1, b2),
        )
    )


def _validate_simple_polygon(points: Tuple[Vector2, ...], epsilon: float) -> None:
    count = len(points)
    for first_index in range(count):
        first_next = (first_index + 1) % count
        for second_index in range(first_index + 1, count):
            second_next = (second_index + 1) % count
            if (
                first_index == second_index
                or first_next == second_index
                or second_next == first_index
            ):
                continue
            if first_index == 0 and second_next == 0:
                continue
            if _segments_intersect(
                points[first_index],
                points[first_next],
                points[second_index],
                points[second_next],
                epsilon,
            ):
                raise TriangulationError(
                    "Polygon boundary self-intersects between edges "
                    f"{first_index}-{first_next} and {second_index}-{second_next}"
                )


def _point_in_triangle(
    point: Vector2,
    first: Vector2,
    second: Vector2,
    third: Vector2,
    orientation_sign: int,
    epsilon: float,
) -> bool:
    values = (
        _cross_2d(first, second, point) * orientation_sign,
        _cross_2d(second, third, point) * orientation_sign,
        _cross_2d(third, first, point) * orientation_sign,
    )
    return all(value >= -epsilon for value in values)


def _ear_clip(
    points: Tuple[Vector2, ...],
    epsilon: float,
) -> Tuple[Tuple[int, int, int], ...]:
    area = _signed_area(points)
    if abs(area) <= epsilon:
        raise TriangulationError("Projected polygon area is zero")
    orientation_sign = 1 if area > 0.0 else -1
    remaining = list(range(len(points)))
    triangles: list[Tuple[int, int, int]] = []

    while len(remaining) > 3:
        ear_position: int | None = None
        for position, current in enumerate(remaining):
            previous = remaining[position - 1]
            following = remaining[(position + 1) % len(remaining)]
            convexity = _cross_2d(
                points[previous],
                points[current],
                points[following],
            ) * orientation_sign
            if convexity <= epsilon:
                continue
            if any(
                _point_in_triangle(
                    points[candidate],
                    points[previous],
                    points[current],
                    points[following],
                    orientation_sign,
                    epsilon,
                )
                for candidate in remaining
                if candidate not in {previous, current, following}
            ):
                continue
            ear_position = position
            triangles.append((previous, current, following))
            break
        if ear_position is None:
            raise TriangulationError(
                "Ear clipping found no valid ear; polygon may contain duplicate, "
                "collinear, or numerically unstable corners"
            )
        del remaining[ear_position]

    final = tuple(remaining)
    final_area = _cross_2d(
        points[final[0]],
        points[final[1]],
        points[final[2]],
    ) * orientation_sign
    if final_area <= epsilon:
        raise TriangulationError("Final triangle is degenerate")
    triangles.append((final[0], final[1], final[2]))
    return tuple(triangles)


def _triangulate_face(
    snapshot: MeshSnapshot,
    face: MeshFace,
    settings: TriangulationSettings,
) -> Tuple[_FaceTriangle, ...]:
    if len(face.loop_ids) == 3:
        return (_FaceTriangle(face, (0, 1, 2)),)
    if len(face.loop_ids) < 3:
        raise TriangulationError(f"Face {face.id.index} has fewer than three corners")

    loop_map = snapshot.loop_by_id()
    vertex_map = snapshot.vertex_by_id()
    loops = tuple(loop_map[loop_id] for loop_id in face.loop_ids)
    vertex_ids = tuple(loop.vertex_id for loop in loops)
    if len(vertex_ids) != len(set(vertex_ids)):
        raise TriangulationError(
            f"Face {face.id.index} repeats a local vertex and is not a simple polygon"
        )

    points_3d = tuple(vertex_map[vertex_id].position for vertex_id in vertex_ids)
    newell = _newell_normal(points_3d)
    newell_length = _length(newell)
    normal = _normalized(
        newell if newell_length > settings.epsilon else face.normal,
        settings.epsilon,
    )
    _validate_planarity(
        points_3d,
        normal,
        absolute_tolerance=settings.planarity_tolerance,
        relative_tolerance=settings.relative_planarity_tolerance,
        epsilon=settings.epsilon,
    )
    if newell_length > settings.epsilon:
        _validate_declared_normal_alignment(
            face.normal,
            normal,
            tolerance_degrees=settings.normal_alignment_tolerance_degrees,
            epsilon=settings.epsilon,
        )
    axis = _projection_axis(normal)
    points_2d = tuple(_project(point, axis) for point in points_3d)
    _validate_simple_polygon(points_2d, settings.epsilon)
    triangles = _ear_clip(points_2d, settings.epsilon)
    _validate_triangle_orientation(
        points_3d,
        triangles,
        normal,
        epsilon=settings.epsilon,
    )
    return tuple(_FaceTriangle(face, triangle) for triangle in triangles)


def _edge_key(first: VertexId, second: VertexId) -> tuple[int, int]:
    first_index = first.index
    second_index = second.index
    return (
        (first_index, second_index)
        if first_index < second_index
        else (second_index, first_index)
    )


def triangulate_snapshot(
    snapshot: MeshSnapshot,
    settings: TriangulationSettings | None = None,
    *,
    snapshot_id: str | None = None,
) -> TriangulationResult:
    """Return a new triangulated snapshot with complete source lineage."""

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    MeshSnapshotValidator().validate_or_raise(snapshot)
    resolved_settings = settings or TriangulationSettings()
    if not isinstance(resolved_settings, TriangulationSettings):
        raise TypeError("settings must be TriangulationSettings")

    face_triangles: list[tuple[MeshFace, Tuple[_FaceTriangle, ...]]] = []
    for face in sorted(snapshot.faces, key=lambda item: item.id.index):
        try:
            triangles = _triangulate_face(snapshot, face, resolved_settings)
        except TriangulationError as exc:
            raise TriangulationError(
                f"Unable to triangulate face {face.id.index} "
                f"(source face {face.source_id.face_index}): {exc}"
            ) from exc
        face_triangles.append((face, triangles))

    original_edge_by_key = {
        _edge_key(*edge.vertex_ids): edge for edge in snapshot.edges
    }
    used_edge_keys: set[tuple[int, int]] = set()
    source_loop_map = snapshot.loop_by_id()
    for _, triangles in face_triangles:
        for triangle in triangles:
            source_loops = tuple(
                source_loop_map[triangle.source_face.loop_ids[index]]
                for index in triangle.corner_indices
            )
            for index, loop in enumerate(source_loops):
                following = source_loops[(index + 1) % 3]
                used_edge_keys.add(_edge_key(loop.vertex_id, following.vertex_id))

    existing_keys = tuple(
        sorted(
            (key for key in used_edge_keys if key in original_edge_by_key),
            key=lambda key: original_edge_by_key[key].id.index,
        )
    )
    generated_keys = tuple(sorted(used_edge_keys - set(existing_keys)))
    ordered_edge_keys = existing_keys + generated_keys
    edge_id_by_key = {
        key: EdgeId(index) for index, key in enumerate(ordered_edge_keys)
    }
    edges = tuple(
        MeshEdge(
            id=edge_id_by_key[key],
            source_id=(
                original_edge_by_key[key].source_id
                if key in original_edge_by_key
                else None
            ),
            vertex_ids=(VertexId(key[0]), VertexId(key[1])),
            seam=(
                original_edge_by_key[key].seam
                if key in original_edge_by_key
                else False
            ),
            sharp=(
                original_edge_by_key[key].sharp
                if key in original_edge_by_key
                else False
            ),
        )
        for key in ordered_edge_keys
    )

    loops: list[MeshLoop] = []
    faces: list[MeshFace] = []
    face_info: list[TriangulatedFaceInfo] = []
    next_loop_index = 0
    next_face_index = 0

    for original_face, triangles in face_triangles:
        output_face_ids: list[FaceId] = []
        for triangle in triangles:
            source_loops = tuple(
                source_loop_map[original_face.loop_ids[index]]
                for index in triangle.corner_indices
            )
            triangle_loop_ids: list[LoopId] = []
            for corner_index, source_loop in enumerate(source_loops):
                following_loop = source_loops[(corner_index + 1) % 3]
                edge_key = _edge_key(source_loop.vertex_id, following_loop.vertex_id)
                loop_id = LoopId(next_loop_index)
                next_loop_index += 1
                triangle_loop_ids.append(loop_id)
                loops.append(
                    MeshLoop(
                        id=loop_id,
                        source_id=source_loop.source_id,
                        vertex_id=source_loop.vertex_id,
                        edge_id=edge_id_by_key[edge_key],
                        uvs=source_loop.uvs,
                    )
                )

            face_id = FaceId(next_face_index)
            next_face_index += 1
            output_face_ids.append(face_id)
            faces.append(
                MeshFace(
                    id=face_id,
                    source_id=original_face.source_id,
                    loop_ids=tuple(triangle_loop_ids),
                    material_index=original_face.material_index,
                    normal=original_face.normal,
                    smooth=original_face.smooth,
                )
            )
        face_info.append(
            TriangulatedFaceInfo(
                original_face_id=original_face.id,
                output_face_ids=tuple(output_face_ids),
                original_corner_count=len(original_face.loop_ids),
            )
        )

    output = MeshSnapshot(
        snapshot_id=snapshot_id or f"{snapshot.snapshot_id}:triangulated",
        source_object_id=snapshot.source_object_id,
        object_name=snapshot.object_name,
        vertices=snapshot.vertices,
        edges=edges,
        loops=tuple(loops),
        faces=tuple(faces),
        uv_layer_names=snapshot.uv_layer_names,
        active_uv_layer=snapshot.active_uv_layer,
        world_matrix=snapshot.world_matrix,
        render_uv_layer=snapshot.render_uv_layer,
    )
    MeshSnapshotValidator().validate_or_raise(output)
    generated_edge_ids = tuple(
        edge_id_by_key[key] for key in generated_keys
    )
    return TriangulationResult(
        source_snapshot_id=snapshot.snapshot_id,
        snapshot=output,
        faces=tuple(face_info),
        generated_edge_ids=generated_edge_ids,
    )
