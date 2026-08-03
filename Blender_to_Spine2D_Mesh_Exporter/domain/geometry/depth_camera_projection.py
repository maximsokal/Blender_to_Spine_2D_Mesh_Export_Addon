"""Build a bounded visible camera-depth relief surface for Spine 2.5D export.

The source MeshSnapshot is evaluated and world-normalized before entering this module.
The algorithm projects its triangulated faces through one active-camera frame, samples
only the front-most surface on a bounded screen lattice, smooths depth without crossing
large discontinuities, and emits a new UV-ready MeshSnapshot.

Output X/Y use the same projected object-local convention as Active Camera geometry.
Output Z keeps absolute Blender camera-local depth: camera space is zero, visible points
in front of the camera are negative, and points nearer the camera have larger values.
The farthest visible point is also recorded as the local relief base, so every relief
value is non-negative and points are displaced only toward the camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import ceil, isfinite, sqrt
from typing import Mapping, Tuple

from ..camera_projection import A1CameraProjectionFrame
from ..projection import A1ProjectedPoint
from .ids import (
    EdgeId,
    FaceId,
    LoopId,
    SourceFaceId,
    SourceLoopId,
    SourceVertexId,
    VertexId,
)
from .model import (
    IDENTITY_MATRIX_4X4,
    LoopUV,
    Matrix4x4,
    MeshEdge,
    MeshFace,
    MeshLoop,
    MeshSnapshot,
    MeshVertex,
    Vector3,
)
from .triangulation import triangulate_snapshot
from .validator import MeshSnapshotValidator


class DepthProjectionBaseMode(str, Enum):
    """Reference used to validate camera-facing relief construction."""

    FARTHEST_VISIBLE = "FARTHEST_VISIBLE"
    OBJECT_ORIGIN = "OBJECT_ORIGIN"


@dataclass(frozen=True, slots=True)
class DepthCameraProjectionSettings:
    """Immutable quality and safety settings for one depth relief surface."""

    smoothing: float = 0.35
    edge_threshold_fraction: float = 0.08
    mesh_error_pixels: float = 4.0
    max_points: int = 128
    base_mode: DepthProjectionBaseMode = DepthProjectionBaseMode.FARTHEST_VISIBLE

    def __post_init__(self) -> None:
        for field_name in (
            "smoothing",
            "edge_threshold_fraction",
            "mesh_error_pixels",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a finite number")
            resolved = float(value)
            if not isfinite(resolved):
                raise ValueError(f"{field_name} must be finite")
            object.__setattr__(self, field_name, resolved)
        if not 0.0 <= self.smoothing <= 1.0:
            raise ValueError("smoothing must be in [0, 1]")
        if not 0.0 <= self.edge_threshold_fraction <= 1.0:
            raise ValueError("edge_threshold_fraction must be in [0, 1]")
        if self.mesh_error_pixels <= 0.0:
            raise ValueError("mesh_error_pixels must be positive")
        if isinstance(self.max_points, bool) or not isinstance(self.max_points, int):
            raise TypeError("max_points must be int")
        if self.max_points < 4:
            raise ValueError("max_points must be at least 4")
        if self.max_points > 4096:
            raise ValueError("max_points cannot exceed 4096")
        if not isinstance(self.base_mode, DepthProjectionBaseMode):
            raise TypeError("base_mode must be DepthProjectionBaseMode")


@dataclass(frozen=True, slots=True)
class DepthCameraProjectionResult:
    """One optimized visible relief surface and deterministic diagnostics."""

    snapshot: MeshSnapshot
    frame: A1CameraProjectionFrame
    projected_origin: A1ProjectedPoint
    base_mode: DepthProjectionBaseMode
    base_depth: float
    farthest_visible_depth: float
    nearest_visible_depth: float
    maximum_relief: float
    requested_spacing_pixels: float
    resolved_spacing_x_pixels: float
    resolved_spacing_y_pixels: float
    source_triangle_count: int
    sampled_point_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, MeshSnapshot):
            raise TypeError("snapshot must be MeshSnapshot")
        if not isinstance(self.frame, A1CameraProjectionFrame):
            raise TypeError("frame must be A1CameraProjectionFrame")
        if not isinstance(self.projected_origin, A1ProjectedPoint):
            raise TypeError("projected_origin must be A1ProjectedPoint")
        if not isinstance(self.base_mode, DepthProjectionBaseMode):
            raise TypeError("base_mode must be DepthProjectionBaseMode")
        for field_name in (
            "base_depth",
            "farthest_visible_depth",
            "nearest_visible_depth",
            "maximum_relief",
            "requested_spacing_pixels",
            "resolved_spacing_x_pixels",
            "resolved_spacing_y_pixels",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a finite number")
            if not isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        if self.nearest_visible_depth < self.farthest_visible_depth:
            raise ValueError("nearest_visible_depth cannot be behind farthest_visible_depth")
        if self.maximum_relief < 0.0:
            raise ValueError("maximum_relief cannot be negative")
        if self.requested_spacing_pixels <= 0.0:
            raise ValueError("requested_spacing_pixels must be positive")
        if self.resolved_spacing_x_pixels <= 0.0:
            raise ValueError("resolved_spacing_x_pixels must be positive")
        if self.resolved_spacing_y_pixels <= 0.0:
            raise ValueError("resolved_spacing_y_pixels must be positive")
        for field_name in ("source_triangle_count", "sampled_point_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.sampled_point_count != len(self.snapshot.vertices):
            raise ValueError("sampled_point_count must match snapshot vertices")


@dataclass(frozen=True, slots=True)
class _ProjectedTriangle:
    face_index: int
    points: Tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        if isinstance(self.face_index, bool) or not isinstance(self.face_index, int):
            raise TypeError("face_index must be int")
        if self.face_index < 0:
            raise ValueError("face_index must be non-negative")
        if not isinstance(self.points, tuple) or len(self.points) != 3:
            raise ValueError("points must contain three projected vertices")
        for point_index, point in enumerate(self.points):
            if not isinstance(point, tuple) or len(point) != 3:
                raise TypeError(f"points[{point_index}] must contain x, y, depth")
            if not all(isfinite(float(value)) for value in point):
                raise ValueError(f"points[{point_index}] contains non-finite values")


@dataclass(frozen=True, slots=True)
class _Sample:
    x: float
    y: float
    depth: float
    source_face_index: int


class DepthCameraProjectionError(ValueError):
    """Raised when a visible bounded depth surface cannot be constructed."""


def _translation_only_origin(matrix: Matrix4x4) -> Vector3:
    if not isinstance(matrix, tuple) or len(matrix) != 16:
        raise TypeError("snapshot.world_matrix must contain 16 values")
    values = tuple(float(value) for value in matrix)
    if not all(isfinite(value) for value in values):
        raise DepthCameraProjectionError("snapshot.world_matrix contains non-finite values")
    tolerance = 1.0e-10
    expected = (
        1.0, 0.0, 0.0, values[3],
        0.0, 1.0, 0.0, values[7],
        0.0, 0.0, 1.0, values[11],
        0.0, 0.0, 0.0, 1.0,
    )
    mismatches = tuple(
        index
        for index, (actual, required) in enumerate(zip(values, expected, strict=True))
        if abs(actual - required) > tolerance
    )
    if mismatches:
        raise DepthCameraProjectionError(
            "snapshot.world_matrix must contain translation only; normalize the "
            f"evaluated mesh first; mismatch_indices={mismatches}"
        )
    return values[3], values[7], values[11]


def _world_point(origin: Vector3, local: Vector3) -> Vector3:
    return (
        float(origin[0]) + float(local[0]),
        float(origin[1]) + float(local[1]),
        float(origin[2]) + float(local[2]),
    )


def _projected_translation_matrix(
    projected_origin: A1ProjectedPoint,
    uniform_scale: float,
) -> Matrix4x4:
    return (
        1.0, 0.0, 0.0, projected_origin.u / uniform_scale,
        0.0, 1.0, 0.0, projected_origin.v / uniform_scale,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def _triangle_from_face(
    snapshot: MeshSnapshot,
    face_index: int,
    projected_by_vertex: Mapping[VertexId, A1ProjectedPoint],
) -> _ProjectedTriangle:
    face = snapshot.faces[face_index]
    loops_by_id = snapshot.loop_by_id()
    vertex_ids = tuple(loops_by_id[loop_id].vertex_id for loop_id in face.loop_ids)
    if len(vertex_ids) != 3:
        raise DepthCameraProjectionError(
            f"triangulated face {face.id.index} does not contain three vertices"
        )
    points = tuple(
        (
            float(projected_by_vertex[vertex_id].u),
            float(projected_by_vertex[vertex_id].v),
            float(projected_by_vertex[vertex_id].depth),
        )
        for vertex_id in vertex_ids
    )
    return _ProjectedTriangle(face_index=face.id.index, points=points)


def _signed_area_twice(points: Tuple[tuple[float, float, float], ...]) -> float:
    first, second, third = points
    return (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _barycentric_depth(
    triangle: _ProjectedTriangle,
    x: float,
    y: float,
    *,
    epsilon: float,
) -> float | None:
    first, second, third = triangle.points
    denominator = _signed_area_twice(triangle.points)
    if abs(denominator) <= epsilon:
        return None
    first_weight = (
        (second[0] - x) * (third[1] - y)
        - (second[1] - y) * (third[0] - x)
    ) / denominator
    second_weight = (
        (third[0] - x) * (first[1] - y)
        - (third[1] - y) * (first[0] - x)
    ) / denominator
    third_weight = 1.0 - first_weight - second_weight
    if min(first_weight, second_weight, third_weight) < -epsilon:
        return None
    depth = (
        first_weight * first[2]
        + second_weight * second[2]
        + third_weight * third[2]
    )
    if not isfinite(depth):
        raise DepthCameraProjectionError("barycentric depth became non-finite")
    return float(depth)


def _grid_dimensions(
    width: float,
    height: float,
    settings: DepthCameraProjectionSettings,
) -> tuple[int, int]:
    columns = max(1, int(ceil(width / settings.mesh_error_pixels)))
    rows = max(1, int(ceil(height / settings.mesh_error_pixels)))
    while (columns + 1) * (rows + 1) > settings.max_points:
        if columns >= rows and columns > 1:
            columns -= 1
        elif rows > 1:
            rows -= 1
        else:
            break
    if (columns + 1) * (rows + 1) > settings.max_points:
        raise DepthCameraProjectionError(
            "max_points is too small to build a two-dimensional depth surface"
        )
    return columns, rows


def _sample_front_surface(
    triangles: Tuple[_ProjectedTriangle, ...],
    *,
    minimum_x: float,
    minimum_y: float,
    width: float,
    height: float,
    columns: int,
    rows: int,
) -> tuple[dict[tuple[int, int], _Sample], float, float]:
    spacing_x = width / float(columns)
    spacing_y = height / float(rows)
    samples: dict[tuple[int, int], _Sample] = {}
    area_epsilon = max(width, height, 1.0) ** 2 * 1.0e-12
    containment_epsilon = 1.0e-8

    for triangle in triangles:
        if abs(_signed_area_twice(triangle.points)) <= area_epsilon:
            continue
        triangle_min_x = min(point[0] for point in triangle.points)
        triangle_max_x = max(point[0] for point in triangle.points)
        triangle_min_y = min(point[1] for point in triangle.points)
        triangle_max_y = max(point[1] for point in triangle.points)
        column_start = max(
            0,
            int((triangle_min_x - minimum_x) // spacing_x),
        )
        column_end = min(
            columns,
            int(ceil((triangle_max_x - minimum_x) / spacing_x)),
        )
        row_start = max(
            0,
            int((triangle_min_y - minimum_y) // spacing_y),
        )
        row_end = min(
            rows,
            int(ceil((triangle_max_y - minimum_y) / spacing_y)),
        )
        for row in range(row_start, row_end + 1):
            y = minimum_y + spacing_y * row
            for column in range(column_start, column_end + 1):
                x = minimum_x + spacing_x * column
                depth = _barycentric_depth(
                    triangle,
                    x,
                    y,
                    epsilon=containment_epsilon,
                )
                if depth is None:
                    continue
                key = (column, row)
                previous = samples.get(key)
                # Camera-local Z is negative in front of the camera. The larger value
                # is therefore the front-most visible surface.
                if previous is None or depth > previous.depth:
                    samples[key] = _Sample(
                        x=float(x),
                        y=float(y),
                        depth=depth,
                        source_face_index=triangle.face_index,
                    )
    return samples, spacing_x, spacing_y


def _triangulated_face_adjacency(
    snapshot: MeshSnapshot,
) -> Mapping[int, frozenset[int]]:
    """Return deterministic shared-edge adjacency for triangulated source faces."""

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    loops_by_id = snapshot.loop_by_id()
    adjacency: dict[int, set[int]] = {
        face.id.index: set() for face in snapshot.faces
    }
    owners_by_edge: dict[tuple[int, int], list[int]] = {}
    for face in snapshot.faces:
        vertex_ids = tuple(
            loops_by_id[loop_id].vertex_id.index for loop_id in face.loop_ids
        )
        if len(vertex_ids) != 3:
            raise DepthCameraProjectionError(
                f"triangulated face {face.id.index} does not contain three vertices"
            )
        for corner_index, first in enumerate(vertex_ids):
            second = vertex_ids[(corner_index + 1) % 3]
            pair = tuple(sorted((first, second)))
            owners_by_edge.setdefault(pair, []).append(face.id.index)

    for owners in owners_by_edge.values():
        unique = tuple(sorted(set(owners)))
        for owner_index, first in enumerate(unique):
            for second in unique[owner_index + 1 :]:
                adjacency[first].add(second)
                adjacency[second].add(first)
    return {
        face_index: frozenset(sorted(neighbors))
        for face_index, neighbors in sorted(adjacency.items())
    }


def _sampled_faces_form_local_patch(
    keys: tuple[tuple[int, int], ...],
    samples: Mapping[tuple[int, int], _Sample],
    face_adjacency: Mapping[int, frozenset[int]],
) -> bool:
    """Return whether sampled source faces form one shared-edge-local patch."""

    source_faces = frozenset(samples[key].source_face_index for key in keys)
    if len(source_faces) <= 1:
        return True
    if any(face_index not in face_adjacency for face_index in source_faces):
        return False

    start = min(source_faces)
    visited = {start}
    pending = [start]
    while pending:
        current = pending.pop()
        for neighbor in sorted(face_adjacency[current]):
            if neighbor not in source_faces or neighbor in visited:
                continue
            visited.add(neighbor)
            pending.append(neighbor)
    return visited == set(source_faces)


def _cell_faces(
    samples: Mapping[tuple[int, int], _Sample],
    *,
    columns: int,
    rows: int,
    edge_threshold: float,
    face_adjacency: Mapping[int, frozenset[int]],
) -> Tuple[tuple[tuple[int, int], tuple[int, int], tuple[int, int]], ...]:
    """Build grid triangles without mistaking a steep connected face for a depth gap."""

    faces: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    for row in range(rows):
        for column in range(columns):
            corners = (
                (column, row),
                (column + 1, row),
                (column + 1, row + 1),
                (column, row + 1),
            )
            valid = tuple(key for key in corners if key in samples)
            if len(valid) < 3:
                continue

            def classify(
                keys: tuple[tuple[int, int], ...],
            ) -> tuple[bool, bool, float]:
                depths = tuple(samples[key].depth for key in keys)
                depth_jump = max(depths) - min(depths)
                if depth_jump <= edge_threshold:
                    return True, False, depth_jump
                topology_preserved = _sampled_faces_form_local_patch(
                    keys,
                    samples,
                    face_adjacency,
                )
                return topology_preserved, topology_preserved, depth_jump

            if len(valid) == 3:
                triangle = (valid[0], valid[1], valid[2])
                accepted, _topology_preserved, _jump = classify(triangle)
                if accepted:
                    faces.append(triangle)
                continue

            diagonal_a = (
                (corners[0], corners[1], corners[2]),
                (corners[0], corners[2], corners[3]),
            )
            diagonal_b = (
                (corners[0], corners[1], corners[3]),
                (corners[1], corners[2], corners[3]),
            )
            classified_a = tuple(classify(triangle) for triangle in diagonal_a)
            classified_b = tuple(classify(triangle) for triangle in diagonal_b)
            valid_a = all(result[0] for result in classified_a)
            valid_b = all(result[0] for result in classified_b)
            if not valid_a and not valid_b:
                continue
            if valid_a and valid_b:
                # Prefer the diagonal that required fewer topology overrides. When both
                # are equivalent, retain the previous minimum-depth-jump behavior.
                a_score = (
                    sum(1 for result in classified_a if result[1]),
                    abs(samples[corners[0]].depth - samples[corners[2]].depth),
                )
                b_score = (
                    sum(1 for result in classified_b if result[1]),
                    abs(samples[corners[1]].depth - samples[corners[3]].depth),
                )
                selected = diagonal_a if a_score <= b_score else diagonal_b
            else:
                selected = diagonal_a if valid_a else diagonal_b
            faces.extend(selected)
    return tuple(faces)


def _smooth_samples(
    samples: Mapping[tuple[int, int], _Sample],
    faces: Tuple[tuple[tuple[int, int], tuple[int, int], tuple[int, int]], ...],
    *,
    strength: float,
    edge_threshold: float,
) -> dict[tuple[int, int], _Sample]:
    if strength <= 0.0:
        return dict(samples)
    adjacency: dict[tuple[int, int], set[tuple[int, int]]] = {
        key: set() for key in samples
    }
    for face in faces:
        for index, first in enumerate(face):
            for second in face[index + 1 :]:
                adjacency[first].add(second)
                adjacency[second].add(first)

    resolved: dict[tuple[int, int], _Sample] = {}
    for key, sample in samples.items():
        neighbor_depths = tuple(
            samples[neighbor].depth
            for neighbor in sorted(adjacency[key])
            if abs(samples[neighbor].depth - sample.depth) <= edge_threshold
        )
        if not neighbor_depths:
            resolved[key] = sample
            continue
        average = sum(neighbor_depths) / float(len(neighbor_depths))
        depth = sample.depth * (1.0 - strength) + average * strength
        resolved[key] = _Sample(
            x=sample.x,
            y=sample.y,
            depth=float(depth),
            source_face_index=sample.source_face_index,
        )
    return resolved


def _dense_surface_snapshot(
    source: MeshSnapshot,
    samples: Mapping[tuple[int, int], _Sample],
    faces: Tuple[tuple[tuple[int, int], tuple[int, int], tuple[int, int]], ...],
    *,
    projected_origin: A1ProjectedPoint,
    uniform_scale: float,
    frame: A1CameraProjectionFrame,
    uv_layer_name: str,
) -> MeshSnapshot:
    used_keys = tuple(
        sorted(
            {key for face in faces for key in face},
            key=lambda item: (item[1], item[0]),
        )
    )
    if len(used_keys) < 3:
        raise DepthCameraProjectionError(
            "depth sampling produced fewer than three connected surface points"
        )
    vertex_index = {key: index for index, key in enumerate(used_keys)}

    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(source.source_object_id, index),
            position=(
                (samples[key].x - projected_origin.u) / uniform_scale,
                -(samples[key].y - projected_origin.v) / uniform_scale,
                samples[key].depth,
            ),
            normal=(0.0, 0.0, 1.0),
        )
        for index, key in enumerate(used_keys)
    )

    dense_faces = tuple(
        tuple(vertex_index[key] for key in face)
        for face in faces
        if all(key in vertex_index for key in face)
    )
    if not dense_faces:
        raise DepthCameraProjectionError("depth sampling produced no valid triangles")

    edge_pairs = tuple(
        sorted(
            {
                tuple(sorted((face[index], face[(index + 1) % 3])))
                for face in dense_faces
                for index in range(3)
            }
        )
    )
    edge_id_by_pair = {
        pair: EdgeId(index) for index, pair in enumerate(edge_pairs)
    }
    edges = tuple(
        MeshEdge(
            id=edge_id_by_pair[pair],
            source_id=None,
            vertex_ids=(VertexId(pair[0]), VertexId(pair[1])),
            seam=False,
            sharp=False,
        )
        for pair in edge_pairs
    )

    loops: list[MeshLoop] = []
    mesh_faces: list[MeshFace] = []
    for face_index, face in enumerate(dense_faces):
        loop_ids: list[LoopId] = []
        for corner_index, vertex_value in enumerate(face):
            following = face[(corner_index + 1) % 3]
            pair = tuple(sorted((vertex_value, following)))
            loop_id = LoopId(len(loops))
            sample = samples[used_keys[vertex_value]]
            pixel_x = sample.x + float(frame.texture_width) / 2.0
            pixel_y = sample.y + float(frame.texture_height) / 2.0
            uv = (
                pixel_x / float(frame.texture_width),
                1.0 - pixel_y / float(frame.texture_height),
            )
            loops.append(
                MeshLoop(
                    id=loop_id,
                    source_id=SourceLoopId(
                        source.source_object_id,
                        face_index,
                        corner_index,
                    ),
                    vertex_id=VertexId(vertex_value),
                    edge_id=edge_id_by_pair[pair],
                    uvs=(LoopUV(layer_name=uv_layer_name, coordinate=uv),),
                )
            )
            loop_ids.append(loop_id)
        mesh_faces.append(
            MeshFace(
                id=FaceId(face_index),
                source_id=SourceFaceId(source.source_object_id, face_index),
                loop_ids=tuple(loop_ids),
                material_index=0,
                normal=(0.0, 0.0, 1.0),
                smooth=True,
            )
        )

    snapshot = MeshSnapshot(
        snapshot_id=f"{source.snapshot_id}:depth-camera-relief",
        source_object_id=source.source_object_id,
        object_name=source.object_name,
        vertices=vertices,
        edges=edges,
        loops=tuple(loops),
        faces=tuple(mesh_faces),
        uv_layer_names=(uv_layer_name,),
        active_uv_layer=uv_layer_name,
        render_uv_layer=uv_layer_name,
        world_matrix=_projected_translation_matrix(projected_origin, uniform_scale),
    )
    MeshSnapshotValidator().validate_or_raise(snapshot)
    return snapshot


def build_depth_camera_projection_surface(
    snapshot: MeshSnapshot,
    frame: A1CameraProjectionFrame,
    *,
    uniform_scale: float,
    uv_layer_name: str,
    settings: DepthCameraProjectionSettings = DepthCameraProjectionSettings(),
) -> DepthCameraProjectionResult:
    """Project, optimize, smooth, and return one visible depth-relief snapshot."""

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    if not isinstance(frame, A1CameraProjectionFrame):
        raise TypeError("frame must be A1CameraProjectionFrame")
    if isinstance(uniform_scale, bool) or not isinstance(uniform_scale, (int, float)):
        raise TypeError("uniform_scale must be a finite positive number")
    resolved_scale = float(uniform_scale)
    if not isfinite(resolved_scale) or resolved_scale <= 0.0:
        raise ValueError("uniform_scale must be a finite positive number")
    if not isinstance(uv_layer_name, str) or not uv_layer_name.strip():
        raise ValueError("uv_layer_name must be a non-empty string")
    if not isinstance(settings, DepthCameraProjectionSettings):
        raise TypeError("settings must be DepthCameraProjectionSettings")

    MeshSnapshotValidator().validate_or_raise(snapshot)
    if not snapshot.faces:
        raise DepthCameraProjectionError("depth projection requires at least one face")

    origin = _translation_only_origin(snapshot.world_matrix)
    projected_origin = frame.project_world_point(origin, field_name="object_origin")
    triangulated = triangulate_snapshot(snapshot).snapshot
    face_adjacency = _triangulated_face_adjacency(triangulated)
    projected_by_vertex = {
        vertex.id: frame.project_world_point(
            _world_point(origin, vertex.position),
            field_name=f"vertex[{vertex.id.index}]",
        )
        for vertex in triangulated.vertices
    }
    triangles = tuple(
        _triangle_from_face(triangulated, face_index, projected_by_vertex)
        for face_index in range(len(triangulated.faces))
    )
    nondegenerate = tuple(
        triangle
        for triangle in triangles
        if abs(_signed_area_twice(triangle.points)) > 1.0e-12
    )
    if not nondegenerate:
        raise DepthCameraProjectionError(
            "all source triangles collapse in active-camera screen space"
        )

    half_width = float(frame.texture_width) / 2.0
    half_height = float(frame.texture_height) / 2.0
    minimum_x = max(
        -half_width,
        min(point[0] for triangle in nondegenerate for point in triangle.points),
    )
    maximum_x = min(
        half_width,
        max(point[0] for triangle in nondegenerate for point in triangle.points),
    )
    minimum_y = max(
        -half_height,
        min(point[1] for triangle in nondegenerate for point in triangle.points),
    )
    maximum_y = min(
        half_height,
        max(point[1] for triangle in nondegenerate for point in triangle.points),
    )
    width = maximum_x - minimum_x
    height = maximum_y - minimum_y
    if width <= 1.0e-9 or height <= 1.0e-9:
        raise DepthCameraProjectionError(
            "visible projected bounds do not contain a two-dimensional surface"
        )

    columns, rows = _grid_dimensions(width, height, settings)
    sampled, spacing_x, spacing_y = _sample_front_surface(
        nondegenerate,
        minimum_x=minimum_x,
        minimum_y=minimum_y,
        width=width,
        height=height,
        columns=columns,
        rows=rows,
    )
    if len(sampled) < 3:
        raise DepthCameraProjectionError(
            "depth lattice did not intersect at least three visible points; reduce "
            "Depth Mesh Error or increase Max Depth Points"
        )

    raw_depths = tuple(sample.depth for sample in sampled.values())
    farthest = min(raw_depths)
    nearest = max(raw_depths)
    depth_span = nearest - farthest
    edge_threshold = max(
        1.0e-8,
        depth_span * settings.edge_threshold_fraction,
    )
    faces = _cell_faces(
        sampled,
        columns=columns,
        rows=rows,
        edge_threshold=edge_threshold,
        face_adjacency=face_adjacency,
    )
    if not faces:
        raise DepthCameraProjectionError(
            "Depth Edge Threshold disconnected every sampled triangle and source "
            "topology could not prove local continuity; increase the threshold or "
            "lower Depth Mesh Error"
        )
    smoothed = _smooth_samples(
        sampled,
        faces,
        strength=settings.smoothing,
        edge_threshold=edge_threshold,
    )
    smoothed_depths = tuple(sample.depth for sample in smoothed.values())
    farthest = min(smoothed_depths)
    nearest = max(smoothed_depths)

    if settings.base_mode is DepthProjectionBaseMode.FARTHEST_VISIBLE:
        base_depth = farthest
    elif settings.base_mode is DepthProjectionBaseMode.OBJECT_ORIGIN:
        base_depth = float(projected_origin.depth)
        tolerance = max(1.0e-8, abs(base_depth) * 1.0e-8)
        behind_origin = tuple(
            sample.depth
            for sample in smoothed.values()
            if sample.depth < base_depth - tolerance
        )
        if behind_origin:
            raise DepthCameraProjectionError(
                "OBJECT_ORIGIN depth base lies in front of visible surface points; "
                "use FARTHEST_VISIBLE or move Object Origin behind the visible surface"
            )
    else:
        raise AssertionError(f"Unhandled depth base mode: {settings.base_mode}")

    maximum_relief = max(sample.depth - base_depth for sample in smoothed.values())
    if maximum_relief < -1.0e-8:
        raise DepthCameraProjectionError(
            "depth relief points extend away from the selected base plane"
        )

    surface = _dense_surface_snapshot(
        snapshot,
        smoothed,
        faces,
        projected_origin=projected_origin,
        uniform_scale=resolved_scale,
        frame=frame,
        uv_layer_name=uv_layer_name,
    )
    if len(surface.vertices) > settings.max_points:
        raise DepthCameraProjectionError(
            f"depth surface exceeded max_points: {len(surface.vertices)} > "
            f"{settings.max_points}"
        )

    return DepthCameraProjectionResult(
        snapshot=surface,
        frame=frame,
        projected_origin=projected_origin,
        base_mode=settings.base_mode,
        base_depth=float(base_depth),
        farthest_visible_depth=float(farthest),
        nearest_visible_depth=float(nearest),
        maximum_relief=max(0.0, float(maximum_relief)),
        requested_spacing_pixels=settings.mesh_error_pixels,
        resolved_spacing_x_pixels=float(spacing_x),
        resolved_spacing_y_pixels=float(spacing_y),
        source_triangle_count=len(nondegenerate),
        sampled_point_count=len(surface.vertices),
    )


__all__ = [
    "DepthCameraProjectionError",
    "DepthCameraProjectionResult",
    "DepthCameraProjectionSettings",
    "DepthProjectionBaseMode",
    "build_depth_camera_projection_surface",
]
