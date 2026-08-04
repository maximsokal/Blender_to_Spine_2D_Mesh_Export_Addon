"""Visible-topology owner for source-bounded camera-depth relief generation.

Fully visible source triangles keep their exact projected vertices and topology. Only
triangles intersected by the active-camera frame are clipped and locally retriangulated.
Dense fully framed sources may still delegate to the established bounded lattice owner,
but camera-clipped low-poly sources never replace the complete object with a grid.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from math import isfinite
from typing import Mapping, Sequence

from ..camera_projection import A1CameraProjectionFrame, A1CameraProjectionKind
from ..projection import A1ProjectedPoint
from .depth_camera_projection import (
    DepthCameraProjectionError,
    DepthCameraProjectionResult,
    DepthCameraProjectionSettings,
    DepthProjectionBaseMode,
    _ProjectedTriangle,
    _Sample,
    _dense_surface_snapshot,
    _signed_area_twice,
    _smooth_samples,
    _translation_only_origin,
)
from .depth_camera_projection_bounded import (
    _depth_at_screen_point,
    _projected_triangles,
    build_depth_camera_projection_surface as _build_bounded_surface,
)
from .ids import VertexId
from .model import MeshSnapshot
from .validator import MeshSnapshotValidator


logger = logging.getLogger(__name__)

_LOCAL_VERTEX_LIMIT = 512
_LOCAL_TRIANGLE_LIMIT = 1024
_AREA_EPSILON = 1.0e-12

_SampleKey = tuple[int, int]
_TriangleKeys = tuple[_SampleKey, _SampleKey, _SampleKey]


class _LocalTopologyUnavailable(RuntimeError):
    """Internal signal that the established bounded owner should handle the source."""


@dataclass(frozen=True, slots=True)
class _ClipPoint:
    """One projected polygon point with optional exact source-vertex provenance."""

    x: float
    y: float
    depth: float
    source_vertex_index: int | None

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.x, self.y, self.depth)):
            raise DepthCameraProjectionError(
                "camera clipping produced a non-finite projected point"
            )
        if self.source_vertex_index is not None:
            if (
                isinstance(self.source_vertex_index, bool)
                or not isinstance(self.source_vertex_index, int)
                or self.source_vertex_index < 0
            ):
                raise TypeError(
                    "source_vertex_index must be a non-negative int or None"
                )


@dataclass(slots=True)
class _TopologyRing:
    """One visible source triangle after optional frame clipping."""

    face_index: int
    keys: list[_SampleKey]
    clipped: bool


@dataclass(frozen=True, slots=True)
class _FrameBoundary:
    axis: int
    value: float
    keep_greater: bool


def _point_coordinate(point: _ClipPoint, axis: int) -> float:
    if axis == 0:
        return point.x
    if axis == 1:
        return point.y
    raise ValueError(f"axis must be 0 or 1; received {axis}")


def _point_inside_boundary(
    point: _ClipPoint,
    boundary: _FrameBoundary,
    *,
    tolerance: float,
) -> bool:
    coordinate = _point_coordinate(point, boundary.axis)
    if boundary.keep_greater:
        return coordinate >= boundary.value - tolerance
    return coordinate <= boundary.value + tolerance


def _interpolate_segment_depth(
    start: _ClipPoint,
    end: _ClipPoint,
    factor: float,
    *,
    kind: A1CameraProjectionKind,
) -> float:
    resolved = min(1.0, max(0.0, float(factor)))
    if kind is A1CameraProjectionKind.ORTHOGRAPHIC:
        depth = start.depth + (end.depth - start.depth) * resolved
    elif kind is A1CameraProjectionKind.PERSPECTIVE:
        if abs(start.depth) <= 1.0e-15 or abs(end.depth) <= 1.0e-15:
            raise DepthCameraProjectionError(
                "perspective clipping cannot interpolate through camera depth zero"
            )
        reciprocal = (
            (1.0 - resolved) / start.depth
            + resolved / end.depth
        )
        if abs(reciprocal) <= 1.0e-15:
            raise DepthCameraProjectionError(
                "perspective clipping produced zero reciprocal depth"
            )
        depth = 1.0 / reciprocal
    else:
        raise AssertionError(f"Unhandled camera projection kind: {kind}")
    if not isfinite(depth):
        raise DepthCameraProjectionError(
            "camera clipping depth interpolation became non-finite"
        )
    return float(depth)


def _boundary_intersection(
    start: _ClipPoint,
    end: _ClipPoint,
    boundary: _FrameBoundary,
    *,
    kind: A1CameraProjectionKind,
    tolerance: float,
) -> _ClipPoint:
    start_coordinate = _point_coordinate(start, boundary.axis)
    end_coordinate = _point_coordinate(end, boundary.axis)
    denominator = end_coordinate - start_coordinate
    if abs(denominator) <= tolerance:
        raise DepthCameraProjectionError(
            "camera clipping encountered a parallel segment without a stable "
            "boundary intersection"
        )
    factor = (boundary.value - start_coordinate) / denominator
    factor = min(1.0, max(0.0, factor))
    if factor <= tolerance:
        return start
    if factor >= 1.0 - tolerance:
        return end

    x = start.x + (end.x - start.x) * factor
    y = start.y + (end.y - start.y) * factor
    if boundary.axis == 0:
        x = boundary.value
    else:
        y = boundary.value
    return _ClipPoint(
        x=float(x),
        y=float(y),
        depth=_interpolate_segment_depth(start, end, factor, kind=kind),
        source_vertex_index=None,
    )


def _points_close(
    first: _ClipPoint,
    second: _ClipPoint,
    *,
    coordinate_tolerance: float,
    depth_tolerance: float,
) -> bool:
    return (
        abs(first.x - second.x) <= coordinate_tolerance
        and abs(first.y - second.y) <= coordinate_tolerance
        and abs(first.depth - second.depth) <= depth_tolerance
    )


def _deduplicate_polygon_points(
    points: Sequence[_ClipPoint],
    *,
    coordinate_tolerance: float,
    depth_tolerance: float,
) -> tuple[_ClipPoint, ...]:
    resolved: list[_ClipPoint] = []
    for point in points:
        if resolved and _points_close(
            resolved[-1],
            point,
            coordinate_tolerance=coordinate_tolerance,
            depth_tolerance=depth_tolerance,
        ):
            if resolved[-1].source_vertex_index is None:
                resolved[-1] = point
            continue
        resolved.append(point)
    if len(resolved) > 1 and _points_close(
        resolved[0],
        resolved[-1],
        coordinate_tolerance=coordinate_tolerance,
        depth_tolerance=depth_tolerance,
    ):
        if resolved[0].source_vertex_index is None:
            resolved[0] = resolved[-1]
        resolved.pop()
    return tuple(resolved)


def _clip_against_boundary(
    polygon: Sequence[_ClipPoint],
    boundary: _FrameBoundary,
    *,
    kind: A1CameraProjectionKind,
    coordinate_tolerance: float,
    depth_tolerance: float,
) -> tuple[_ClipPoint, ...]:
    if not polygon:
        return ()
    output: list[_ClipPoint] = []
    previous = polygon[-1]
    previous_inside = _point_inside_boundary(
        previous,
        boundary,
        tolerance=coordinate_tolerance,
    )
    for current in polygon:
        current_inside = _point_inside_boundary(
            current,
            boundary,
            tolerance=coordinate_tolerance,
        )
        if current_inside:
            if not previous_inside:
                output.append(
                    _boundary_intersection(
                        previous,
                        current,
                        boundary,
                        kind=kind,
                        tolerance=coordinate_tolerance,
                    )
                )
            output.append(current)
        elif previous_inside:
            output.append(
                _boundary_intersection(
                    previous,
                    current,
                    boundary,
                    kind=kind,
                    tolerance=coordinate_tolerance,
                )
            )
        previous = current
        previous_inside = current_inside
    return _deduplicate_polygon_points(
        output,
        coordinate_tolerance=coordinate_tolerance,
        depth_tolerance=depth_tolerance,
    )


def _clip_triangle_to_frame(
    triangle: _ProjectedTriangle,
    vertex_ids: tuple[VertexId, VertexId, VertexId],
    frame: A1CameraProjectionFrame,
) -> tuple[tuple[_ClipPoint, ...], bool]:
    half_width = float(frame.texture_width) / 2.0
    half_height = float(frame.texture_height) / 2.0
    coordinate_tolerance = max(
        float(frame.texture_width),
        float(frame.texture_height),
        1.0,
    ) * 1.0e-10
    depth_scale = max(abs(point[2]) for point in triangle.points)
    depth_tolerance = max(1.0e-10, depth_scale * 1.0e-10)
    polygon: tuple[_ClipPoint, ...] = tuple(
        _ClipPoint(
            x=float(point[0]),
            y=float(point[1]),
            depth=float(point[2]),
            source_vertex_index=vertex_id.index,
        )
        for point, vertex_id in zip(triangle.points, vertex_ids, strict=True)
    )
    boundaries = (
        _FrameBoundary(axis=0, value=-half_width, keep_greater=True),
        _FrameBoundary(axis=0, value=half_width, keep_greater=False),
        _FrameBoundary(axis=1, value=-half_height, keep_greater=True),
        _FrameBoundary(axis=1, value=half_height, keep_greater=False),
    )
    for boundary in boundaries:
        polygon = _clip_against_boundary(
            polygon,
            boundary,
            kind=frame.kind,
            coordinate_tolerance=coordinate_tolerance,
            depth_tolerance=depth_tolerance,
        )
        if len(polygon) < 3:
            return (), True

    clipped = (
        len(polygon) != 3
        or any(point.source_vertex_index is None for point in polygon)
        or tuple(point.source_vertex_index for point in polygon)
        != tuple(vertex_id.index for vertex_id in vertex_ids)
    )
    return polygon, clipped


def _triangle_vertex_ids(
    triangulated: MeshSnapshot,
) -> Mapping[int, tuple[VertexId, VertexId, VertexId]]:
    loops_by_id = triangulated.loop_by_id()
    resolved: dict[int, tuple[VertexId, VertexId, VertexId]] = {}
    for face in triangulated.faces:
        vertex_ids = tuple(
            loops_by_id[loop_id].vertex_id for loop_id in face.loop_ids
        )
        if len(vertex_ids) != 3:
            raise DepthCameraProjectionError(
                f"triangulated face {face.id.index} does not contain three vertices"
            )
        resolved[face.id.index] = (
            vertex_ids[0],
            vertex_ids[1],
            vertex_ids[2],
        )
    return resolved


def _polygon_probe_points(
    polygon: Sequence[_ClipPoint],
) -> tuple[tuple[float, float], ...]:
    count = float(len(polygon))
    centroid_x = sum(point.x for point in polygon) / count
    centroid_y = sum(point.y for point in polygon) / count
    probes: list[tuple[float, float]] = [(centroid_x, centroid_y)]
    for point in polygon:
        probes.append(
            (
                centroid_x * 0.15 + point.x * 0.85,
                centroid_y * 0.15 + point.y * 0.85,
            )
        )
    for index, point in enumerate(polygon):
        following = polygon[(index + 1) % len(polygon)]
        probes.append(
            (
                centroid_x * 0.10 + (point.x + following.x) * 0.45,
                centroid_y * 0.10 + (point.y + following.y) * 0.45,
            )
        )
    return tuple(probes)


def _visible_clipped_face_indices(
    triangles: Sequence[_ProjectedTriangle],
    polygons: Mapping[int, tuple[_ClipPoint, ...]],
    *,
    kind: A1CameraProjectionKind,
) -> tuple[int, ...]:
    extent = max(
        (
            max(
                max(point.x for point in polygon) - min(point.x for point in polygon),
                max(point.y for point in polygon) - min(point.y for point in polygon),
            )
            for polygon in polygons.values()
        ),
        default=1.0,
    )
    containment_epsilon = max(1.0e-8, extent * 1.0e-10)
    depth_scale = max(
        (abs(point[2]) for triangle in triangles for point in triangle.points),
        default=1.0,
    )
    depth_tolerance = max(1.0e-8, depth_scale * 1.0e-8)
    triangle_by_face = {triangle.face_index: triangle for triangle in triangles}

    visible: list[int] = []
    for face_index in sorted(polygons):
        triangle = triangle_by_face[face_index]
        for x, y in _polygon_probe_points(polygons[face_index]):
            expected_depth = _depth_at_screen_point(
                triangle,
                x,
                y,
                kind=kind,
                epsilon=containment_epsilon,
            )
            if expected_depth is None:
                continue
            front_depth: float | None = None
            for candidate in triangles:
                depth = _depth_at_screen_point(
                    candidate,
                    x,
                    y,
                    kind=kind,
                    epsilon=containment_epsilon,
                )
                if depth is not None and (
                    front_depth is None or depth > front_depth
                ):
                    front_depth = depth
            if (
                front_depth is not None
                and expected_depth >= front_depth - depth_tolerance
            ):
                visible.append(face_index)
                break
    return tuple(visible)


def _generated_signature(
    point: _ClipPoint,
    *,
    coordinate_quantum: float,
    depth_quantum: float,
) -> tuple[int, int, int]:
    return (
        round(point.x / coordinate_quantum),
        round(point.y / coordinate_quantum),
        round(point.depth / depth_quantum),
    )


def _ring_area(
    keys: Sequence[_SampleKey],
    samples: Mapping[_SampleKey, _Sample],
) -> float:
    return 0.5 * sum(
        samples[key].x * samples[keys[(index + 1) % len(keys)]].y
        - samples[keys[(index + 1) % len(keys)]].x * samples[key].y
        for index, key in enumerate(keys)
    )


def _clean_key_ring(keys: Sequence[_SampleKey]) -> list[_SampleKey]:
    resolved: list[_SampleKey] = []
    for key in keys:
        if not resolved or resolved[-1] != key:
            resolved.append(key)
    if len(resolved) > 1 and resolved[0] == resolved[-1]:
        resolved.pop()
    return resolved


def _build_samples_and_rings(
    triangulated: MeshSnapshot,
    projected_by_vertex: Mapping[VertexId, A1ProjectedPoint],
    polygons: Mapping[int, tuple[_ClipPoint, ...]],
    clipped_by_face: Mapping[int, bool],
    visible_face_indices: Sequence[int],
    frame: A1CameraProjectionFrame,
) -> tuple[dict[_SampleKey, _Sample], list[_TopologyRing]]:
    coordinate_quantum = max(
        float(frame.texture_width),
        float(frame.texture_height),
        1.0,
    ) * 1.0e-10
    depth_scale = max(
        (
            abs(point.depth)
            for face_index in visible_face_indices
            for point in polygons[face_index]
        ),
        default=1.0,
    )
    depth_quantum = max(1.0e-10, depth_scale * 1.0e-10)

    generated_points: dict[tuple[int, int, int], _ClipPoint] = {}
    generated_owner: dict[tuple[int, int, int], int] = {}
    for face_index in visible_face_indices:
        for point in polygons[face_index]:
            if point.source_vertex_index is not None:
                continue
            signature = _generated_signature(
                point,
                coordinate_quantum=coordinate_quantum,
                depth_quantum=depth_quantum,
            )
            generated_points.setdefault(signature, point)
            generated_owner[signature] = min(
                face_index,
                generated_owner.get(signature, face_index),
            )

    generated_key_by_signature = {
        signature: (len(triangulated.vertices) + index, 1)
        for index, signature in enumerate(sorted(generated_points))
    }
    samples: dict[_SampleKey, _Sample] = {}
    rings: list[_TopologyRing] = []
    for face_index in visible_face_indices:
        keys: list[_SampleKey] = []
        for point in polygons[face_index]:
            if point.source_vertex_index is not None:
                key = (point.source_vertex_index, 0)
                projected = projected_by_vertex[VertexId(point.source_vertex_index)]
                samples.setdefault(
                    key,
                    _Sample(
                        x=float(projected.u),
                        y=float(projected.v),
                        depth=float(projected.depth),
                        source_face_index=face_index,
                    ),
                )
            else:
                signature = _generated_signature(
                    point,
                    coordinate_quantum=coordinate_quantum,
                    depth_quantum=depth_quantum,
                )
                key = generated_key_by_signature[signature]
                samples.setdefault(
                    key,
                    _Sample(
                        x=point.x,
                        y=point.y,
                        depth=point.depth,
                        source_face_index=generated_owner[signature],
                    ),
                )
            keys.append(key)
        cleaned = _clean_key_ring(keys)
        if len(cleaned) < 3:
            continue
        if abs(_ring_area(cleaned, samples)) <= _AREA_EPSILON:
            continue
        rings.append(
            _TopologyRing(
                face_index=face_index,
                keys=cleaned,
                clipped=bool(clipped_by_face[face_index]),
            )
        )
    if not rings:
        raise DepthCameraProjectionError(
            "active-camera clipping retained no visible two-dimensional polygons"
        )
    return samples, rings


def _used_keys(rings: Sequence[_TopologyRing]) -> set[_SampleKey]:
    return {key for ring in rings for key in ring.keys}


def _removal_cost(
    ring: _TopologyRing,
    index: int,
    samples: Mapping[_SampleKey, _Sample],
) -> float:
    previous = samples[ring.keys[index - 1]]
    current = samples[ring.keys[index]]
    following = samples[ring.keys[(index + 1) % len(ring.keys)]]
    return abs(
        (current.x - previous.x) * (following.y - previous.y)
        - (current.y - previous.y) * (following.x - previous.x)
    )


def _ring_without_key_is_valid(
    ring: _TopologyRing,
    key: _SampleKey,
    samples: Mapping[_SampleKey, _Sample],
) -> bool:
    if key not in ring.keys:
        return True
    if not ring.clipped or len(ring.keys) <= 3 or ring.keys.count(key) != 1:
        return False
    previous_area = _ring_area(ring.keys, samples)
    reduced = [candidate for candidate in ring.keys if candidate != key]
    reduced_area = _ring_area(reduced, samples)
    if abs(reduced_area) <= _AREA_EPSILON:
        return False
    return previous_area * reduced_area > 0.0


def _fit_clipped_rings_to_budget(
    samples: Mapping[_SampleKey, _Sample],
    rings: list[_TopologyRing],
    *,
    allowed_points: int,
) -> None:
    """Simplify only frame-clipped rings until the global source budget is met."""

    protected = {
        key
        for ring in rings
        if not ring.clipped
        for key in ring.keys
    }
    while len(_used_keys(rings)) > allowed_points:
        used = _used_keys(rings)
        candidates: list[tuple[int, float, _SampleKey]] = []
        for key in sorted(used):
            if key in protected:
                continue
            owning_rings = tuple(ring for ring in rings if key in ring.keys)
            if not owning_rings or not all(
                _ring_without_key_is_valid(ring, key, samples)
                for ring in owning_rings
            ):
                continue
            cost = sum(
                _removal_cost(ring, ring.keys.index(key), samples)
                for ring in owning_rings
            )
            source_penalty = 0 if key[1] == 1 else 1
            candidates.append((source_penalty, cost, key))
        if not candidates:
            raise DepthCameraProjectionError(
                "camera-clipped source topology cannot satisfy the vertex budget "
                "without modifying fully visible polygons; increase Max Depth Points "
                "or keep more of the object inside the active-camera frame"
            )
        _penalty, _cost, selected = min(candidates)
        for ring in rings:
            if selected in ring.keys:
                ring.keys = [key for key in ring.keys if key != selected]
        logger.debug(
            "Locally simplified clipped boundary vertex %s; remaining=%d budget=%d",
            selected,
            len(_used_keys(rings)),
            allowed_points,
        )


def _triangulate_rings(
    rings: Sequence[_TopologyRing],
    samples: Mapping[_SampleKey, _Sample],
) -> tuple[_TriangleKeys, ...]:
    faces: list[_TriangleKeys] = []
    for ring in rings:
        anchor = ring.keys[0]
        for index in range(1, len(ring.keys) - 1):
            triangle = (anchor, ring.keys[index], ring.keys[index + 1])
            projected = tuple(
                (
                    samples[key].x,
                    samples[key].y,
                    samples[key].depth,
                )
                for key in triangle
            )
            probe = _ProjectedTriangle(
                face_index=ring.face_index,
                points=projected,
            )
            if abs(_signed_area_twice(probe.points)) <= _AREA_EPSILON:
                continue
            faces.append(triangle)
    if not faces:
        raise DepthCameraProjectionError(
            "local camera clipping produced no valid triangles"
        )
    return tuple(faces)


def _result_from_local_topology(
    snapshot: MeshSnapshot,
    frame: A1CameraProjectionFrame,
    *,
    projected_origin: A1ProjectedPoint,
    samples: Mapping[_SampleKey, _Sample],
    faces: tuple[_TriangleKeys, ...],
    uniform_scale: float,
    uv_layer_name: str,
    settings: DepthCameraProjectionSettings,
    source_triangle_count: int,
) -> DepthCameraProjectionResult:
    used = {key for face in faces for key in face}
    pruned_samples = {key: samples[key] for key in used}
    raw_depths = tuple(sample.depth for sample in pruned_samples.values())
    raw_farthest = min(raw_depths)
    raw_nearest = max(raw_depths)
    edge_threshold = max(
        1.0e-8,
        (raw_nearest - raw_farthest) * settings.edge_threshold_fraction,
    )
    smoothed = _smooth_samples(
        pruned_samples,
        faces,
        strength=settings.smoothing,
        edge_threshold=edge_threshold,
    )
    farthest = min(sample.depth for sample in smoothed.values())
    nearest = max(sample.depth for sample in smoothed.values())

    if settings.base_mode is DepthProjectionBaseMode.FARTHEST_VISIBLE:
        base_depth = farthest
    elif settings.base_mode is DepthProjectionBaseMode.OBJECT_ORIGIN:
        base_depth = float(projected_origin.depth)
        tolerance = max(1.0e-8, abs(base_depth) * 1.0e-8)
        if any(sample.depth < base_depth - tolerance for sample in smoothed.values()):
            raise DepthCameraProjectionError(
                "OBJECT_ORIGIN depth base lies in front of visible surface points; "
                "use FARTHEST_VISIBLE or move Object Origin behind the visible surface"
            )
    else:
        raise AssertionError(f"Unhandled depth base mode: {settings.base_mode}")

    surface = _dense_surface_snapshot(
        snapshot,
        smoothed,
        faces,
        projected_origin=projected_origin,
        uniform_scale=uniform_scale,
        frame=frame,
        uv_layer_name=uv_layer_name,
    )
    maximum_relief = max(
        float(vertex.position[2]) - base_depth for vertex in surface.vertices
    )
    if maximum_relief < -1.0e-8:
        raise DepthCameraProjectionError(
            "depth relief points extend away from the selected base plane"
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
        resolved_spacing_x_pixels=settings.mesh_error_pixels,
        resolved_spacing_y_pixels=settings.mesh_error_pixels,
        source_triangle_count=source_triangle_count,
        sampled_point_count=len(surface.vertices),
    )


def _build_visible_topology_surface(
    snapshot: MeshSnapshot,
    frame: A1CameraProjectionFrame,
    *,
    uniform_scale: float,
    uv_layer_name: str,
    settings: DepthCameraProjectionSettings,
) -> DepthCameraProjectionResult:
    if len(snapshot.vertices) > _LOCAL_VERTEX_LIMIT:
        raise _LocalTopologyUnavailable(
            f"source has more than {_LOCAL_VERTEX_LIMIT} vertices"
        )
    origin = _translation_only_origin(snapshot.world_matrix)
    projected_origin = frame.project_world_point(origin, field_name="object_origin")
    triangulated, projected_by_vertex, triangles = _projected_triangles(
        snapshot,
        frame,
        origin,
    )
    if len(triangles) > _LOCAL_TRIANGLE_LIMIT:
        raise _LocalTopologyUnavailable(
            f"source has more than {_LOCAL_TRIANGLE_LIMIT} projected triangles"
        )
    vertex_ids_by_face = _triangle_vertex_ids(triangulated)
    polygons: dict[int, tuple[_ClipPoint, ...]] = {}
    clipped_by_face: dict[int, bool] = {}
    for triangle in triangles:
        polygon, clipped = _clip_triangle_to_frame(
            triangle,
            vertex_ids_by_face[triangle.face_index],
            frame,
        )
        if polygon:
            polygons[triangle.face_index] = polygon
            clipped_by_face[triangle.face_index] = clipped
    if not polygons:
        raise DepthCameraProjectionError(
            "active camera frame does not intersect the projected source surface"
        )

    visible_face_indices = _visible_clipped_face_indices(
        triangles,
        polygons,
        kind=frame.kind,
    )
    samples, rings = _build_samples_and_rings(
        triangulated,
        projected_by_vertex,
        polygons,
        clipped_by_face,
        visible_face_indices,
        frame,
    )
    allowed_points = min(settings.max_points, len(snapshot.vertices))
    if len(_used_keys(rings)) > allowed_points:
        if not any(ring.clipped for ring in rings):
            raise _LocalTopologyUnavailable(
                "fully framed visible topology exceeds Max Depth Points"
            )
        _fit_clipped_rings_to_budget(
            samples,
            rings,
            allowed_points=allowed_points,
        )
    faces = _triangulate_rings(rings, samples)
    result = _result_from_local_topology(
        snapshot,
        frame,
        projected_origin=projected_origin,
        samples=samples,
        faces=faces,
        uniform_scale=uniform_scale,
        uv_layer_name=uv_layer_name,
        settings=settings,
        source_triangle_count=len(triangles),
    )
    generated = len(result.snapshot.vertices)
    if generated > allowed_points:
        raise DepthCameraProjectionError(
            "local camera clipping violated the source-bounded vertex contract: "
            f"generated={generated}, source={len(snapshot.vertices)}, "
            f"max_points={settings.max_points}"
        )
    return result


def build_depth_camera_projection_surface(
    snapshot: MeshSnapshot,
    frame: A1CameraProjectionFrame,
    *,
    uniform_scale: float,
    uv_layer_name: str,
    settings: DepthCameraProjectionSettings = DepthCameraProjectionSettings(),
) -> DepthCameraProjectionResult:
    """Preserve visible topology and repair only polygons clipped by the camera."""

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
    if len(snapshot.vertices) < 3:
        raise DepthCameraProjectionError(
            "depth projection requires at least three evaluated source vertices"
        )
    try:
        return _build_visible_topology_surface(
            snapshot,
            frame,
            uniform_scale=resolved_scale,
            uv_layer_name=uv_layer_name,
            settings=settings,
        )
    except _LocalTopologyUnavailable as exc:
        logger.debug(
            "Visible-topology path unavailable for '%s': %s; delegating to "
            "the bounded lattice owner",
            snapshot.source_object_id,
            exc,
        )
        return _build_bounded_surface(
            snapshot,
            frame,
            uniform_scale=resolved_scale,
            uv_layer_name=uv_layer_name,
            settings=settings,
        )


__all__ = ["build_depth_camera_projection_surface"]
