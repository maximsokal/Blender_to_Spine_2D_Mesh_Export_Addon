"""Source-bounded owner for camera-depth relief generation.

Low-poly objects keep exact projected source vertices and visible source triangles, so a
regular sampling lattice cannot shrink pointed silhouettes. Dense or camera-clipped
objects continue through the established lattice implementation, but its point budget is
capped by the evaluated source vertex count. Generated depth vertices therefore never
outnumber either the evaluated source mesh or the user-selected Max Depth Points value.
"""

from __future__ import annotations

from dataclasses import replace
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
    _world_point,
    build_depth_camera_projection_surface as _build_lattice_surface,
)
from .ids import FaceId, VertexId
from .model import MeshSnapshot
from .triangulation import triangulate_snapshot
from .validator import MeshSnapshotValidator


logger = logging.getLogger(__name__)

_DIRECT_VERTEX_LIMIT = 512
_DIRECT_TRIANGLE_LIMIT = 1024
_AREA_EPSILON = 1.0e-12
_VISIBILITY_PROBES = (
    (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    (0.80, 0.10, 0.10),
    (0.10, 0.80, 0.10),
    (0.10, 0.10, 0.80),
    (0.98, 0.01, 0.01),
    (0.01, 0.98, 0.01),
    (0.01, 0.01, 0.98),
    (0.495, 0.495, 0.01),
    (0.495, 0.01, 0.495),
    (0.01, 0.495, 0.495),
)


class _SourceTopologyUnavailable(RuntimeError):
    """Internal signal that the source-bounded lattice fallback must own the mesh."""


def _projected_triangles(
    snapshot: MeshSnapshot,
    frame: A1CameraProjectionFrame,
    origin: tuple[float, float, float],
) -> tuple[
    MeshSnapshot,
    Mapping[VertexId, A1ProjectedPoint],
    tuple[_ProjectedTriangle, ...],
]:
    triangulated = triangulate_snapshot(snapshot).snapshot
    projected_by_vertex = {
        vertex.id: frame.project_world_point(
            _world_point(origin, vertex.position),
            field_name=f"vertex[{vertex.id.index}]",
        )
        for vertex in triangulated.vertices
    }
    loops_by_id = triangulated.loop_by_id()
    triangles: list[_ProjectedTriangle] = []
    for face in sorted(triangulated.faces, key=lambda item: item.id.index):
        vertex_ids = tuple(
            loops_by_id[loop_id].vertex_id for loop_id in face.loop_ids
        )
        if len(vertex_ids) != 3:
            raise DepthCameraProjectionError(
                f"triangulated face {face.id.index} does not contain three vertices"
            )
        triangle = _ProjectedTriangle(
            face_index=face.id.index,
            points=tuple(
                (
                    float(projected_by_vertex[vertex_id].u),
                    float(projected_by_vertex[vertex_id].v),
                    float(projected_by_vertex[vertex_id].depth),
                )
                for vertex_id in vertex_ids
            ),
        )
        if abs(_signed_area_twice(triangle.points)) > _AREA_EPSILON:
            triangles.append(triangle)
    if not triangles:
        raise DepthCameraProjectionError(
            "all source triangles collapse in active-camera screen space"
        )
    return triangulated, projected_by_vertex, tuple(triangles)


def _inside_frame(
    projected: Mapping[VertexId, A1ProjectedPoint],
    frame: A1CameraProjectionFrame,
) -> bool:
    half_width = float(frame.texture_width) / 2.0
    half_height = float(frame.texture_height) / 2.0
    tolerance = (
        max(float(frame.texture_width), float(frame.texture_height), 1.0)
        * 1.0e-9
    )
    return all(
        -half_width - tolerance <= point.u <= half_width + tolerance
        and -half_height - tolerance <= point.v <= half_height + tolerance
        for point in projected.values()
    )


def _screen_barycentric_weights(
    triangle: _ProjectedTriangle,
    x: float,
    y: float,
    *,
    epsilon: float,
) -> tuple[float, float, float] | None:
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
    return float(first_weight), float(second_weight), float(third_weight)


def _depth_at_screen_point(
    triangle: _ProjectedTriangle,
    x: float,
    y: float,
    *,
    kind: A1CameraProjectionKind,
    epsilon: float,
) -> float | None:
    weights = _screen_barycentric_weights(
        triangle,
        x,
        y,
        epsilon=epsilon,
    )
    if weights is None:
        return None
    depths = tuple(float(point[2]) for point in triangle.points)
    if kind is A1CameraProjectionKind.ORTHOGRAPHIC:
        depth = sum(weights[index] * depths[index] for index in range(3))
    elif kind is A1CameraProjectionKind.PERSPECTIVE:
        reciprocal = sum(weights[index] / depths[index] for index in range(3))
        if abs(reciprocal) <= 1.0e-15:
            raise DepthCameraProjectionError(
                "perspective-correct depth interpolation produced zero reciprocal depth"
            )
        depth = 1.0 / reciprocal
    else:
        raise AssertionError(f"Unhandled camera projection kind: {kind}")
    if not isfinite(depth):
        raise DepthCameraProjectionError(
            "front-most visibility depth interpolation became non-finite"
        )
    return float(depth)


def _probe_xy(
    triangle: _ProjectedTriangle,
    weights: tuple[float, float, float],
) -> tuple[float, float]:
    return (
        sum(weights[index] * triangle.points[index][0] for index in range(3)),
        sum(weights[index] * triangle.points[index][1] for index in range(3)),
    )


def _visible_face_indices(
    triangles: Sequence[_ProjectedTriangle],
    *,
    kind: A1CameraProjectionKind,
) -> tuple[int, ...]:
    """Return faces that own at least one deterministic front-most interior probe."""

    extent = max(
        (
            max(
                max(point[0] for point in triangle.points)
                - min(point[0] for point in triangle.points),
                max(point[1] for point in triangle.points)
                - min(point[1] for point in triangle.points),
            )
            for triangle in triangles
        ),
        default=1.0,
    )
    containment_epsilon = max(1.0e-8, extent * 1.0e-10)
    depth_scale = max(
        (abs(point[2]) for triangle in triangles for point in triangle.points),
        default=1.0,
    )
    depth_tolerance = max(1.0e-8, depth_scale * 1.0e-8)

    visible: list[int] = []
    for triangle in triangles:
        for weights in _VISIBILITY_PROBES:
            x, y = _probe_xy(triangle, weights)
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
                visible.append(triangle.face_index)
                break
    return tuple(sorted(set(visible)))


def _direct_samples_and_faces(
    triangulated: MeshSnapshot,
    projected_by_vertex: Mapping[VertexId, A1ProjectedPoint],
    visible_face_indices: tuple[int, ...],
) -> tuple[
    dict[tuple[int, int], _Sample],
    tuple[tuple[tuple[int, int], tuple[int, int], tuple[int, int]], ...],
]:
    faces_by_id = triangulated.face_by_id()
    loops_by_id = triangulated.loop_by_id()
    owner_by_vertex: dict[VertexId, int] = {}
    direct_faces: list[
        tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    ] = []
    for face_index in visible_face_indices:
        face = faces_by_id[FaceId(face_index)]
        vertex_ids = tuple(
            loops_by_id[loop_id].vertex_id for loop_id in face.loop_ids
        )
        if len(vertex_ids) != 3:
            raise DepthCameraProjectionError(
                f"visible source face {face_index} is not triangulated"
            )
        for vertex_id in vertex_ids:
            owner_by_vertex.setdefault(vertex_id, face_index)
        direct_faces.append(
            tuple(
                (vertex_id.index, 0) for vertex_id in vertex_ids
            )  # type: ignore[arg-type]
        )

    samples = {
        (vertex_id.index, 0): _Sample(
            x=float(projected_by_vertex[vertex_id].u),
            y=float(projected_by_vertex[vertex_id].v),
            depth=float(projected_by_vertex[vertex_id].depth),
            source_face_index=owner,
        )
        for vertex_id, owner in sorted(
            owner_by_vertex.items(),
            key=lambda item: item[0].index,
        )
    }
    if len(samples) < 3 or not direct_faces:
        raise _SourceTopologyUnavailable(
            "front-most source topology retained no two-dimensional surface"
        )
    return samples, tuple(direct_faces)


def _build_source_topology_surface(
    snapshot: MeshSnapshot,
    frame: A1CameraProjectionFrame,
    *,
    uniform_scale: float,
    uv_layer_name: str,
    settings: DepthCameraProjectionSettings,
) -> DepthCameraProjectionResult:
    if len(snapshot.vertices) > _DIRECT_VERTEX_LIMIT:
        raise _SourceTopologyUnavailable(
            f"source has more than {_DIRECT_VERTEX_LIMIT} vertices"
        )
    origin = _translation_only_origin(snapshot.world_matrix)
    projected_origin = frame.project_world_point(origin, field_name="object_origin")
    triangulated, projected_by_vertex, triangles = _projected_triangles(
        snapshot,
        frame,
        origin,
    )
    if len(triangles) > _DIRECT_TRIANGLE_LIMIT:
        raise _SourceTopologyUnavailable(
            f"source has more than {_DIRECT_TRIANGLE_LIMIT} projected triangles"
        )
    if not _inside_frame(projected_by_vertex, frame):
        raise _SourceTopologyUnavailable(
            "projected source crosses the camera frame and requires clipping"
        )

    visible_face_indices = _visible_face_indices(triangles, kind=frame.kind)
    samples, faces = _direct_samples_and_faces(
        triangulated,
        projected_by_vertex,
        visible_face_indices,
    )
    if len(samples) > settings.max_points:
        raise _SourceTopologyUnavailable(
            "visible source topology exceeds the selected Max Depth Points"
        )

    raw_depths = tuple(sample.depth for sample in samples.values())
    raw_farthest = min(raw_depths)
    raw_nearest = max(raw_depths)
    edge_threshold = max(
        1.0e-8,
        (raw_nearest - raw_farthest) * settings.edge_threshold_fraction,
    )
    smoothed = _smooth_samples(
        samples,
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
        source_triangle_count=len(triangles),
        sampled_point_count=len(surface.vertices),
    )


def _validate_result_budget(
    result: DepthCameraProjectionResult,
    *,
    source_vertex_count: int,
    user_max_points: int,
) -> DepthCameraProjectionResult:
    generated = len(result.snapshot.vertices)
    allowed = min(source_vertex_count, user_max_points)
    if generated > allowed:
        raise DepthCameraProjectionError(
            "Depth Camera Projection violated its source-bounded vertex contract: "
            f"generated={generated}, source={source_vertex_count}, "
            f"max_points={user_max_points}"
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
    """Build a visible relief with no more vertices than its evaluated source mesh."""

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
    source_vertex_count = len(snapshot.vertices)
    if source_vertex_count < 3:
        raise DepthCameraProjectionError(
            "depth projection requires at least three evaluated source vertices"
        )
    effective_max_points = min(settings.max_points, source_vertex_count)

    try:
        direct = _build_source_topology_surface(
            snapshot,
            frame,
            uniform_scale=resolved_scale,
            uv_layer_name=uv_layer_name,
            settings=settings,
        )
    except _SourceTopologyUnavailable as exc:
        logger.debug(
            "Depth source-topology path unavailable for '%s': %s; using "
            "source-bounded lattice fallback",
            snapshot.source_object_id,
            exc,
        )
    else:
        return _validate_result_budget(
            direct,
            source_vertex_count=source_vertex_count,
            user_max_points=settings.max_points,
        )

    if effective_max_points < 4:
        raise DepthCameraProjectionError(
            "camera-clipped depth projection needs at least four source vertices; "
            f"evaluated source has {source_vertex_count}"
        )
    fallback = _build_lattice_surface(
        snapshot,
        frame,
        uniform_scale=resolved_scale,
        uv_layer_name=uv_layer_name,
        settings=replace(settings, max_points=effective_max_points),
    )
    return _validate_result_budget(
        fallback,
        source_vertex_count=source_vertex_count,
        user_max_points=settings.max_points,
    )


__all__ = ["build_depth_camera_projection_surface"]
