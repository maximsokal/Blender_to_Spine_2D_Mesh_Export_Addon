"""Budgeted component-envelope fallback for sparse camera-depth geometry.

Dense Blender objects can consist of many tiny disconnected islands. A regular screen
lattice may miss those islands entirely even though their projected bounds are valid.
This module preserves their complete screen footprint without deleting source geometry:
connected source components are represented by independent projected envelope quads.
When the component count exceeds the user point budget, deterministic spatial partitioning
merges nearby components into a bounded number of envelope clusters.

The fallback is intentionally narrow. It is used only after the exact source-topology and
regular lattice routes report sparse/empty sampling. It never imports Blender APIs and it
never exceeds ``DepthCameraProjectionSettings.max_points`` or the evaluated source vertex
count.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

from ..camera_projection import A1CameraProjectionFrame
from ..projection import A1ProjectedPoint
from .depth_camera_projection import (
    DepthCameraProjectionError,
    DepthCameraProjectionResult,
    DepthCameraProjectionSettings,
    DepthProjectionBaseMode,
    _ProjectedTriangle,
    _Sample,
    _dense_surface_snapshot,
    _smooth_samples,
    _translation_only_origin,
    _triangulated_face_adjacency,
)
from .depth_camera_projection_bounded import _projected_triangles
from .ids import FaceId, VertexId
from .model import MeshSnapshot
from .validator import MeshSnapshotValidator


_RECOVERABLE_LATTICE_MESSAGES = (
    "depth lattice did not intersect at least three visible points",
    "Depth Edge Threshold disconnected every sampled triangle",
    "depth sampling produced fewer than three connected surface points",
    "depth sampling produced no valid triangles",
)


class _ComponentEnvelopeUnavailable(RuntimeError):
    """Internal signal that the component-envelope route cannot form a safe surface."""


@dataclass(frozen=True, slots=True)
class _ProjectedComponent:
    face_indices: tuple[int, ...]
    vertex_ids: tuple[VertexId, ...]
    minimum_x: float
    maximum_x: float
    minimum_y: float
    maximum_y: float
    centroid_x: float
    centroid_y: float

    def __post_init__(self) -> None:
        if not isinstance(self.face_indices, tuple) or not self.face_indices:
            raise ValueError("face_indices must be a non-empty tuple")
        if not isinstance(self.vertex_ids, tuple) or len(self.vertex_ids) < 3:
            raise ValueError("vertex_ids must contain at least three vertices")
        numeric_fields = (
            "minimum_x",
            "maximum_x",
            "minimum_y",
            "maximum_y",
            "centroid_x",
            "centroid_y",
        )
        for field_name in numeric_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be numeric")
            if not isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        if self.maximum_x <= self.minimum_x:
            raise ValueError("component width must be positive")
        if self.maximum_y <= self.minimum_y:
            raise ValueError("component height must be positive")


@dataclass(frozen=True, slots=True)
class _ComponentCluster:
    components: tuple[_ProjectedComponent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.components, tuple) or not self.components:
            raise ValueError("components must be a non-empty tuple")

    @property
    def minimum_x(self) -> float:
        return min(component.minimum_x for component in self.components)

    @property
    def maximum_x(self) -> float:
        return max(component.maximum_x for component in self.components)

    @property
    def minimum_y(self) -> float:
        return min(component.minimum_y for component in self.components)

    @property
    def maximum_y(self) -> float:
        return max(component.maximum_y for component in self.components)

    @property
    def width(self) -> float:
        return self.maximum_x - self.minimum_x

    @property
    def height(self) -> float:
        return self.maximum_y - self.minimum_y

    @property
    def face_indices(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    face_index
                    for component in self.components
                    for face_index in component.face_indices
                }
            )
        )

    @property
    def vertex_ids(self) -> tuple[VertexId, ...]:
        return tuple(
            sorted(
                {
                    vertex_id
                    for component in self.components
                    for vertex_id in component.vertex_ids
                },
                key=lambda item: item.index,
            )
        )


def is_sparse_lattice_failure(error: BaseException) -> bool:
    """Return whether a bounded lattice failed only because samples were too sparse."""

    if not isinstance(error, DepthCameraProjectionError):
        return False
    message = str(error)
    return any(token in message for token in _RECOVERABLE_LATTICE_MESSAGES)


def _connected_face_components(
    triangulated: MeshSnapshot,
    triangles: Sequence[_ProjectedTriangle],
) -> tuple[tuple[int, ...], ...]:
    """Return deterministic shared-edge components for non-degenerate projected faces."""

    triangle_faces = frozenset(triangle.face_index for triangle in triangles)
    if not triangle_faces:
        raise _ComponentEnvelopeUnavailable(
            "projected source contains no non-degenerate triangles"
        )

    adjacency = _triangulated_face_adjacency(triangulated)
    remaining = set(triangle_faces)
    components: list[tuple[int, ...]] = []

    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        pending = [seed]
        accepted = [seed]

        while pending:
            current = pending.pop()
            for neighbour in sorted(adjacency.get(current, frozenset())):
                if neighbour not in remaining or neighbour not in triangle_faces:
                    continue
                remaining.remove(neighbour)
                pending.append(neighbour)
                accepted.append(neighbour)

        components.append(tuple(sorted(accepted)))

    return tuple(
        sorted(
            components,
            key=lambda component: (component[0], len(component)),
        )
    )


def _projected_components(
    triangulated: MeshSnapshot,
    projected_by_vertex: Mapping[VertexId, A1ProjectedPoint],
    triangles: Sequence[_ProjectedTriangle],
) -> tuple[_ProjectedComponent, ...]:
    """Materialize component bounds and exact source vertex ownership."""

    faces_by_id = triangulated.face_by_id()
    loops_by_id = triangulated.loop_by_id()
    result: list[_ProjectedComponent] = []

    for face_indices in _connected_face_components(triangulated, triangles):
        vertex_ids = tuple(
            sorted(
                {
                    loops_by_id[loop_id].vertex_id
                    for face_index in face_indices
                    for loop_id in faces_by_id[FaceId(face_index)].loop_ids
                },
                key=lambda item: item.index,
            )
        )
        if len(vertex_ids) < 3:
            raise _ComponentEnvelopeUnavailable(
                "projected component contains fewer than three unique vertices"
            )

        projected_points = tuple(projected_by_vertex[vertex_id] for vertex_id in vertex_ids)
        minimum_x = min(float(point.u) for point in projected_points)
        maximum_x = max(float(point.u) for point in projected_points)
        minimum_y = min(float(point.v) for point in projected_points)
        maximum_y = max(float(point.v) for point in projected_points)
        width = maximum_x - minimum_x
        height = maximum_y - minimum_y
        extent_tolerance = max(abs(width), abs(height), 1.0) * 1.0e-12
        if width <= extent_tolerance or height <= extent_tolerance:
            raise _ComponentEnvelopeUnavailable(
                "projected component does not contain a two-dimensional envelope; "
                f"faces={face_indices}, width={width}, height={height}"
            )

        result.append(
            _ProjectedComponent(
                face_indices=face_indices,
                vertex_ids=vertex_ids,
                minimum_x=minimum_x,
                maximum_x=maximum_x,
                minimum_y=minimum_y,
                maximum_y=maximum_y,
                centroid_x=(minimum_x + maximum_x) * 0.5,
                centroid_y=(minimum_y + maximum_y) * 0.5,
            )
        )

    if not result:
        raise _ComponentEnvelopeUnavailable(
            "projected source contains no usable connected components"
        )
    return tuple(result)


def _cluster_sort_key(cluster: _ComponentCluster) -> tuple[float, int, int]:
    """Prioritize the spatially largest splittable cluster deterministically."""

    area = max(cluster.width, 0.0) * max(cluster.height, 0.0)
    first_face = min(component.face_indices[0] for component in cluster.components)
    return area, len(cluster.components), -first_face


def _split_cluster(cluster: _ComponentCluster) -> tuple[_ComponentCluster, _ComponentCluster]:
    """Split one cluster at its deterministic spatial median."""

    if len(cluster.components) < 2:
        raise _ComponentEnvelopeUnavailable(
            "cannot split a cluster containing fewer than two components"
        )

    use_x = cluster.width >= cluster.height
    ordered = tuple(
        sorted(
            cluster.components,
            key=lambda component: (
                component.centroid_x if use_x else component.centroid_y,
                component.centroid_y if use_x else component.centroid_x,
                component.face_indices[0],
            ),
        )
    )
    midpoint = len(ordered) // 2
    if midpoint <= 0 or midpoint >= len(ordered):
        raise _ComponentEnvelopeUnavailable(
            "component spatial partition produced an empty cluster"
        )
    return (
        _ComponentCluster(ordered[:midpoint]),
        _ComponentCluster(ordered[midpoint:]),
    )


def _partition_components(
    components: tuple[_ProjectedComponent, ...],
    *,
    cluster_budget: int,
) -> tuple[_ComponentCluster, ...]:
    """Partition all components into at most ``cluster_budget`` envelope groups."""

    if isinstance(cluster_budget, bool) or not isinstance(cluster_budget, int):
        raise TypeError("cluster_budget must be int")
    if cluster_budget < 1:
        raise ValueError("cluster_budget must be positive")
    if not components:
        raise _ComponentEnvelopeUnavailable("components cannot be empty")

    if len(components) <= cluster_budget:
        return tuple(_ComponentCluster((component,)) for component in components)

    clusters: list[_ComponentCluster] = [_ComponentCluster(components)]
    while len(clusters) < cluster_budget:
        splittable = tuple(
            (index, cluster)
            for index, cluster in enumerate(clusters)
            if len(cluster.components) > 1
        )
        if not splittable:
            break
        selected_index, selected = max(
            splittable,
            key=lambda item: (*_cluster_sort_key(item[1]), -item[0]),
        )
        first, second = _split_cluster(selected)
        clusters[selected_index : selected_index + 1] = [first, second]

    return tuple(
        sorted(
            clusters,
            key=lambda cluster: (
                cluster.minimum_y,
                cluster.minimum_x,
                cluster.face_indices[0],
            ),
        )
    )


def _nearest_projected_vertex_depth(
    x: float,
    y: float,
    vertex_ids: Sequence[VertexId],
    projected_by_vertex: Mapping[VertexId, A1ProjectedPoint],
) -> float:
    """Return exact source depth from the nearest projected component vertex."""

    if not vertex_ids:
        raise _ComponentEnvelopeUnavailable(
            "cannot resolve envelope depth without source vertices"
        )

    best: tuple[float, float, int, float] | None = None
    for vertex_id in vertex_ids:
        projected = projected_by_vertex[vertex_id]
        delta_x = float(projected.u) - float(x)
        delta_y = float(projected.v) - float(y)
        distance_squared = delta_x * delta_x + delta_y * delta_y
        depth = float(projected.depth)
        if not isfinite(distance_squared) or not isfinite(depth):
            raise _ComponentEnvelopeUnavailable(
                "component envelope depth lookup became non-finite"
            )
        candidate = (distance_squared, -depth, vertex_id.index, depth)
        if best is None or candidate[:3] < best[:3]:
            best = candidate

    if best is None:
        raise _ComponentEnvelopeUnavailable(
            "component envelope depth lookup found no candidate"
        )
    return float(best[3])


def _samples_and_faces(
    clusters: tuple[_ComponentCluster, ...],
    projected_by_vertex: Mapping[VertexId, A1ProjectedPoint],
) -> tuple[
    dict[tuple[int, int], _Sample],
    tuple[tuple[tuple[int, int], tuple[int, int], tuple[int, int]], ...],
    float,
    float,
]:
    """Create one disconnected screen-space quad for every envelope cluster."""

    samples: dict[tuple[int, int], _Sample] = {}
    faces: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    widths: list[float] = []
    heights: list[float] = []

    for cluster_index, cluster in enumerate(clusters):
        width = cluster.width
        height = cluster.height
        if not isfinite(width) or not isfinite(height) or width <= 0.0 or height <= 0.0:
            raise _ComponentEnvelopeUnavailable(
                "component cluster has invalid projected bounds; "
                f"cluster={cluster_index}, width={width}, height={height}"
            )
        widths.append(width)
        heights.append(height)

        corners = (
            (cluster.minimum_x, cluster.minimum_y),
            (cluster.maximum_x, cluster.minimum_y),
            (cluster.maximum_x, cluster.maximum_y),
            (cluster.minimum_x, cluster.maximum_y),
        )
        keys = tuple((cluster_index * 4 + corner_index, 0) for corner_index in range(4))
        owner = cluster.face_indices[0]
        for key, (x, y) in zip(keys, corners, strict=True):
            samples[key] = _Sample(
                x=float(x),
                y=float(y),
                depth=_nearest_projected_vertex_depth(
                    x,
                    y,
                    cluster.vertex_ids,
                    projected_by_vertex,
                ),
                source_face_index=owner,
            )
        faces.extend(
            (
                (keys[0], keys[1], keys[2]),
                (keys[0], keys[2], keys[3]),
            )
        )

    if len(samples) < 4 or not faces:
        raise _ComponentEnvelopeUnavailable(
            "component envelopes produced no two-dimensional surface"
        )
    return samples, tuple(faces), max(widths), max(heights)


def _resolve_base_depth(
    samples: Mapping[tuple[int, int], _Sample],
    projected_origin: A1ProjectedPoint,
    settings: DepthCameraProjectionSettings,
) -> tuple[float, float, float]:
    """Resolve the same depth-base contract as the regular lattice route."""

    depths = tuple(float(sample.depth) for sample in samples.values())
    if not depths or any(not isfinite(depth) for depth in depths):
        raise _ComponentEnvelopeUnavailable(
            "component envelope contains no finite depth samples"
        )
    farthest = min(depths)
    nearest = max(depths)

    if settings.base_mode is DepthProjectionBaseMode.FARTHEST_VISIBLE:
        base_depth = farthest
    elif settings.base_mode is DepthProjectionBaseMode.OBJECT_ORIGIN:
        base_depth = float(projected_origin.depth)
        tolerance = max(1.0e-8, abs(base_depth) * 1.0e-8)
        if any(depth < base_depth - tolerance for depth in depths):
            raise DepthCameraProjectionError(
                "OBJECT_ORIGIN depth base lies in front of visible surface points; "
                "use FARTHEST_VISIBLE or move Object Origin behind the visible surface"
            )
    else:
        raise AssertionError(f"Unhandled depth base mode: {settings.base_mode}")

    return float(base_depth), float(farthest), float(nearest)


def _build_component_envelope_surface(
    snapshot: MeshSnapshot,
    frame: A1CameraProjectionFrame,
    *,
    uniform_scale: float,
    uv_layer_name: str,
    settings: DepthCameraProjectionSettings,
) -> DepthCameraProjectionResult:
    """Build a source-bounded disconnected envelope surface after sparse lattice failure."""

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    if not isinstance(frame, A1CameraProjectionFrame):
        raise TypeError("frame must be A1CameraProjectionFrame")
    if isinstance(uniform_scale, bool) or not isinstance(uniform_scale, (int, float)):
        raise TypeError("uniform_scale must be numeric")
    resolved_scale = float(uniform_scale)
    if not isfinite(resolved_scale) or resolved_scale <= 0.0:
        raise ValueError("uniform_scale must be finite and positive")
    if not isinstance(uv_layer_name, str) or not uv_layer_name.strip():
        raise ValueError("uv_layer_name must be a non-empty string")
    if not isinstance(settings, DepthCameraProjectionSettings):
        raise TypeError("settings must be DepthCameraProjectionSettings")

    MeshSnapshotValidator().validate_or_raise(snapshot)
    source_vertex_count = len(snapshot.vertices)
    effective_point_budget = min(source_vertex_count, settings.max_points)
    cluster_budget = effective_point_budget // 4
    if cluster_budget < 1:
        raise _ComponentEnvelopeUnavailable(
            "component envelopes require at least four available points"
        )

    origin = _translation_only_origin(snapshot.world_matrix)
    projected_origin = frame.project_world_point(origin, field_name="object_origin")
    triangulated, projected_by_vertex, triangles = _projected_triangles(
        snapshot,
        frame,
        origin,
    )
    components = _projected_components(
        triangulated,
        projected_by_vertex,
        triangles,
    )
    clusters = _partition_components(
        components,
        cluster_budget=cluster_budget,
    )
    samples, faces, resolved_spacing_x, resolved_spacing_y = _samples_and_faces(
        clusters,
        projected_by_vertex,
    )
    if len(samples) > effective_point_budget:
        raise DepthCameraProjectionError(
            "component envelope violated the Depth point budget; "
            f"generated={len(samples)}, available={effective_point_budget}"
        )

    raw_depths = tuple(sample.depth for sample in samples.values())
    depth_span = max(raw_depths) - min(raw_depths)
    edge_threshold = max(
        1.0e-8,
        depth_span * settings.edge_threshold_fraction,
    )
    smoothed = _smooth_samples(
        samples,
        faces,
        strength=settings.smoothing,
        edge_threshold=edge_threshold,
    )
    base_depth, farthest, nearest = _resolve_base_depth(
        smoothed,
        projected_origin,
        settings,
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
    maximum_relief = max(
        float(vertex.position[2]) - base_depth
        for vertex in surface.vertices
    )
    if maximum_relief < -1.0e-8:
        raise DepthCameraProjectionError(
            "component envelope relief extends away from the selected base plane"
        )

    return DepthCameraProjectionResult(
        snapshot=surface,
        frame=frame,
        projected_origin=projected_origin,
        base_mode=settings.base_mode,
        base_depth=base_depth,
        farthest_visible_depth=farthest,
        nearest_visible_depth=nearest,
        maximum_relief=max(0.0, float(maximum_relief)),
        requested_spacing_pixels=settings.mesh_error_pixels,
        resolved_spacing_x_pixels=max(float(resolved_spacing_x), 1.0e-12),
        resolved_spacing_y_pixels=max(float(resolved_spacing_y), 1.0e-12),
        source_triangle_count=len(triangles),
        sampled_point_count=len(surface.vertices),
    )


__all__ = [
    "_ComponentEnvelopeUnavailable",
    "_build_component_envelope_surface",
    "is_sparse_lattice_failure",
]
