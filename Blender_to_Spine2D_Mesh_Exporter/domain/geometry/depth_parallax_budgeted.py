"""Budget-aware parallax owner for dense Blender geometry.

The legacy parallax builder is exact when the complete front plus reserve topology fits
``Max Depth Points``. Dense assets may contain hundreds of tiny reserve faces while the
front surface already consumes the point budget. Failing after materializing that complete
union is both slow and unhelpful.

This owner keeps the exact path when it fits. Otherwise each active reserve view receives
a deterministic screen-space proxy built from existing front vertices. The proxy adds no
new union points, while ``DepthParallaxReserveSurface.source_face_indices`` retains every
original evaluated polygon that must be isolated for the reserve texture render.

Front-most face discovery uses a projected screen grid instead of comparing every probe
against every triangle. The result is exact with respect to the same barycentric depth
predicate, but candidate lookup is local for dense meshes.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
import logging
from math import atan2, floor, isfinite, sqrt
from time import perf_counter
from typing import Mapping, Sequence

from ..camera_projection import A1CameraProjectionFrame, A1CameraProjectionKind
from .depth_camera_projection import (
    DepthCameraProjectionError,
    DepthCameraProjectionResult,
    _ProjectedTriangle,
)
from .depth_camera_projection_bounded import (
    _depth_at_screen_point,
    _projected_triangles,
)
from .depth_camera_projection_visible_topology import (
    _clip_triangle_to_frame,
    _polygon_probe_points,
    _triangle_vertex_ids,
)
from .depth_parallax import (
    DepthParallaxCameraView,
    DepthParallaxGeometryPackage,
    DepthParallaxReserveSurface,
    DepthParallaxViewId,
    _FaceGeometry,
    _FaceRecord,
    _VIEW_ORDER,
    _accumulated_horizon_costs,
    _dihedral_angle,
    _face_adjacency,
    _face_geometry,
    _front_records,
    _reserve_record,
    _snapshot_from_records,
    _subset_material,
    _translation_only_origin,
    _view_for_face,
)
from .ids import SourceVertexId
from .model import MeshSnapshot, MeshVertex
from .validator import MeshSnapshotValidator


logger = logging.getLogger(__name__)

_GRID_TARGET_OCCUPANCY = 12
_GRID_MAX_AXIS = 96
_PROXY_AREA_EPSILON = 1.0e-14


@dataclass(frozen=True, slots=True)
class _ScreenGrid:
    """Deterministic projected-triangle candidate index."""

    minimum_x: float
    minimum_y: float
    cell_width: float
    cell_height: float
    columns: int
    rows: int
    triangle_by_face: Mapping[int, _ProjectedTriangle]
    buckets: Mapping[tuple[int, int], tuple[int, ...]]

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        column = int(floor((x - self.minimum_x) / self.cell_width))
        row = int(floor((y - self.minimum_y) / self.cell_height))
        return (
            min(self.columns - 1, max(0, column)),
            min(self.rows - 1, max(0, row)),
        )

    def candidates(self, x: float, y: float) -> tuple[_ProjectedTriangle, ...]:
        column, row = self._cell(x, y)
        face_indices: set[int] = set()
        for candidate_row in range(max(0, row - 1), min(self.rows, row + 2)):
            for candidate_column in range(
                max(0, column - 1),
                min(self.columns, column + 2),
            ):
                face_indices.update(
                    self.buckets.get((candidate_column, candidate_row), ())
                )
        if not face_indices:
            face_indices.update(self.triangle_by_face)
        return tuple(
            self.triangle_by_face[index]
            for index in sorted(face_indices)
        )


def _build_screen_grid(
    triangles: Sequence[_ProjectedTriangle],
    polygons: Mapping[int, Sequence[object]],
    frame: A1CameraProjectionFrame,
) -> _ScreenGrid:
    if not triangles:
        raise DepthCameraProjectionError("parallax visibility grid needs triangles")

    triangle_by_face = {triangle.face_index: triangle for triangle in triangles}
    count = max(1, len(polygons))
    axis = max(1, min(_GRID_MAX_AXIS, int(sqrt(count / _GRID_TARGET_OCCUPANCY)) + 1))
    aspect = float(frame.texture_width) / float(frame.texture_height)
    if aspect >= 1.0:
        columns = max(1, min(_GRID_MAX_AXIS, int(round(axis * sqrt(aspect)))))
        rows = max(1, min(_GRID_MAX_AXIS, int(round(axis / sqrt(aspect)))))
    else:
        columns = max(1, min(_GRID_MAX_AXIS, int(round(axis * sqrt(aspect)))))
        rows = max(1, min(_GRID_MAX_AXIS, int(round(axis / sqrt(aspect)))))

    minimum_x = -float(frame.texture_width) / 2.0
    maximum_x = float(frame.texture_width) / 2.0
    minimum_y = -float(frame.texture_height) / 2.0
    maximum_y = float(frame.texture_height) / 2.0
    cell_width = max((maximum_x - minimum_x) / float(columns), 1.0e-12)
    cell_height = max((maximum_y - minimum_y) / float(rows), 1.0e-12)

    pending: dict[tuple[int, int], set[int]] = {}
    for face_index, polygon in polygons.items():
        points = tuple(polygon)
        if not points or face_index not in triangle_by_face:
            continue
        polygon_min_x = min(float(getattr(point, "x")) for point in points)
        polygon_max_x = max(float(getattr(point, "x")) for point in points)
        polygon_min_y = min(float(getattr(point, "y")) for point in points)
        polygon_max_y = max(float(getattr(point, "y")) for point in points)
        first_column = min(
            columns - 1,
            max(0, int(floor((polygon_min_x - minimum_x) / cell_width))),
        )
        last_column = min(
            columns - 1,
            max(0, int(floor((polygon_max_x - minimum_x) / cell_width))),
        )
        first_row = min(
            rows - 1,
            max(0, int(floor((polygon_min_y - minimum_y) / cell_height))),
        )
        last_row = min(
            rows - 1,
            max(0, int(floor((polygon_max_y - minimum_y) / cell_height))),
        )
        for row in range(first_row, last_row + 1):
            for column in range(first_column, last_column + 1):
                pending.setdefault((column, row), set()).add(face_index)

    buckets = {
        key: tuple(sorted(values))
        for key, values in sorted(pending.items())
    }
    return _ScreenGrid(
        minimum_x=minimum_x,
        minimum_y=minimum_y,
        cell_width=cell_width,
        cell_height=cell_height,
        columns=columns,
        rows=rows,
        triangle_by_face=triangle_by_face,
        buckets=buckets,
    )


def _front_visible_face_indices_fast(
    snapshot: MeshSnapshot,
    frame: A1CameraProjectionFrame,
) -> tuple[int, ...]:
    """Resolve exact probe visibility with local projected candidates."""

    origin = _translation_only_origin(snapshot.world_matrix)
    triangulated, _projected, triangles = _projected_triangles(
        snapshot,
        frame,
        origin,
    )
    vertex_ids = _triangle_vertex_ids(triangulated)
    polygons: dict[int, tuple[object, ...]] = {}
    for triangle in triangles:
        polygon, _clipped = _clip_triangle_to_frame(
            triangle,
            vertex_ids[triangle.face_index],
            frame,
        )
        if polygon:
            polygons[triangle.face_index] = tuple(polygon)
    if not polygons:
        raise DepthCameraProjectionError(
            "active camera retains no visible source triangles for parallax expansion"
        )

    extent = max(
        (
            max(
                max(float(getattr(point, "x")) for point in polygon)
                - min(float(getattr(point, "x")) for point in polygon),
                max(float(getattr(point, "y")) for point in polygon)
                - min(float(getattr(point, "y")) for point in polygon),
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
    grid = _build_screen_grid(triangles, polygons, frame)

    visible: list[int] = []
    for face_index in sorted(polygons):
        triangle = triangle_by_face[face_index]
        for x, y in _polygon_probe_points(polygons[face_index]):
            expected_depth = _depth_at_screen_point(
                triangle,
                x,
                y,
                kind=frame.kind,
                epsilon=containment_epsilon,
            )
            if expected_depth is None:
                continue
            front_depth: float | None = None
            for candidate in grid.candidates(x, y):
                depth = _depth_at_screen_point(
                    candidate,
                    x,
                    y,
                    kind=frame.kind,
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
    if not visible:
        raise DepthCameraProjectionError(
            "active camera retains no front-most source triangles for parallax expansion"
        )
    return tuple(sorted(set(visible)))


def _accumulated_horizon_costs_cached(
    geometry: Mapping[int, _FaceGeometry],
    adjacency: Mapping[int, tuple[int, ...]],
    seeds: Sequence[int],
    limit: float,
) -> Mapping[int, float]:
    """Dijkstra expansion with one cached dihedral value per shared edge."""

    resolved_seeds = tuple(sorted(set(seeds)))
    if not resolved_seeds:
        raise DepthCameraProjectionError(
            "parallax horizon expansion requires at least one visible seed face"
        )
    unknown = tuple(index for index in resolved_seeds if index not in geometry)
    if unknown:
        raise DepthCameraProjectionError(
            f"visible seed faces are absent from triangulated geometry: {unknown}"
        )

    costs: dict[int, float] = {index: 0.0 for index in resolved_seeds}
    pending: list[tuple[float, int]] = []
    for index in resolved_seeds:
        heappush(pending, (0.0, index))
    edge_costs: dict[tuple[int, int], float] = {}
    tolerance = max(1.0e-12, limit * 1.0e-10)

    while pending:
        current_cost, face_index = heappop(pending)
        if current_cost > costs.get(face_index, float("inf")) + tolerance:
            continue
        for neighbor in adjacency.get(face_index, ()):
            edge = (min(face_index, neighbor), max(face_index, neighbor))
            bend = edge_costs.get(edge)
            if bend is None:
                bend = _dihedral_angle(geometry[face_index], geometry[neighbor])
                edge_costs[edge] = bend
            candidate = current_cost + bend
            if candidate > limit + tolerance:
                continue
            previous = costs.get(neighbor)
            if previous is None or candidate + tolerance < previous:
                costs[neighbor] = candidate
                heappush(pending, (candidate, neighbor))
    return costs


def _signed_area_xy(vertices: Sequence[MeshVertex]) -> float:
    return 0.5 * sum(
        float(vertex.position[0]) * float(vertices[(index + 1) % len(vertices)].position[1])
        - float(vertices[(index + 1) % len(vertices)].position[0])
        * float(vertex.position[1])
        for index, vertex in enumerate(vertices)
    )


def _proxy_vertices(
    front: MeshSnapshot,
    targets: Sequence[tuple[float, float]],
) -> tuple[MeshVertex, ...]:
    """Choose distinct existing front vertices nearest the requested envelope corners."""

    available = tuple(front.vertices)
    if len(available) < 3:
        raise DepthCameraProjectionError(
            "parallax proxy requires at least three front vertices"
        )

    selected: list[MeshVertex] = []
    used: set[int] = set()
    for target_x, target_y in targets:
        candidates = tuple(
            vertex
            for vertex in available
            if vertex.id.index not in used
        )
        if not candidates:
            break
        vertex = min(
            candidates,
            key=lambda item: (
                (float(item.position[0]) - target_x) ** 2
                + (float(item.position[1]) - target_y) ** 2,
                item.id.index,
            ),
        )
        selected.append(vertex)
        used.add(vertex.id.index)

    if len(selected) < 3:
        center_x = sum(float(vertex.position[0]) for vertex in available) / len(available)
        center_y = sum(float(vertex.position[1]) for vertex in available) / len(available)
        for vertex in sorted(
            available,
            key=lambda item: (
                -(
                    (float(item.position[0]) - center_x) ** 2
                    + (float(item.position[1]) - center_y) ** 2
                ),
                item.id.index,
            ),
        ):
            if vertex.id.index in used:
                continue
            selected.append(vertex)
            used.add(vertex.id.index)
            if len(selected) >= 3:
                break

    center_x = sum(float(vertex.position[0]) for vertex in selected) / len(selected)
    center_y = sum(float(vertex.position[1]) for vertex in selected) / len(selected)
    ordered = sorted(
        selected,
        key=lambda item: (
            atan2(
                float(item.position[1]) - center_y,
                float(item.position[0]) - center_x,
            ),
            item.id.index,
        ),
    )
    area = _signed_area_xy(ordered)
    if abs(area) <= _PROXY_AREA_EPSILON:
        raise DepthCameraProjectionError(
            "parallax proxy front vertices are collinear"
        )
    if area < 0.0:
        ordered.reverse()
    return tuple(ordered)


def _proxy_records_for_view(
    front: MeshSnapshot,
    faces: Sequence[_FaceGeometry],
    view: DepthParallaxCameraView,
    front_frame: A1CameraProjectionFrame,
    projected_origin: object,
    uniform_scale: float,
) -> tuple[_FaceRecord, ...]:
    """Build one view-owned proxy using only existing front vertex positions."""

    if not faces:
        return ()
    front_points: list[tuple[float, float, float]] = []
    reserve_uvs: list[tuple[float, float]] = []
    for face in faces:
        for point_index, world_point in enumerate(face.world_points):
            projected_front = front_frame.project_world_point(
                world_point,
                field_name=f"proxy.face[{face.face_index}].front[{point_index}]",
            )
            projected_reserve = view.frame.project_world_point(
                world_point,
                field_name=(
                    f"proxy.face[{face.face_index}].{view.view_id.value}[{point_index}]"
                ),
            )
            front_points.append(
                (
                    (float(projected_front.u) - float(getattr(projected_origin, "u")))
                    / uniform_scale,
                    -(float(projected_front.v) - float(getattr(projected_origin, "v")))
                    / uniform_scale,
                    float(projected_front.depth),
                )
            )
            u = (
                float(projected_reserve.u) + float(view.frame.texture_width) / 2.0
            ) / float(view.frame.texture_width)
            v = 1.0 - (
                float(projected_reserve.v) + float(view.frame.texture_height) / 2.0
            ) / float(view.frame.texture_height)
            reserve_uvs.append(
                (min(1.0, max(0.0, u)), min(1.0, max(0.0, v)))
            )

    minimum_x = min(point[0] for point in front_points)
    maximum_x = max(point[0] for point in front_points)
    minimum_y = min(point[1] for point in front_points)
    maximum_y = max(point[1] for point in front_points)
    if maximum_x - minimum_x <= 1.0e-12 or maximum_y - minimum_y <= 1.0e-12:
        raise DepthCameraProjectionError(
            f"parallax proxy view {view.view_id.value} has collapsed front bounds"
        )
    minimum_u = min(uv[0] for uv in reserve_uvs)
    maximum_u = max(uv[0] for uv in reserve_uvs)
    minimum_v = min(uv[1] for uv in reserve_uvs)
    maximum_v = max(uv[1] for uv in reserve_uvs)

    selected = _proxy_vertices(
        front,
        (
            (minimum_x, minimum_y),
            (maximum_x, minimum_y),
            (maximum_x, maximum_y),
            (minimum_x, maximum_y),
        ),
    )
    width = maximum_x - minimum_x
    height = maximum_y - minimum_y
    vertex_uv: dict[int, tuple[float, float]] = {}
    for vertex in selected:
        factor_x = min(
            1.0,
            max(0.0, (float(vertex.position[0]) - minimum_x) / width),
        )
        factor_y = min(
            1.0,
            max(0.0, (float(vertex.position[1]) - minimum_y) / height),
        )
        vertex_uv[vertex.id.index] = (
            minimum_u + (maximum_u - minimum_u) * factor_x,
            minimum_v + (maximum_v - minimum_v) * factor_y,
        )

    owner = min(face.source_face_index for face in faces)
    triangles = tuple(
        (selected[0], selected[index], selected[index + 1])
        for index in range(1, len(selected) - 1)
    )
    records = tuple(
        _FaceRecord(
            material_index=view.material_index,
            source_face_index=owner,
            source_vertex_ids=(
                triangle[0].source_id,
                triangle[1].source_id,
                triangle[2].source_id,
            ),
            positions=(
                tuple(float(value) for value in triangle[0].position),
                tuple(float(value) for value in triangle[1].position),
                tuple(float(value) for value in triangle[2].position),
            ),
            uvs=(
                vertex_uv[triangle[0].id.index],
                vertex_uv[triangle[1].id.index],
                vertex_uv[triangle[2].id.index],
            ),
        )
        for triangle in triangles
    )
    if not records:
        raise DepthCameraProjectionError(
            f"parallax proxy view {view.view_id.value} produced no triangles"
        )
    return records


def _evaluated_owner_indices(
    geometry: Mapping[int, _FaceGeometry],
    face_indices: Sequence[int],
) -> tuple[int, ...]:
    return tuple(
        sorted({geometry[index].source_face_index for index in face_indices})
    )


def build_depth_parallax_geometry_package(
    source: MeshSnapshot,
    front_result: DepthCameraProjectionResult,
    reserve_views: Sequence[DepthParallaxCameraView],
    *,
    uniform_scale: float,
    uv_layer_name: str,
    horizon_angle_radians: float,
    max_points: int,
) -> DepthParallaxGeometryPackage:
    """Build exact reserve topology when possible and front-shared proxies otherwise."""

    started = perf_counter()
    if not isinstance(source, MeshSnapshot):
        raise TypeError("source must be MeshSnapshot")
    if not isinstance(front_result, DepthCameraProjectionResult):
        raise TypeError("front_result must be DepthCameraProjectionResult")
    if isinstance(uniform_scale, bool) or not isinstance(uniform_scale, (int, float)):
        raise TypeError("uniform_scale must be numeric")
    scale = float(uniform_scale)
    if not isfinite(scale) or scale <= 0.0:
        raise ValueError("uniform_scale must be finite and positive")
    if not isinstance(uv_layer_name, str) or not uv_layer_name.strip():
        raise ValueError("uv_layer_name must be a non-empty string")
    if isinstance(horizon_angle_radians, bool) or not isinstance(
        horizon_angle_radians,
        (int, float),
    ):
        raise TypeError("horizon_angle_radians must be numeric")
    angle = float(horizon_angle_radians)
    if not isfinite(angle) or angle < 0.0 or angle >= 1.5707963267948966:
        raise ValueError("horizon_angle_radians must be finite in [0, pi/2)")
    if isinstance(max_points, bool) or not isinstance(max_points, int):
        raise TypeError("max_points must be int")
    if max_points < 4:
        raise ValueError("max_points must be at least four")

    MeshSnapshotValidator().validate_or_raise(source)
    MeshSnapshotValidator().validate_or_raise(front_result.snapshot)
    if len(front_result.snapshot.vertices) > max_points:
        raise DepthCameraProjectionError(
            "front depth surface already exceeds Max Depth Points; "
            f"front={len(front_result.snapshot.vertices)}, max_points={max_points}"
        )

    visibility_started = perf_counter()
    front_faces = _front_visible_face_indices_fast(source, front_result.frame)
    logger.info(
        "Depth parallax visibility for '%s': source_faces=%d front_faces=%d elapsed=%.3fs",
        source.source_object_id,
        len(source.faces),
        len(front_faces),
        perf_counter() - visibility_started,
    )

    if angle <= 1.0e-12:
        front = front_result.snapshot
        return DepthParallaxGeometryPackage(
            front_result=front_result,
            union_snapshot=front,
            front_snapshot=front,
            reserve_surfaces=(),
            horizon_angle_radians=0.0,
            front_face_indices=front_faces,
            reserve_face_indices=(),
        )

    available: dict[DepthParallaxViewId, DepthParallaxCameraView] = {}
    for view in reserve_views:
        if not isinstance(view, DepthParallaxCameraView):
            raise TypeError("reserve_views must contain DepthParallaxCameraView values")
        if view.view_id in available:
            raise ValueError(f"duplicate reserve view {view.view_id.value}")
        available[view.view_id] = view
    missing = tuple(view_id.value for view_id in _VIEW_ORDER if view_id not in available)
    if missing:
        raise ValueError(
            "positive parallax horizon requires all eight reserve views; "
            f"missing={missing}"
        )

    geometry_started = perf_counter()
    geometry = _face_geometry(source)
    adjacency = _face_adjacency(geometry)
    costs = _accumulated_horizon_costs_cached(
        geometry,
        adjacency,
        front_faces,
        angle,
    )
    reserve_indices = tuple(sorted(set(costs) - set(front_faces)))
    logger.info(
        "Depth parallax horizon for '%s': geometry_faces=%d reserve_faces=%d elapsed=%.3fs",
        source.source_object_id,
        len(geometry),
        len(reserve_indices),
        perf_counter() - geometry_started,
    )
    if not reserve_indices:
        front = front_result.snapshot
        return DepthParallaxGeometryPackage(
            front_result=front_result,
            union_snapshot=front,
            front_snapshot=front,
            reserve_surfaces=(),
            horizon_angle_radians=angle,
            front_face_indices=front_faces,
            reserve_face_indices=(),
        )

    assigned: dict[DepthParallaxViewId, list[int]] = {
        view_id: [] for view_id in _VIEW_ORDER
    }
    for face_index in reserve_indices:
        view = _view_for_face(geometry[face_index], front_result.frame, available)
        assigned[view.view_id].append(face_index)

    front_records = _front_records(front_result.snapshot, uv_layer_name)
    unique_reserve_vertices = {
        source_id
        for face_index in reserve_indices
        for source_id in geometry[face_index].source_vertex_ids
    }
    exact_upper_bound = len(front_result.snapshot.vertices) + len(unique_reserve_vertices)
    projected_origin = front_result.projected_origin
    records = list(front_records)
    compacted = exact_upper_bound > max_points

    if not compacted:
        for view_id in _VIEW_ORDER:
            view = available[view_id]
            for face_index in assigned[view_id]:
                records.append(
                    _reserve_record(
                        geometry[face_index],
                        view,
                        front_result.frame,
                        projected_origin,
                        scale,
                    )
                )
    else:
        for view_id in _VIEW_ORDER:
            face_indices = tuple(sorted(assigned[view_id]))
            if not face_indices:
                continue
            records.extend(
                _proxy_records_for_view(
                    front_result.snapshot,
                    tuple(geometry[index] for index in face_indices),
                    available[view_id],
                    front_result.frame,
                    projected_origin,
                    scale,
                )
            )

    union = _snapshot_from_records(
        front_result.snapshot,
        records,
        uv_layer_name=uv_layer_name,
        snapshot_suffix=(
            "parallax-budget-proxy" if compacted else "parallax-union"
        ),
        preserve_source_vertex_ids=False,
    )
    if len(union.vertices) > max_points:
        raise DepthCameraProjectionError(
            "Budgeted parallax union exceeded Max Depth Points; "
            f"points={len(union.vertices)}, max_points={max_points}, "
            f"mode={'proxy' if compacted else 'exact'}"
        )

    front = _subset_material(
        union,
        0,
        uv_layer_name=uv_layer_name,
        suffix="parallax-front",
    )
    surfaces: list[DepthParallaxReserveSurface] = []
    for view_id in _VIEW_ORDER:
        face_indices = tuple(sorted(assigned[view_id]))
        if not face_indices:
            continue
        surface = _subset_material(
            union,
            available[view_id].material_index,
            uv_layer_name=uv_layer_name,
            suffix=f"parallax-{view_id.value.lower()}",
        )
        surfaces.append(
            DepthParallaxReserveSurface(
                view=available[view_id],
                snapshot=surface,
                source_face_indices=_evaluated_owner_indices(geometry, face_indices),
                maximum_accumulated_angle_radians=max(
                    costs[index] for index in face_indices
                ),
            )
        )

    logger.info(
        "Depth parallax package for '%s': mode=%s front_points=%d exact_upper=%d "
        "union_points=%d reserve_faces=%d views=%d elapsed=%.3fs",
        source.source_object_id,
        "PROXY" if compacted else "EXACT",
        len(front_result.snapshot.vertices),
        exact_upper_bound,
        len(union.vertices),
        len(reserve_indices),
        len(surfaces),
        perf_counter() - started,
    )
    return DepthParallaxGeometryPackage(
        front_result=front_result,
        union_snapshot=union,
        front_snapshot=front,
        reserve_surfaces=tuple(surfaces),
        horizon_angle_radians=angle,
        front_face_indices=front_faces,
        reserve_face_indices=reserve_indices,
    )


__all__ = ["build_depth_parallax_geometry_package"]
