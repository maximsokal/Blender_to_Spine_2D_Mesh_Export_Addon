"""Budget-aware parallax owner for dense Blender geometry.

The exact parallax builder is retained whenever front plus reserve topology fits the
shared ``Max Depth Points`` limit. Dense assets may contain hundreds of tiny reserve
faces while the front relief already consumes most of that budget. In that case every
active virtual view receives an isolated three- or four-point screen-space proxy.

Proxy vertices are generated in a separate topology domain; they never reuse FRONT
vertices or edges. The proxy therefore cannot create coplanar duplicate faces or edges
with more than two owners. ``DepthParallaxReserveSurface.source_face_indices`` retains
all evaluated Blender polygons that must be isolated for the reserve texture render.

Front-most source-face discovery uses a deterministic projected screen grid instead of
comparing every probe against every triangle. Shared-edge dihedral values are cached once
for horizon expansion. The module is Blender-independent and emits timing diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
import logging
from math import floor, isfinite, pi, sqrt
from time import perf_counter
from typing import Mapping, Sequence

from ..camera_projection import A1CameraProjectionFrame
from ..projection import A1ProjectedPoint
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
from .model import MeshSnapshot
from .validator import MeshSnapshotValidator


logger = logging.getLogger(__name__)

_GRID_TARGET_OCCUPANCY = 12
_GRID_MAX_AXIS = 96
_PROXY_MINIMUM_POINTS = 3
_PROXY_MAXIMUM_POINTS = 4
_COORDINATE_EPSILON = 1.0e-12


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

    def __post_init__(self) -> None:
        for field_name in (
            "minimum_x",
            "minimum_y",
            "cell_width",
            "cell_height",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be numeric")
            if not isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        if self.cell_width <= 0.0 or self.cell_height <= 0.0:
            raise ValueError("screen-grid cell dimensions must be positive")
        if self.columns < 1 or self.rows < 1:
            raise ValueError("screen-grid dimensions must be positive")

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        column = int(floor((x - self.minimum_x) / self.cell_width))
        row = int(floor((y - self.minimum_y) / self.cell_height))
        return (
            min(self.columns - 1, max(0, column)),
            min(self.rows - 1, max(0, row)),
        )

    def candidates(self, x: float, y: float) -> tuple[_ProjectedTriangle, ...]:
        column, row = self._cell(float(x), float(y))
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
    """Index each clipped polygon into every screen cell touched by its bounds."""

    if not isinstance(frame, A1CameraProjectionFrame):
        raise TypeError("frame must be A1CameraProjectionFrame")
    if not triangles:
        raise DepthCameraProjectionError("parallax visibility grid needs triangles")

    triangle_by_face = {triangle.face_index: triangle for triangle in triangles}
    count = max(1, len(polygons))
    base_axis = max(
        1,
        min(
            _GRID_MAX_AXIS,
            int(sqrt(count / _GRID_TARGET_OCCUPANCY)) + 1,
        ),
    )
    aspect = float(frame.texture_width) / float(frame.texture_height)
    columns = max(
        1,
        min(_GRID_MAX_AXIS, int(round(base_axis * sqrt(aspect)))),
    )
    rows = max(
        1,
        min(_GRID_MAX_AXIS, int(round(base_axis / sqrt(aspect)))),
    )

    minimum_x = -float(frame.texture_width) / 2.0
    maximum_x = float(frame.texture_width) / 2.0
    minimum_y = -float(frame.texture_height) / 2.0
    maximum_y = float(frame.texture_height) / 2.0
    cell_width = max((maximum_x - minimum_x) / columns, _COORDINATE_EPSILON)
    cell_height = max((maximum_y - minimum_y) / rows, _COORDINATE_EPSILON)

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

    return _ScreenGrid(
        minimum_x=minimum_x,
        minimum_y=minimum_y,
        cell_width=cell_width,
        cell_height=cell_height,
        columns=columns,
        rows=rows,
        triangle_by_face=triangle_by_face,
        buckets={
            key: tuple(sorted(values))
            for key, values in sorted(pending.items())
        },
    )


def _front_visible_face_indices_fast(
    snapshot: MeshSnapshot,
    frame: A1CameraProjectionFrame,
) -> tuple[int, ...]:
    """Resolve exact probe visibility using local projected candidates."""

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
    """Run Dijkstra expansion with one cached dihedral value per shared edge."""

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


def _circular_view_distance(
    first: DepthParallaxViewId,
    second: DepthParallaxViewId,
) -> int:
    distance = abs(first.ordinal - second.ordinal)
    return min(distance, len(_VIEW_ORDER) - distance)


def _merge_view_assignments(
    assigned: Mapping[DepthParallaxViewId, Sequence[int]],
    *,
    maximum_view_count: int,
) -> Mapping[DepthParallaxViewId, tuple[int, ...]]:
    """Merge low-budget directions into the nearest retained virtual views."""

    if isinstance(maximum_view_count, bool) or not isinstance(maximum_view_count, int):
        raise TypeError("maximum_view_count must be int")
    if maximum_view_count < 1:
        raise ValueError("maximum_view_count must be positive")

    active = tuple(
        view_id
        for view_id in _VIEW_ORDER
        if tuple(assigned.get(view_id, ()))
    )
    if not active:
        return {}
    if len(active) <= maximum_view_count:
        return {
            view_id: tuple(sorted(set(assigned[view_id])))
            for view_id in active
        }

    retained = tuple(
        sorted(
            sorted(
                active,
                key=lambda view_id: (
                    -len(tuple(assigned[view_id])),
                    view_id.ordinal,
                ),
            )[:maximum_view_count],
            key=lambda view_id: view_id.ordinal,
        )
    )
    merged: dict[DepthParallaxViewId, set[int]] = {
        view_id: set() for view_id in retained
    }
    for source_view in active:
        target = min(
            retained,
            key=lambda candidate: (
                _circular_view_distance(source_view, candidate),
                candidate.ordinal,
            ),
        )
        merged[target].update(assigned[source_view])
    return {
        view_id: tuple(sorted(values))
        for view_id, values in sorted(
            merged.items(),
            key=lambda item: item[0].ordinal,
        )
        if values
    }


def _nearest_depth(
    points: Sequence[tuple[float, float, float]],
    x: float,
    y: float,
) -> float:
    if not points:
        raise DepthCameraProjectionError("parallax proxy has no depth samples")
    return float(
        min(
            points,
            key=lambda point: (
                (point[0] - x) ** 2 + (point[1] - y) ** 2,
                point[2],
            ),
        )[2]
    )


def _proxy_records_for_view(
    faces: Sequence[_FaceGeometry],
    view: DepthParallaxCameraView,
    front_frame: A1CameraProjectionFrame,
    projected_origin: A1ProjectedPoint,
    uniform_scale: float,
    *,
    point_count: int,
    generated_source_vertex_base: int,
    source_object_id: str,
) -> tuple[_FaceRecord, ...]:
    """Build an isolated envelope proxy with three or four generated vertices."""

    if point_count not in (_PROXY_MINIMUM_POINTS, _PROXY_MAXIMUM_POINTS):
        raise ValueError("point_count must be three or four")
    if generated_source_vertex_base < 0:
        raise ValueError("generated_source_vertex_base must be non-negative")
    if not isinstance(source_object_id, str) or not source_object_id.strip():
        raise ValueError("source_object_id must be a non-empty string")
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
                    (float(projected_front.u) - float(projected_origin.u))
                    / uniform_scale,
                    -(float(projected_front.v) - float(projected_origin.v))
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
            tolerance = 1.0e-7
            if (
                u < -tolerance
                or u > 1.0 + tolerance
                or v < -tolerance
                or v > 1.0 + tolerance
            ):
                raise DepthCameraProjectionError(
                    "virtual parallax camera did not frame budget proxy ownership; "
                    f"view={view.view_id.value}, face={face.face_index}, uv={(u, v)}"
                )
            reserve_uvs.append(
                (min(1.0, max(0.0, u)), min(1.0, max(0.0, v)))
            )

    minimum_x = min(point[0] for point in front_points)
    maximum_x = max(point[0] for point in front_points)
    minimum_y = min(point[1] for point in front_points)
    maximum_y = max(point[1] for point in front_points)
    width = maximum_x - minimum_x
    height = maximum_y - minimum_y
    if width <= _COORDINATE_EPSILON or height <= _COORDINATE_EPSILON:
        raise DepthCameraProjectionError(
            f"parallax proxy view {view.view_id.value} has collapsed front bounds"
        )

    minimum_u = min(uv[0] for uv in reserve_uvs)
    maximum_u = max(uv[0] for uv in reserve_uvs)
    minimum_v = min(uv[1] for uv in reserve_uvs)
    maximum_v = max(uv[1] for uv in reserve_uvs)

    if point_count == 4:
        targets = (
            (minimum_x, minimum_y),
            (maximum_x, minimum_y),
            (maximum_x, maximum_y),
            (minimum_x, maximum_y),
        )
        target_uvs = (
            (minimum_u, minimum_v),
            (maximum_u, minimum_v),
            (maximum_u, maximum_v),
            (minimum_u, maximum_v),
        )
        triangles = ((0, 1, 2), (0, 2, 3))
    else:
        center_x = (minimum_x + maximum_x) / 2.0
        targets = (
            (minimum_x, minimum_y),
            (maximum_x, minimum_y),
            (center_x, maximum_y + height),
        )
        target_uvs = (
            (minimum_u, minimum_v),
            (maximum_u, minimum_v),
            ((minimum_u + maximum_u) / 2.0, maximum_v),
        )
        triangles = ((0, 1, 2),)

    positions = tuple(
        (x, y, _nearest_depth(front_points, x, y))
        for x, y in targets
    )
    source_ids = tuple(
        SourceVertexId(
            source_object_id,
            generated_source_vertex_base + index,
        )
        for index in range(point_count)
    )
    owner = min(face.source_face_index for face in faces)
    return tuple(
        _FaceRecord(
            material_index=view.material_index,
            source_face_index=owner,
            source_vertex_ids=(
                source_ids[triangle[0]],
                source_ids[triangle[1]],
                source_ids[triangle[2]],
            ),
            positions=(
                positions[triangle[0]],
                positions[triangle[1]],
                positions[triangle[2]],
            ),
            uvs=(
                target_uvs[triangle[0]],
                target_uvs[triangle[1]],
                target_uvs[triangle[2]],
            ),
        )
        for triangle in triangles
    )


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
    """Build exact reserve topology or an isolated budgeted proxy package."""

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
    if not isfinite(angle) or angle < 0.0 or angle >= pi / 2.0:
        raise ValueError("horizon_angle_radians must be finite in [0, pi/2)")
    if isinstance(max_points, bool) or not isinstance(max_points, int):
        raise TypeError("max_points must be int")
    if max_points < 4:
        raise ValueError("max_points must be at least four")

    MeshSnapshotValidator().validate_or_raise(source)
    MeshSnapshotValidator().validate_or_raise(front_result.snapshot)
    front_point_count = len(front_result.snapshot.vertices)
    if front_point_count > max_points:
        raise DepthCameraProjectionError(
            "front depth surface already exceeds Max Depth Points; "
            f"front={front_point_count}, max_points={max_points}"
        )

    visibility_started = perf_counter()
    front_faces = _front_visible_face_indices_fast(source, front_result.frame)
    logger.info(
        "Depth parallax visibility for '%s': source_faces=%d front_faces=%d "
        "elapsed=%.3fs",
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
        "Depth parallax horizon for '%s': geometry_faces=%d reserve_faces=%d "
        "elapsed=%.3fs",
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
    exact_upper_bound = front_point_count + len(unique_reserve_vertices)
    compacted = exact_upper_bound > max_points
    projected_origin = front_result.projected_origin
    records = list(front_records)
    resolved_assignments: Mapping[DepthParallaxViewId, tuple[int, ...]]

    if not compacted:
        resolved_assignments = {
            view_id: tuple(sorted(indices))
            for view_id, indices in assigned.items()
            if indices
        }
        for view_id in _VIEW_ORDER:
            for face_index in resolved_assignments.get(view_id, ()):
                records.append(
                    _reserve_record(
                        geometry[face_index],
                        available[view_id],
                        front_result.frame,
                        projected_origin,
                        scale,
                    )
                )
    else:
        reserve_budget = max_points - front_point_count
        maximum_view_count = reserve_budget // _PROXY_MINIMUM_POINTS
        if maximum_view_count < 1:
            raise DepthCameraProjectionError(
                "Positive Parallax Horizon has no reserve point budget after FRONT; "
                f"front_points={front_point_count}, max_points={max_points}. "
                "Lower FRONT quality or increase Max Depth Points."
            )
        resolved_assignments = _merge_view_assignments(
            assigned,
            maximum_view_count=maximum_view_count,
        )
        view_count = len(resolved_assignments)
        if view_count < 1:
            raise DepthCameraProjectionError(
                "parallax reserve proxy retained no virtual view"
            )
        points_per_view = (
            _PROXY_MAXIMUM_POINTS
            if reserve_budget >= _PROXY_MAXIMUM_POINTS * view_count
            else _PROXY_MINIMUM_POINTS
        )
        required_points = points_per_view * view_count
        if required_points > reserve_budget:
            raise DepthCameraProjectionError(
                "parallax reserve proxy cannot fit the remaining point budget; "
                f"required={required_points}, available={reserve_budget}, "
                f"views={view_count}"
            )
        highest_source_index = max(
            (
                vertex.source_id.vertex_index
                for vertex in front_result.snapshot.vertices
            ),
            default=-1,
        )
        generated_base = highest_source_index + 1
        for output_index, view_id in enumerate(
            sorted(resolved_assignments, key=lambda value: value.ordinal)
        ):
            face_indices = resolved_assignments[view_id]
            records.extend(
                _proxy_records_for_view(
                    tuple(geometry[index] for index in face_indices),
                    available[view_id],
                    front_result.frame,
                    projected_origin,
                    scale,
                    point_count=points_per_view,
                    generated_source_vertex_base=(
                        generated_base + output_index * points_per_view
                    ),
                    source_object_id=source.source_object_id,
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
    for view_id in sorted(resolved_assignments, key=lambda value: value.ordinal):
        face_indices = resolved_assignments[view_id]
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
        "union_points=%d reserve_faces=%d source_views=%d output_views=%d "
        "elapsed=%.3fs",
        source.source_object_id,
        "PROXY" if compacted else "EXACT",
        front_point_count,
        exact_upper_bound,
        len(union.vertices),
        len(reserve_indices),
        sum(1 for indices in assigned.values() if indices),
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


__all__ = [
    "_ScreenGrid",
    "_accumulated_horizon_costs_cached",
    "_build_screen_grid",
    "_front_visible_face_indices_fast",
    "_merge_view_assignments",
    "_proxy_records_for_view",
    "build_depth_parallax_geometry_package",
]
