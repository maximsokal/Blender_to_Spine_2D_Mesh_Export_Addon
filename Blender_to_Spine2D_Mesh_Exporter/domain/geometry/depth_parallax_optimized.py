"""Single-pass, budget-aware parallax owner for dense evaluated geometry.

The previous budgeted owner fixed the final point-count failure but still rebuilt and
reprojected the complete source topology several times. Its screen grid also covered the
entire camera frame and queried a 3x3 neighbourhood for every probe. A small object inside
a large frame could therefore place most triangles into only a few occupied cells and
silently approach quadratic visibility work.

This owner performs one source triangulation and one active-camera projection. The same
analysis supplies front-most visibility, local evaluated adjacency, horizon expansion,
and reserve ownership. The visibility grid is fitted to the occupied clipped bounds and
queries the current cell; neighbouring cells are consulted only as a numerical fallback.
Geometric adjacency is built from local ``VertexId`` values, while ``Source*Id`` values
remain lineage only.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from math import floor, isfinite, pi, sqrt
from time import perf_counter
from typing import Mapping, Sequence

from ..camera_projection import (
    A1CameraProjectionFrame,
    A1CameraProjectionKind,
)
from ..projection import A1ProjectedPoint
from .depth_camera_projection import (
    DepthCameraProjectionError,
    DepthCameraProjectionResult,
    _ProjectedTriangle,
    _signed_area_twice,
    _translation_only_origin,
    _world_point,
)
from .depth_camera_projection_visible_topology import (
    _clip_triangle_to_frame,
    _polygon_probe_points,
)
from .depth_parallax import (
    DepthParallaxCameraView,
    DepthParallaxGeometryPackage,
    DepthParallaxReserveSurface,
    DepthParallaxViewId,
    _FaceGeometry,
    _VIEW_ORDER,
    _front_records,
    _reserve_record,
    _snapshot_from_records,
    _subset_material,
    _view_for_face,
)
from .depth_parallax_budgeted import (
    _PROXY_MAXIMUM_POINTS,
    _PROXY_MINIMUM_POINTS,
    _accumulated_horizon_costs_cached,
    _evaluated_owner_indices,
    _merge_view_assignments,
    _proxy_records_for_view,
)
from .ids import SourceVertexId, VertexId
from .model import MeshSnapshot
from .triangulation import triangulate_snapshot
from .validator import MeshSnapshotValidator


logger = logging.getLogger(__name__)

_GRID_TARGET_OCCUPANCY = 8
_GRID_MAX_AXIS = 192
_AREA_EPSILON = 1.0e-12
_COORDINATE_EPSILON = 1.0e-12


@dataclass(frozen=True, slots=True)
class _VisibilityTriangle:
    """Projected triangle with cached containment and depth coefficients."""

    face_index: int
    points: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    denominator: float
    minimum_x: float
    maximum_x: float
    minimum_y: float
    maximum_y: float

    @classmethod
    def from_projected(
        cls,
        triangle: _ProjectedTriangle,
    ) -> "_VisibilityTriangle":
        if not isinstance(triangle, _ProjectedTriangle):
            raise TypeError("triangle must be _ProjectedTriangle")
        denominator = float(_signed_area_twice(triangle.points))
        if not isfinite(denominator) or abs(denominator) <= _AREA_EPSILON:
            raise DepthCameraProjectionError(
                f"projected face {triangle.face_index} is degenerate"
            )
        return cls(
            face_index=triangle.face_index,
            points=(
                triangle.points[0],
                triangle.points[1],
                triangle.points[2],
            ),
            denominator=denominator,
            minimum_x=min(point[0] for point in triangle.points),
            maximum_x=max(point[0] for point in triangle.points),
            minimum_y=min(point[1] for point in triangle.points),
            maximum_y=max(point[1] for point in triangle.points),
        )

    def depth_at(
        self,
        x: float,
        y: float,
        *,
        kind: A1CameraProjectionKind,
        epsilon: float,
    ) -> float | None:
        if (
            x < self.minimum_x - epsilon
            or x > self.maximum_x + epsilon
            or y < self.minimum_y - epsilon
            or y > self.maximum_y + epsilon
        ):
            return None

        first, second, third = self.points
        first_weight = (
            (second[0] - x) * (third[1] - y)
            - (second[1] - y) * (third[0] - x)
        ) / self.denominator
        second_weight = (
            (third[0] - x) * (first[1] - y)
            - (third[1] - y) * (first[0] - x)
        ) / self.denominator
        third_weight = 1.0 - first_weight - second_weight
        if min(first_weight, second_weight, third_weight) < -epsilon:
            return None

        depths = (first[2], second[2], third[2])
        if kind is A1CameraProjectionKind.ORTHOGRAPHIC:
            depth = (
                first_weight * depths[0]
                + second_weight * depths[1]
                + third_weight * depths[2]
            )
        elif kind is A1CameraProjectionKind.PERSPECTIVE:
            reciprocal = (
                first_weight / depths[0]
                + second_weight / depths[1]
                + third_weight / depths[2]
            )
            if abs(reciprocal) <= 1.0e-15:
                raise DepthCameraProjectionError(
                    "perspective visibility interpolation produced zero reciprocal depth"
                )
            depth = 1.0 / reciprocal
        else:
            raise AssertionError(f"Unhandled camera projection kind: {kind}")
        if not isfinite(depth):
            raise DepthCameraProjectionError(
                "optimized parallax visibility depth became non-finite"
            )
        return float(depth)


@dataclass(frozen=True, slots=True)
class _OccupiedScreenGrid:
    """Triangle candidate index fitted to occupied clipped screen bounds."""

    minimum_x: float
    minimum_y: float
    cell_width: float
    cell_height: float
    columns: int
    rows: int
    triangle_by_face: Mapping[int, _VisibilityTriangle]
    buckets: Mapping[tuple[int, int], tuple[int, ...]]

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        column = int(floor((float(x) - self.minimum_x) / self.cell_width))
        row = int(floor((float(y) - self.minimum_y) / self.cell_height))
        return (
            min(self.columns - 1, max(0, column)),
            min(self.rows - 1, max(0, row)),
        )

    def candidates(
        self,
        x: float,
        y: float,
        *,
        expected_face_index: int,
    ) -> tuple[_VisibilityTriangle, ...]:
        column, row = self._cell(x, y)
        face_indices = set(self.buckets.get((column, row), ()))

        # Bounds insertion guarantees that the current cell contains every triangle
        # whose projected AABB contains the probe. Neighbours are only a numerical
        # fallback for values lying exactly on a quantized cell boundary.
        if not face_indices:
            for candidate_row in range(max(0, row - 1), min(self.rows, row + 2)):
                for candidate_column in range(
                    max(0, column - 1),
                    min(self.columns, column + 2),
                ):
                    face_indices.update(
                        self.buckets.get((candidate_column, candidate_row), ())
                    )

        if expected_face_index in self.triangle_by_face:
            face_indices.add(expected_face_index)
        if not face_indices:
            raise DepthCameraProjectionError(
                "optimized visibility grid found no candidates for a source probe"
            )
        return tuple(
            self.triangle_by_face[index]
            for index in sorted(face_indices)
        )


@dataclass(frozen=True, slots=True)
class _ParallaxSourceAnalysis:
    triangulated: MeshSnapshot
    geometry: Mapping[int, _FaceGeometry]
    adjacency: Mapping[int, tuple[int, ...]]
    triangles: tuple[_ProjectedTriangle, ...]
    visibility_triangles: Mapping[int, _VisibilityTriangle]
    clipped_polygons: Mapping[int, tuple[object, ...]]
    grid: _OccupiedScreenGrid


def _normalized(
    vector: tuple[float, float, float],
    *,
    field_name: str,
) -> tuple[float, float, float]:
    length_squared = sum(component * component for component in vector)
    if not isfinite(length_squared) or length_squared <= 1.0e-30:
        raise DepthCameraProjectionError(f"{field_name} collapses")
    inverse = 1.0 / sqrt(length_squared)
    return tuple(float(component * inverse) for component in vector)  # type: ignore[return-value]


def _build_occupied_grid(
    triangles: Mapping[int, _VisibilityTriangle],
    polygons: Mapping[int, tuple[object, ...]],
) -> _OccupiedScreenGrid:
    if not triangles or not polygons:
        raise DepthCameraProjectionError(
            "optimized visibility grid requires projected polygons"
        )

    minimum_x = min(
        float(getattr(point, "x"))
        for polygon in polygons.values()
        for point in polygon
    )
    maximum_x = max(
        float(getattr(point, "x"))
        for polygon in polygons.values()
        for point in polygon
    )
    minimum_y = min(
        float(getattr(point, "y"))
        for polygon in polygons.values()
        for point in polygon
    )
    maximum_y = max(
        float(getattr(point, "y"))
        for polygon in polygons.values()
        for point in polygon
    )
    width = max(maximum_x - minimum_x, _COORDINATE_EPSILON)
    height = max(maximum_y - minimum_y, _COORDINATE_EPSILON)
    count = max(1, len(polygons))
    base_axis = max(
        1,
        min(
            _GRID_MAX_AXIS,
            int(sqrt(count / _GRID_TARGET_OCCUPANCY)) + 1,
        ),
    )
    aspect = width / height
    columns = max(
        1,
        min(_GRID_MAX_AXIS, int(round(base_axis * sqrt(aspect)))),
    )
    rows = max(
        1,
        min(_GRID_MAX_AXIS, int(round(base_axis / sqrt(aspect)))),
    )
    cell_width = max(width / columns, _COORDINATE_EPSILON)
    cell_height = max(height / rows, _COORDINATE_EPSILON)
    boundary_epsilon = max(width, height, 1.0) * 1.0e-12

    pending: dict[tuple[int, int], set[int]] = {}
    for face_index, triangle in triangles.items():
        first_column = min(
            columns - 1,
            max(
                0,
                int(
                    floor(
                        (
                            triangle.minimum_x
                            - boundary_epsilon
                            - minimum_x
                        )
                        / cell_width
                    )
                ),
            ),
        )
        last_column = min(
            columns - 1,
            max(
                0,
                int(
                    floor(
                        (
                            triangle.maximum_x
                            + boundary_epsilon
                            - minimum_x
                        )
                        / cell_width
                    )
                ),
            ),
        )
        first_row = min(
            rows - 1,
            max(
                0,
                int(
                    floor(
                        (
                            triangle.minimum_y
                            - boundary_epsilon
                            - minimum_y
                        )
                        / cell_height
                    )
                ),
            ),
        )
        last_row = min(
            rows - 1,
            max(
                0,
                int(
                    floor(
                        (
                            triangle.maximum_y
                            + boundary_epsilon
                            - minimum_y
                        )
                        / cell_height
                    )
                ),
            ),
        )
        for row in range(first_row, last_row + 1):
            for column in range(first_column, last_column + 1):
                pending.setdefault((column, row), set()).add(face_index)

    return _OccupiedScreenGrid(
        minimum_x=minimum_x,
        minimum_y=minimum_y,
        cell_width=cell_width,
        cell_height=cell_height,
        columns=columns,
        rows=rows,
        triangle_by_face=dict(triangles),
        buckets={
            key: tuple(sorted(values))
            for key, values in sorted(pending.items())
        },
    )


def _build_source_analysis(
    source: MeshSnapshot,
    frame: A1CameraProjectionFrame,
) -> _ParallaxSourceAnalysis:
    """Triangulate, project, and index the complete source exactly once."""

    origin = _translation_only_origin(source.world_matrix)
    triangulated = triangulate_snapshot(source).snapshot
    loops = triangulated.loop_by_id()
    vertices = triangulated.vertex_by_id()
    projected_by_vertex = {
        vertex.id: frame.project_world_point(
            _world_point(origin, vertex.position),
            field_name=f"parallax.vertex[{vertex.id.index}]",
        )
        for vertex in triangulated.vertices
    }

    geometry: dict[int, _FaceGeometry] = {}
    vertex_ids_by_face: dict[int, tuple[VertexId, VertexId, VertexId]] = {}
    triangles: list[_ProjectedTriangle] = []
    visibility_triangles: dict[int, _VisibilityTriangle] = {}
    polygons: dict[int, tuple[object, ...]] = {}

    for face in sorted(triangulated.faces, key=lambda value: value.id.index):
        face_loops = tuple(loops[loop_id] for loop_id in face.loop_ids)
        if len(face_loops) != 3:
            raise DepthCameraProjectionError(
                f"triangulated face {face.id.index} is not triangular"
            )
        face_vertices = tuple(vertices[loop.vertex_id] for loop in face_loops)
        local_vertex_ids = (
            face_vertices[0].id,
            face_vertices[1].id,
            face_vertices[2].id,
        )
        vertex_ids_by_face[face.id.index] = local_vertex_ids
        world_points = tuple(
            _world_point(origin, vertex.position)
            for vertex in face_vertices
        )
        first_edge = tuple(
            world_points[1][axis] - world_points[0][axis]
            for axis in range(3)
        )
        second_edge = tuple(
            world_points[2][axis] - world_points[0][axis]
            for axis in range(3)
        )
        cross = (
            first_edge[1] * second_edge[2] - first_edge[2] * second_edge[1],
            first_edge[2] * second_edge[0] - first_edge[0] * second_edge[2],
            first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0],
        )
        normal = _normalized(
            cross,
            field_name=f"parallax.face[{face.id.index}].normal",
        )
        centroid = tuple(
            sum(point[axis] for point in world_points) / 3.0
            for axis in range(3)
        )
        geometry[face.id.index] = _FaceGeometry(
            face_index=face.id.index,
            source_face_index=face.source_id.face_index,
            source_vertex_ids=(
                face_vertices[0].source_id,
                face_vertices[1].source_id,
                face_vertices[2].source_id,
            ),
            world_points=(world_points[0], world_points[1], world_points[2]),
            normal_world=normal,
            centroid_world=(centroid[0], centroid[1], centroid[2]),
        )

        triangle = _ProjectedTriangle(
            face_index=face.id.index,
            points=tuple(
                (
                    float(projected_by_vertex[vertex_id].u),
                    float(projected_by_vertex[vertex_id].v),
                    float(projected_by_vertex[vertex_id].depth),
                )
                for vertex_id in local_vertex_ids
            ),
        )
        if abs(_signed_area_twice(triangle.points)) <= _AREA_EPSILON:
            continue
        triangles.append(triangle)
        visibility_triangle = _VisibilityTriangle.from_projected(triangle)
        polygon, _clipped = _clip_triangle_to_frame(
            triangle,
            local_vertex_ids,
            frame,
        )
        if polygon:
            visibility_triangles[face.id.index] = visibility_triangle
            polygons[face.id.index] = tuple(polygon)

    if not triangles or not polygons:
        raise DepthCameraProjectionError(
            "active camera retains no visible source triangles for parallax expansion"
        )

    owners_by_edge: dict[tuple[int, int], list[int]] = {}
    for face_index, vertex_ids in vertex_ids_by_face.items():
        for corner_index, first in enumerate(vertex_ids):
            second = vertex_ids[(corner_index + 1) % 3]
            pair = tuple(sorted((first.index, second.index)))
            owners_by_edge.setdefault(pair, []).append(face_index)
    adjacency: dict[int, set[int]] = {
        face_index: set() for face_index in geometry
    }
    for owners in owners_by_edge.values():
        unique = tuple(sorted(set(owners)))
        for owner_index, first in enumerate(unique):
            for second in unique[owner_index + 1 :]:
                adjacency[first].add(second)
                adjacency[second].add(first)

    return _ParallaxSourceAnalysis(
        triangulated=triangulated,
        geometry=geometry,
        adjacency={
            face_index: tuple(sorted(neighbours))
            for face_index, neighbours in sorted(adjacency.items())
        },
        triangles=tuple(triangles),
        visibility_triangles=visibility_triangles,
        clipped_polygons=polygons,
        grid=_build_occupied_grid(visibility_triangles, polygons),
    )


def _front_visible_face_indices(
    analysis: _ParallaxSourceAnalysis,
    frame: A1CameraProjectionFrame,
) -> tuple[int, ...]:
    extent = max(
        (
            max(
                max(float(getattr(point, "x")) for point in polygon)
                - min(float(getattr(point, "x")) for point in polygon),
                max(float(getattr(point, "y")) for point in polygon)
                - min(float(getattr(point, "y")) for point in polygon),
            )
            for polygon in analysis.clipped_polygons.values()
        ),
        default=1.0,
    )
    containment_epsilon = max(1.0e-8, extent * 1.0e-10)
    depth_scale = max(
        (
            abs(point[2])
            for triangle in analysis.triangles
            for point in triangle.points
        ),
        default=1.0,
    )
    depth_tolerance = max(1.0e-8, depth_scale * 1.0e-8)

    visible: list[int] = []
    for face_index in sorted(analysis.clipped_polygons):
        triangle = analysis.visibility_triangles[face_index]
        polygon = analysis.clipped_polygons[face_index]
        for x, y in _polygon_probe_points(polygon):
            expected_depth = triangle.depth_at(
                x,
                y,
                kind=frame.kind,
                epsilon=containment_epsilon,
            )
            if expected_depth is None:
                continue
            front_depth: float | None = None
            for candidate in analysis.grid.candidates(
                x,
                y,
                expected_face_index=face_index,
            ):
                depth = candidate.depth_at(
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
    """Build exact or proxy reserve topology from one shared source analysis."""

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

    analysis_started = perf_counter()
    analysis = _build_source_analysis(source, front_result.frame)
    front_faces = _front_visible_face_indices(analysis, front_result.frame)
    analysis_elapsed = perf_counter() - analysis_started
    logger.info(
        "Optimized depth parallax analysis for '%s': source_faces=%d "
        "projected_faces=%d front_faces=%d grid=%dx%d elapsed=%.3fs",
        source.source_object_id,
        len(analysis.geometry),
        len(analysis.visibility_triangles),
        len(front_faces),
        analysis.grid.columns,
        analysis.grid.rows,
        analysis_elapsed,
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

    horizon_started = perf_counter()
    costs = _accumulated_horizon_costs_cached(
        analysis.geometry,
        analysis.adjacency,
        front_faces,
        angle,
    )
    reserve_indices = tuple(sorted(set(costs) - set(front_faces)))
    logger.info(
        "Optimized depth parallax horizon for '%s': reserve_faces=%d elapsed=%.3fs",
        source.source_object_id,
        len(reserve_indices),
        perf_counter() - horizon_started,
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
        view = _view_for_face(
            analysis.geometry[face_index],
            front_result.frame,
            available,
        )
        assigned[view.view_id].append(face_index)

    front_records = _front_records(front_result.snapshot, uv_layer_name)
    unique_reserve_vertices = {
        source_id
        for face_index in reserve_indices
        for source_id in analysis.geometry[face_index].source_vertex_ids
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
                        analysis.geometry[face_index],
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
                    tuple(
                        analysis.geometry[index]
                        for index in face_indices
                    ),
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
            "Optimized parallax union exceeded Max Depth Points; "
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
                source_face_indices=_evaluated_owner_indices(
                    analysis.geometry,
                    face_indices,
                ),
                maximum_accumulated_angle_radians=max(
                    costs[index] for index in face_indices
                ),
            )
        )

    logger.info(
        "Optimized depth parallax package for '%s': mode=%s front_points=%d "
        "exact_upper=%d union_points=%d reserve_faces=%d output_views=%d "
        "analysis=%.3fs total=%.3fs",
        source.source_object_id,
        "PROXY" if compacted else "EXACT",
        front_point_count,
        exact_upper_bound,
        len(union.vertices),
        len(reserve_indices),
        len(surfaces),
        analysis_elapsed,
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
    "_OccupiedScreenGrid",
    "_ParallaxSourceAnalysis",
    "_VisibilityTriangle",
    "_build_occupied_grid",
    "_build_source_analysis",
    "_front_visible_face_indices",
    "build_depth_parallax_geometry_package",
]
