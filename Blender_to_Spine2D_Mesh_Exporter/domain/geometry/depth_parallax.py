"""Angular parallax-reserve geometry for Depth Camera Projection 0.90.0.

The front surface remains the exact active-camera result. When a positive horizon angle
is requested, this module expands from front-visible source triangles through shared-edge
adjacency using accumulated unsigned dihedral angle. Extra faces are assigned to one of
eight deterministic virtual camera views, projected into the front-camera rig space, and
receive UVs from the virtual view that textures them.

One union MeshSnapshot owns every retained vertex and face. Material index zero denotes
the front attachment; positive material indices denote reserve views. This keeps one
shared Z-group/vertex-bone domain while allowing document assembly to emit independent
front and reserve attachments with independent rendered images.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from heapq import heappop, heappush
from math import acos, atan2, isfinite, pi, sqrt
from typing import Mapping, Sequence

from ..camera_projection import A1CameraProjectionFrame
from ..projection import A1ProjectedPoint
from .depth_camera_projection import (
    DepthCameraProjectionError,
    DepthCameraProjectionResult,
    _ProjectedTriangle,
    _translation_only_origin,
    _world_point,
)
from .depth_camera_projection_bounded import _projected_triangles
from .depth_camera_projection_visible_topology import (
    _clip_triangle_to_frame,
    _triangle_vertex_ids,
    _visible_clipped_face_indices,
)
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
    LoopUV,
    Matrix4x4,
    MeshEdge,
    MeshFace,
    MeshLoop,
    MeshSnapshot,
    MeshVertex,
)
from .triangulation import triangulate_snapshot
from .validator import MeshSnapshotValidator


_AREA_EPSILON = 1.0e-12
_COORDINATE_QUANTUM = 1.0e-9


class DepthParallaxViewId(str, Enum):
    """Deterministic reserve texture directions around the active-camera view."""

    RIGHT = "RIGHT"
    UP_RIGHT = "UP_RIGHT"
    UP = "UP"
    UP_LEFT = "UP_LEFT"
    LEFT = "LEFT"
    DOWN_LEFT = "DOWN_LEFT"
    DOWN = "DOWN"
    DOWN_RIGHT = "DOWN_RIGHT"

    @property
    def ordinal(self) -> int:
        return _VIEW_ORDER.index(self)


_VIEW_ORDER = (
    DepthParallaxViewId.RIGHT,
    DepthParallaxViewId.UP_RIGHT,
    DepthParallaxViewId.UP,
    DepthParallaxViewId.UP_LEFT,
    DepthParallaxViewId.LEFT,
    DepthParallaxViewId.DOWN_LEFT,
    DepthParallaxViewId.DOWN,
    DepthParallaxViewId.DOWN_RIGHT,
)


@dataclass(frozen=True, slots=True)
class DepthParallaxCameraView:
    """One virtual texture view and its exact render-camera override."""

    view_id: DepthParallaxViewId
    yaw_radians: float
    pitch_radians: float
    frame: A1CameraProjectionFrame
    camera_world_matrix: Matrix4x4
    lens_scale: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.view_id, DepthParallaxViewId):
            raise TypeError("view_id must be DepthParallaxViewId")
        for field_name in ("yaw_radians", "pitch_radians", "lens_scale"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a finite number")
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError(f"{field_name} must be finite")
            object.__setattr__(self, field_name, numeric)
        if self.lens_scale <= 0.0 or self.lens_scale > 1.0:
            raise ValueError("lens_scale must be in (0, 1]")
        if not isinstance(self.frame, A1CameraProjectionFrame):
            raise TypeError("frame must be A1CameraProjectionFrame")
        if not isinstance(self.camera_world_matrix, tuple) or len(
            self.camera_world_matrix
        ) != 16:
            raise TypeError("camera_world_matrix must contain sixteen values")
        if not all(isfinite(float(value)) for value in self.camera_world_matrix):
            raise ValueError("camera_world_matrix contains non-finite values")

    @property
    def material_index(self) -> int:
        return self.view_id.ordinal + 1


@dataclass(frozen=True, slots=True)
class DepthParallaxReserveSurface:
    """One view-owned reserve attachment extracted from the union snapshot."""

    view: DepthParallaxCameraView
    snapshot: MeshSnapshot
    source_face_indices: tuple[int, ...]
    maximum_accumulated_angle_radians: float

    def __post_init__(self) -> None:
        if not isinstance(self.view, DepthParallaxCameraView):
            raise TypeError("view must be DepthParallaxCameraView")
        if not isinstance(self.snapshot, MeshSnapshot):
            raise TypeError("snapshot must be MeshSnapshot")
        if not isinstance(self.source_face_indices, tuple) or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in self.source_face_indices
        ):
            raise TypeError("source_face_indices must contain non-negative ints")
        if tuple(sorted(set(self.source_face_indices))) != self.source_face_indices:
            raise ValueError("source_face_indices must be sorted and unique")
        if (
            isinstance(self.maximum_accumulated_angle_radians, bool)
            or not isinstance(
                self.maximum_accumulated_angle_radians,
                (int, float),
            )
        ):
            raise TypeError("maximum_accumulated_angle_radians must be numeric")
        angle = float(self.maximum_accumulated_angle_radians)
        if not isfinite(angle) or angle < 0.0:
            raise ValueError(
                "maximum_accumulated_angle_radians must be finite and non-negative"
            )
        object.__setattr__(self, "maximum_accumulated_angle_radians", angle)


@dataclass(frozen=True, slots=True)
class DepthParallaxGeometryPackage:
    """Front result plus all angular reserve surfaces in one shared rig snapshot."""

    front_result: DepthCameraProjectionResult
    union_snapshot: MeshSnapshot
    front_snapshot: MeshSnapshot
    reserve_surfaces: tuple[DepthParallaxReserveSurface, ...]
    horizon_angle_radians: float
    front_face_indices: tuple[int, ...]
    reserve_face_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.front_result, DepthCameraProjectionResult):
            raise TypeError("front_result must be DepthCameraProjectionResult")
        for field_name in ("union_snapshot", "front_snapshot"):
            if not isinstance(getattr(self, field_name), MeshSnapshot):
                raise TypeError(f"{field_name} must be MeshSnapshot")
        if not isinstance(self.reserve_surfaces, tuple) or not all(
            isinstance(value, DepthParallaxReserveSurface)
            for value in self.reserve_surfaces
        ):
            raise TypeError(
                "reserve_surfaces must contain DepthParallaxReserveSurface values"
            )
        if (
            isinstance(self.horizon_angle_radians, bool)
            or not isinstance(self.horizon_angle_radians, (int, float))
        ):
            raise TypeError("horizon_angle_radians must be numeric")
        angle = float(self.horizon_angle_radians)
        if not isfinite(angle) or angle < 0.0 or angle >= pi / 2.0:
            raise ValueError("horizon_angle_radians must be finite in [0, pi/2)")
        object.__setattr__(self, "horizon_angle_radians", angle)
        for field_name in ("front_face_indices", "reserve_face_indices"):
            values = getattr(self, field_name)
            if not isinstance(values, tuple) or not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in values
            ):
                raise TypeError(f"{field_name} must contain non-negative ints")
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{field_name} must be sorted and unique")
        MeshSnapshotValidator().validate_or_raise(self.union_snapshot)
        MeshSnapshotValidator().validate_or_raise(self.front_snapshot)
        if self.front_snapshot.source_object_id != self.union_snapshot.source_object_id:
            raise ValueError("front and union snapshots must share source_object_id")

    @property
    def reserve_enabled(self) -> bool:
        return bool(self.reserve_surfaces)

    @property
    def attachment_count(self) -> int:
        return 1 + len(self.reserve_surfaces)


@dataclass(frozen=True, slots=True)
class _FaceGeometry:
    face_index: int
    vertex_ids: tuple[VertexId, VertexId, VertexId]
    world_points: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    normal_world: tuple[float, float, float]
    centroid_world: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class _FaceRecord:
    material_index: int
    source_face_index: int
    positions: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    uvs: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]


def _subtract(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    )


def _cross(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _dot(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    return sum(a * b for a, b in zip(first, second, strict=True))


def _normalized(
    vector: tuple[float, float, float],
    *,
    field_name: str,
) -> tuple[float, float, float]:
    length_squared = _dot(vector, vector)
    if not isfinite(length_squared) or length_squared <= 1.0e-30:
        raise DepthCameraProjectionError(f"{field_name} collapses")
    inverse = 1.0 / sqrt(length_squared)
    return (
        float(vector[0] * inverse),
        float(vector[1] * inverse),
        float(vector[2] * inverse),
    )


def _face_geometry(
    snapshot: MeshSnapshot,
) -> Mapping[int, _FaceGeometry]:
    triangulated = triangulate_snapshot(snapshot).snapshot
    loops = triangulated.loop_by_id()
    vertices = triangulated.vertex_by_id()
    origin = _translation_only_origin(triangulated.world_matrix)
    result: dict[int, _FaceGeometry] = {}
    for face in sorted(triangulated.faces, key=lambda value: value.id.index):
        vertex_ids = tuple(loops[loop_id].vertex_id for loop_id in face.loop_ids)
        if len(vertex_ids) != 3:
            raise DepthCameraProjectionError(
                f"triangulated face {face.id.index} is not triangular"
            )
        world_points = tuple(
            _world_point(origin, vertices[vertex_id].position)
            for vertex_id in vertex_ids
        )
        first_edge = _subtract(world_points[1], world_points[0])
        second_edge = _subtract(world_points[2], world_points[0])
        normal = _normalized(
            _cross(first_edge, second_edge),
            field_name=f"face[{face.id.index}].normal",
        )
        centroid = tuple(
            sum(point[axis] for point in world_points) / 3.0
            for axis in range(3)
        )
        result[face.id.index] = _FaceGeometry(
            face_index=face.id.index,
            vertex_ids=(vertex_ids[0], vertex_ids[1], vertex_ids[2]),
            world_points=(world_points[0], world_points[1], world_points[2]),
            normal_world=normal,
            centroid_world=(centroid[0], centroid[1], centroid[2]),
        )
    return result


def _face_adjacency(
    geometry: Mapping[int, _FaceGeometry],
) -> Mapping[int, tuple[int, ...]]:
    owners: dict[tuple[int, int], list[int]] = {}
    for face_index, face in geometry.items():
        values = tuple(vertex_id.index for vertex_id in face.vertex_ids)
        for corner, first in enumerate(values):
            second = values[(corner + 1) % 3]
            owners.setdefault(tuple(sorted((first, second))), []).append(face_index)
    adjacency: dict[int, set[int]] = {face_index: set() for face_index in geometry}
    for edge_owners in owners.values():
        unique = tuple(sorted(set(edge_owners)))
        for owner_index, first in enumerate(unique):
            for second in unique[owner_index + 1 :]:
                adjacency[first].add(second)
                adjacency[second].add(first)
    return {
        face_index: tuple(sorted(neighbors))
        for face_index, neighbors in sorted(adjacency.items())
    }


def _dihedral_angle(first: _FaceGeometry, second: _FaceGeometry) -> float:
    # Ignore accidental winding inversion while retaining the geometric plane bend.
    cosine = min(1.0, max(0.0, abs(_dot(first.normal_world, second.normal_world))))
    return float(acos(cosine))


def _front_visible_face_indices(
    snapshot: MeshSnapshot,
    frame: A1CameraProjectionFrame,
) -> tuple[int, ...]:
    origin = _translation_only_origin(snapshot.world_matrix)
    triangulated, _projected, triangles = _projected_triangles(
        snapshot,
        frame,
        origin,
    )
    vertex_ids = _triangle_vertex_ids(triangulated)
    polygons = {}
    for triangle in triangles:
        polygon, _clipped = _clip_triangle_to_frame(
            triangle,
            vertex_ids[triangle.face_index],
            frame,
        )
        if polygon:
            polygons[triangle.face_index] = polygon
    if not polygons:
        raise DepthCameraProjectionError(
            "active camera retains no visible source triangles for parallax expansion"
        )
    visible = _visible_clipped_face_indices(
        triangles,
        polygons,
        kind=frame.kind,
    )
    if not visible:
        raise DepthCameraProjectionError(
            "active camera retains no front-most source triangles for parallax expansion"
        )
    return tuple(sorted(set(visible)))


def _accumulated_horizon_costs(
    geometry: Mapping[int, _FaceGeometry],
    adjacency: Mapping[int, tuple[int, ...]],
    seeds: Sequence[int],
    limit: float,
) -> Mapping[int, float]:
    costs: dict[int, float] = {face_index: 0.0 for face_index in seeds}
    pending: list[tuple[float, int]] = [(0.0, face_index) for face_index in seeds]
    for item in pending:
        heappush([], item)
    pending = []
    for face_index in sorted(set(seeds)):
        heappush(pending, (0.0, face_index))
    tolerance = max(1.0e-12, limit * 1.0e-10)
    while pending:
        current_cost, face_index = heappop(pending)
        if current_cost > costs.get(face_index, float("inf")) + tolerance:
            continue
        for neighbor in adjacency.get(face_index, ()):
            candidate = current_cost + _dihedral_angle(
                geometry[face_index],
                geometry[neighbor],
            )
            if candidate > limit + tolerance:
                continue
            previous = costs.get(neighbor)
            if previous is None or candidate + tolerance < previous:
                costs[neighbor] = candidate
                heappush(pending, (candidate, neighbor))
    return costs


def _view_for_face(
    face: _FaceGeometry,
    frame: A1CameraProjectionFrame,
    available: Mapping[DepthParallaxViewId, DepthParallaxCameraView],
) -> DepthParallaxCameraView:
    normal = frame.transform_world_direction(
        face.normal_world,
        field_name=f"face[{face.face_index}].normal",
    )
    centroid = frame.world_to_camera_point(
        face.centroid_world,
        field_name=f"face[{face.face_index}].centroid",
    )
    toward_camera = _normalized(
        (-centroid[0], -centroid[1], -centroid[2]),
        field_name=f"face[{face.face_index}].toward_camera",
    )
    if _dot(normal, toward_camera) < 0.0:
        normal = (-normal[0], -normal[1], -normal[2])
    if abs(normal[0]) <= 1.0e-12 and abs(normal[1]) <= 1.0e-12:
        # A nearly front-facing reserve face is assigned by stable face index so the
        # result stays deterministic even on coplanar non-manifold fixtures.
        view_id = _VIEW_ORDER[face.face_index % len(_VIEW_ORDER)]
        return available[view_id]
    azimuth = atan2(normal[1], normal[0])
    sector = int(round(azimuth / (pi / 4.0))) % 8
    view_id = _VIEW_ORDER[sector]
    return available[view_id]


def _front_records(front: MeshSnapshot, uv_layer_name: str) -> list[_FaceRecord]:
    loops = front.loop_by_id()
    vertices = front.vertex_by_id()
    records: list[_FaceRecord] = []
    for face in sorted(front.faces, key=lambda value: value.id.index):
        if len(face.loop_ids) != 3:
            raise DepthCameraProjectionError(
                f"front depth face {face.id.index} is not triangulated"
            )
        positions = []
        uvs = []
        for loop_id in face.loop_ids:
            loop = loops[loop_id]
            uv = loop.uv(uv_layer_name)
            if uv is None:
                raise DepthCameraProjectionError(
                    f"front depth loop {loop.id.index} has no {uv_layer_name!r} UV"
                )
            vertex = vertices[loop.vertex_id]
            positions.append(tuple(float(value) for value in vertex.position))
            uvs.append((float(uv[0]), float(uv[1])))
        records.append(
            _FaceRecord(
                material_index=0,
                source_face_index=face.id.index,
                positions=(positions[0], positions[1], positions[2]),
                uvs=(uvs[0], uvs[1], uvs[2]),
            )
        )
    return records


def _reserve_record(
    face: _FaceGeometry,
    view: DepthParallaxCameraView,
    front_frame: A1CameraProjectionFrame,
    projected_origin: A1ProjectedPoint,
    uniform_scale: float,
) -> _FaceRecord:
    positions = []
    uvs = []
    for point_index, world_point in enumerate(face.world_points):
        front = front_frame.project_world_point(
            world_point,
            field_name=f"face[{face.face_index}].front[{point_index}]",
        )
        reserve = view.frame.project_world_point(
            world_point,
            field_name=(
                f"face[{face.face_index}].{view.view_id.value}[{point_index}]"
            ),
        )
        positions.append(
            (
                (float(front.u) - float(projected_origin.u)) / uniform_scale,
                -(float(front.v) - float(projected_origin.v)) / uniform_scale,
                float(front.depth),
            )
        )
        u = (float(reserve.u) + float(view.frame.texture_width) / 2.0) / float(
            view.frame.texture_width
        )
        v = 1.0 - (
            float(reserve.v) + float(view.frame.texture_height) / 2.0
        ) / float(view.frame.texture_height)
        tolerance = 1.0e-7
        if u < -tolerance or u > 1.0 + tolerance or v < -tolerance or v > 1.0 + tolerance:
            raise DepthCameraProjectionError(
                "virtual parallax camera did not frame its assigned reserve surface; "
                f"view={view.view_id.value}, face={face.face_index}, uv={(u, v)}"
            )
        uvs.append((min(1.0, max(0.0, u)), min(1.0, max(0.0, v))))
    return _FaceRecord(
        material_index=view.material_index,
        source_face_index=face.face_index,
        positions=(positions[0], positions[1], positions[2]),
        uvs=(uvs[0], uvs[1], uvs[2]),
    )


def _vertex_signature(position: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(round(float(value) / _COORDINATE_QUANTUM) for value in position)  # type: ignore[return-value]


def _snapshot_from_records(
    source: MeshSnapshot,
    records: Sequence[_FaceRecord],
    *,
    uv_layer_name: str,
    snapshot_suffix: str,
) -> MeshSnapshot:
    if not records:
        raise DepthCameraProjectionError("parallax snapshot records cannot be empty")
    vertex_index_by_signature: dict[tuple[int, int, int], int] = {}
    vertex_positions: list[tuple[float, float, float]] = []
    face_vertex_indices: list[tuple[int, int, int]] = []
    for record in records:
        indices = []
        for position in record.positions:
            signature = _vertex_signature(position)
            index = vertex_index_by_signature.get(signature)
            if index is None:
                index = len(vertex_positions)
                vertex_index_by_signature[signature] = index
                vertex_positions.append(position)
            indices.append(index)
        if len(set(indices)) != 3:
            continue
        face_vertex_indices.append((indices[0], indices[1], indices[2]))
    if not face_vertex_indices:
        raise DepthCameraProjectionError(
            "parallax records collapsed to no non-degenerate triangles"
        )

    edge_pairs = tuple(
        sorted(
            {
                tuple(sorted((face[index], face[(index + 1) % 3])))
                for face in face_vertex_indices
                for index in range(3)
            }
        )
    )
    edge_id_by_pair = {pair: EdgeId(index) for index, pair in enumerate(edge_pairs)}
    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(source.source_object_id, index),
            position=(float(position[0]), float(position[1]), float(position[2])),
            normal=(0.0, 0.0, 1.0),
        )
        for index, position in enumerate(vertex_positions)
    )
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
    faces: list[MeshFace] = []
    retained_record_index = 0
    for record in records:
        indices = tuple(
            vertex_index_by_signature[_vertex_signature(position)]
            for position in record.positions
        )
        if len(set(indices)) != 3:
            continue
        loop_ids = []
        for corner_index, vertex_index in enumerate(indices):
            following = indices[(corner_index + 1) % 3]
            pair = tuple(sorted((vertex_index, following)))
            loop_id = LoopId(len(loops))
            loops.append(
                MeshLoop(
                    id=loop_id,
                    source_id=SourceLoopId(
                        source.source_object_id,
                        retained_record_index,
                        corner_index,
                    ),
                    vertex_id=VertexId(vertex_index),
                    edge_id=edge_id_by_pair[pair],
                    uvs=(
                        LoopUV(
                            layer_name=uv_layer_name,
                            coordinate=record.uvs[corner_index],
                        ),
                    ),
                )
            )
            loop_ids.append(loop_id)
        faces.append(
            MeshFace(
                id=FaceId(len(faces)),
                source_id=SourceFaceId(
                    source.source_object_id,
                    record.source_face_index,
                ),
                loop_ids=tuple(loop_ids),
                material_index=record.material_index,
                normal=(0.0, 0.0, 1.0),
                smooth=True,
            )
        )
        retained_record_index += 1

    snapshot = MeshSnapshot(
        snapshot_id=f"{source.snapshot_id}:{snapshot_suffix}",
        source_object_id=source.source_object_id,
        object_name=source.object_name,
        vertices=vertices,
        edges=edges,
        loops=tuple(loops),
        faces=tuple(faces),
        uv_layer_names=(uv_layer_name,),
        active_uv_layer=uv_layer_name,
        render_uv_layer=uv_layer_name,
        world_matrix=source.world_matrix,
    )
    MeshSnapshotValidator().validate_or_raise(snapshot)
    return snapshot


def _subset_material(
    union: MeshSnapshot,
    material_index: int,
    *,
    uv_layer_name: str,
    suffix: str,
) -> MeshSnapshot:
    loops = union.loop_by_id()
    vertices = union.vertex_by_id()
    records = []
    for face in sorted(union.faces, key=lambda value: value.id.index):
        if face.material_index != material_index:
            continue
        positions = []
        uvs = []
        for loop_id in face.loop_ids:
            loop = loops[loop_id]
            vertex = vertices[loop.vertex_id]
            uv = loop.uv(uv_layer_name)
            if uv is None:
                raise DepthCameraProjectionError(
                    f"union loop {loop.id.index} has no {uv_layer_name!r} UV"
                )
            positions.append(vertex.position)
            uvs.append(uv)
        records.append(
            _FaceRecord(
                material_index=0,
                source_face_index=face.source_id.face_index,
                positions=(positions[0], positions[1], positions[2]),
                uvs=(uvs[0], uvs[1], uvs[2]),
            )
        )
    if not records:
        raise DepthCameraProjectionError(
            f"union snapshot has no faces for material index {material_index}"
        )
    return _snapshot_from_records(
        union,
        records,
        uv_layer_name=uv_layer_name,
        snapshot_suffix=suffix,
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
    """Build front plus angular reserve attachments with one shared point budget."""

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
    if (
        isinstance(horizon_angle_radians, bool)
        or not isinstance(horizon_angle_radians, (int, float))
    ):
        raise TypeError("horizon_angle_radians must be numeric")
    angle = float(horizon_angle_radians)
    if not isfinite(angle) or angle < 0.0 or angle >= pi / 2.0:
        raise ValueError("horizon_angle_radians must be finite in [0, pi/2)")
    if isinstance(max_points, bool) or not isinstance(max_points, int) or max_points < 4:
        raise ValueError("max_points must be an integer of at least four")

    MeshSnapshotValidator().validate_or_raise(source)
    MeshSnapshotValidator().validate_or_raise(front_result.snapshot)
    front_faces = _front_visible_face_indices(source, front_result.frame)
    if angle <= 1.0e-12:
        front = _snapshot_from_records(
            front_result.snapshot,
            _front_records(front_result.snapshot, uv_layer_name),
            uv_layer_name=uv_layer_name,
            snapshot_suffix="parallax-front",
        )
        return DepthParallaxGeometryPackage(
            front_result=front_result,
            union_snapshot=front,
            front_snapshot=front,
            reserve_surfaces=(),
            horizon_angle_radians=0.0,
            front_face_indices=front_faces,
            reserve_face_indices=(),
        )

    available = {view.view_id: view for view in reserve_views}
    missing = tuple(view_id.value for view_id in _VIEW_ORDER if view_id not in available)
    if missing:
        raise ValueError(
            "positive parallax horizon requires all eight reserve views; "
            f"missing={missing}"
        )
    geometry = _face_geometry(source)
    adjacency = _face_adjacency(geometry)
    costs = _accumulated_horizon_costs(
        geometry,
        adjacency,
        front_faces,
        angle,
    )
    reserve_indices = tuple(sorted(set(costs) - set(front_faces)))
    records = _front_records(front_result.snapshot, uv_layer_name)
    assigned: dict[DepthParallaxViewId, list[int]] = {
        view_id: [] for view_id in _VIEW_ORDER
    }
    origin = _translation_only_origin(source.world_matrix)
    projected_origin = front_result.frame.project_world_point(
        origin,
        field_name="object_origin",
    )
    for face_index in reserve_indices:
        face = geometry[face_index]
        view = _view_for_face(face, front_result.frame, available)
        records.append(
            _reserve_record(
                face,
                view,
                front_result.frame,
                projected_origin,
                scale,
            )
        )
        assigned[view.view_id].append(face_index)

    union = _snapshot_from_records(
        front_result.snapshot,
        records,
        uv_layer_name=uv_layer_name,
        snapshot_suffix="parallax-union",
    )
    if len(union.vertices) > max_points:
        raise DepthCameraProjectionError(
            "Parallax Horizon Angle exceeds Max Depth Points after adding reserve "
            f"coverage: points={len(union.vertices)}, max_points={max_points}. "
            "Reduce the horizon angle or increase Max Depth Points."
        )
    front = _subset_material(
        union,
        0,
        uv_layer_name=uv_layer_name,
        suffix="parallax-front",
    )
    surfaces = []
    for view_id in _VIEW_ORDER:
        face_indices = tuple(sorted(assigned[view_id]))
        if not face_indices:
            continue
        view = available[view_id]
        surface = _subset_material(
            union,
            view.material_index,
            uv_layer_name=uv_layer_name,
            suffix=f"parallax-{view_id.value.lower()}",
        )
        surfaces.append(
            DepthParallaxReserveSurface(
                view=view,
                snapshot=surface,
                source_face_indices=face_indices,
                maximum_accumulated_angle_radians=max(
                    costs[face_index] for face_index in face_indices
                ),
            )
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
    "DepthParallaxCameraView",
    "DepthParallaxGeometryPackage",
    "DepthParallaxReserveSurface",
    "DepthParallaxViewId",
    "build_depth_parallax_geometry_package",
]
