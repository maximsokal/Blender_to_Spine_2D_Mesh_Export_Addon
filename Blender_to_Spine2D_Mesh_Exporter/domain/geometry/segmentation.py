"""Deterministic, Blender-independent mesh segmentation."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from math import acos, degrees, isfinite, sqrt
from typing import Iterable, Tuple

from .correspondence import extract_face_subset
from .ids import EdgeId, FaceId, SourceEdgeId, SourceFaceId, VertexId
from .model import MeshFace, MeshLoop, MeshSnapshot, Vector2, Vector3
from .validator import MeshSnapshotValidator


class SegmentBoundaryReason(str, Enum):
    MESH_BOUNDARY = "MESH_BOUNDARY"
    NON_MANIFOLD = "NON_MANIFOLD"
    SEAM = "SEAM"
    SHARP = "SHARP"
    MATERIAL = "MATERIAL"
    ANGLE = "ANGLE"
    INVALID_NORMAL = "INVALID_NORMAL"
    UV_DISCONTINUITY = "UV_DISCONTINUITY"


@dataclass(frozen=True, slots=True)
class SegmentationSettings:
    split_by_angle: bool = True
    angle_limit_degrees: float = 30.0
    respect_seams: bool = True
    split_sharp_edges: bool = False
    split_materials: bool = True
    split_uv_boundaries: bool = True
    uv_layer_name: str | None = None
    uv_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if not isinstance(self.split_by_angle, bool):
            raise TypeError("split_by_angle must be bool")
        if not isinstance(self.respect_seams, bool):
            raise TypeError("respect_seams must be bool")
        if not isinstance(self.split_sharp_edges, bool):
            raise TypeError("split_sharp_edges must be bool")
        if not isinstance(self.split_materials, bool):
            raise TypeError("split_materials must be bool")
        if not isinstance(self.split_uv_boundaries, bool):
            raise TypeError("split_uv_boundaries must be bool")
        if not isinstance(self.angle_limit_degrees, (int, float)) or not isfinite(
            float(self.angle_limit_degrees)
        ):
            raise ValueError("angle_limit_degrees must be finite")
        if self.angle_limit_degrees < 0.0 or self.angle_limit_degrees > 180.0:
            raise ValueError("angle_limit_degrees must be in the range [0, 180]")
        if not isinstance(self.uv_tolerance, (int, float)) or not isfinite(
            float(self.uv_tolerance)
        ):
            raise ValueError("uv_tolerance must be finite")
        if self.uv_tolerance < 0.0:
            raise ValueError("uv_tolerance cannot be negative")
        if self.uv_layer_name is not None and not self.uv_layer_name.strip():
            raise ValueError("uv_layer_name cannot be empty")


@dataclass(frozen=True, slots=True)
class SegmentTopology:
    vertex_count: int
    edge_count: int
    face_count: int
    euler_characteristic: int
    boundary_edge_count: int
    boundary_component_count: int
    manifold: bool


@dataclass(frozen=True, slots=True)
class MeshSegment:
    segment_id: int
    face_ids: Tuple[FaceId, ...]
    source_face_ids: Tuple[SourceFaceId, ...]
    topology: SegmentTopology

    def __post_init__(self) -> None:
        if not isinstance(self.segment_id, int) or self.segment_id < 0:
            raise ValueError("segment_id must be a non-negative integer")
        if not self.face_ids:
            raise ValueError("face_ids cannot be empty")
        if len(self.face_ids) != len(set(self.face_ids)):
            raise ValueError("face_ids cannot contain duplicates")


@dataclass(frozen=True, slots=True)
class SegmentBoundaryEdge:
    edge_id: EdgeId
    source_edge_id: SourceEdgeId | None
    linked_face_ids: Tuple[FaceId, ...]
    segment_ids: Tuple[int, ...]
    reasons: Tuple[SegmentBoundaryReason, ...]

    def __post_init__(self) -> None:
        if not self.reasons:
            raise ValueError("boundary edge must contain at least one reason")


@dataclass(frozen=True, slots=True)
class SegmentationPlan:
    snapshot_id: str
    settings: SegmentationSettings
    segments: Tuple[MeshSegment, ...]
    boundary_edges: Tuple[SegmentBoundaryEdge, ...]

    def segment_for_face(self) -> dict[FaceId, int]:
        return {
            face_id: segment.segment_id
            for segment in self.segments
            for face_id in segment.face_ids
        }


class SegmentationError(ValueError):
    """Raised when a valid snapshot cannot be segmented deterministically."""


def _vector_length(value: Vector3) -> float:
    return sqrt(sum(component * component for component in value))


def _normal_angle_degrees(first: Vector3, second: Vector3) -> float | None:
    first_length = _vector_length(first)
    second_length = _vector_length(second)
    if first_length <= 1e-12 or second_length <= 1e-12:
        return None
    dot = sum(a * b for a, b in zip(first, second)) / (
        first_length * second_length
    )
    dot = max(-1.0, min(1.0, dot))
    return degrees(acos(dot))


def _uv_equal(first: Vector2, second: Vector2, tolerance: float) -> bool:
    return (
        abs(first[0] - second[0]) <= tolerance
        and abs(first[1] - second[1]) <= tolerance
    )


def _face_edge_uvs(
    face: MeshFace,
    edge_id: EdgeId,
    loop_map: dict,
    layer_name: str,
) -> dict[VertexId, Vector2]:
    face_loops = [loop_map[loop_id] for loop_id in face.loop_ids]
    for index, loop in enumerate(face_loops):
        if loop.edge_id != edge_id:
            continue
        next_loop: MeshLoop = face_loops[(index + 1) % len(face_loops)]
        first_uv = loop.uv(layer_name)
        second_uv = next_loop.uv(layer_name)
        if first_uv is None or second_uv is None:
            raise SegmentationError(
                f"Face {face.id.index} is missing UV layer '{layer_name}' on edge "
                f"{edge_id.index}"
            )
        return {loop.vertex_id: first_uv, next_loop.vertex_id: second_uv}
    raise SegmentationError(
        f"Face {face.id.index} does not reference linked edge {edge_id.index}"
    )


def _is_uv_discontinuous(
    first_face: MeshFace,
    second_face: MeshFace,
    edge_id: EdgeId,
    loop_map: dict,
    layer_name: str,
    tolerance: float,
) -> bool:
    first = _face_edge_uvs(first_face, edge_id, loop_map, layer_name)
    second = _face_edge_uvs(second_face, edge_id, loop_map, layer_name)
    if set(first) != set(second):
        raise SegmentationError(
            f"Faces {first_face.id.index} and {second_face.id.index} disagree on "
            f"the vertices of edge {edge_id.index}"
        )
    return any(
        not _uv_equal(first[vertex], second[vertex], tolerance) for vertex in first
    )


def _boundary_components(
    boundary_edge_ids: Iterable[EdgeId],
    edge_map: dict,
) -> tuple[int, bool]:
    adjacency: dict[VertexId, set[VertexId]] = defaultdict(set)
    for edge_id in boundary_edge_ids:
        edge = edge_map[edge_id]
        first, second = edge.vertex_ids
        adjacency[first].add(second)
        adjacency[second].add(first)

    if not adjacency:
        return 0, True

    manifold = all(len(neighbours) == 2 for neighbours in adjacency.values())
    visited: set[VertexId] = set()
    component_count = 0
    for seed in sorted(adjacency, key=lambda item: item.index):
        if seed in visited:
            continue
        component_count += 1
        queue = deque([seed])
        visited.add(seed)
        while queue:
            current = queue.popleft()
            for neighbour in sorted(adjacency[current], key=lambda item: item.index):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
    return component_count, manifold


def _segment_topology(
    face_ids: Tuple[FaceId, ...],
    face_map: dict,
    loop_map: dict,
    edge_map: dict,
    edge_to_faces: dict[EdgeId, Tuple[FaceId, ...]],
) -> SegmentTopology:
    face_id_set = set(face_ids)
    loop_ids = {
        loop_id for face_id in face_ids for loop_id in face_map[face_id].loop_ids
    }
    edge_ids = {loop_map[loop_id].edge_id for loop_id in loop_ids}
    vertex_ids = {loop_map[loop_id].vertex_id for loop_id in loop_ids}
    boundary_edge_ids = tuple(
        edge_id
        for edge_id in sorted(edge_ids, key=lambda item: item.index)
        if sum(face_id in face_id_set for face_id in edge_to_faces[edge_id]) == 1
    )
    boundary_components, boundary_manifold = _boundary_components(
        boundary_edge_ids, edge_map
    )
    globally_manifold = all(len(edge_to_faces[edge_id]) <= 2 for edge_id in edge_ids)
    return SegmentTopology(
        vertex_count=len(vertex_ids),
        edge_count=len(edge_ids),
        face_count=len(face_ids),
        euler_characteristic=len(vertex_ids) - len(edge_ids) + len(face_ids),
        boundary_edge_count=len(boundary_edge_ids),
        boundary_component_count=boundary_components,
        manifold=globally_manifold and boundary_manifold,
    )


def segment_mesh(
    snapshot: MeshSnapshot,
    settings: SegmentationSettings | None = None,
) -> SegmentationPlan:
    """Split faces into deterministic connected components under cut policies."""

    MeshSnapshotValidator().validate_or_raise(snapshot)
    resolved_settings = settings or SegmentationSettings()
    face_map = snapshot.face_by_id()
    edge_map = snapshot.edge_by_id()
    loop_map = snapshot.loop_by_id()

    edge_to_faces_mutable: dict[EdgeId, list[FaceId]] = defaultdict(list)
    for face in snapshot.faces:
        for loop_id in face.loop_ids:
            edge_id = loop_map[loop_id].edge_id
            if face.id not in edge_to_faces_mutable[edge_id]:
                edge_to_faces_mutable[edge_id].append(face.id)
    edge_to_faces: dict[EdgeId, Tuple[FaceId, ...]] = {
        edge_id: tuple(sorted(face_ids, key=lambda item: item.index))
        for edge_id, face_ids in edge_to_faces_mutable.items()
    }

    resolved_uv_layer = resolved_settings.uv_layer_name or snapshot.active_uv_layer
    if (
        resolved_settings.split_uv_boundaries
        and resolved_uv_layer is not None
        and resolved_uv_layer not in snapshot.uv_layer_names
    ):
        raise SegmentationError(
            f"UV layer '{resolved_uv_layer}' is not present in snapshot"
        )

    adjacency: dict[FaceId, set[FaceId]] = {
        face.id: set() for face in snapshot.faces
    }
    boundary_reasons: dict[EdgeId, set[SegmentBoundaryReason]] = defaultdict(set)

    for edge in snapshot.edges:
        linked_faces = edge_to_faces.get(edge.id, ())
        if len(linked_faces) == 0:
            continue
        if len(linked_faces) == 1:
            boundary_reasons[edge.id].add(SegmentBoundaryReason.MESH_BOUNDARY)
            continue
        if len(linked_faces) != 2:
            boundary_reasons[edge.id].add(SegmentBoundaryReason.NON_MANIFOLD)
            continue

        first_face = face_map[linked_faces[0]]
        second_face = face_map[linked_faces[1]]
        reasons = boundary_reasons[edge.id]

        if resolved_settings.respect_seams and edge.seam:
            reasons.add(SegmentBoundaryReason.SEAM)
        if resolved_settings.split_sharp_edges and edge.sharp:
            reasons.add(SegmentBoundaryReason.SHARP)
        if (
            resolved_settings.split_materials
            and first_face.material_index != second_face.material_index
        ):
            reasons.add(SegmentBoundaryReason.MATERIAL)
        if resolved_settings.split_by_angle:
            angle = _normal_angle_degrees(first_face.normal, second_face.normal)
            if angle is None:
                reasons.add(SegmentBoundaryReason.INVALID_NORMAL)
            elif angle >= resolved_settings.angle_limit_degrees:
                reasons.add(SegmentBoundaryReason.ANGLE)
        if resolved_settings.split_uv_boundaries and resolved_uv_layer is not None:
            if _is_uv_discontinuous(
                first_face,
                second_face,
                edge.id,
                loop_map,
                resolved_uv_layer,
                resolved_settings.uv_tolerance,
            ):
                reasons.add(SegmentBoundaryReason.UV_DISCONTINUITY)

        if not reasons:
            adjacency[first_face.id].add(second_face.id)
            adjacency[second_face.id].add(first_face.id)

    components: list[Tuple[FaceId, ...]] = []
    visited: set[FaceId] = set()
    for seed in sorted(face_map, key=lambda item: item.index):
        if seed in visited:
            continue
        queue = deque([seed])
        visited.add(seed)
        component: list[FaceId] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbour in sorted(adjacency[current], key=lambda item: item.index):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        components.append(tuple(sorted(component, key=lambda item: item.index)))

    segments = tuple(
        MeshSegment(
            segment_id=segment_id,
            face_ids=face_ids,
            source_face_ids=tuple(face_map[face_id].source_id for face_id in face_ids),
            topology=_segment_topology(
                face_ids,
                face_map,
                loop_map,
                edge_map,
                edge_to_faces,
            ),
        )
        for segment_id, face_ids in enumerate(components)
    )
    face_to_segment = {
        face_id: segment.segment_id
        for segment in segments
        for face_id in segment.face_ids
    }

    boundary_edges = tuple(
        SegmentBoundaryEdge(
            edge_id=edge_id,
            source_edge_id=edge_map[edge_id].source_id,
            linked_face_ids=edge_to_faces.get(edge_id, ()),
            segment_ids=tuple(
                sorted(
                    {
                        face_to_segment[face_id]
                        for face_id in edge_to_faces.get(edge_id, ())
                    }
                )
            ),
            reasons=tuple(sorted(reasons, key=lambda reason: reason.value)),
        )
        for edge_id, reasons in sorted(
            boundary_reasons.items(), key=lambda item: item[0].index
        )
        if reasons
    )

    return SegmentationPlan(
        snapshot_id=snapshot.snapshot_id,
        settings=resolved_settings,
        segments=segments,
        boundary_edges=boundary_edges,
    )


def materialize_segment_snapshots(
    snapshot: MeshSnapshot,
    plan: SegmentationPlan,
    *,
    snapshot_id_prefix: str | None = None,
    object_name_prefix: str | None = None,
) -> Tuple[MeshSnapshot, ...]:
    """Convert a plan to immutable segment snapshots without Blender datablocks."""

    if plan.snapshot_id != snapshot.snapshot_id:
        raise SegmentationError("plan does not belong to the supplied snapshot")
    id_prefix = snapshot_id_prefix or f"{snapshot.snapshot_id}:segment"
    name_prefix = object_name_prefix or snapshot.object_name
    return tuple(
        extract_face_subset(
            snapshot,
            segment.face_ids,
            snapshot_id=f"{id_prefix}:{segment.segment_id:03d}",
            object_name=f"{name_prefix}_Segment_{segment.segment_id:03d}",
        )
        for segment in plan.segments
    )
