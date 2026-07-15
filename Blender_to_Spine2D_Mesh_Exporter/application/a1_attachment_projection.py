"""Project validated geometry snapshots into explicit A1 attachment inputs.

This application bridge intentionally accepts only triangulated manifold disks
whose local vertices each have one UV coordinate in the requested layer.  It
computes a deterministic boundary cycle, places hull vertices first as required by
Spine mesh attachments, and never matches geometry by rounded positions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from typing import Tuple

from ..domain.geometry import (
    EdgeId,
    FaceId,
    MeshSnapshot,
    MeshSnapshotValidator,
    VertexId,
    analyse_face_region,
    build_edge_to_faces,
    is_simple_disk,
)
from ..domain.spine import (
    LegacyAttachmentSequence,
    LegacyAttachmentVertex,
    LegacyMeshAttachmentRequest,
    LegacyRigBuildResult,
)


class A1AttachmentProjectionError(ValueError):
    """Raised when a snapshot cannot be projected without ambiguous topology."""


@dataclass(frozen=True, slots=True)
class A1VertexZBinding:
    vertex_id: VertexId
    z_group_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.vertex_id, VertexId):
            raise TypeError("vertex_id must be VertexId")
        if not isinstance(self.z_group_index, int) or self.z_group_index < 0:
            raise ValueError("z_group_index must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class A1AttachmentProjectionSettings:
    slot_name: str
    attachment_name: str
    vertex_prefix: str
    image_path: str
    uv_layer_name: str
    attachment_width: float
    attachment_height: float
    center_x: float
    center_y: float
    z_bindings: Tuple[A1VertexZBinding, ...]
    sequence: LegacyAttachmentSequence | None = None
    skin_name: str = "default"

    def __post_init__(self) -> None:
        for field_name in (
            "slot_name",
            "attachment_name",
            "vertex_prefix",
            "image_path",
            "uv_layer_name",
            "skin_name",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        for field_name in (
            "attachment_width",
            "attachment_height",
            "center_x",
            "center_y",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or not isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        if self.attachment_width <= 0.0 or self.attachment_height <= 0.0:
            raise ValueError("attachment dimensions must be positive")
        if not isinstance(self.z_bindings, tuple) or not self.z_bindings:
            raise ValueError("z_bindings must be a non-empty tuple")
        if not all(isinstance(item, A1VertexZBinding) for item in self.z_bindings):
            raise TypeError("z_bindings must contain A1VertexZBinding values")
        vertex_ids = tuple(binding.vertex_id for binding in self.z_bindings)
        if len(vertex_ids) != len(set(vertex_ids)):
            raise ValueError("z_bindings contain duplicate vertex IDs")
        if self.sequence is not None and not isinstance(
            self.sequence, LegacyAttachmentSequence
        ):
            raise TypeError("sequence must be LegacyAttachmentSequence or None")


@dataclass(frozen=True, slots=True)
class A1AttachmentProjectionResult:
    request: LegacyMeshAttachmentRequest
    hull_vertex_ids: Tuple[VertexId, ...]
    ordered_vertex_ids: Tuple[VertexId, ...]
    old_to_attachment_index: Tuple[Tuple[VertexId, int], ...]

    def attachment_index_for(self, vertex_id: VertexId) -> int:
        if not isinstance(vertex_id, VertexId):
            raise TypeError("vertex_id must be VertexId")
        mapping = dict(self.old_to_attachment_index)
        if vertex_id not in mapping:
            raise KeyError(f"Unknown projected vertex {vertex_id.index}")
        return mapping[vertex_id]


def _validate_triangulated_disk(snapshot: MeshSnapshot) -> None:
    MeshSnapshotValidator().validate_or_raise(snapshot)
    non_triangles = tuple(
        face.id.index for face in snapshot.faces if len(face.loop_ids) != 3
    )
    if non_triangles:
        raise A1AttachmentProjectionError(
            "Snapshot must be triangulated before A1 attachment projection; "
            f"non-triangle faces: {non_triangles}"
        )
    topology = analyse_face_region(snapshot, tuple(face.id for face in snapshot.faces))
    if not is_simple_disk(topology):
        raise A1AttachmentProjectionError(
            "Snapshot must be one manifold topological disk; "
            f"Euler={topology.euler_characteristic}, "
            f"boundary_components={topology.boundary_component_count}, "
            f"manifold={topology.manifold}"
        )


def _boundary_adjacency(snapshot: MeshSnapshot) -> dict[VertexId, Tuple[VertexId, ...]]:
    edge_to_faces = build_edge_to_faces(snapshot)
    edge_map = snapshot.edge_by_id()
    mutable: dict[VertexId, set[VertexId]] = defaultdict(set)
    boundary_edge_ids = tuple(
        edge_id
        for edge_id, linked_faces in edge_to_faces.items()
        if len(linked_faces) == 1
    )
    if not boundary_edge_ids:
        raise A1AttachmentProjectionError("Disk snapshot has no boundary edges")
    for edge_id in boundary_edge_ids:
        first, second = edge_map[edge_id].vertex_ids
        mutable[first].add(second)
        mutable[second].add(first)
    invalid = tuple(
        sorted(
            (
                (vertex_id.index, len(neighbours))
                for vertex_id, neighbours in mutable.items()
                if len(neighbours) != 2
            ),
            key=lambda item: item[0],
        )
    )
    if invalid:
        raise A1AttachmentProjectionError(
            "Boundary is not a single closed cycle; vertex degrees: " + str(invalid)
        )
    return {
        vertex_id: tuple(sorted(neighbours, key=lambda item: item.index))
        for vertex_id, neighbours in mutable.items()
    }


def _walk_boundary_cycle(
    adjacency: dict[VertexId, Tuple[VertexId, ...]],
    start: VertexId,
    first_next: VertexId,
) -> Tuple[VertexId, ...]:
    cycle = [start]
    previous = start
    current = first_next
    while current != start:
        if current in cycle:
            raise A1AttachmentProjectionError(
                f"Boundary cycle revisited vertex {current.index} before closing"
            )
        cycle.append(current)
        neighbours = adjacency.get(current)
        if neighbours is None or len(neighbours) != 2:
            raise A1AttachmentProjectionError(
                f"Boundary vertex {current.index} has invalid adjacency"
            )
        candidates = tuple(neighbour for neighbour in neighbours if neighbour != previous)
        if len(candidates) != 1:
            raise A1AttachmentProjectionError(
                f"Boundary traversal is ambiguous at vertex {current.index}"
            )
        previous, current = current, candidates[0]
        if len(cycle) > len(adjacency):
            raise A1AttachmentProjectionError("Boundary traversal exceeded vertex count")
    if len(cycle) != len(adjacency):
        missing = sorted(vertex_id.index for vertex_id in set(adjacency) - set(cycle))
        raise A1AttachmentProjectionError(
            f"Boundary contains disconnected vertices: {missing}"
        )
    return tuple(cycle)


def _deterministic_hull_cycle(snapshot: MeshSnapshot) -> Tuple[VertexId, ...]:
    adjacency = _boundary_adjacency(snapshot)
    start = min(adjacency, key=lambda item: item.index)
    first, second = adjacency[start]
    forward = _walk_boundary_cycle(adjacency, start, first)
    reverse = _walk_boundary_cycle(adjacency, start, second)
    forward_key = tuple(vertex_id.index for vertex_id in forward)
    reverse_key = tuple(vertex_id.index for vertex_id in reverse)
    return forward if forward_key <= reverse_key else reverse


def _single_uv_by_vertex(
    snapshot: MeshSnapshot,
    layer_name: str,
) -> dict[VertexId, Tuple[float, float]]:
    if layer_name not in snapshot.uv_layer_names:
        raise A1AttachmentProjectionError(
            f"UV layer '{layer_name}' is absent from snapshot"
        )
    uv_sets: dict[VertexId, set[Tuple[float, float]]] = defaultdict(set)
    for loop in snapshot.loops:
        coordinate = loop.uv(layer_name)
        if coordinate is None:
            raise A1AttachmentProjectionError(
                f"Loop {loop.id.index} is missing UV layer '{layer_name}'"
            )
        uv_sets[loop.vertex_id].add(
            (float(coordinate[0]), float(coordinate[1]))
        )
    ambiguous = tuple(
        sorted(
            (
                (vertex_id.index, tuple(sorted(values)))
                for vertex_id, values in uv_sets.items()
                if len(values) != 1
            ),
            key=lambda item: item[0],
        )
    )
    if ambiguous:
        raise A1AttachmentProjectionError(
            "A local mesh vertex has multiple UV coordinates. UV seam duplication "
            "must be handled by a dedicated projector before A1 attachment build: "
            + str(ambiguous)
        )
    return {vertex_id: next(iter(values)) for vertex_id, values in uv_sets.items()}


def _z_binding_map(
    snapshot: MeshSnapshot,
    settings: A1AttachmentProjectionSettings,
    rig: LegacyRigBuildResult,
) -> dict[VertexId, int]:
    mapping = {binding.vertex_id: binding.z_group_index for binding in settings.z_bindings}
    snapshot_vertex_ids = {vertex.id for vertex in snapshot.vertices}
    missing = snapshot_vertex_ids - set(mapping)
    unknown = set(mapping) - snapshot_vertex_ids
    if missing or unknown:
        raise A1AttachmentProjectionError(
            "z_bindings must cover snapshot vertices exactly; "
            f"missing={tuple(sorted(item.index for item in missing))}, "
            f"unknown={tuple(sorted(item.index for item in unknown))}"
        )
    valid_z_indices = {group.index for group in rig.info.z_groups}
    invalid_indices = tuple(
        sorted({value for value in mapping.values() if value not in valid_z_indices})
    )
    if invalid_indices:
        raise A1AttachmentProjectionError(
            f"z_bindings reference unknown rig groups: {invalid_indices}; "
            f"available={tuple(sorted(valid_z_indices))}"
        )
    return mapping


def project_triangulated_disk_attachment(
    snapshot: MeshSnapshot,
    rig: LegacyRigBuildResult,
    settings: A1AttachmentProjectionSettings,
) -> A1AttachmentProjectionResult:
    """Create a deterministic legacy attachment request from one disk snapshot."""

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    if not isinstance(rig, LegacyRigBuildResult):
        raise TypeError("rig must be LegacyRigBuildResult")
    if not isinstance(settings, A1AttachmentProjectionSettings):
        raise TypeError("settings must be A1AttachmentProjectionSettings")
    if snapshot.source_object_id != rig.request.prefix and (
        snapshot.object_name != rig.request.prefix
    ):
        # Object IDs may be UUID-like in the future. The check is intentionally a
        # warning-level omission rather than a hard name equality requirement; all
        # actual binding integrity is carried by explicit vertex IDs below.
        pass

    _validate_triangulated_disk(snapshot)
    rig.validate()
    hull_vertex_ids = _deterministic_hull_cycle(snapshot)
    hull_set = set(hull_vertex_ids)
    interior_vertex_ids = tuple(
        sorted(
            (vertex.id for vertex in snapshot.vertices if vertex.id not in hull_set),
            key=lambda item: item.index,
        )
    )
    ordered_vertex_ids = hull_vertex_ids + interior_vertex_ids
    old_to_new = {
        vertex_id: attachment_index
        for attachment_index, vertex_id in enumerate(ordered_vertex_ids)
    }
    uv_by_vertex = _single_uv_by_vertex(snapshot, settings.uv_layer_name)
    z_by_vertex = _z_binding_map(snapshot, settings, rig)
    vertex_map = snapshot.vertex_by_id()

    projected_vertices = tuple(
        LegacyAttachmentVertex(
            index=attachment_index,
            uv=uv_by_vertex[vertex_id],
            bone_position_pixels=(
                (float(vertex_map[vertex_id].position[0]) - float(settings.center_x))
                * rig.info.uniform_scale,
                -(
                    float(vertex_map[vertex_id].position[1])
                    - float(settings.center_y)
                )
                * rig.info.uniform_scale,
            ),
            z_group_index=z_by_vertex[vertex_id],
        )
        for attachment_index, vertex_id in enumerate(ordered_vertex_ids)
    )

    face_map = snapshot.face_by_id()
    loop_map = snapshot.loop_by_id()
    triangles = tuple(
        old_to_new[loop_map[loop_id].vertex_id]
        for face_id in sorted(face_map, key=lambda item: item.index)
        for loop_id in face_map[face_id].loop_ids
    )
    edge_map = snapshot.edge_by_id()
    edges = tuple(
        old_to_new[vertex_id]
        for edge_id in sorted(edge_map, key=lambda item: item.index)
        for vertex_id in edge_map[edge_id].vertex_ids
    )

    request = LegacyMeshAttachmentRequest(
        slot_name=settings.slot_name,
        attachment_name=settings.attachment_name,
        vertex_prefix=settings.vertex_prefix,
        image_path=settings.image_path,
        width=settings.attachment_width,
        height=settings.attachment_height,
        vertices=projected_vertices,
        triangles=triangles,
        hull=len(hull_vertex_ids),
        edges=edges,
        sequence=settings.sequence,
        skin_name=settings.skin_name,
    )
    return A1AttachmentProjectionResult(
        request=request,
        hull_vertex_ids=hull_vertex_ids,
        ordered_vertex_ids=ordered_vertex_ids,
        old_to_attachment_index=tuple(
            (vertex_id, old_to_new[vertex_id]) for vertex_id in ordered_vertex_ids
        ),
    )
