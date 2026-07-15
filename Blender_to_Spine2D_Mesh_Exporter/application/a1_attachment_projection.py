"""Project triangulated disk snapshots into explicit A1 attachment inputs.

Blender UV coordinates belong to mesh loops, not vertices. A single geometric
vertex may therefore require several Spine attachment vertices when UV seams split
its incident corners. This module creates one deterministic attachment vertex for
every unique ``(VertexId, UV)`` pair while preserving the original geometric
position and Z-group binding.

The first attachment vertices form the ordered physical mesh hull required by the
Spine runtime. When a UV seam reaches the external boundary, both UV variants are
placed consecutively at the same geometric position so the physical hull order is
preserved without merging distinct texture coordinates.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from typing import Tuple

from ..domain.geometry import (
    EdgeId,
    FaceId,
    LoopId,
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
class A1AttachmentVertexKey:
    """Identity of one Spine attachment vertex after UV seam duplication."""

    vertex_id: VertexId
    uv: Tuple[float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.vertex_id, VertexId):
            raise TypeError("vertex_id must be VertexId")
        if (
            not isinstance(self.uv, tuple)
            or len(self.uv) != 2
            or not all(
                isinstance(value, (int, float)) and isfinite(float(value))
                for value in self.uv
            )
        ):
            raise ValueError("uv must contain two finite numeric values")


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
    hull_vertex_keys: Tuple[A1AttachmentVertexKey, ...]
    ordered_vertex_keys: Tuple[A1AttachmentVertexKey, ...]
    loop_to_attachment_index: Tuple[Tuple[LoopId, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, LegacyMeshAttachmentRequest):
            raise TypeError("request must be LegacyMeshAttachmentRequest")
        for field_name in ("hull_vertex_keys", "ordered_vertex_keys"):
            value = getattr(self, field_name)
            if not isinstance(value, tuple) or not all(
                isinstance(item, A1AttachmentVertexKey) for item in value
            ):
                raise TypeError(
                    f"{field_name} must be a tuple of A1AttachmentVertexKey values"
                )
        if not self.ordered_vertex_keys:
            raise ValueError("ordered_vertex_keys cannot be empty")
        if len(self.ordered_vertex_keys) != len(set(self.ordered_vertex_keys)):
            raise ValueError("ordered_vertex_keys contain duplicate attachment keys")
        if self.ordered_vertex_keys[: len(self.hull_vertex_keys)] != (
            self.hull_vertex_keys
        ):
            raise ValueError("hull_vertex_keys must be the ordered vertex prefix")
        if self.request.hull != len(self.hull_vertex_keys):
            raise ValueError("request.hull does not match hull_vertex_keys")
        if len(self.request.vertices) != len(self.ordered_vertex_keys):
            raise ValueError(
                "request vertex count does not match ordered_vertex_keys"
            )
        if not isinstance(self.loop_to_attachment_index, tuple):
            raise TypeError("loop_to_attachment_index must be tuple")
        loop_ids: list[LoopId] = []
        for loop_id, attachment_index in self.loop_to_attachment_index:
            if not isinstance(loop_id, LoopId):
                raise TypeError("loop_to_attachment_index keys must be LoopId")
            if (
                not isinstance(attachment_index, int)
                or attachment_index < 0
                or attachment_index >= len(self.ordered_vertex_keys)
            ):
                raise ValueError(
                    "loop_to_attachment_index contains an invalid attachment index"
                )
            loop_ids.append(loop_id)
        if len(loop_ids) != len(set(loop_ids)):
            raise ValueError("loop_to_attachment_index contains duplicate LoopId values")

    @property
    def hull_vertex_ids(self) -> Tuple[VertexId, ...]:
        """Compatibility view; UV-split boundary vertices may appear more than once."""

        return tuple(key.vertex_id for key in self.hull_vertex_keys)

    @property
    def ordered_vertex_ids(self) -> Tuple[VertexId, ...]:
        """Compatibility view; UV-split vertices may appear more than once."""

        return tuple(key.vertex_id for key in self.ordered_vertex_keys)

    @property
    def old_to_attachment_index(self) -> Tuple[Tuple[VertexId, int], ...]:
        """Return the legacy one-to-one map only when no UV duplication exists."""

        mapping: list[Tuple[VertexId, int]] = []
        seen: set[VertexId] = set()
        for attachment_index, key in enumerate(self.ordered_vertex_keys):
            if key.vertex_id in seen:
                raise A1AttachmentProjectionError(
                    "UV seam duplication makes old_to_attachment_index one-to-many; "
                    "use attachment_indices_for() or attachment_index_for_loop()"
                )
            seen.add(key.vertex_id)
            mapping.append((key.vertex_id, attachment_index))
        return tuple(mapping)

    def attachment_indices_for(self, vertex_id: VertexId) -> Tuple[int, ...]:
        if not isinstance(vertex_id, VertexId):
            raise TypeError("vertex_id must be VertexId")
        return tuple(
            attachment_index
            for attachment_index, key in enumerate(self.ordered_vertex_keys)
            if key.vertex_id == vertex_id
        )

    def attachment_index_for(
        self,
        vertex_id: VertexId,
        *,
        uv: Tuple[float, float] | None = None,
    ) -> int:
        if not isinstance(vertex_id, VertexId):
            raise TypeError("vertex_id must be VertexId")
        if uv is not None:
            key = A1AttachmentVertexKey(
                vertex_id=vertex_id,
                uv=(float(uv[0]), float(uv[1])),
            )
            try:
                return self.ordered_vertex_keys.index(key)
            except ValueError as exc:
                raise KeyError(f"Unknown attachment vertex key {key}") from exc

        matches = self.attachment_indices_for(vertex_id)
        if not matches:
            raise KeyError(f"Unknown projected vertex {vertex_id.index}")
        if len(matches) != 1:
            raise A1AttachmentProjectionError(
                f"Vertex {vertex_id.index} has {len(matches)} UV-specific attachment "
                "vertices; provide uv or resolve through a LoopId"
            )
        return matches[0]

    def attachment_index_for_loop(self, loop_id: LoopId) -> int:
        if not isinstance(loop_id, LoopId):
            raise TypeError("loop_id must be LoopId")
        mapping = dict(self.loop_to_attachment_index)
        try:
            return mapping[loop_id]
        except KeyError as exc:
            raise KeyError(f"Unknown projected loop {loop_id.index}") from exc


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
        candidates = tuple(
            neighbour for neighbour in neighbours if neighbour != previous
        )
        if len(candidates) != 1:
            raise A1AttachmentProjectionError(
                f"Boundary traversal is ambiguous at vertex {current.index}"
            )
        previous, current = current, candidates[0]
        if len(cycle) > len(adjacency):
            raise A1AttachmentProjectionError(
                "Boundary traversal exceeded vertex count"
            )
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


def _loop_attachment_keys(
    snapshot: MeshSnapshot,
    layer_name: str,
) -> dict[LoopId, A1AttachmentVertexKey]:
    if layer_name not in snapshot.uv_layer_names:
        raise A1AttachmentProjectionError(
            f"UV layer '{layer_name}' is absent from snapshot"
        )

    result: dict[LoopId, A1AttachmentVertexKey] = {}
    for loop in sorted(snapshot.loops, key=lambda item: item.id.index):
        coordinate = loop.uv(layer_name)
        if coordinate is None:
            raise A1AttachmentProjectionError(
                f"Loop {loop.id.index} is missing UV layer '{layer_name}'"
            )
        if loop.id in result:
            raise A1AttachmentProjectionError(
                f"Duplicate local LoopId {loop.id.index} in snapshot"
            )
        result[loop.id] = A1AttachmentVertexKey(
            vertex_id=loop.vertex_id,
            uv=(float(coordinate[0]), float(coordinate[1])),
        )
    return result


def _face_loop_by_vertex(snapshot: MeshSnapshot) -> dict[Tuple[FaceId, VertexId], LoopId]:
    loop_map = snapshot.loop_by_id()
    result: dict[Tuple[FaceId, VertexId], LoopId] = {}
    for face in sorted(snapshot.faces, key=lambda item: item.id.index):
        for loop_id in face.loop_ids:
            vertex_id = loop_map[loop_id].vertex_id
            key = (face.id, vertex_id)
            if key in result:
                raise A1AttachmentProjectionError(
                    f"Face {face.id.index} references vertex {vertex_id.index} more "
                    "than once"
                )
            result[key] = loop_id
    return result


def _normalized_vertex_pair(
    first: VertexId,
    second: VertexId,
) -> Tuple[VertexId, VertexId]:
    if first == second:
        raise A1AttachmentProjectionError(
            f"Degenerate edge references vertex {first.index} twice"
        )
    return (first, second) if first.index < second.index else (second, first)


def _edge_by_vertex_pair(
    snapshot: MeshSnapshot,
) -> dict[Tuple[VertexId, VertexId], EdgeId]:
    result: dict[Tuple[VertexId, VertexId], EdgeId] = {}
    for edge in snapshot.edges:
        pair = _normalized_vertex_pair(*edge.vertex_ids)
        if pair in result:
            raise A1AttachmentProjectionError(
                f"Multiple edges connect vertices {pair[0].index} and {pair[1].index}"
            )
        result[pair] = edge.id
    return result


def _boundary_endpoint_key(
    *,
    first: VertexId,
    second: VertexId,
    endpoint: VertexId,
    edge_by_pair: dict[Tuple[VertexId, VertexId], EdgeId],
    edge_to_faces: dict[EdgeId, Tuple[FaceId, ...]],
    face_loop_by_vertex: dict[Tuple[FaceId, VertexId], LoopId],
    loop_keys: dict[LoopId, A1AttachmentVertexKey],
) -> A1AttachmentVertexKey:
    pair = _normalized_vertex_pair(first, second)
    edge_id = edge_by_pair.get(pair)
    if edge_id is None:
        raise A1AttachmentProjectionError(
            f"Boundary cycle references missing edge {first.index}-{second.index}"
        )
    linked_faces = edge_to_faces.get(edge_id, ())
    if len(linked_faces) != 1:
        raise A1AttachmentProjectionError(
            f"Expected boundary edge {edge_id.index} to have one face, found "
            f"{len(linked_faces)}"
        )
    face_id = linked_faces[0]
    loop_id = face_loop_by_vertex.get((face_id, endpoint))
    if loop_id is None:
        raise A1AttachmentProjectionError(
            f"Boundary face {face_id.index} has no loop for vertex {endpoint.index}"
        )
    return loop_keys[loop_id]


def _ordered_hull_attachment_keys(
    snapshot: MeshSnapshot,
    geometric_cycle: Tuple[VertexId, ...],
    loop_keys: dict[LoopId, A1AttachmentVertexKey],
) -> Tuple[A1AttachmentVertexKey, ...]:
    edge_by_pair = _edge_by_vertex_pair(snapshot)
    edge_to_faces = build_edge_to_faces(snapshot)
    face_loop_by_vertex = _face_loop_by_vertex(snapshot)
    hull: list[A1AttachmentVertexKey] = []

    for index, vertex_id in enumerate(geometric_cycle):
        previous_vertex = geometric_cycle[index - 1]
        next_vertex = geometric_cycle[(index + 1) % len(geometric_cycle)]
        incoming_key = _boundary_endpoint_key(
            first=previous_vertex,
            second=vertex_id,
            endpoint=vertex_id,
            edge_by_pair=edge_by_pair,
            edge_to_faces=edge_to_faces,
            face_loop_by_vertex=face_loop_by_vertex,
            loop_keys=loop_keys,
        )
        outgoing_key = _boundary_endpoint_key(
            first=vertex_id,
            second=next_vertex,
            endpoint=vertex_id,
            edge_by_pair=edge_by_pair,
            edge_to_faces=edge_to_faces,
            face_loop_by_vertex=face_loop_by_vertex,
            loop_keys=loop_keys,
        )
        hull.append(incoming_key)
        if outgoing_key != incoming_key:
            # The UV seam reaches the physical hull. Both keys must remain hull
            # vertices, consecutively, to keep the runtime's ordered hull polygon.
            hull.append(outgoing_key)

    resolved = tuple(hull)
    if len(resolved) != len(set(resolved)):
        raise A1AttachmentProjectionError(
            "UV-specific hull contains a repeated attachment vertex key"
        )
    return resolved


def _ordered_unique_loop_keys(
    snapshot: MeshSnapshot,
    loop_keys: dict[LoopId, A1AttachmentVertexKey],
) -> Tuple[A1AttachmentVertexKey, ...]:
    ordered: list[A1AttachmentVertexKey] = []
    seen: set[A1AttachmentVertexKey] = set()
    for face in sorted(snapshot.faces, key=lambda item: item.id.index):
        for loop_id in face.loop_ids:
            key = loop_keys[loop_id]
            if key in seen:
                continue
            seen.add(key)
            ordered.append(key)
    return tuple(ordered)


def _z_binding_map(
    snapshot: MeshSnapshot,
    settings: A1AttachmentProjectionSettings,
    rig: LegacyRigBuildResult,
) -> dict[VertexId, int]:
    mapping = {
        binding.vertex_id: binding.z_group_index for binding in settings.z_bindings
    }
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


def _attachment_edges(
    snapshot: MeshSnapshot,
    loop_keys: dict[LoopId, A1AttachmentVertexKey],
    key_to_index: dict[A1AttachmentVertexKey, int],
) -> Tuple[int, ...]:
    edges: list[int] = []
    seen_pairs: set[Tuple[int, int]] = set()
    for face in sorted(snapshot.faces, key=lambda item: item.id.index):
        for corner_index, first_loop_id in enumerate(face.loop_ids):
            second_loop_id = face.loop_ids[(corner_index + 1) % len(face.loop_ids)]
            first_index = key_to_index[loop_keys[first_loop_id]]
            second_index = key_to_index[loop_keys[second_loop_id]]
            if first_index == second_index:
                raise A1AttachmentProjectionError(
                    f"Face {face.id.index} produced a degenerate attachment edge"
                )
            normalized = (
                (first_index, second_index)
                if first_index < second_index
                else (second_index, first_index)
            )
            if normalized in seen_pairs:
                continue
            seen_pairs.add(normalized)
            edges.extend((first_index, second_index))
    return tuple(edges)


def project_triangulated_disk_attachment(
    snapshot: MeshSnapshot,
    rig: LegacyRigBuildResult,
    settings: A1AttachmentProjectionSettings,
) -> A1AttachmentProjectionResult:
    """Create a deterministic A1 attachment with exact loop-level UV identity."""

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    if not isinstance(rig, LegacyRigBuildResult):
        raise TypeError("rig must be LegacyRigBuildResult")
    if not isinstance(settings, A1AttachmentProjectionSettings):
        raise TypeError("settings must be A1AttachmentProjectionSettings")

    _validate_triangulated_disk(snapshot)
    rig.validate()
    loop_keys = _loop_attachment_keys(snapshot, settings.uv_layer_name)
    geometric_hull = _deterministic_hull_cycle(snapshot)
    hull_keys = _ordered_hull_attachment_keys(snapshot, geometric_hull, loop_keys)
    hull_set = set(hull_keys)
    all_keys = _ordered_unique_loop_keys(snapshot, loop_keys)
    ordered_keys = hull_keys + tuple(key for key in all_keys if key not in hull_set)
    if set(ordered_keys) != set(all_keys):
        missing = set(all_keys) - set(ordered_keys)
        unknown = set(ordered_keys) - set(all_keys)
        raise A1AttachmentProjectionError(
            f"Attachment key coverage mismatch; missing={missing}, unknown={unknown}"
        )

    key_to_index = {
        key: attachment_index for attachment_index, key in enumerate(ordered_keys)
    }
    z_by_vertex = _z_binding_map(snapshot, settings, rig)
    vertex_map = snapshot.vertex_by_id()

    projected_vertices = tuple(
        LegacyAttachmentVertex(
            index=attachment_index,
            uv=key.uv,
            bone_position_pixels=(
                (
                    float(vertex_map[key.vertex_id].position[0])
                    - float(settings.center_x)
                )
                * rig.info.uniform_scale,
                -(
                    float(vertex_map[key.vertex_id].position[1])
                    - float(settings.center_y)
                )
                * rig.info.uniform_scale,
            ),
            z_group_index=z_by_vertex[key.vertex_id],
        )
        for attachment_index, key in enumerate(ordered_keys)
    )

    face_map = snapshot.face_by_id()
    triangles = tuple(
        key_to_index[loop_keys[loop_id]]
        for face_id in sorted(face_map, key=lambda item: item.index)
        for loop_id in face_map[face_id].loop_ids
    )
    edges = _attachment_edges(snapshot, loop_keys, key_to_index)
    loop_to_attachment_index = tuple(
        (loop_id, key_to_index[key])
        for loop_id, key in sorted(
            loop_keys.items(),
            key=lambda item: item[0].index,
        )
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
        hull=len(hull_keys),
        edges=edges,
        sequence=settings.sequence,
        skin_name=settings.skin_name,
    )
    return A1AttachmentProjectionResult(
        request=request,
        hull_vertex_keys=hull_keys,
        ordered_vertex_keys=ordered_keys,
        loop_to_attachment_index=loop_to_attachment_index,
    )
