"""Reusable topology analysis for immutable face regions.

The legacy implementation calculated ``holes = max(0, 1 - Euler)`` in several
places. That is insufficient for disconnected, closed, or non-manifold input.
This module keeps the calculation in one deterministic implementation and
reports both Euler characteristic and boundary-component count.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable, Tuple

from .ids import EdgeId, FaceId, VertexId
from .model import MeshSnapshot
from .segmentation import SegmentTopology
from .validator import MeshSnapshotValidator


class RegionTopologyError(ValueError):
    """Raised when a requested face region cannot be analysed safely."""


def build_edge_to_faces(snapshot: MeshSnapshot) -> dict[EdgeId, Tuple[FaceId, ...]]:
    """Validate once and build a stable edge-to-face map from loop connectivity."""

    MeshSnapshotValidator().validate_or_raise(snapshot)
    loop_map = snapshot.loop_by_id()
    mutable: dict[EdgeId, list[FaceId]] = defaultdict(list)
    for face in snapshot.faces:
        for loop_id in face.loop_ids:
            edge_id = loop_map[loop_id].edge_id
            if face.id not in mutable[edge_id]:
                mutable[edge_id].append(face.id)
    return {
        edge_id: tuple(sorted(face_ids, key=lambda item: item.index))
        for edge_id, face_ids in mutable.items()
    }


def face_edge_ids(snapshot: MeshSnapshot, face_id: FaceId) -> Tuple[EdgeId, ...]:
    """Return the ordered unique edge IDs used by one face."""

    face = snapshot.face_by_id().get(face_id)
    if face is None:
        raise RegionTopologyError(f"Unknown face id {face_id.index}")
    loop_map = snapshot.loop_by_id()
    ordered: list[EdgeId] = []
    seen: set[EdgeId] = set()
    for loop_id in face.loop_ids:
        edge_id = loop_map[loop_id].edge_id
        if edge_id not in seen:
            seen.add(edge_id)
            ordered.append(edge_id)
    return tuple(ordered)


def _boundary_component_count(
    boundary_edge_ids: Iterable[EdgeId],
    snapshot: MeshSnapshot,
) -> tuple[int, bool]:
    edge_map = snapshot.edge_by_id()
    adjacency: dict[VertexId, set[VertexId]] = defaultdict(set)
    for edge_id in boundary_edge_ids:
        edge = edge_map[edge_id]
        first, second = edge.vertex_ids
        adjacency[first].add(second)
        adjacency[second].add(first)

    if not adjacency:
        return 0, True

    # A manifold polygonal boundary is a collection of closed cycles. Every
    # boundary vertex therefore has degree exactly two inside the boundary graph.
    boundary_manifold = all(len(neighbours) == 2 for neighbours in adjacency.values())
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
    return component_count, boundary_manifold


def analyse_face_region(
    snapshot: MeshSnapshot,
    face_ids: Iterable[FaceId],
    *,
    edge_to_faces: dict[EdgeId, Tuple[FaceId, ...]] | None = None,
) -> SegmentTopology:
    """Calculate topology invariants for a connected or disconnected face subset.

    Passing a precomputed ``edge_to_faces`` map means the caller has already
    validated the snapshot. This is used by decomposition, where hundreds of
    candidate regions may be analysed against one immutable snapshot.
    """

    if edge_to_faces is None:
        resolved_edge_to_faces = build_edge_to_faces(snapshot)
    else:
        resolved_edge_to_faces = edge_to_faces

    ordered_face_ids = tuple(sorted(set(face_ids), key=lambda item: item.index))
    if not ordered_face_ids:
        raise RegionTopologyError("face_ids cannot be empty")

    face_map = snapshot.face_by_id()
    unknown = [face_id.index for face_id in ordered_face_ids if face_id not in face_map]
    if unknown:
        raise RegionTopologyError(f"Unknown face ids: {unknown}")

    loop_map = snapshot.loop_by_id()
    region_face_set = set(ordered_face_ids)
    loop_ids = {
        loop_id
        for face_id in ordered_face_ids
        for loop_id in face_map[face_id].loop_ids
    }
    edge_ids = {loop_map[loop_id].edge_id for loop_id in loop_ids}
    vertex_ids = {loop_map[loop_id].vertex_id for loop_id in loop_ids}

    region_link_count = {
        edge_id: sum(
            face_id in region_face_set
            for face_id in resolved_edge_to_faces.get(edge_id, ())
        )
        for edge_id in edge_ids
    }
    boundary_edge_ids = tuple(
        sorted(
            (edge_id for edge_id, count in region_link_count.items() if count == 1),
            key=lambda item: item.index,
        )
    )
    boundary_components, boundary_manifold = _boundary_component_count(
        boundary_edge_ids,
        snapshot,
    )

    # An edge is considered safe only when the source mesh itself and the region
    # both use it with ordinary manifold multiplicity.
    edge_manifold = all(
        1 <= region_link_count[edge_id] <= 2
        and len(resolved_edge_to_faces.get(edge_id, ())) <= 2
        for edge_id in edge_ids
    )

    return SegmentTopology(
        vertex_count=len(vertex_ids),
        edge_count=len(edge_ids),
        face_count=len(ordered_face_ids),
        euler_characteristic=len(vertex_ids) - len(edge_ids) + len(ordered_face_ids),
        boundary_edge_count=len(boundary_edge_ids),
        boundary_component_count=boundary_components,
        manifold=edge_manifold and boundary_manifold,
    )


def is_simple_disk(topology: SegmentTopology) -> bool:
    """Return whether a face region is a manifold topological disk."""

    if not isinstance(topology, SegmentTopology):
        raise TypeError("topology must be SegmentTopology")
    return (
        topology.manifold
        and topology.euler_characteristic == 1
        and topology.boundary_component_count == 1
    )


def build_face_adjacency(
    snapshot: MeshSnapshot,
    face_ids: Iterable[FaceId],
    *,
    blocked_edge_ids: Iterable[EdgeId] = (),
    edge_to_faces: dict[EdgeId, Tuple[FaceId, ...]] | None = None,
) -> dict[FaceId, Tuple[FaceId, ...]]:
    """Build deterministic adjacency for a face subset while respecting cuts."""

    ordered_face_ids = tuple(sorted(set(face_ids), key=lambda item: item.index))
    region = set(ordered_face_ids)
    blocked = set(blocked_edge_ids)
    resolved_edge_to_faces = edge_to_faces or build_edge_to_faces(snapshot)
    adjacency: dict[FaceId, set[FaceId]] = {
        face_id: set() for face_id in ordered_face_ids
    }

    for edge_id, linked_faces in resolved_edge_to_faces.items():
        if edge_id in blocked:
            continue
        inside = [face_id for face_id in linked_faces if face_id in region]
        if len(inside) != 2:
            continue
        first, second = inside
        adjacency[first].add(second)
        adjacency[second].add(first)

    return {
        face_id: tuple(sorted(neighbours, key=lambda item: item.index))
        for face_id, neighbours in adjacency.items()
    }
