"""Incremental topology state for hot manifold-disk growth and merge loops."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Tuple

from .ids import EdgeId, FaceId, VertexId
from .model import MeshSnapshot
from .segmentation import SegmentTopology
from .topology import RegionTopologyError, build_edge_to_faces, is_simple_disk


class DiskTopologyIndex:
    """Precomputed immutable connectivity used by all disk states of one snapshot."""

    __slots__ = (
        "_snapshot",
        "_edge_to_faces",
        "_edge_map",
        "_face_edges",
        "_face_vertices",
    )

    def __init__(
        self,
        snapshot: MeshSnapshot,
        *,
        edge_to_faces: dict[EdgeId, Tuple[FaceId, ...]] | None = None,
    ) -> None:
        if not isinstance(snapshot, MeshSnapshot):
            raise TypeError("snapshot must be MeshSnapshot")
        self._snapshot = snapshot
        self._edge_to_faces = (
            build_edge_to_faces(snapshot)
            if edge_to_faces is None
            else edge_to_faces
        )
        self._edge_map = snapshot.edge_by_id()
        loop_map = snapshot.loop_by_id()
        face_edges: dict[FaceId, Tuple[EdgeId, ...]] = {}
        face_vertices: dict[FaceId, Tuple[VertexId, ...]] = {}
        for face in snapshot.faces:
            ordered_edges: list[EdgeId] = []
            ordered_vertices: list[VertexId] = []
            seen_edges: set[EdgeId] = set()
            seen_vertices: set[VertexId] = set()
            for loop_id in face.loop_ids:
                loop = loop_map[loop_id]
                if loop.edge_id not in seen_edges:
                    seen_edges.add(loop.edge_id)
                    ordered_edges.append(loop.edge_id)
                if loop.vertex_id not in seen_vertices:
                    seen_vertices.add(loop.vertex_id)
                    ordered_vertices.append(loop.vertex_id)
            face_edges[face.id] = tuple(ordered_edges)
            face_vertices[face.id] = tuple(ordered_vertices)
        self._face_edges = face_edges
        self._face_vertices = face_vertices

    @property
    def snapshot(self) -> MeshSnapshot:
        return self._snapshot

    @property
    def edge_to_faces(self) -> dict[EdgeId, Tuple[FaceId, ...]]:
        return self._edge_to_faces

    @property
    def edge_map(self) -> dict:
        return self._edge_map

    def face_edge_ids(self, face_id: FaceId) -> Tuple[EdgeId, ...]:
        try:
            return self._face_edges[face_id]
        except KeyError as exc:
            raise RegionTopologyError(f"Unknown face id {face_id.index}") from exc

    def face_vertex_ids(self, face_id: FaceId) -> Tuple[VertexId, ...]:
        try:
            return self._face_vertices[face_id]
        except KeyError as exc:
            raise RegionTopologyError(f"Unknown face id {face_id.index}") from exc


@dataclass(frozen=True, slots=True)
class DiskRegionAddition:
    """A topology-checked local face-addition delta for one disk state revision."""

    revision: int
    face_id: FaceId
    edge_ids: Tuple[EdgeId, ...]
    vertex_ids: Tuple[VertexId, ...]
    boundary_edges_to_add: Tuple[EdgeId, ...]
    boundary_edges_to_remove: Tuple[EdgeId, ...]
    boundary_degree_deltas: Tuple[tuple[VertexId, int], ...]
    topology: SegmentTopology


def _cyclic_positions_form_single_proper_run(
    positions: set[int],
    size: int,
) -> bool:
    """Return whether selected cyclic positions form one non-empty proper interval."""

    if not positions or len(positions) >= size:
        return False
    selected = [index in positions for index in range(size)]
    starts = sum(
        selected[index] and not selected[index - 1]
        for index in range(size)
    )
    return starts == 1


def _edge_set_is_open_path(
    edge_ids: Iterable[EdgeId],
    edge_map: dict,
) -> bool:
    """Return whether distinct edges form one connected non-cyclic manifold path.

    Vertex degree is counted by incident ``EdgeId`` values rather than by unique
    neighbouring vertices. This matters for malformed or non-manifold snapshots
    containing parallel edges between the same two vertices: two parallel edges
    form a cycle of length two, not one open edge.
    """

    resolved = tuple(edge_ids)
    if not resolved or len(resolved) != len(set(resolved)):
        return False

    adjacency: dict[VertexId, list[tuple[EdgeId, VertexId]]] = defaultdict(list)
    for edge_id in resolved:
        edge = edge_map.get(edge_id)
        if edge is None:
            return False
        first, second = edge.vertex_ids
        adjacency[first].append((edge_id, second))
        adjacency[second].append((edge_id, first))

    if any(len(links) > 2 for links in adjacency.values()):
        return False
    endpoints = [
        vertex_id
        for vertex_id, links in adjacency.items()
        if len(links) == 1
    ]
    if len(endpoints) != 2:
        return False

    visited_vertices = {endpoints[0]}
    visited_edges: set[EdgeId] = set()
    queue = deque([endpoints[0]])
    while queue:
        current = queue.popleft()
        for edge_id, neighbour in adjacency[current]:
            if edge_id in visited_edges:
                continue
            visited_edges.add(edge_id)
            if neighbour not in visited_vertices:
                visited_vertices.add(neighbour)
                queue.append(neighbour)
    return (
        len(visited_edges) == len(resolved)
        and len(visited_vertices) == len(adjacency)
    )


class DiskRegionState:
    """Mutable incremental state for a connected manifold topological disk.

    Growth starts from one valid face. A candidate face is accepted only when its
    intersection with the current disk is one proper boundary-edge interval and
    it has no additional vertex-only contact. Candidate checks then touch only
    the candidate corners and cached boundary state, not every face in the region.
    """

    __slots__ = (
        "_index",
        "_face_ids",
        "_edge_face_counts",
        "_vertex_ids",
        "_boundary_edge_ids",
        "_boundary_degrees",
        "_invalid_edge_count",
        "_invalid_boundary_degree_count",
        "_minimum_face_index",
        "_maximum_face_index",
        "_revision",
    )

    def __init__(
        self,
        topology_index: DiskTopologyIndex,
        face_ids: Iterable[FaceId],
        edge_face_counts: dict[EdgeId, int],
        vertex_ids: Iterable[VertexId],
        boundary_edge_ids: Iterable[EdgeId],
        boundary_degrees: dict[VertexId, int],
        *,
        revision: int = 0,
    ) -> None:
        if not isinstance(topology_index, DiskTopologyIndex):
            raise TypeError("topology_index must be DiskTopologyIndex")
        resolved_faces = set(face_ids)
        if not resolved_faces:
            raise RegionTopologyError("DiskRegionState requires at least one face")
        self._index = topology_index
        self._face_ids = resolved_faces
        self._edge_face_counts = dict(edge_face_counts)
        self._vertex_ids = set(vertex_ids)
        self._boundary_edge_ids = set(boundary_edge_ids)
        self._boundary_degrees = {
            vertex_id: degree
            for vertex_id, degree in boundary_degrees.items()
            if degree
        }
        self._invalid_edge_count = sum(
            not self._edge_incidence_is_valid(edge_id, count)
            for edge_id, count in self._edge_face_counts.items()
        )
        self._invalid_boundary_degree_count = sum(
            degree != 2 for degree in self._boundary_degrees.values()
        )
        self._minimum_face_index = min(face_id.index for face_id in resolved_faces)
        self._maximum_face_index = max(face_id.index for face_id in resolved_faces)
        self._revision = revision

    def _edge_incidence_is_valid(self, edge_id: EdgeId, count: int) -> bool:
        return (
            1 <= count <= 2
            and len(self._index.edge_to_faces.get(edge_id, ())) <= 2
        )

    @classmethod
    def from_face(
        cls,
        snapshot: MeshSnapshot,
        face_id: FaceId,
        *,
        edge_to_faces: dict[EdgeId, Tuple[FaceId, ...]] | None = None,
        topology_index: DiskTopologyIndex | None = None,
    ) -> "DiskRegionState":
        """Create a disk state from one validated polygon face."""

        if not isinstance(snapshot, MeshSnapshot):
            raise TypeError("snapshot must be MeshSnapshot")
        if not isinstance(face_id, FaceId):
            raise TypeError("face_id must be FaceId")
        if topology_index is not None and edge_to_faces is not None:
            raise ValueError("pass topology_index or edge_to_faces, not both")
        resolved_index = topology_index or DiskTopologyIndex(
            snapshot,
            edge_to_faces=edge_to_faces,
        )
        if resolved_index.snapshot is not snapshot:
            raise RegionTopologyError("topology_index belongs to another snapshot")

        edge_ids = resolved_index.face_edge_ids(face_id)
        vertex_ids = resolved_index.face_vertex_ids(face_id)
        if any(
            len(resolved_index.edge_to_faces.get(edge_id, ())) > 2
            for edge_id in edge_ids
        ):
            raise RegionTopologyError(
                f"Face {face_id.index} touches a non-manifold source edge"
            )
        boundary_degrees: dict[VertexId, int] = defaultdict(int)
        for edge_id in edge_ids:
            first, second = resolved_index.edge_map[edge_id].vertex_ids
            boundary_degrees[first] += 1
            boundary_degrees[second] += 1
        state = cls(
            resolved_index,
            (face_id,),
            {edge_id: 1 for edge_id in edge_ids},
            vertex_ids,
            edge_ids,
            dict(boundary_degrees),
        )
        if not is_simple_disk(state.topology):
            raise RegionTopologyError(
                f"Face {face_id.index} is not a valid manifold disk by itself"
            )
        return state

    @property
    def face_ids(self) -> Tuple[FaceId, ...]:
        return tuple(sorted(self._face_ids, key=lambda item: item.index))

    @property
    def face_count(self) -> int:
        return len(self._face_ids)

    @property
    def minimum_face_index(self) -> int:
        return self._minimum_face_index

    @property
    def maximum_face_index(self) -> int:
        return self._maximum_face_index

    @property
    def boundary_edge_ids(self) -> Tuple[EdgeId, ...]:
        return tuple(sorted(self._boundary_edge_ids, key=lambda item: item.index))

    @property
    def topology(self) -> SegmentTopology:
        return SegmentTopology(
            vertex_count=len(self._vertex_ids),
            edge_count=len(self._edge_face_counts),
            face_count=len(self._face_ids),
            euler_characteristic=(
                len(self._vertex_ids)
                - len(self._edge_face_counts)
                + len(self._face_ids)
            ),
            boundary_edge_count=len(self._boundary_edge_ids),
            boundary_component_count=1 if self._boundary_edge_ids else 0,
            manifold=(
                self._invalid_edge_count == 0
                and self._invalid_boundary_degree_count == 0
            ),
        )

    def preview_add_face(self, face_id: FaceId) -> DiskRegionAddition | None:
        """Return an applicable local delta, or ``None`` when union is not a disk."""

        if not isinstance(face_id, FaceId):
            raise TypeError("face_id must be FaceId")
        if face_id in self._face_ids:
            return None

        edge_ids = self._index.face_edge_ids(face_id)
        vertex_ids = self._index.face_vertex_ids(face_id)
        edge_map = self._index.edge_map
        current_counts = tuple(
            self._edge_face_counts.get(edge_id, 0) for edge_id in edge_ids
        )
        if any(count >= 2 for count in current_counts):
            return None
        if any(
            len(self._index.edge_to_faces.get(edge_id, ())) > 2
            for edge_id in edge_ids
        ):
            return None

        shared_positions = {
            index for index, count in enumerate(current_counts) if count == 1
        }
        if not _cyclic_positions_form_single_proper_run(
            shared_positions,
            len(edge_ids),
        ):
            return None

        boundary_edges_to_remove = tuple(
            edge_ids[index] for index in sorted(shared_positions)
        )
        boundary_edges_to_add = tuple(
            edge_id
            for edge_id, count in zip(edge_ids, current_counts)
            if count == 0
        )
        interface_vertices = {
            vertex_id
            for edge_id in boundary_edges_to_remove
            for vertex_id in edge_map[edge_id].vertex_ids
        }
        candidate_vertex_set = set(vertex_ids)
        # Additional vertex-only contact creates a pinched/non-manifold union.
        if candidate_vertex_set & self._vertex_ids != interface_vertices:
            return None

        degree_deltas: dict[VertexId, int] = defaultdict(int)
        for edge_id in boundary_edges_to_remove:
            for vertex_id in edge_map[edge_id].vertex_ids:
                degree_deltas[vertex_id] -= 1
        for edge_id in boundary_edges_to_add:
            for vertex_id in edge_map[edge_id].vertex_ids:
                degree_deltas[vertex_id] += 1
        for vertex_id, delta in degree_deltas.items():
            resulting_degree = self._boundary_degrees.get(vertex_id, 0) + delta
            if resulting_degree not in (0, 2):
                return None

        resulting_vertex_count = len(self._vertex_ids) + sum(
            vertex_id not in self._vertex_ids for vertex_id in vertex_ids
        )
        resulting_edge_count = (
            len(self._edge_face_counts) + len(boundary_edges_to_add)
        )
        resulting_face_count = len(self._face_ids) + 1
        resulting_euler = (
            resulting_vertex_count - resulting_edge_count + resulting_face_count
        )
        resulting_boundary_count = (
            len(self._boundary_edge_ids)
            + len(boundary_edges_to_add)
            - len(boundary_edges_to_remove)
        )
        if resulting_euler != 1 or resulting_boundary_count <= 0:
            return None

        return DiskRegionAddition(
            revision=self._revision,
            face_id=face_id,
            edge_ids=edge_ids,
            vertex_ids=vertex_ids,
            boundary_edges_to_add=boundary_edges_to_add,
            boundary_edges_to_remove=boundary_edges_to_remove,
            boundary_degree_deltas=tuple(
                sorted(degree_deltas.items(), key=lambda item: item[0].index)
            ),
            topology=SegmentTopology(
                vertex_count=resulting_vertex_count,
                edge_count=resulting_edge_count,
                face_count=resulting_face_count,
                euler_characteristic=resulting_euler,
                boundary_edge_count=resulting_boundary_count,
                boundary_component_count=1,
                manifold=True,
            ),
        )

    def apply_addition(self, addition: DiskRegionAddition) -> None:
        """Apply a delta produced by :meth:`preview_add_face`."""

        if not isinstance(addition, DiskRegionAddition):
            raise TypeError("addition must be DiskRegionAddition")
        if addition.revision != self._revision:
            raise RegionTopologyError("DiskRegionAddition belongs to a stale state")
        if addition.face_id in self._face_ids:
            raise RegionTopologyError(
                f"Face {addition.face_id.index} is already present in disk state"
            )

        self._face_ids.add(addition.face_id)
        self._minimum_face_index = min(
            self._minimum_face_index,
            addition.face_id.index,
        )
        self._maximum_face_index = max(
            self._maximum_face_index,
            addition.face_id.index,
        )
        self._vertex_ids.update(addition.vertex_ids)
        for edge_id in addition.edge_ids:
            previous_count = self._edge_face_counts.get(edge_id, 0)
            previous_invalid = not self._edge_incidence_is_valid(
                edge_id,
                previous_count,
            ) if previous_count else False
            resulting_count = previous_count + 1
            resulting_invalid = not self._edge_incidence_is_valid(
                edge_id,
                resulting_count,
            )
            self._invalid_edge_count += int(resulting_invalid) - int(previous_invalid)
            self._edge_face_counts[edge_id] = resulting_count

        self._boundary_edge_ids.difference_update(
            addition.boundary_edges_to_remove
        )
        self._boundary_edge_ids.update(addition.boundary_edges_to_add)
        for vertex_id, delta in addition.boundary_degree_deltas:
            previous_degree = self._boundary_degrees.get(vertex_id, 0)
            previous_invalid = previous_degree not in (0, 2)
            resulting_degree = previous_degree + delta
            resulting_invalid = resulting_degree not in (0, 2)
            self._invalid_boundary_degree_count += (
                int(resulting_invalid) - int(previous_invalid)
            )
            if resulting_degree:
                self._boundary_degrees[vertex_id] = resulting_degree
            else:
                self._boundary_degrees.pop(vertex_id, None)
        self._revision += 1

        if self.topology != addition.topology:
            raise RegionTopologyError(
                "Incremental disk topology drifted from the validated addition"
            )

    def _merge_interface(
        self,
        other: "DiskRegionState",
    ) -> Tuple[EdgeId, ...] | None:
        if not isinstance(other, DiskRegionState):
            raise TypeError("other must be DiskRegionState")
        if self._index.snapshot is not other._index.snapshot:
            return None
        if self._face_ids & other._face_ids:
            return None

        shared_boundary_edges = (
            self._boundary_edge_ids & other._boundary_edge_ids
        )
        if not shared_boundary_edges:
            return None

        edge_map = self._index.edge_map
        if not _edge_set_is_open_path(shared_boundary_edges, edge_map):
            return None
        interface_vertices = {
            vertex_id
            for edge_id in shared_boundary_edges
            for vertex_id in edge_map[edge_id].vertex_ids
        }
        # Two regions touching at any extra vertex would create a pinch after merge.
        if self._vertex_ids & other._vertex_ids != interface_vertices:
            return None

        shared_incidence: dict[VertexId, int] = defaultdict(int)
        for edge_id in shared_boundary_edges:
            for vertex_id in edge_map[edge_id].vertex_ids:
                shared_incidence[vertex_id] += 1
        for vertex_id in interface_vertices:
            resulting_degree = (
                self._boundary_degrees.get(vertex_id, 0)
                + other._boundary_degrees.get(vertex_id, 0)
                - 2 * shared_incidence[vertex_id]
            )
            if resulting_degree not in (0, 2):
                return None

        resulting_vertex_count = (
            len(self._vertex_ids)
            + len(other._vertex_ids)
            - len(interface_vertices)
        )
        resulting_edge_count = (
            len(self._edge_face_counts)
            + len(other._edge_face_counts)
            - len(shared_boundary_edges)
        )
        resulting_face_count = len(self._face_ids) + len(other._face_ids)
        if (
            resulting_vertex_count
            - resulting_edge_count
            + resulting_face_count
            != 1
        ):
            return None
        resulting_boundary_count = (
            len(self._boundary_edge_ids)
            + len(other._boundary_edge_ids)
            - 2 * len(shared_boundary_edges)
        )
        if resulting_boundary_count <= 0:
            return None
        return tuple(
            sorted(shared_boundary_edges, key=lambda item: item.index)
        )

    def merge_compatibility(self, other: "DiskRegionState") -> int:
        """Return shared interface edge count when two disks can merge, else zero."""

        interface = self._merge_interface(other)
        return len(interface) if interface is not None else 0

    def merged_with(self, other: "DiskRegionState") -> "DiskRegionState" | None:
        """Return the incrementally merged disk, or ``None`` when incompatible."""

        interface = self._merge_interface(other)
        if interface is None:
            return None

        edge_face_counts = dict(self._edge_face_counts)
        for edge_id, count in other._edge_face_counts.items():
            edge_face_counts[edge_id] = edge_face_counts.get(edge_id, 0) + count
        boundary_edge_ids = self._boundary_edge_ids ^ other._boundary_edge_ids
        edge_map = self._index.edge_map
        boundary_degrees: dict[VertexId, int] = defaultdict(int)
        for edge_id in boundary_edge_ids:
            first, second = edge_map[edge_id].vertex_ids
            boundary_degrees[first] += 1
            boundary_degrees[second] += 1

        merged = DiskRegionState(
            self._index,
            self._face_ids | other._face_ids,
            edge_face_counts,
            self._vertex_ids | other._vertex_ids,
            boundary_edge_ids,
            dict(boundary_degrees),
        )
        if not is_simple_disk(merged.topology):
            raise RegionTopologyError(
                "Incremental region merge produced a non-disk topology"
            )
        return merged
