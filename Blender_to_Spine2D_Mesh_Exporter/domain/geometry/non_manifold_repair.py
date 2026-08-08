"""Conservative topology repair for non-manifold source edges.

Spine attachments do not require Blender's full manifold connectivity to remain joined.
When one source edge is used by more than two faces, there is no unique manifold pairing
that an exporter may infer safely. The conservative repair is therefore to cut that edge
for every incident face while preserving all geometric/source lineage.

The source Blender mesh is never mutated. Only local ``EdgeId`` identity changes inside a
new immutable ``MeshSnapshot``; duplicated edges retain the same ``SourceEdgeId``, seam
and sharp flags. Vertices, loops, faces, UV payloads and source IDs are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Tuple

from .ids import EdgeId, FaceId, LoopId
from .model import MeshEdge, MeshLoop, MeshSnapshot
from .topology import build_edge_to_faces
from .validator import MeshSnapshotValidator


logger = logging.getLogger(__name__)


class NonManifoldRepairError(ValueError):
    """Raised when an immutable non-manifold edge cut cannot preserve topology."""


@dataclass(frozen=True, slots=True)
class NonManifoldEdgeSplitReport:
    """Diagnostics for one conservative non-manifold edge split."""

    source_snapshot_id: str
    output_snapshot_id: str
    split_edge_ids: Tuple[EdgeId, ...]
    input_edge_count: int
    output_edge_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.source_snapshot_id, str) or not self.source_snapshot_id:
            raise ValueError("source_snapshot_id must be a non-empty string")
        if not isinstance(self.output_snapshot_id, str) or not self.output_snapshot_id:
            raise ValueError("output_snapshot_id must be a non-empty string")
        if not isinstance(self.split_edge_ids, tuple) or not all(
            isinstance(edge_id, EdgeId) for edge_id in self.split_edge_ids
        ):
            raise TypeError("split_edge_ids must contain EdgeId values")
        if len(self.split_edge_ids) != len(set(self.split_edge_ids)):
            raise ValueError("split_edge_ids cannot contain duplicates")
        for field_name in ("input_edge_count", "output_edge_count"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.output_edge_count < self.input_edge_count:
            raise ValueError("output_edge_count cannot be smaller than input_edge_count")

    @property
    def changed(self) -> bool:
        return bool(self.split_edge_ids)

    @property
    def created_edge_count(self) -> int:
        return self.output_edge_count - self.input_edge_count


def _loop_owner_by_id(snapshot: MeshSnapshot) -> dict[LoopId, FaceId]:
    """Return the unique owning face for every local loop."""

    owners: dict[LoopId, FaceId] = {}
    for face in snapshot.faces:
        for loop_id in face.loop_ids:
            previous = owners.get(loop_id)
            if previous is not None and previous != face.id:
                raise NonManifoldRepairError(
                    "A loop is referenced by more than one face; conservative edge "
                    f"splitting is ambiguous: loop={loop_id.index}, "
                    f"faces=({previous.index}, {face.id.index})"
                )
            owners[loop_id] = face.id
    if len(owners) != len(snapshot.loops):
        missing = tuple(
            loop.id.index for loop in snapshot.loops if loop.id not in owners
        )
        raise NonManifoldRepairError(
            f"Cannot repair snapshot with unowned loops: {missing}"
        )
    return owners


def split_non_manifold_edges(
    snapshot: MeshSnapshot,
    *,
    snapshot_id: str | None = None,
) -> tuple[MeshSnapshot, NonManifoldEdgeSplitReport]:
    """Cut every edge used by more than two faces into per-face local edge identities.

    The operation is intentionally maximally conservative. It never guesses which two
    faces should remain connected across a non-manifold edge. Every incident face gets a
    distinct local edge, turning that source edge into an export boundary while keeping
    the same ``SourceEdgeId`` provenance on all copies.
    """

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    if snapshot_id is not None and (
        not isinstance(snapshot_id, str) or not snapshot_id.strip()
    ):
        raise ValueError("snapshot_id must be a non-empty string or None")

    MeshSnapshotValidator().validate_or_raise(snapshot)
    edge_to_faces = build_edge_to_faces(snapshot)
    split_edge_ids = tuple(
        sorted(
            (
                edge_id
                for edge_id, face_ids in edge_to_faces.items()
                if len(face_ids) > 2
            ),
            key=lambda edge_id: edge_id.index,
        )
    )
    resolved_output_id = snapshot_id or snapshot.snapshot_id
    if not split_edge_ids:
        report = NonManifoldEdgeSplitReport(
            source_snapshot_id=snapshot.snapshot_id,
            output_snapshot_id=resolved_output_id,
            split_edge_ids=(),
            input_edge_count=len(snapshot.edges),
            output_edge_count=len(snapshot.edges),
        )
        if resolved_output_id == snapshot.snapshot_id:
            return snapshot, report
        copied = MeshSnapshot(
            snapshot_id=resolved_output_id,
            source_object_id=snapshot.source_object_id,
            object_name=snapshot.object_name,
            vertices=snapshot.vertices,
            edges=snapshot.edges,
            loops=snapshot.loops,
            faces=snapshot.faces,
            uv_layer_names=snapshot.uv_layer_names,
            active_uv_layer=snapshot.active_uv_layer,
            world_matrix=snapshot.world_matrix,
            render_uv_layer=snapshot.render_uv_layer,
        )
        return copied, report

    split_set = set(split_edge_ids)
    loop_owner = _loop_owner_by_id(snapshot)

    # Rebuild the edge domain densely. Ordinary manifold edges keep one local copy.
    # Every non-manifold edge receives one copy per incident face.
    edge_id_map: dict[tuple[EdgeId, FaceId | None], EdgeId] = {}
    rebuilt_edges: list[MeshEdge] = []
    for edge in snapshot.edges:
        if edge.id not in split_set:
            new_id = EdgeId(len(rebuilt_edges))
            edge_id_map[(edge.id, None)] = new_id
            rebuilt_edges.append(
                MeshEdge(
                    id=new_id,
                    source_id=edge.source_id,
                    vertex_ids=edge.vertex_ids,
                    seam=edge.seam,
                    sharp=edge.sharp,
                )
            )
            continue

        incident_faces = edge_to_faces.get(edge.id, ())
        if len(incident_faces) <= 2:
            raise NonManifoldRepairError(
                f"Edge {edge.id.index} was selected for repair without >2 incident faces"
            )
        for face_id in incident_faces:
            new_id = EdgeId(len(rebuilt_edges))
            edge_id_map[(edge.id, face_id)] = new_id
            rebuilt_edges.append(
                MeshEdge(
                    id=new_id,
                    source_id=edge.source_id,
                    vertex_ids=edge.vertex_ids,
                    seam=edge.seam,
                    sharp=edge.sharp,
                )
            )

    rebuilt_loops: list[MeshLoop] = []
    for loop in snapshot.loops:
        owner = loop_owner[loop.id]
        key = (
            (loop.edge_id, owner)
            if loop.edge_id in split_set
            else (loop.edge_id, None)
        )
        replacement_edge_id = edge_id_map.get(key)
        if replacement_edge_id is None:
            raise NonManifoldRepairError(
                "Failed to resolve repaired edge for loop: "
                f"loop={loop.id.index}, edge={loop.edge_id.index}, face={owner.index}"
            )
        rebuilt_loops.append(
            MeshLoop(
                id=loop.id,
                source_id=loop.source_id,
                vertex_id=loop.vertex_id,
                edge_id=replacement_edge_id,
                uvs=loop.uvs,
            )
        )

    repaired = MeshSnapshot(
        snapshot_id=resolved_output_id,
        source_object_id=snapshot.source_object_id,
        object_name=snapshot.object_name,
        vertices=snapshot.vertices,
        edges=tuple(rebuilt_edges),
        loops=tuple(rebuilt_loops),
        faces=snapshot.faces,
        uv_layer_names=snapshot.uv_layer_names,
        active_uv_layer=snapshot.active_uv_layer,
        world_matrix=snapshot.world_matrix,
        render_uv_layer=snapshot.render_uv_layer,
    )
    MeshSnapshotValidator().validate_or_raise(repaired)

    repaired_edge_to_faces = build_edge_to_faces(repaired)
    remaining_non_manifold = tuple(
        sorted(
            (
                (edge_id.index, tuple(face_id.index for face_id in face_ids))
                for edge_id, face_ids in repaired_edge_to_faces.items()
                if len(face_ids) > 2
            )
        )
    )
    if remaining_non_manifold:
        raise NonManifoldRepairError(
            "Non-manifold edge split left >2-face edge incidence: "
            f"{remaining_non_manifold}"
        )

    report = NonManifoldEdgeSplitReport(
        source_snapshot_id=snapshot.snapshot_id,
        output_snapshot_id=repaired.snapshot_id,
        split_edge_ids=split_edge_ids,
        input_edge_count=len(snapshot.edges),
        output_edge_count=len(repaired.edges),
    )
    logger.warning(
        "Split %d non-manifold source edges for '%s' into per-face boundaries: "
        "input_edges=%d output_edges=%d created_edges=%d",
        len(report.split_edge_ids),
        snapshot.source_object_id,
        report.input_edge_count,
        report.output_edge_count,
        report.created_edge_count,
    )
    return repaired, report


__all__ = [
    "NonManifoldEdgeSplitReport",
    "NonManifoldRepairError",
    "split_non_manifold_edges",
]
