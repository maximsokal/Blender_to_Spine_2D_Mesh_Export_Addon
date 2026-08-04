"""Canonical evaluated-topology identity for modifier-derived Mesh snapshots.

Blender modifiers such as Array may duplicate one stamped source element several times.
That propagation is useful for proving provenance, but duplicate ``Source*Id`` values are
not a safe working identity for segmentation, depth adjacency, UV transfer, or virtual
view face ownership: unrelated modifier copies could otherwise collapse into one logical
surface.

This module runs only after the evaluated-lineage report has been validated. It preserves
all evaluated geometry and material/UV data while rebasing source identities to the dense
local evaluated topology. The original Blender object and Mesh datablocks are never
mutated.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import Iterable

from .ids import (
    LoopId,
    SourceEdgeId,
    SourceFaceId,
    SourceLoopId,
    SourceVertexId,
)
from .model import MeshSnapshot
from .validator import MeshSnapshotValidator


logger = logging.getLogger(__name__)


class EvaluatedIdentityRebaseError(ValueError):
    """Raised when an evaluated snapshot cannot receive coherent local identity."""


@dataclass(frozen=True, slots=True)
class EvaluatedIdentityRebaseResult:
    """Rebased snapshot and diagnostics about collisions in the incoming lineage."""

    snapshot: MeshSnapshot
    changed: bool
    duplicate_vertex_source_id_count: int
    duplicate_edge_source_id_count: int
    duplicate_face_source_id_count: int
    duplicate_loop_source_id_count: int
    missing_edge_source_id_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, MeshSnapshot):
            raise TypeError("snapshot must be MeshSnapshot")
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        for field_name in (
            "duplicate_vertex_source_id_count",
            "duplicate_edge_source_id_count",
            "duplicate_face_source_id_count",
            "duplicate_loop_source_id_count",
            "missing_edge_source_id_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative int")



def _duplicate_occurrence_count(values: Iterable[object | None]) -> int:
    """Count repeated occurrences, ignoring optional ``None`` edge lineage."""

    known = tuple(value for value in values if value is not None)
    return len(known) - len(set(known))



def _loop_local_identity(
    snapshot: MeshSnapshot,
) -> dict[LoopId, tuple[int, int]]:
    """Map every local loop to ``(local face index, local corner index)`` exactly once."""

    ownership: dict[LoopId, tuple[int, int]] = {}
    for face in snapshot.faces:
        for corner_index, loop_id in enumerate(face.loop_ids):
            previous = ownership.get(loop_id)
            if previous is not None:
                raise EvaluatedIdentityRebaseError(
                    "Evaluated loop is referenced by more than one face; "
                    f"loop={loop_id.index}, first={previous}, "
                    f"second={(face.id.index, corner_index)}"
                )
            ownership[loop_id] = (face.id.index, corner_index)

    expected = {loop.id for loop in snapshot.loops}
    actual = set(ownership)
    if actual != expected:
        missing = tuple(sorted(loop_id.index for loop_id in expected - actual))
        unknown = tuple(sorted(loop_id.index for loop_id in actual - expected))
        raise EvaluatedIdentityRebaseError(
            "Evaluated loop ownership is incomplete; "
            f"missing={missing}, unknown={unknown}"
        )
    return ownership



def rebase_mesh_snapshot_to_evaluated_identity(
    snapshot: MeshSnapshot,
) -> EvaluatedIdentityRebaseResult:
    """Use dense evaluated topology as the canonical source identity.

    This function must be called only after modifier-lineage validation. Geometry,
    normals, materials, UV coordinates, local IDs, object metadata, and ``world_matrix``
    remain byte-for-byte equivalent. Only ``Source*Id`` values are canonicalized.
    """

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")

    validator = MeshSnapshotValidator()
    validator.validate_or_raise(snapshot)

    try:
        loop_identity = _loop_local_identity(snapshot)
        object_id = snapshot.source_object_id

        vertices = tuple(
            replace(
                vertex,
                source_id=SourceVertexId(object_id, vertex.id.index),
            )
            for vertex in snapshot.vertices
        )
        edges = tuple(
            replace(
                edge,
                source_id=SourceEdgeId(object_id, edge.id.index),
            )
            for edge in snapshot.edges
        )
        loops = tuple(
            replace(
                loop,
                source_id=SourceLoopId(
                    object_id,
                    loop_identity[loop.id][0],
                    loop_identity[loop.id][1],
                ),
            )
            for loop in snapshot.loops
        )
        faces = tuple(
            replace(
                face,
                source_id=SourceFaceId(object_id, face.id.index),
            )
            for face in snapshot.faces
        )

        changed = (
            vertices != snapshot.vertices
            or edges != snapshot.edges
            or loops != snapshot.loops
            or faces != snapshot.faces
        )
        resolved_snapshot = (
            replace(
                snapshot,
                vertices=vertices,
                edges=edges,
                loops=loops,
                faces=faces,
            )
            if changed
            else snapshot
        )
        validator.validate_or_raise(resolved_snapshot)

        result = EvaluatedIdentityRebaseResult(
            snapshot=resolved_snapshot,
            changed=changed,
            duplicate_vertex_source_id_count=_duplicate_occurrence_count(
                vertex.source_id for vertex in snapshot.vertices
            ),
            duplicate_edge_source_id_count=_duplicate_occurrence_count(
                edge.source_id for edge in snapshot.edges
            ),
            duplicate_face_source_id_count=_duplicate_occurrence_count(
                face.source_id for face in snapshot.faces
            ),
            duplicate_loop_source_id_count=_duplicate_occurrence_count(
                loop.source_id for loop in snapshot.loops
            ),
            missing_edge_source_id_count=sum(
                edge.source_id is None for edge in snapshot.edges
            ),
        )
        logger.info(
            "Canonicalized evaluated identity for '%s': changed=%s "
            "vertices=%d edges=%d loops=%d faces=%d collisions=(%d,%d,%d,%d) "
            "generated_edges=%d",
            snapshot.source_object_id,
            result.changed,
            len(snapshot.vertices),
            len(snapshot.edges),
            len(snapshot.loops),
            len(snapshot.faces),
            result.duplicate_vertex_source_id_count,
            result.duplicate_edge_source_id_count,
            result.duplicate_loop_source_id_count,
            result.duplicate_face_source_id_count,
            result.missing_edge_source_id_count,
        )
        return result
    except EvaluatedIdentityRebaseError:
        raise
    except Exception as exc:
        logger.exception(
            "Failed to canonicalize evaluated topology identity for '%s'",
            snapshot.source_object_id,
        )
        raise EvaluatedIdentityRebaseError(
            "Unable to canonicalize evaluated topology identity for "
            f"'{snapshot.source_object_id}': {exc}"
        ) from exc


__all__ = [
    "EvaluatedIdentityRebaseError",
    "EvaluatedIdentityRebaseResult",
    "rebase_mesh_snapshot_to_evaluated_identity",
]
