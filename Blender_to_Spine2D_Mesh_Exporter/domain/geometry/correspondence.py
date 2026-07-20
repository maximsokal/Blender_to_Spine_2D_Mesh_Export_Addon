"""Exact geometry and UV correspondence based on stable source lineage."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Iterable, Tuple

from .contracts import (
    require_exact_type,
    require_finite_number,
    require_finite_vector,
    require_identity,
    require_integer,
    require_non_empty_string,
    require_tuple_items,
)
from .ids import EdgeId, FaceId, LoopId, SourceLoopId, VertexId
from .model import LoopUV, MeshEdge, MeshFace, MeshLoop, MeshSnapshot, MeshVertex, Vector2
from .validator import MeshSnapshotValidator


class CorrespondenceError(ValueError):
    """Base class for exact source-lineage correspondence failures."""


class MissingSourceLoopError(CorrespondenceError):
    def __init__(self, source_loop_ids: Iterable[SourceLoopId]):
        resolved = tuple(source_loop_ids)
        for index, source_loop_id in enumerate(resolved):
            require_exact_type(
                source_loop_id,
                SourceLoopId,
                f"source_loop_ids[{index}]",
            )
        self.source_loop_ids = tuple(sorted(set(resolved)))
        super().__init__(
            "No source UV was found for SourceLoopId values: "
            + ", ".join(
                f"{item.object_id}:{item.face_index}:{item.corner_index}"
                for item in self.source_loop_ids
            )
        )


class ConflictingSourceLoopUVError(CorrespondenceError):
    def __init__(self, source_loop_id: SourceLoopId, first: Vector2, second: Vector2):
        require_exact_type(source_loop_id, SourceLoopId, "source_loop_id")
        require_finite_vector(first, 2, "first")
        require_finite_vector(second, 2, "second")
        self.source_loop_id = source_loop_id
        self.first = first
        self.second = second
        super().__init__(
            "Conflicting UV coordinates for SourceLoopId "
            f"{source_loop_id.object_id}:{source_loop_id.face_index}:"
            f"{source_loop_id.corner_index}: {first} != {second}"
        )


@dataclass(frozen=True, slots=True)
class SourceLoopUV:
    source_loop_id: SourceLoopId
    coordinate: Vector2

    def __post_init__(self) -> None:
        require_exact_type(self.source_loop_id, SourceLoopId, "source_loop_id")
        require_finite_vector(self.coordinate, 2, "coordinate")


@dataclass(frozen=True, slots=True)
class UvCorrespondenceMap:
    layer_name: str
    entries: Tuple[SourceLoopUV, ...]

    def __post_init__(self) -> None:
        require_non_empty_string(self.layer_name, "layer_name")
        require_tuple_items(self.entries, SourceLoopUV, "entries")
        source_loop_ids = tuple(entry.source_loop_id for entry in self.entries)
        if len(source_loop_ids) != len(set(source_loop_ids)):
            raise ValueError("entries contain duplicate SourceLoopId values")

    def as_dict(self) -> dict[SourceLoopId, Vector2]:
        return {entry.source_loop_id: entry.coordinate for entry in self.entries}


@dataclass(frozen=True, slots=True)
class UvTransferReport:
    source_layer_name: str
    target_layer_name: str
    updated_loop_count: int
    missing_source_loop_ids: Tuple[SourceLoopId, ...]
    unused_source_loop_ids: Tuple[SourceLoopId, ...]

    def __post_init__(self) -> None:
        require_non_empty_string(self.source_layer_name, "source_layer_name")
        require_non_empty_string(self.target_layer_name, "target_layer_name")
        require_integer(self.updated_loop_count, "updated_loop_count", minimum=0)
        require_tuple_items(
            self.missing_source_loop_ids,
            SourceLoopId,
            "missing_source_loop_ids",
        )
        require_tuple_items(
            self.unused_source_loop_ids,
            SourceLoopId,
            "unused_source_loop_ids",
        )
        for field_name, values in (
            ("missing_source_loop_ids", self.missing_source_loop_ids),
            ("unused_source_loop_ids", self.unused_source_loop_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} cannot contain duplicates")
        overlap = set(self.missing_source_loop_ids).intersection(
            self.unused_source_loop_ids
        )
        if overlap:
            raise ValueError(
                "missing_source_loop_ids and unused_source_loop_ids cannot overlap: "
                + str(tuple(sorted(overlap)))
            )

    @property
    def complete(self) -> bool:
        return not self.missing_source_loop_ids


def _coordinates_equal(first: Vector2, second: Vector2, tolerance: float) -> bool:
    require_finite_vector(first, 2, "first")
    require_finite_vector(second, 2, "second")
    resolved_tolerance = require_finite_number(tolerance, "tolerance")
    if resolved_tolerance < 0.0:
        raise ValueError("tolerance cannot be negative")
    return isclose(
        first[0],
        second[0],
        rel_tol=0.0,
        abs_tol=resolved_tolerance,
    ) and isclose(
        first[1],
        second[1],
        rel_tol=0.0,
        abs_tol=resolved_tolerance,
    )


def build_uv_correspondence(
    snapshot: MeshSnapshot,
    layer_name: str,
    *,
    duplicate_tolerance: float = 0.0,
) -> UvCorrespondenceMap:
    """Build one exact ``SourceLoopId -> UV`` lookup.

    Repeated SourceLoopId values are legal in derived topology, for example after
    triangulating an n-gon. They must resolve to the same UV coordinate; conflicting
    values are rejected instead of silently selecting one by rounded position.
    """

    require_exact_type(snapshot, MeshSnapshot, "snapshot")
    require_non_empty_string(layer_name, "layer_name")
    resolved_tolerance = require_finite_number(
        duplicate_tolerance,
        "duplicate_tolerance",
    )
    if resolved_tolerance < 0.0:
        raise ValueError("duplicate_tolerance cannot be negative")
    MeshSnapshotValidator().validate_or_raise(snapshot)
    if layer_name not in snapshot.uv_layer_names:
        raise KeyError(f"UV layer '{layer_name}' is not present in snapshot")

    lookup: dict[SourceLoopId, Vector2] = {}
    for loop in snapshot.loops:
        coordinate = loop.uv(layer_name)
        if coordinate is None:
            raise KeyError(
                f"Loop {loop.id.index} does not contain declared UV layer "
                f"'{layer_name}'"
            )
        previous = lookup.get(loop.source_id)
        if previous is not None and not _coordinates_equal(
            previous,
            coordinate,
            resolved_tolerance,
        ):
            raise ConflictingSourceLoopUVError(loop.source_id, previous, coordinate)
        lookup[loop.source_id] = coordinate

    return UvCorrespondenceMap(
        layer_name=layer_name,
        entries=tuple(
            SourceLoopUV(source_loop_id=source_id, coordinate=coordinate)
            for source_id, coordinate in sorted(lookup.items())
        ),
    )


def transfer_uv_by_source_loop(
    source: MeshSnapshot,
    target: MeshSnapshot,
    *,
    source_layer_name: str,
    target_layer_name: str,
    require_complete: bool = True,
    duplicate_tolerance: float = 0.0,
) -> tuple[MeshSnapshot, UvTransferReport]:
    """Transfer UVs without coordinate rounding or nearest-point matching."""

    require_exact_type(source, MeshSnapshot, "source")
    require_exact_type(target, MeshSnapshot, "target")
    require_non_empty_string(source_layer_name, "source_layer_name")
    require_non_empty_string(target_layer_name, "target_layer_name")
    if not isinstance(require_complete, bool):
        raise TypeError("require_complete must be bool")
    resolved_tolerance = require_finite_number(
        duplicate_tolerance,
        "duplicate_tolerance",
    )
    if resolved_tolerance < 0.0:
        raise ValueError("duplicate_tolerance cannot be negative")
    MeshSnapshotValidator().validate_or_raise(source)
    MeshSnapshotValidator().validate_or_raise(target)
    if source.source_object_id != target.source_object_id:
        raise CorrespondenceError(
            "source and target snapshots must originate from the same source object"
        )

    correspondence = build_uv_correspondence(
        source,
        source_layer_name,
        duplicate_tolerance=resolved_tolerance,
    )
    lookup = correspondence.as_dict()
    used: set[SourceLoopId] = set()
    missing: set[SourceLoopId] = set()
    updated_loops: list[MeshLoop] = []
    updated_count = 0

    for loop in target.loops:
        coordinate = lookup.get(loop.source_id)
        if coordinate is None:
            missing.add(loop.source_id)
            fallback = loop.uv(target_layer_name)
            if fallback is None and target.active_uv_layer is not None:
                fallback = loop.uv(target.active_uv_layer)
            if fallback is None and loop.uvs:
                fallback = loop.uvs[0].coordinate
            if fallback is None:
                fallback = (0.0, 0.0)
            updated_loops.append(loop.with_uv(target_layer_name, fallback))
            continue
        used.add(loop.source_id)
        updated_loops.append(loop.with_uv(target_layer_name, coordinate))
        updated_count += 1

    if missing and require_complete:
        raise MissingSourceLoopError(missing)

    uv_layer_names = tuple(sorted(set(target.uv_layer_names) | {target_layer_name}))
    updated_snapshot = MeshSnapshot(
        snapshot_id=target.snapshot_id,
        source_object_id=target.source_object_id,
        object_name=target.object_name,
        vertices=target.vertices,
        edges=target.edges,
        loops=tuple(updated_loops),
        faces=target.faces,
        uv_layer_names=uv_layer_names,
        active_uv_layer=target_layer_name,
        world_matrix=target.world_matrix,
        render_uv_layer=target.render_uv_layer,
    )
    MeshSnapshotValidator().validate_or_raise(updated_snapshot)

    report = UvTransferReport(
        source_layer_name=source_layer_name,
        target_layer_name=target_layer_name,
        updated_loop_count=updated_count,
        missing_source_loop_ids=tuple(sorted(missing)),
        unused_source_loop_ids=tuple(sorted(set(lookup) - used)),
    )
    return updated_snapshot, report


def extract_face_subset(
    snapshot: MeshSnapshot,
    face_ids: Iterable[FaceId],
    *,
    snapshot_id: str,
    object_name: str | None = None,
) -> MeshSnapshot:
    """Create a densely reindexed segment while preserving all source IDs."""

    selected_ids = set(face_ids)
    face_map = snapshot.face_by_id()
    unknown = selected_ids - set(face_map)
    if unknown:
        raise KeyError(
            "Unknown FaceId values: "
            + ", ".join(str(face_id.index) for face_id in sorted(unknown))
        )
    if not selected_ids:
        raise ValueError("face_ids cannot be empty")

    selected_faces = [face for face in snapshot.faces if face.id in selected_ids]
    loop_map = snapshot.loop_by_id()
    edge_map = snapshot.edge_by_id()
    vertex_map = snapshot.vertex_by_id()

    selected_old_loop_ids = [
        loop_id for face in selected_faces for loop_id in face.loop_ids
    ]
    selected_loops = [loop_map[loop_id] for loop_id in selected_old_loop_ids]
    selected_old_edge_ids = sorted(
        {loop.edge_id for loop in selected_loops}, key=lambda item: item.index
    )
    selected_edges = [edge_map[edge_id] for edge_id in selected_old_edge_ids]
    selected_old_vertex_ids = sorted(
        {vertex_id for edge in selected_edges for vertex_id in edge.vertex_ids},
        key=lambda item: item.index,
    )
    selected_vertices = [vertex_map[vertex_id] for vertex_id in selected_old_vertex_ids]

    vertex_reindex = {
        vertex.id: VertexId(new_index)
        for new_index, vertex in enumerate(selected_vertices)
    }
    edge_reindex = {
        edge.id: EdgeId(new_index) for new_index, edge in enumerate(selected_edges)
    }
    loop_reindex = {
        loop.id: LoopId(new_index) for new_index, loop in enumerate(selected_loops)
    }

    new_vertices = tuple(
        MeshVertex(
            id=vertex_reindex[vertex.id],
            source_id=vertex.source_id,
            position=vertex.position,
            normal=vertex.normal,
        )
        for vertex in selected_vertices
    )
    new_edges = tuple(
        MeshEdge(
            id=edge_reindex[edge.id],
            source_id=edge.source_id,
            vertex_ids=(
                vertex_reindex[edge.vertex_ids[0]],
                vertex_reindex[edge.vertex_ids[1]],
            ),
            seam=edge.seam,
            sharp=edge.sharp,
        )
        for edge in selected_edges
    )
    new_loops = tuple(
        MeshLoop(
            id=loop_reindex[loop.id],
            source_id=loop.source_id,
            vertex_id=vertex_reindex[loop.vertex_id],
            edge_id=edge_reindex[loop.edge_id],
            uvs=loop.uvs,
        )
        for loop in selected_loops
    )
    new_faces = tuple(
        MeshFace(
            id=FaceId(new_index),
            source_id=face.source_id,
            loop_ids=tuple(loop_reindex[loop_id] for loop_id in face.loop_ids),
            material_index=face.material_index,
            normal=face.normal,
            smooth=face.smooth,
        )
        for new_index, face in enumerate(selected_faces)
    )

    result = MeshSnapshot(
        snapshot_id=snapshot_id,
        source_object_id=snapshot.source_object_id,
        object_name=object_name or snapshot.object_name,
        vertices=new_vertices,
        edges=new_edges,
        loops=new_loops,
        faces=new_faces,
        uv_layer_names=snapshot.uv_layer_names,
        active_uv_layer=snapshot.active_uv_layer,
        world_matrix=snapshot.world_matrix,
        render_uv_layer=snapshot.render_uv_layer,
    )
    MeshSnapshotValidator().validate_or_raise(result)
    return result
