"""Deterministic decomposition of complex face regions into exportable disks.

The legacy exporter attempted random spatial k-means and could produce different
results for the same mesh.  This module grows connected topological disks, then
merges adjacent disks whenever the union remains a disk.  It guarantees that
faces are neither lost nor duplicated and contains no random state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Tuple

from .correspondence import extract_face_subset
from .ids import EdgeId, FaceId, SourceEdgeId, SourceFaceId
from .model import MeshSnapshot
from .segmentation import SegmentTopology, SegmentationPlan
from .topology import (
    analyse_face_region,
    build_edge_to_faces,
    build_face_adjacency,
    is_simple_disk,
)
from .validator import MeshSnapshotValidator


class DecompositionReason(str, Enum):
    MULTIPLE_BOUNDARIES = "MULTIPLE_BOUNDARIES"
    CLOSED_SURFACE = "CLOSED_SURFACE"
    NON_DISK_EULER = "NON_DISK_EULER"


@dataclass(frozen=True, slots=True)
class DecompositionSettings:
    """Policies for deterministic complex-region decomposition."""

    merge_compatible_regions: bool = True
    reject_non_manifold: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.merge_compatible_regions, bool):
            raise TypeError("merge_compatible_regions must be bool")
        if not isinstance(self.reject_non_manifold, bool):
            raise TypeError("reject_non_manifold must be bool")


@dataclass(frozen=True, slots=True)
class DecomposedRegion:
    region_id: int
    source_segment_id: int
    face_ids: Tuple[FaceId, ...]
    source_face_ids: Tuple[SourceFaceId, ...]
    topology: SegmentTopology

    def __post_init__(self) -> None:
        if self.region_id < 0 or self.source_segment_id < 0:
            raise ValueError("region_id and source_segment_id must be non-negative")
        if not self.face_ids:
            raise ValueError("face_ids cannot be empty")
        if len(self.face_ids) != len(set(self.face_ids)):
            raise ValueError("face_ids cannot contain duplicates")


@dataclass(frozen=True, slots=True)
class DecompositionCut:
    edge_id: EdgeId
    source_edge_id: SourceEdgeId | None
    linked_face_ids: Tuple[FaceId, ...]
    region_ids: Tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SegmentDecompositionDiagnostic:
    source_segment_id: int
    original_topology: SegmentTopology
    reasons: Tuple[DecompositionReason, ...]
    output_region_ids: Tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MeshDecompositionPlan:
    snapshot_id: str
    source_segment_count: int
    regions: Tuple[DecomposedRegion, ...]
    cuts: Tuple[DecompositionCut, ...]
    diagnostics: Tuple[SegmentDecompositionDiagnostic, ...]

    def region_for_face(self) -> dict[FaceId, int]:
        return {
            face_id: region.region_id
            for region in self.regions
            for face_id in region.face_ids
        }


class DecompositionError(ValueError):
    """Raised when a complex segment cannot be decomposed without data loss."""


class _RegionTopologyCache:
    """Memoize topology checks performed by deterministic region growth."""

    def __init__(self, snapshot: MeshSnapshot) -> None:
        self._snapshot = snapshot
        self._edge_to_faces = build_edge_to_faces(snapshot)
        self._cache: dict[frozenset[FaceId], SegmentTopology] = {}

    @property
    def edge_to_faces(self) -> dict[EdgeId, Tuple[FaceId, ...]]:
        return self._edge_to_faces

    def topology(self, face_ids: Iterable[FaceId]) -> SegmentTopology:
        key = frozenset(face_ids)
        if not key:
            raise DecompositionError("Cannot analyse an empty region")
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        topology = analyse_face_region(
            self._snapshot,
            key,
            edge_to_faces=self._edge_to_faces,
        )
        self._cache[key] = topology
        return topology


def _decomposition_reasons(topology: SegmentTopology) -> Tuple[DecompositionReason, ...]:
    reasons: list[DecompositionReason] = []
    if topology.boundary_component_count > 1:
        reasons.append(DecompositionReason.MULTIPLE_BOUNDARIES)
    if topology.boundary_component_count == 0:
        reasons.append(DecompositionReason.CLOSED_SURFACE)
    if topology.euler_characteristic != 1:
        reasons.append(DecompositionReason.NON_DISK_EULER)
    return tuple(reasons)


def _shared_neighbour_count(
    candidate: FaceId,
    region: set[FaceId],
    adjacency: dict[FaceId, Tuple[FaceId, ...]],
) -> int:
    return sum(neighbour in region for neighbour in adjacency[candidate])


def _grow_disk_regions(
    face_ids: Tuple[FaceId, ...],
    adjacency: dict[FaceId, Tuple[FaceId, ...]],
    topology_cache: _RegionTopologyCache,
) -> list[set[FaceId]]:
    """Grow maximal deterministic disk regions from the lowest unassigned face."""

    remaining = set(face_ids)
    regions: list[set[FaceId]] = []

    while remaining:
        seed = min(remaining, key=lambda item: item.index)
        region = {seed}
        remaining.remove(seed)

        if not is_simple_disk(topology_cache.topology(region)):
            raise DecompositionError(
                f"Face {seed.index} is not a valid manifold disk by itself"
            )

        while True:
            candidates = {
                neighbour
                for face_id in region
                for neighbour in adjacency[face_id]
                if neighbour in remaining
            }
            if not candidates:
                break

            ordered_candidates = sorted(
                candidates,
                key=lambda face_id: (
                    -_shared_neighbour_count(face_id, region, adjacency),
                    face_id.index,
                ),
            )
            accepted: FaceId | None = None
            for candidate in ordered_candidates:
                candidate_region = set(region)
                candidate_region.add(candidate)
                if is_simple_disk(topology_cache.topology(candidate_region)):
                    accepted = candidate
                    break
            if accepted is None:
                break
            region.add(accepted)
            remaining.remove(accepted)

        regions.append(region)

    return regions


def _regions_are_adjacent(
    first: set[FaceId],
    second: set[FaceId],
    adjacency: dict[FaceId, Tuple[FaceId, ...]],
) -> int:
    return sum(
        neighbour in second
        for face_id in first
        for neighbour in adjacency[face_id]
    )


def _merge_compatible_regions(
    regions: list[set[FaceId]],
    adjacency: dict[FaceId, Tuple[FaceId, ...]],
    topology_cache: _RegionTopologyCache,
) -> list[set[FaceId]]:
    """Greedily merge the largest compatible adjacent pair until convergence."""

    working = [set(region) for region in regions]
    while True:
        candidates: list[tuple[tuple[int, int, int, int], int, int, set[FaceId]]] = []
        for first_index in range(len(working)):
            for second_index in range(first_index + 1, len(working)):
                first = working[first_index]
                second = working[second_index]
                shared = _regions_are_adjacent(first, second, adjacency)
                if shared == 0:
                    continue
                merged = first | second
                if not is_simple_disk(topology_cache.topology(merged)):
                    continue
                score = (
                    -len(merged),
                    -shared,
                    min(face_id.index for face_id in merged),
                    max(face_id.index for face_id in merged),
                )
                candidates.append((score, first_index, second_index, merged))

        if not candidates:
            break
        _, first_index, second_index, merged = min(candidates, key=lambda item: item[0])
        working[first_index] = merged
        del working[second_index]

    return working


def _decompose_segment_faces(
    snapshot: MeshSnapshot,
    face_ids: Tuple[FaceId, ...],
    topology_cache: _RegionTopologyCache,
    settings: DecompositionSettings,
) -> Tuple[Tuple[FaceId, ...], ...]:
    adjacency = build_face_adjacency(
        snapshot,
        face_ids,
        edge_to_faces=topology_cache.edge_to_faces,
    )
    regions = _grow_disk_regions(face_ids, adjacency, topology_cache)
    if settings.merge_compatible_regions:
        regions = _merge_compatible_regions(regions, adjacency, topology_cache)

    ordered = tuple(
        tuple(sorted(region, key=lambda item: item.index))
        for region in sorted(
            regions,
            key=lambda item: (min(face_id.index for face_id in item), len(item)),
        )
    )
    covered = [face_id for region in ordered for face_id in region]
    if len(covered) != len(set(covered)):
        raise DecompositionError("Decomposition produced overlapping regions")
    if set(covered) != set(face_ids):
        missing = sorted(face_id.index for face_id in set(face_ids) - set(covered))
        raise DecompositionError(f"Decomposition lost faces: {missing}")
    for region in ordered:
        if not is_simple_disk(topology_cache.topology(region)):
            raise DecompositionError(
                "Decomposition produced a region that is not a manifold disk: "
                + str([face_id.index for face_id in region])
            )
    return ordered


def decompose_complex_segments(
    snapshot: MeshSnapshot,
    segmentation_plan: SegmentationPlan,
    settings: DecompositionSettings | None = None,
) -> MeshDecompositionPlan:
    """Convert all segmentation regions into disjoint exportable disk regions."""

    MeshSnapshotValidator().validate_or_raise(snapshot)
    if segmentation_plan.snapshot_id != snapshot.snapshot_id:
        raise DecompositionError("segmentation_plan does not belong to snapshot")
    resolved_settings = settings or DecompositionSettings()
    topology_cache = _RegionTopologyCache(snapshot)
    face_map = snapshot.face_by_id()

    region_records: list[tuple[int, Tuple[FaceId, ...], SegmentTopology]] = []
    pending_diagnostics: list[
        tuple[int, SegmentTopology, Tuple[DecompositionReason, ...], list[int]]
    ] = []

    for source_segment in segmentation_plan.segments:
        topology = topology_cache.topology(source_segment.face_ids)
        if resolved_settings.reject_non_manifold and not topology.manifold:
            raise DecompositionError(
                f"Segment {source_segment.segment_id} is non-manifold and cannot be "
                "safely decomposed without an explicit repair policy"
            )

        if is_simple_disk(topology):
            face_regions = (source_segment.face_ids,)
            reasons: Tuple[DecompositionReason, ...] = ()
        else:
            reasons = _decomposition_reasons(topology)
            face_regions = _decompose_segment_faces(
                snapshot,
                source_segment.face_ids,
                topology_cache,
                resolved_settings,
            )

        output_ids: list[int] = []
        for face_region in face_regions:
            output_ids.append(len(region_records))
            region_records.append(
                (
                    source_segment.segment_id,
                    face_region,
                    topology_cache.topology(face_region),
                )
            )
        if reasons:
            pending_diagnostics.append(
                (source_segment.segment_id, topology, reasons, output_ids)
            )

    regions = tuple(
        DecomposedRegion(
            region_id=region_id,
            source_segment_id=source_segment_id,
            face_ids=face_ids,
            source_face_ids=tuple(face_map[face_id].source_id for face_id in face_ids),
            topology=topology,
        )
        for region_id, (source_segment_id, face_ids, topology) in enumerate(region_records)
    )
    face_to_region = {
        face_id: region.region_id
        for region in regions
        for face_id in region.face_ids
    }
    edge_map = snapshot.edge_by_id()
    cuts: list[DecompositionCut] = []
    for edge_id, linked_faces in sorted(
        topology_cache.edge_to_faces.items(),
        key=lambda item: item[0].index,
    ):
        linked_region_ids = tuple(
            sorted(
                {
                    face_to_region[face_id]
                    for face_id in linked_faces
                    if face_id in face_to_region
                }
            )
        )
        if len(linked_region_ids) <= 1:
            continue
        cuts.append(
            DecompositionCut(
                edge_id=edge_id,
                source_edge_id=edge_map[edge_id].source_id,
                linked_face_ids=linked_faces,
                region_ids=linked_region_ids,
            )
        )

    diagnostics = tuple(
        SegmentDecompositionDiagnostic(
            source_segment_id=segment_id,
            original_topology=topology,
            reasons=reasons,
            output_region_ids=tuple(output_ids),
        )
        for segment_id, topology, reasons, output_ids in pending_diagnostics
    )

    expected_faces = {face.id for face in snapshot.faces}
    covered_faces = set(face_to_region)
    if covered_faces != expected_faces:
        missing = sorted(face_id.index for face_id in expected_faces - covered_faces)
        extra = sorted(face_id.index for face_id in covered_faces - expected_faces)
        raise DecompositionError(
            f"Final decomposition coverage mismatch; missing={missing}, extra={extra}"
        )

    return MeshDecompositionPlan(
        snapshot_id=snapshot.snapshot_id,
        source_segment_count=len(segmentation_plan.segments),
        regions=regions,
        cuts=tuple(cuts),
        diagnostics=diagnostics,
    )


def materialize_decomposed_snapshots(
    snapshot: MeshSnapshot,
    plan: MeshDecompositionPlan,
    *,
    snapshot_id_prefix: str | None = None,
    object_name_prefix: str | None = None,
) -> Tuple[MeshSnapshot, ...]:
    """Materialize immutable region snapshots without touching Blender state."""

    if plan.snapshot_id != snapshot.snapshot_id:
        raise DecompositionError("plan does not belong to snapshot")
    id_prefix = snapshot_id_prefix or f"{snapshot.snapshot_id}:region"
    name_prefix = object_name_prefix or snapshot.object_name
    return tuple(
        extract_face_subset(
            snapshot,
            region.face_ids,
            snapshot_id=f"{id_prefix}:{region.region_id:03d}",
            object_name=f"{name_prefix}_Region_{region.region_id:03d}",
        )
        for region in plan.regions
    )
