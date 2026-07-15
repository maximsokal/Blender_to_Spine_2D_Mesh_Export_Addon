"""Pure geometry preparation for the A1 single-object export pipeline.

The stage performs deterministic seed-normal segmentation, manifold-disk
decomposition, immutable region materialization, and lineage-preserving
triangulation. It creates no Blender objects and performs no UV operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..domain.geometry import (
    DecompositionSettings,
    MeshDecompositionPlan,
    MeshSnapshot,
    MeshSnapshotValidator,
    SegmentationPlan,
    SegmentationSettings,
    SegmentTopology,
    TriangulationResult,
    TriangulationSettings,
    analyse_face_region,
    decompose_complex_segments,
    is_simple_disk,
    materialize_decomposed_snapshots,
    segment_mesh_a1,
    triangulate_snapshot,
)


class A1GeometryPreparationError(ValueError):
    """Raised when a geometry stage cannot produce complete disjoint disk regions."""


@dataclass(frozen=True, slots=True)
class A1GeometryPreparationSettings:
    segmentation: SegmentationSettings = SegmentationSettings()
    decomposition: DecompositionSettings = DecompositionSettings()
    triangulation: TriangulationSettings = TriangulationSettings()

    def __post_init__(self) -> None:
        if not isinstance(self.segmentation, SegmentationSettings):
            raise TypeError("segmentation must be SegmentationSettings")
        if not isinstance(self.decomposition, DecompositionSettings):
            raise TypeError("decomposition must be DecompositionSettings")
        if not isinstance(self.triangulation, TriangulationSettings):
            raise TypeError("triangulation must be TriangulationSettings")


@dataclass(frozen=True, slots=True)
class A1PreparedRegion:
    region_index: int
    decomposition_region_id: int
    source_segment_id: int
    source_face_ids: tuple
    topology_before_triangulation: SegmentTopology
    triangulation: TriangulationResult

    def __post_init__(self) -> None:
        if not isinstance(self.region_index, int) or self.region_index < 0:
            raise ValueError("region_index must be a non-negative integer")
        if not isinstance(self.decomposition_region_id, int) or self.decomposition_region_id < 0:
            raise ValueError("decomposition_region_id must be a non-negative integer")
        if not isinstance(self.source_segment_id, int) or self.source_segment_id < 0:
            raise ValueError("source_segment_id must be a non-negative integer")
        if not isinstance(self.source_face_ids, tuple) or not self.source_face_ids:
            raise ValueError("source_face_ids must be a non-empty tuple")
        if not isinstance(self.topology_before_triangulation, SegmentTopology):
            raise TypeError("topology_before_triangulation must be SegmentTopology")
        if not isinstance(self.triangulation, TriangulationResult):
            raise TypeError("triangulation must be TriangulationResult")

    @property
    def snapshot(self) -> MeshSnapshot:
        return self.triangulation.snapshot


@dataclass(frozen=True, slots=True)
class A1GeometryPreparationResult:
    source_snapshot_id: str
    settings: A1GeometryPreparationSettings
    segmentation: SegmentationPlan
    decomposition: MeshDecompositionPlan
    regions: Tuple[A1PreparedRegion, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_snapshot_id, str) or not self.source_snapshot_id.strip():
            raise ValueError("source_snapshot_id must be a non-empty string")
        if not isinstance(self.settings, A1GeometryPreparationSettings):
            raise TypeError("settings must be A1GeometryPreparationSettings")
        if not isinstance(self.segmentation, SegmentationPlan):
            raise TypeError("segmentation must be SegmentationPlan")
        if not isinstance(self.decomposition, MeshDecompositionPlan):
            raise TypeError("decomposition must be MeshDecompositionPlan")
        if not isinstance(self.regions, tuple) or not self.regions:
            raise ValueError("regions must be a non-empty tuple")
        actual_indices = tuple(region.region_index for region in self.regions)
        if actual_indices != tuple(range(len(self.regions))):
            raise ValueError("region indices must be ordered and dense from zero")


def _validate_prepared_coverage(
    source_snapshot: MeshSnapshot,
    decomposition: MeshDecompositionPlan,
    prepared_regions: Tuple[A1PreparedRegion, ...],
) -> None:
    source_face_ids = {face.id for face in source_snapshot.faces}
    decomposition_face_ids = [
        face_id for region in decomposition.regions for face_id in region.face_ids
    ]
    prepared_source_face_ids = [
        source_face_id
        for region in prepared_regions
        for source_face_id in region.source_face_ids
    ]
    if len(decomposition_face_ids) != len(set(decomposition_face_ids)):
        raise A1GeometryPreparationError(
            "Decomposition regions overlap in local face coverage"
        )
    if set(decomposition_face_ids) != source_face_ids:
        missing = source_face_ids - set(decomposition_face_ids)
        unknown = set(decomposition_face_ids) - source_face_ids
        raise A1GeometryPreparationError(
            "Decomposition does not cover source faces exactly; "
            f"missing={tuple(sorted(item.index for item in missing))}, "
            f"unknown={tuple(sorted(item.index for item in unknown))}"
        )
    if len(prepared_source_face_ids) != len(set(prepared_source_face_ids)):
        raise A1GeometryPreparationError(
            "Prepared regions overlap in SourceFaceId coverage"
        )
    expected_source_ids = {face.source_id for face in source_snapshot.faces}
    if set(prepared_source_face_ids) != expected_source_ids:
        missing = expected_source_ids - set(prepared_source_face_ids)
        unknown = set(prepared_source_face_ids) - expected_source_ids
        raise A1GeometryPreparationError(
            "Prepared regions do not cover source lineage exactly; "
            f"missing={missing}, unknown={unknown}"
        )


def prepare_a1_geometry_regions(
    source_snapshot: MeshSnapshot,
    settings: A1GeometryPreparationSettings | None = None,
) -> A1GeometryPreparationResult:
    """Prepare triangulated manifold-disk regions for later UV unwrap and export."""

    if not isinstance(source_snapshot, MeshSnapshot):
        raise TypeError("source_snapshot must be MeshSnapshot")
    MeshSnapshotValidator().validate_or_raise(source_snapshot)
    resolved_settings = settings or A1GeometryPreparationSettings()
    if not isinstance(resolved_settings, A1GeometryPreparationSettings):
        raise TypeError("settings must be A1GeometryPreparationSettings")

    segmentation = segment_mesh_a1(
        source_snapshot,
        resolved_settings.segmentation,
    )
    decomposition = decompose_complex_segments(
        source_snapshot,
        segmentation,
        resolved_settings.decomposition,
    )
    region_snapshots = materialize_decomposed_snapshots(
        source_snapshot,
        decomposition,
    )
    if len(region_snapshots) != len(decomposition.regions):
        raise A1GeometryPreparationError(
            "Decomposition materialization count does not match region plan"
        )

    prepared: list[A1PreparedRegion] = []
    for region_index, (region, region_snapshot) in enumerate(
        zip(decomposition.regions, region_snapshots)
    ):
        if not is_simple_disk(region.topology):
            raise A1GeometryPreparationError(
                f"Region {region.region_id} is not a manifold disk before triangulation"
            )
        triangulation = triangulate_snapshot(
            region_snapshot,
            resolved_settings.triangulation,
            snapshot_id=f"{source_snapshot.snapshot_id}:region-{region_index:03d}:tri",
        )
        triangulated_topology = analyse_face_region(
            triangulation.snapshot,
            tuple(face.id for face in triangulation.snapshot.faces),
        )
        if not is_simple_disk(triangulated_topology):
            raise A1GeometryPreparationError(
                f"Triangulation changed region {region.region_id} from a disk to "
                f"Euler={triangulated_topology.euler_characteristic}, "
                f"boundaries={triangulated_topology.boundary_component_count}, "
                f"manifold={triangulated_topology.manifold}"
            )
        prepared.append(
            A1PreparedRegion(
                region_index=region_index,
                decomposition_region_id=region.region_id,
                source_segment_id=region.source_segment_id,
                source_face_ids=region.source_face_ids,
                topology_before_triangulation=region.topology,
                triangulation=triangulation,
            )
        )

    prepared_regions = tuple(prepared)
    _validate_prepared_coverage(
        source_snapshot,
        decomposition,
        prepared_regions,
    )
    return A1GeometryPreparationResult(
        source_snapshot_id=source_snapshot.snapshot_id,
        settings=resolved_settings,
        segmentation=segmentation,
        decomposition=decomposition,
        regions=prepared_regions,
    )
