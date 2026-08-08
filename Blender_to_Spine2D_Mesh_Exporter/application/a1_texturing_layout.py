"""Shared A1 texturing topology and exact UV propagation to prepared regions.

Legacy-compatible baking unwraps one complete object copy with every segmentation
and decomposition cut marked as a seam. The baked UV layer is then transferred to
all exported regions. This module implements that ordering with immutable snapshots
and exact source-lineage correspondence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

from ..domain.geometry import (
    EdgeId,
    MeshSnapshot,
    MeshSnapshotValidator,
    SourceEdgeId,
    UvTransferReport,
    transfer_uv_by_source_loop,
)
from .a1_geometry_preparation import (
    A1GeometryPreparationResult,
    A1PreparedRegion,
)


class A1TexturingLayoutError(ValueError):
    """Raised when the shared texturing mesh or UV propagation is inconsistent."""


@dataclass(frozen=True, slots=True)
class A1TexturingTopology:
    source_snapshot_id: str
    snapshot: MeshSnapshot
    existing_seam_edge_ids: Tuple[EdgeId, ...]
    segmentation_seam_edge_ids: Tuple[EdgeId, ...]
    decomposition_seam_edge_ids: Tuple[EdgeId, ...]
    all_seam_edge_ids: Tuple[EdgeId, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_snapshot_id, str) or not self.source_snapshot_id:
            raise ValueError("source_snapshot_id must be a non-empty string")
        if not isinstance(self.snapshot, MeshSnapshot):
            raise TypeError("snapshot must be MeshSnapshot")
        for field_name in (
            "existing_seam_edge_ids",
            "segmentation_seam_edge_ids",
            "decomposition_seam_edge_ids",
            "all_seam_edge_ids",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, tuple) or not all(
                isinstance(edge_id, EdgeId) for edge_id in value
            ):
                raise TypeError(f"{field_name} must be a tuple of EdgeId values")


@dataclass(frozen=True, slots=True)
class A1UvReadyRegion:
    prepared_region: A1PreparedRegion
    snapshot: MeshSnapshot
    transfer_report: UvTransferReport

    def __post_init__(self) -> None:
        if not isinstance(self.prepared_region, A1PreparedRegion):
            raise TypeError("prepared_region must be A1PreparedRegion")
        if not isinstance(self.snapshot, MeshSnapshot):
            raise TypeError("snapshot must be MeshSnapshot")
        if not isinstance(self.transfer_report, UvTransferReport):
            raise TypeError("transfer_report must be UvTransferReport")
        if not self.transfer_report.complete:
            raise ValueError("A1UvReadyRegion requires a complete UV transfer")


@dataclass(frozen=True, slots=True)
class A1UvPropagationResult:
    source_snapshot_id: str
    source_layer_name: str
    target_layer_name: str
    regions: Tuple[A1UvReadyRegion, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_snapshot_id, str) or not self.source_snapshot_id:
            raise ValueError("source_snapshot_id must be a non-empty string")
        for field_name in ("source_layer_name", "target_layer_name"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.regions, tuple) or not self.regions:
            raise ValueError("regions must be a non-empty tuple")
        if not all(isinstance(region, A1UvReadyRegion) for region in self.regions):
            raise TypeError("regions must contain A1UvReadyRegion values")

    @property
    def snapshots(self) -> Tuple[MeshSnapshot, ...]:
        return tuple(region.snapshot for region in self.regions)


_GeometryEdgeReference = tuple[EdgeId, SourceEdgeId | None]


def _internal_segmentation_cut_references(
    geometry: A1GeometryPreparationResult,
) -> Tuple[_GeometryEdgeReference, ...]:
    """Return internal segmentation cuts with stable source lineage."""

    return tuple(
        sorted(
            {
                (boundary.edge_id, boundary.source_edge_id)
                for boundary in geometry.segmentation.boundary_edges
                if len(boundary.linked_face_ids) == 2
                and len(boundary.segment_ids) == 2
            },
            key=lambda item: item[0].index,
        )
    )


def _decomposition_cut_references(
    geometry: A1GeometryPreparationResult,
) -> Tuple[_GeometryEdgeReference, ...]:
    """Return internal decomposition cuts with stable source lineage."""

    return tuple(
        sorted(
            {
                (cut.edge_id, cut.source_edge_id)
                for cut in geometry.decomposition.cuts
                if len(cut.linked_face_ids) == 2 and len(cut.region_ids) == 2
            },
            key=lambda item: item[0].index,
        )
    )


def _source_edge_lookup(
    source_snapshot: MeshSnapshot,
) -> dict[SourceEdgeId, Tuple[EdgeId, ...]]:
    """Build ``SourceEdgeId -> local EdgeId`` correspondence for texturing topology."""

    grouped: dict[SourceEdgeId, list[EdgeId]] = {}
    for edge in source_snapshot.edges:
        if edge.source_id is None:
            continue
        grouped.setdefault(edge.source_id, []).append(edge.id)
    return {
        source_id: tuple(sorted(edge_ids, key=lambda item: item.index))
        for source_id, edge_ids in grouped.items()
    }


def _resolve_geometry_cut_edge_ids(
    source_snapshot: MeshSnapshot,
    references: Tuple[_GeometryEdgeReference, ...],
    *,
    label: str,
) -> Tuple[EdgeId, ...]:
    """Translate repaired geometry cut IDs onto the full texturing snapshot.

    Geometry preparation may conservatively split a non-manifold edge into multiple
    local ``EdgeId`` values. The full texturing/unwrap snapshot intentionally remains the
    unrepaired source topology, so local IDs cannot be compared directly after repair.
    Stable ``SourceEdgeId`` lineage is authoritative for source edges. Generated edges
    have no source lineage and are accepted only when the exact local ID still exists as
    a generated edge in the texturing snapshot; the non-manifold repair preserves those
    original IDs and appends only additional edge copies.
    """

    if not isinstance(source_snapshot, MeshSnapshot):
        raise TypeError("source_snapshot must be MeshSnapshot")
    if not isinstance(references, tuple):
        raise TypeError("references must be a tuple")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a non-empty string")

    edge_map = source_snapshot.edge_by_id()
    source_lookup = _source_edge_lookup(source_snapshot)
    resolved: set[EdgeId] = set()

    for edge_id, source_edge_id in references:
        if not isinstance(edge_id, EdgeId):
            raise TypeError(f"{label} cut edge_id must be EdgeId")
        if source_edge_id is not None and not isinstance(source_edge_id, SourceEdgeId):
            raise TypeError(f"{label} cut source_edge_id must be SourceEdgeId or None")

        if source_edge_id is not None:
            matches = source_lookup.get(source_edge_id, ())
            if not matches:
                raise A1TexturingLayoutError(
                    f"{label} seam cannot resolve SourceEdgeId "
                    f"{source_edge_id.object_id}:{source_edge_id.edge_index} "
                    "onto the texturing snapshot"
                )
            resolved.update(matches)
            continue

        # A generated edge has no SourceEdgeId. Its local identity is safe only when the
        # exact generated edge still exists in the unrepaired snapshot. Repair keeps all
        # original IDs stable specifically to make this deterministic.
        source_edge = edge_map.get(edge_id)
        if source_edge is None:
            raise A1TexturingLayoutError(
                f"{label} generated seam edge {edge_id.index} has no SourceEdgeId and "
                "does not exist in the texturing snapshot"
            )
        if source_edge.source_id is not None:
            raise A1TexturingLayoutError(
                f"{label} seam edge {edge_id.index} lost source lineage: geometry has "
                "source_edge_id=None but texturing edge has "
                f"{source_edge.source_id.object_id}:{source_edge.source_id.edge_index}"
            )
        resolved.add(edge_id)

    return tuple(sorted(resolved, key=lambda item: item.index))


def build_a1_texturing_topology(
    source_snapshot: MeshSnapshot,
    geometry: A1GeometryPreparationResult,
    *,
    snapshot_id: str | None = None,
    object_name: str | None = None,
) -> A1TexturingTopology:
    """Mark every internal export cut as a seam on one full-object snapshot."""

    if not isinstance(source_snapshot, MeshSnapshot):
        raise TypeError("source_snapshot must be MeshSnapshot")
    if not isinstance(geometry, A1GeometryPreparationResult):
        raise TypeError("geometry must be A1GeometryPreparationResult")
    MeshSnapshotValidator().validate_or_raise(source_snapshot)
    if geometry.source_snapshot_id != source_snapshot.snapshot_id:
        raise A1TexturingLayoutError(
            "geometry preparation does not belong to source_snapshot"
        )

    existing = tuple(
        sorted(
            (edge.id for edge in source_snapshot.edges if edge.seam),
            key=lambda item: item.index,
        )
    )
    segmentation = _resolve_geometry_cut_edge_ids(
        source_snapshot,
        _internal_segmentation_cut_references(geometry),
        label="Segmentation",
    )
    decomposition = _resolve_geometry_cut_edge_ids(
        source_snapshot,
        _decomposition_cut_references(geometry),
        label="Decomposition",
    )
    requested = set(existing) | set(segmentation) | set(decomposition)

    # All resolved seam IDs must now belong to the actual topology being unwrapped.
    edge_map = source_snapshot.edge_by_id()
    unknown = requested - set(edge_map)
    if unknown:
        raise A1TexturingLayoutError(
            "Resolved texturing seam plan references unknown edges: "
            + str(tuple(sorted(edge_id.index for edge_id in unknown)))
        )

    updated_edges = tuple(
        replace(edge, seam=edge.id in requested)
        for edge in source_snapshot.edges
    )
    updated_snapshot = replace(
        source_snapshot,
        snapshot_id=snapshot_id or f"{source_snapshot.snapshot_id}:texturing",
        object_name=object_name or f"{source_snapshot.object_name}_Texturing",
        edges=updated_edges,
    )
    MeshSnapshotValidator().validate_or_raise(updated_snapshot)
    all_seams = tuple(sorted(requested, key=lambda item: item.index))
    return A1TexturingTopology(
        source_snapshot_id=source_snapshot.snapshot_id,
        snapshot=updated_snapshot,
        existing_seam_edge_ids=existing,
        segmentation_seam_edge_ids=segmentation,
        decomposition_seam_edge_ids=decomposition,
        all_seam_edge_ids=all_seams,
    )


def propagate_texturing_uv_to_regions(
    textured_snapshot: MeshSnapshot,
    geometry: A1GeometryPreparationResult,
    *,
    source_layer_name: str,
    target_layer_name: str | None = None,
    duplicate_tolerance: float = 0.0,
) -> A1UvPropagationResult:
    """Transfer one globally unwrapped UV layer to every triangulated region."""

    if not isinstance(textured_snapshot, MeshSnapshot):
        raise TypeError("textured_snapshot must be MeshSnapshot")
    if not isinstance(geometry, A1GeometryPreparationResult):
        raise TypeError("geometry must be A1GeometryPreparationResult")
    MeshSnapshotValidator().validate_or_raise(textured_snapshot)
    if textured_snapshot.source_object_id != geometry.regions[0].snapshot.source_object_id:
        raise A1TexturingLayoutError(
            "textured_snapshot and prepared regions have different source_object_id"
        )
    if not isinstance(source_layer_name, str) or not source_layer_name.strip():
        raise ValueError("source_layer_name must be a non-empty string")
    resolved_target_layer = target_layer_name or source_layer_name
    if not isinstance(resolved_target_layer, str) or not resolved_target_layer.strip():
        raise ValueError("target_layer_name must be a non-empty string")
    if duplicate_tolerance < 0.0:
        raise ValueError("duplicate_tolerance cannot be negative")

    transferred: list[A1UvReadyRegion] = []
    for prepared_region in geometry.regions:
        try:
            updated, report = transfer_uv_by_source_loop(
                textured_snapshot,
                prepared_region.snapshot,
                source_layer_name=source_layer_name,
                target_layer_name=resolved_target_layer,
                require_complete=True,
                duplicate_tolerance=duplicate_tolerance,
            )
        except Exception as exc:
            raise A1TexturingLayoutError(
                f"Unable to transfer global UVs to region "
                f"{prepared_region.region_index}: {exc}"
            ) from exc
        if report.updated_loop_count != len(prepared_region.snapshot.loops):
            raise A1TexturingLayoutError(
                f"Region {prepared_region.region_index} received "
                f"{report.updated_loop_count} UV loops for "
                f"{len(prepared_region.snapshot.loops)} mesh loops"
            )
        transferred.append(
            A1UvReadyRegion(
                prepared_region=prepared_region,
                snapshot=updated,
                transfer_report=report,
            )
        )

    return A1UvPropagationResult(
        source_snapshot_id=geometry.source_snapshot_id,
        source_layer_name=source_layer_name,
        target_layer_name=resolved_target_layer,
        regions=tuple(transferred),
    )
