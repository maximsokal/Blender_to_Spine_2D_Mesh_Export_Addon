"""Immutable UV layouts keyed by local and source loop identity."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

from ..geometry import (
    LoopId,
    MeshSnapshot,
    MeshSnapshotValidator,
    SourceLoopId,
)
from ..geometry.contracts import (
    require_exact_type,
    require_finite_vector,
    require_identity,
    require_non_empty_string,
    require_tuple_items,
)


class UvLayoutError(ValueError):
    """Raised when a UV layout does not match its target snapshot."""


@dataclass(frozen=True, slots=True)
class UvLoopCoordinate:
    loop_id: LoopId
    source_loop_id: SourceLoopId
    coordinate: Tuple[float, float]

    def __post_init__(self) -> None:
        require_exact_type(self.loop_id, LoopId, "loop_id")
        require_exact_type(self.source_loop_id, SourceLoopId, "source_loop_id")
        require_finite_vector(self.coordinate, 2, "coordinate")


@dataclass(frozen=True, slots=True)
class UvLayout:
    snapshot_id: str
    layer_name: str
    coordinates: Tuple[UvLoopCoordinate, ...]

    def __post_init__(self) -> None:
        require_identity(self.snapshot_id, "snapshot_id")
        require_non_empty_string(self.layer_name, "layer_name")
        require_tuple_items(self.coordinates, UvLoopCoordinate, "coordinates")
        loop_ids = tuple(entry.loop_id for entry in self.coordinates)
        if len(loop_ids) != len(set(loop_ids)):
            raise ValueError("coordinates contain duplicate local LoopId values")

    def by_loop_id(self) -> dict[LoopId, UvLoopCoordinate]:
        return {entry.loop_id: entry for entry in self.coordinates}


def build_uv_layout(
    snapshot: MeshSnapshot,
    layer_name: str,
) -> UvLayout:
    """Capture one existing UV layer without coordinate-based matching."""

    require_exact_type(snapshot, MeshSnapshot, "snapshot")
    require_non_empty_string(layer_name, "layer_name")
    MeshSnapshotValidator().validate_or_raise(snapshot)
    if layer_name not in snapshot.uv_layer_names:
        raise UvLayoutError(f"UV layer '{layer_name}' is absent from snapshot")
    coordinates: list[UvLoopCoordinate] = []
    for loop in snapshot.loops:
        coordinate = loop.uv(layer_name)
        if coordinate is None:
            raise UvLayoutError(
                f"Loop {loop.id.index} is missing UV layer '{layer_name}'"
            )
        coordinates.append(
            UvLoopCoordinate(
                loop_id=loop.id,
                source_loop_id=loop.source_id,
                coordinate=coordinate,
            )
        )
    return UvLayout(
        snapshot_id=snapshot.snapshot_id,
        layer_name=layer_name,
        coordinates=tuple(coordinates),
    )


def apply_uv_layout(
    snapshot: MeshSnapshot,
    layout: UvLayout,
    *,
    require_complete: bool = True,
) -> MeshSnapshot:
    """Return a new snapshot with a layout applied by exact local ``LoopId``.

    A partial layout can only update a layer that already exists on every omitted
    loop. The immutable :class:`MeshSnapshot` contract declares UV layers globally,
    so silently introducing a new layer on only part of the loops would create an
    invalid snapshot and defer the real error to a later validator.
    """

    require_exact_type(snapshot, MeshSnapshot, "snapshot")
    require_exact_type(layout, UvLayout, "layout")
    if not isinstance(require_complete, bool):
        raise TypeError("require_complete must be bool")
    MeshSnapshotValidator().validate_or_raise(snapshot)
    if layout.snapshot_id != snapshot.snapshot_id:
        raise UvLayoutError("layout does not belong to the supplied snapshot")

    layout_by_loop = layout.by_loop_id()
    snapshot_loop_ids = {loop.id for loop in snapshot.loops}
    unknown_layout_ids = set(layout_by_loop) - snapshot_loop_ids
    if unknown_layout_ids:
        raise UvLayoutError(
            "layout references unknown loops: "
            + str(sorted(loop_id.index for loop_id in unknown_layout_ids))
        )

    missing = snapshot_loop_ids - set(layout_by_loop)
    if missing and require_complete:
        raise UvLayoutError(
            "layout is incomplete; missing loops: "
            + str(sorted(loop_id.index for loop_id in missing))
        )
    if missing:
        loop_map = snapshot.loop_by_id()
        missing_without_target_layer = tuple(
            sorted(
                (
                    loop_id.index
                    for loop_id in missing
                    if loop_map[loop_id].uv(layout.layer_name) is None
                )
            )
        )
        if missing_without_target_layer:
            raise UvLayoutError(
                "a partial layout cannot introduce a new UV layer on only some "
                "loops; omitted loops without layer "
                f"'{layout.layer_name}': {missing_without_target_layer}"
            )

    updated_loops = []
    for loop in snapshot.loops:
        entry = layout_by_loop.get(loop.id)
        if entry is None:
            updated_loops.append(loop)
            continue
        if entry.source_loop_id != loop.source_id:
            raise UvLayoutError(
                f"Loop {loop.id.index} source lineage changed from {loop.source_id} "
                f"to {entry.source_loop_id}"
            )
        updated_loops.append(loop.with_uv(layout.layer_name, entry.coordinate))

    layer_names = tuple(
        sorted(set(snapshot.uv_layer_names) | {layout.layer_name})
    )
    updated = replace(
        snapshot,
        loops=tuple(updated_loops),
        uv_layer_names=layer_names,
        active_uv_layer=layout.layer_name,
    )
    MeshSnapshotValidator().validate_or_raise(updated)
    return updated
