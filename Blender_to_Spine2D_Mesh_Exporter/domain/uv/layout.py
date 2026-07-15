"""Immutable UV layouts keyed by local and source loop identity."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Iterable, Tuple

from ..geometry import (
    LoopId,
    MeshSnapshot,
    MeshSnapshotValidator,
    SourceLoopId,
)


class UvLayoutError(ValueError):
    """Raised when a UV layout does not match its target snapshot."""


@dataclass(frozen=True, slots=True)
class UvLoopCoordinate:
    loop_id: LoopId
    source_loop_id: SourceLoopId
    coordinate: Tuple[float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate, tuple) or len(self.coordinate) != 2:
            raise ValueError("coordinate must contain two values")
        if not all(
            isinstance(value, (int, float)) and isfinite(float(value))
            for value in self.coordinate
        ):
            raise ValueError("coordinate must contain finite numeric values")


@dataclass(frozen=True, slots=True)
class UvLayout:
    snapshot_id: str
    layer_name: str
    coordinates: Tuple[UvLoopCoordinate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_id, str) or not self.snapshot_id.strip():
            raise ValueError("snapshot_id must be a non-empty string")
        if not isinstance(self.layer_name, str) or not self.layer_name.strip():
            raise ValueError("layer_name must be a non-empty string")
        if not isinstance(self.coordinates, tuple):
            raise TypeError("coordinates must be tuple")
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
    """Return a new snapshot with a layout applied by exact local LoopId."""

    MeshSnapshotValidator().validate_or_raise(snapshot)
    if not isinstance(layout, UvLayout):
        raise TypeError("layout must be UvLayout")
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
        updated_loops.append(
            loop.with_uv(layout.layer_name, entry.coordinate)
        )

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
