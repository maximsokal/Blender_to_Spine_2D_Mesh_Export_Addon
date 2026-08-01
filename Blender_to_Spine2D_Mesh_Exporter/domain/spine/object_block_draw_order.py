"""Plan and apply deterministic setup slot order for complete object blocks.

Spine renders later setup slots above earlier slots. Multi-object A1 composition therefore
serializes whole per-object slot blocks from far to near. Geometry ownership, internal
segment order, skins, weighted bone indices, and bone/constraint order remain unchanged.

Existing non-empty draw-order animation timelines are rejected before setup slots move:
their offset payloads are relative to the old setup indices and require a separate explicit
rebasing implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import isfinite
from typing import Tuple

from .composition import SpineDocumentComponent
from .model import Slot, SpineDocument
from .validator import SpineValidator


class SpineObjectBlockDrawOrderError(ValueError):
    """Raised when complete object slot blocks cannot be reordered safely."""


@dataclass(frozen=True, slots=True)
class SpineObjectBlockDepth:
    """Projected world-depth contract for one complete object slot block."""

    component_id: str
    source_input_index: int
    nearest_vertex_index: int
    nearest_vertex_depth: float
    farthest_vertex_index: int
    farthest_vertex_depth: float

    def __post_init__(self) -> None:
        if not isinstance(self.component_id, str) or not self.component_id.strip():
            raise ValueError("component_id must be a non-empty string")
        for field_name in (
            "source_input_index",
            "nearest_vertex_index",
            "farthest_vertex_index",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be int")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        for field_name in (
            "nearest_vertex_depth",
            "farthest_vertex_depth",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a finite number")
            if not isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        if self.farthest_vertex_depth > self.nearest_vertex_depth:
            raise ValueError(
                "farthest_vertex_depth cannot exceed nearest_vertex_depth"
            )


def _require_depth_tolerance(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("depth_tolerance must be a finite number")
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError("depth_tolerance must be finite")
    if resolved < 0.0:
        raise ValueError("depth_tolerance must be non-negative")
    return resolved


def _validate_depth_entries(
    entries: Tuple[SpineObjectBlockDepth, ...],
) -> None:
    if not isinstance(entries, tuple) or not entries:
        raise ValueError("entries must be a non-empty tuple")
    if not all(isinstance(item, SpineObjectBlockDepth) for item in entries):
        raise TypeError("entries must contain SpineObjectBlockDepth values")
    component_ids = tuple(item.component_id for item in entries)
    if len(component_ids) != len(set(component_ids)):
        raise SpineObjectBlockDrawOrderError(
            "Object-block depth entries contain duplicate component IDs"
        )


def object_block_draw_order_component_ids(
    entries: Tuple[SpineObjectBlockDepth, ...],
    *,
    depth_tolerance: float = 1.0e-4,
) -> Tuple[str, ...]:
    """Return deterministic far-to-near component IDs.

    Canonical depth increases toward the observer, so smaller nearest-vertex depth is
    emitted first. Entries whose nearest depths differ from the cluster anchor by at most
    ``depth_tolerance`` retain source input order, with ``component_id`` as the final
    deterministic fallback.
    """

    _validate_depth_entries(entries)
    tolerance = _require_depth_tolerance(depth_tolerance)
    by_depth = tuple(
        sorted(
            entries,
            key=lambda item: (
                float(item.nearest_vertex_depth),
                item.source_input_index,
                item.component_id,
            ),
        )
    )

    ordered: list[SpineObjectBlockDepth] = []
    cluster: list[SpineObjectBlockDepth] = []
    cluster_anchor: float | None = None

    def flush_cluster() -> None:
        if not cluster:
            return
        ordered.extend(
            sorted(
                cluster,
                key=lambda item: (
                    item.source_input_index,
                    item.component_id,
                ),
            )
        )
        cluster.clear()

    for entry in by_depth:
        depth = float(entry.nearest_vertex_depth)
        if cluster_anchor is None:
            cluster_anchor = depth
            cluster.append(entry)
            continue
        if depth - cluster_anchor <= tolerance:
            cluster.append(entry)
            continue
        flush_cluster()
        cluster_anchor = depth
        cluster.append(entry)
    flush_cluster()

    result = tuple(item.component_id for item in ordered)
    if len(result) != len(entries) or set(result) != {
        item.component_id for item in entries
    }:
        raise SpineObjectBlockDrawOrderError(
            "Object-block draw-order planning lost or duplicated a component"
        )
    return result


def _require_no_unrebased_draworder_timelines(
    components: Tuple[SpineDocumentComponent, ...],
) -> None:
    for component in components:
        for animation_name, animation in component.document.animations.items():
            if not isinstance(animation, Mapping):
                continue
            draworder_keys = tuple(
                key
                for key in animation
                if str(key).replace("_", "").casefold() == "draworder"
            )
            for key in draworder_keys:
                timeline = animation[key]
                if timeline:
                    raise SpineObjectBlockDrawOrderError(
                        "Setup object-block slot reordering cannot preserve an existing "
                        "draw-order timeline until component offsets are explicitly "
                        f"rebased; component={component.component_id!r}, "
                        f"animation={str(animation_name)!r}, key={str(key)!r}"
                    )


def apply_object_block_setup_draw_order(
    document: SpineDocument,
    components: Tuple[SpineDocumentComponent, ...],
    entries: Tuple[SpineObjectBlockDepth, ...],
    *,
    depth_tolerance: float = 1.0e-4,
) -> SpineDocument:
    """Move complete component slot blocks without changing any slot payload."""

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    if not isinstance(components, tuple) or not components:
        raise ValueError("components must be a non-empty tuple")
    if not all(isinstance(item, SpineDocumentComponent) for item in components):
        raise TypeError("components must contain SpineDocumentComponent values")

    component_by_id = {item.component_id: item for item in components}
    if len(component_by_id) != len(components):
        raise SpineObjectBlockDrawOrderError(
            "Object-block components contain duplicate component IDs"
        )
    ordered_component_ids = object_block_draw_order_component_ids(
        entries,
        depth_tolerance=depth_tolerance,
    )
    if set(ordered_component_ids) != set(component_by_id):
        missing_depths = tuple(
            sorted(set(component_by_id) - set(ordered_component_ids))
        )
        unknown_depths = tuple(
            sorted(set(ordered_component_ids) - set(component_by_id))
        )
        raise SpineObjectBlockDrawOrderError(
            "Object-block component/depth ownership mismatch; "
            f"missing={missing_depths}, unknown={unknown_depths}"
        )

    _require_no_unrebased_draworder_timelines(components)

    slots_by_name: dict[str, Slot] = {}
    for slot in document.slots:
        if slot.name in slots_by_name:
            raise SpineObjectBlockDrawOrderError(
                f"Composed document repeats slot '{slot.name}'"
            )
        slots_by_name[slot.name] = slot

    ordered_slot_names: list[str] = []
    owner_by_slot: dict[str, str] = {}
    for component_id in ordered_component_ids:
        component = component_by_id[component_id]
        for slot in component.document.slots:
            previous_owner = owner_by_slot.get(slot.name)
            if previous_owner is not None:
                raise SpineObjectBlockDrawOrderError(
                    f"Slot '{slot.name}' is owned by both '{previous_owner}' and "
                    f"'{component_id}'"
                )
            owner_by_slot[slot.name] = component_id
            ordered_slot_names.append(slot.name)

    composed_names = tuple(slot.name for slot in document.slots)
    expected_names = tuple(ordered_slot_names)
    if (
        len(composed_names) != len(expected_names)
        or set(composed_names) != set(expected_names)
    ):
        unowned = tuple(sorted(set(composed_names) - set(expected_names)))
        missing = tuple(sorted(set(expected_names) - set(composed_names)))
        raise SpineObjectBlockDrawOrderError(
            "Object-block setup ordering cannot account for every composed slot; "
            f"unowned={unowned}, missing={missing}"
        )

    reordered_slots = tuple(slots_by_name[name] for name in expected_names)
    if reordered_slots == document.slots:
        return document

    reordered = replace(document, slots=reordered_slots)
    try:
        SpineValidator().validate_or_raise(reordered)
    except Exception as exc:
        raise SpineObjectBlockDrawOrderError(
            f"Object-block setup ordering produced an invalid Spine document: {exc}"
        ) from exc
    return reordered


__all__ = [
    "SpineObjectBlockDepth",
    "SpineObjectBlockDrawOrderError",
    "apply_object_block_setup_draw_order",
    "object_block_draw_order_component_ids",
]
