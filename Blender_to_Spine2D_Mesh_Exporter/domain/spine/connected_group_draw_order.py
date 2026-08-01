"""Resolve and apply setup-pose slot draw order for connected A1 objects.

Connected hierarchy layers are grouped by projected Object Origin depth. Setup slot order
is a separate visual contract: when projected object-block depth analysis is available,
whole object blocks are emitted from far to near by nearest evaluated vertex. Legacy
callers without that analysis retain the historical origin-layer ordering.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Tuple

from .composition import SpineDocumentComponent
from .connected_group_contracts import (
    ConnectedObjectDocument,
    ConnectedObjectPlacement,
)
from .connected_group_error import ConnectedGroupBuildError
from .model import Slot, SpineDocument
from .object_block_draw_order import (
    SpineObjectBlockDepth,
    SpineObjectBlockDrawOrderError,
    apply_object_block_setup_draw_order,
)


def connected_draw_order_component_ids(
    placements: Tuple[ConnectedObjectPlacement, ...],
) -> Tuple[str, ...]:
    """Return the historical origin-layer back-to-front component order."""

    if not isinstance(placements, tuple) or not placements:
        raise ValueError("placements must be a non-empty tuple")
    if not all(isinstance(item, ConnectedObjectPlacement) for item in placements):
        raise TypeError("placements must contain ConnectedObjectPlacement values")

    component_ids = tuple(item.component_id for item in placements)
    if len(component_ids) != len(set(component_ids)):
        raise ConnectedGroupBuildError(
            "Connected draw order placements contain duplicate component IDs"
        )
    input_order = {
        placement.component_id: index
        for index, placement in enumerate(placements)
    }
    return tuple(
        placement.component_id
        for placement in sorted(
            placements,
            # layer_index 0 is the highest/front Object Origin cluster. Later Spine
            # slots draw on top, therefore larger/back layer indices come first.
            key=lambda item: (
                -item.layer_index,
                input_order[item.component_id],
            ),
        )
    )


def _require_no_unrebased_draworder_timelines(
    objects: Tuple[ConnectedObjectDocument, ...],
) -> None:
    """Fail before setup slots move if animation offsets would need rebasing."""

    for item in objects:
        for animation_name, animation in item.document.animations.items():
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
                    raise ConnectedGroupBuildError(
                        "Connected setup slot reordering cannot preserve an existing "
                        "draworder timeline until component offsets are explicitly "
                        f"rebased; component={item.component_id!r}, "
                        f"animation={str(animation_name)!r}, key={str(key)!r}"
                    )


def _apply_nearest_vertex_draw_order(
    document: SpineDocument,
    objects: Tuple[ConnectedObjectDocument, ...],
    object_block_depths: Tuple[SpineObjectBlockDepth, ...],
    *,
    depth_tolerance: float,
) -> SpineDocument:
    """Apply the common object-block planner to connected source documents."""

    components = tuple(
        SpineDocumentComponent(
            component_id=item.component_id,
            document=item.document,
            animation_namespace=(item.animation_namespace or item.component_id),
        )
        for item in objects
    )
    try:
        return apply_object_block_setup_draw_order(
            document,
            components,
            object_block_depths,
            depth_tolerance=depth_tolerance,
        )
    except SpineObjectBlockDrawOrderError as exc:
        raise ConnectedGroupBuildError(
            f"Connected nearest-vertex setup draw order failed: {exc}"
        ) from exc


def _apply_legacy_layer_draw_order(
    document: SpineDocument,
    objects: Tuple[ConnectedObjectDocument, ...],
    placements: Tuple[ConnectedObjectPlacement, ...],
) -> SpineDocument:
    """Preserve historical Object Origin layer ordering for legacy direct callers."""

    _require_no_unrebased_draworder_timelines(objects)
    object_by_component = {item.component_id: item for item in objects}
    if len(object_by_component) != len(objects):
        raise ConnectedGroupBuildError(
            "Connected draw order objects contain duplicate component IDs"
        )
    ordered_components = connected_draw_order_component_ids(placements)
    if set(ordered_components) != set(object_by_component):
        missing_placements = tuple(
            sorted(set(object_by_component) - set(ordered_components))
        )
        unknown_placements = tuple(
            sorted(set(ordered_components) - set(object_by_component))
        )
        raise ConnectedGroupBuildError(
            "Connected draw order object/placement ownership mismatch; "
            f"missing={missing_placements}, unknown={unknown_placements}"
        )

    slots_by_name: dict[str, Slot] = {}
    for slot in document.slots:
        if slot.name in slots_by_name:
            raise ConnectedGroupBuildError(
                f"Composed connected document repeats slot '{slot.name}'"
            )
        slots_by_name[slot.name] = slot

    ordered_slot_names: list[str] = []
    owner_by_slot: dict[str, str] = {}
    for component_id in ordered_components:
        component = object_by_component[component_id]
        for slot in component.document.slots:
            previous_owner = owner_by_slot.get(slot.name)
            if previous_owner is not None:
                raise ConnectedGroupBuildError(
                    f"Slot '{slot.name}' is owned by both '{previous_owner}' and "
                    f"'{component_id}'"
                )
            owner_by_slot[slot.name] = component_id
            ordered_slot_names.append(slot.name)

    composed_names = tuple(slot.name for slot in document.slots)
    expected_names = tuple(ordered_slot_names)
    if set(composed_names) != set(expected_names) or len(composed_names) != len(
        expected_names
    ):
        unowned = tuple(sorted(set(composed_names) - set(expected_names)))
        missing = tuple(sorted(set(expected_names) - set(composed_names)))
        raise ConnectedGroupBuildError(
            "Connected setup draw order cannot account for every composed slot; "
            f"unowned={unowned}, missing={missing}"
        )

    reordered = tuple(slots_by_name[name] for name in expected_names)
    if reordered == document.slots:
        return document
    return replace(document, slots=reordered)


def apply_connected_setup_draw_order(
    document: SpineDocument,
    objects: Tuple[ConnectedObjectDocument, ...],
    placements: Tuple[ConnectedObjectPlacement, ...],
    *,
    object_block_depths: Tuple[SpineObjectBlockDepth, ...] | None = None,
    depth_tolerance: float = 1.0e-4,
) -> SpineDocument:
    """Reorder complete connected slot blocks without changing slot payloads.

    Production projected composition supplies ``object_block_depths`` and therefore uses
    nearest evaluated vertex depth. ``None`` exists only for retained direct legacy tests
    and callers that have not yet supplied projected depth analysis.
    """

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    if not isinstance(objects, tuple) or not objects:
        raise ValueError("objects must be a non-empty tuple")
    if not all(isinstance(item, ConnectedObjectDocument) for item in objects):
        raise TypeError("objects must contain ConnectedObjectDocument values")
    if object_block_depths is not None:
        return _apply_nearest_vertex_draw_order(
            document,
            objects,
            object_block_depths,
            depth_tolerance=depth_tolerance,
        )
    return _apply_legacy_layer_draw_order(document, objects, placements)


__all__ = [
    "apply_connected_setup_draw_order",
    "connected_draw_order_component_ids",
]
