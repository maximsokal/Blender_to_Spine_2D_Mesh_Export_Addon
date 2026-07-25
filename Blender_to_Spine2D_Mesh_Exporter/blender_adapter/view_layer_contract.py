"""Active View Layer validation for source-only camera projection."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Iterator, Tuple


logger = logging.getLogger(__name__)


class ViewLayerContractError(ValueError):
    """Raised when a source object has no direct camera-render path."""


@dataclass(frozen=True, slots=True)
class SourceLayerCollectionState:
    collection_name: str
    excluded: bool
    holdout: bool
    indirect_only: bool

    def __post_init__(self) -> None:
        if not isinstance(self.collection_name, str) or not self.collection_name.strip():
            raise ValueError("collection_name must be a non-empty string")
        for field_name in ("excluded", "holdout", "indirect_only"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")

    @property
    def direct_camera_renderable(self) -> bool:
        return not self.excluded and not self.holdout and not self.indirect_only


def _name(value: Any) -> str:
    return str(
        getattr(value, "name_full", None)
        or getattr(value, "name", None)
        or ""
    ).strip()


def _rna_identity(value: Any) -> int:
    pointer = getattr(value, "as_pointer", None)
    if callable(pointer):
        try:
            resolved = int(pointer())
            if resolved:
                return resolved
        except Exception:
            logger.debug("Unable to read Blender RNA pointer", exc_info=True)
    return id(value)


def _iter_layer_collection_states(
    layer_collection: Any,
    *,
    parent_excluded: bool = False,
    parent_holdout: bool = False,
    parent_indirect_only: bool = False,
) -> Iterator[tuple[Any, SourceLayerCollectionState]]:
    if layer_collection is None:
        return
    excluded = parent_excluded or bool(getattr(layer_collection, "exclude", False))
    holdout = parent_holdout or bool(getattr(layer_collection, "holdout", False))
    indirect_only = parent_indirect_only or bool(
        getattr(layer_collection, "indirect_only", False)
    )
    collection = getattr(layer_collection, "collection", None)
    name = _name(collection) or _name(layer_collection) or "Scene Collection"
    yield collection, SourceLayerCollectionState(
        collection_name=name,
        excluded=excluded,
        holdout=holdout,
        indirect_only=indirect_only,
    )
    try:
        children = tuple(getattr(layer_collection, "children", ()))
    except Exception as exc:
        raise ViewLayerContractError(
            f"Unable to inspect child Layer Collections below '{name}'"
        ) from exc
    for child in children:
        yield from _iter_layer_collection_states(
            child,
            parent_excluded=excluded,
            parent_holdout=holdout,
            parent_indirect_only=indirect_only,
        )


def source_layer_collection_states(
    source_obj: Any,
    view_layer: Any,
) -> Tuple[SourceLayerCollectionState, ...]:
    if source_obj is None:
        raise TypeError("source_obj cannot be None")
    if view_layer is None:
        raise ViewLayerContractError("An active Blender View Layer is required")
    try:
        source_collections = tuple(getattr(source_obj, "users_collection", ()))
    except Exception as exc:
        raise ViewLayerContractError(
            "Unable to inspect source object collection membership"
        ) from exc
    source_ids = {_rna_identity(collection) for collection in source_collections}
    states = tuple(
        state
        for collection, state in _iter_layer_collection_states(
            getattr(view_layer, "layer_collection", None)
        )
        if collection is not None and _rna_identity(collection) in source_ids
    )
    return tuple(
        sorted(
            states,
            key=lambda state: (
                state.collection_name.casefold(),
                state.excluded,
                state.holdout,
                state.indirect_only,
            ),
        )
    )


def validate_source_view_layer_for_camera_projection(
    source_obj: Any,
    view_layer: Any,
) -> Tuple[SourceLayerCollectionState, ...]:
    """Require one non-excluded, non-holdout, non-indirect source collection path."""

    if source_obj is None:
        raise TypeError("source_obj cannot be None")
    if view_layer is None:
        raise ViewLayerContractError("An active Blender View Layer is required")
    source_name = _name(source_obj) or "Source"
    try:
        view_objects = getattr(view_layer, "objects", ())
        in_view_layer = any(
            item is source_obj or _rna_identity(item) == _rna_identity(source_obj)
            for item in view_objects
        )
    except Exception as exc:
        raise ViewLayerContractError(
            f"Unable to inspect objects in View Layer '{_name(view_layer)}'"
        ) from exc
    if not in_view_layer:
        raise ViewLayerContractError(
            f"Source object '{source_name}' is excluded from active View Layer "
            f"'{_name(view_layer)}'"
        )

    states = source_layer_collection_states(source_obj, view_layer)
    if not states:
        raise ViewLayerContractError(
            f"Source object '{source_name}' has no Layer Collection path in active "
            f"View Layer '{_name(view_layer)}'"
        )
    if not any(state.direct_camera_renderable for state in states):
        details = tuple(
            (
                state.collection_name,
                state.excluded,
                state.holdout,
                state.indirect_only,
            )
            for state in states
        )
        raise ViewLayerContractError(
            f"Source object '{source_name}' is only available through excluded, Holdout, "
            f"or Indirect Only Layer Collections: {details}"
        )
    return states
