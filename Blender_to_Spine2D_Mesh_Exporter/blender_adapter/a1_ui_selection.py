"""Capture deterministic Blender object selection and immutable object profiles."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Tuple


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _ObjectExportProfile:
    """Live Blender object handle plus values captured before preparation starts."""

    source_object: Any
    object_name: str
    sequence_start_frame: int
    sequence_frame_count: int
    connect_enabled: bool

    def __post_init__(self) -> None:
        if self.source_object is None:
            raise ValueError("source_object cannot be None")
        if not isinstance(self.object_name, str) or not self.object_name.strip():
            raise ValueError("object_name must be a non-empty string")
        if (
            not isinstance(self.sequence_start_frame, int)
            or self.sequence_start_frame < 0
        ):
            raise ValueError(
                "sequence_start_frame must be a non-negative integer"
            )
        if (
            not isinstance(self.sequence_frame_count, int)
            or self.sequence_frame_count < 0
        ):
            raise ValueError(
                "sequence_frame_count must be a non-negative integer"
            )
        if not isinstance(self.connect_enabled, bool):
            raise TypeError("connect_enabled must be bool")


def _object_name(obj: Any) -> str:
    value = str(
        getattr(obj, "name_full", None)
        or getattr(obj, "name", None)
        or ""
    ).strip()
    if not value:
        raise ValueError("Selected mesh object has an empty name")
    return value


def _rna_identity(value: Any) -> tuple[str, object]:
    """Return stable identity across transient Blender RNA wrapper instances."""

    pointer = getattr(value, "as_pointer", None)
    if callable(pointer):
        try:
            resolved = int(pointer())
            if resolved:
                return ("RNA_POINTER", resolved)
        except Exception:
            logger.debug("Unable to read Blender RNA pointer", exc_info=True)
    name = str(
        getattr(value, "name_full", None)
        or getattr(value, "name", None)
        or ""
    ).strip()
    if name:
        return ("RNA_NAME", name)
    return ("PYTHON_ID", id(value))


def _active_mesh(context: Any) -> Any:
    obj = getattr(context, "active_object", None)
    if obj is None:
        raise ValueError("There is no active object")
    if getattr(obj, "type", None) != "MESH":
        raise ValueError(
            f"The active object '{_object_name(obj)}' is not a Mesh"
        )
    if getattr(obj, "data", None) is None:
        raise ValueError(
            f"The active Mesh object '{_object_name(obj)}' has no data"
        )
    return obj


def _ordered_selected_meshes(context: Any) -> Tuple[Any, ...]:
    """Return active Mesh first and remaining unique meshes in deterministic order."""

    raw_selected = tuple(
        obj
        for obj in getattr(context, "selected_objects", ())
        if getattr(obj, "type", None) == "MESH"
    )
    unique_by_identity: dict[tuple[str, object], Any] = {}
    for obj in raw_selected:
        unique_by_identity.setdefault(_rna_identity(obj), obj)
    selected = tuple(unique_by_identity.values())
    if len(selected) < 2:
        raise ValueError("Select at least two Mesh objects for multi-export")

    active = getattr(context, "active_object", None)
    active_identity = None if active is None else _rna_identity(active)
    active_match = next(
        (obj for obj in selected if _rna_identity(obj) == active_identity),
        None,
    )
    ordered: list[Any] = []
    if active_match is not None:
        ordered.append(active_match)
    ordered.extend(
        sorted(
            (
                obj
                for obj in selected
                if active_match is None
                or _rna_identity(obj) != _rna_identity(active_match)
            ),
            key=lambda obj: (_object_name(obj).casefold(), _object_name(obj)),
        )
    )
    return tuple(ordered)


def _connect_enabled(obj: Any) -> bool:
    settings = getattr(obj, "spine2d_connect_settings", None)
    return bool(settings is not None and getattr(settings, "enabled", False))


def _capture_object_profile(
    obj: Any,
    *,
    sequence_start_frame: int,
    sequence_frame_count: int,
    connect_enabled: bool,
) -> _ObjectExportProfile:
    return _ObjectExportProfile(
        source_object=obj,
        object_name=_object_name(obj),
        sequence_start_frame=max(0, int(sequence_start_frame)),
        sequence_frame_count=max(0, int(sequence_frame_count)),
        connect_enabled=bool(connect_enabled),
    )


__all__ = [
    "_ObjectExportProfile",
    "_active_mesh",
    "_capture_object_profile",
    "_connect_enabled",
    "_object_name",
    "_ordered_selected_meshes",
    "_rna_identity",
]
