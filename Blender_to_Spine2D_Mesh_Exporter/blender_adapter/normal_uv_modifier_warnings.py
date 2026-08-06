"""Detect Blender modifiers ignored by the public Normal / UV Segments route.

The public Normal route intentionally exports the original Mesh datablock. Blender
modifiers may therefore change the viewport appearance without changing the geometry
serialized to Spine. This module contains no bpy dependency so the detection policy is
unit-testable and can be reused by UI/readiness diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from ..domain.baking import A1TextureExportMode


@dataclass(frozen=True, slots=True)
class IgnoredNormalUvModifier:
    """One enabled modifier whose evaluated geometry is absent from Normal export."""

    object_name: str
    modifier_name: str
    modifier_type: str
    show_viewport: bool
    show_render: bool

    def __post_init__(self) -> None:
        for field_name in ("object_name", "modifier_name", "modifier_type"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.show_viewport, bool):
            raise TypeError("show_viewport must be bool")
        if not isinstance(self.show_render, bool):
            raise TypeError("show_render must be bool")
        if not (self.show_viewport or self.show_render):
            raise ValueError("ignored modifier must be enabled in viewport or render")


def _texture_mode_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip().upper()


def _object_name(obj: Any) -> str:
    return str(
        getattr(obj, "name_full", None)
        or getattr(obj, "name", None)
        or "<unnamed mesh>"
    ).strip()


def _modifier_name(modifier: Any) -> str:
    return str(getattr(modifier, "name", None) or "<unnamed modifier>").strip()


def _modifier_type(modifier: Any) -> str:
    return str(getattr(modifier, "type", None) or "UNKNOWN").strip().upper()


def collect_normal_uv_ignored_modifiers(
    objects: Iterable[Any],
    texture_export_mode: A1TextureExportMode | str,
) -> Tuple[IgnoredNormalUvModifier, ...]:
    """Return enabled modifiers ignored by Normal / UV Segments.

    Order is deterministic and follows the incoming object order and each Blender
    modifier-stack order. Modifiers disabled for both viewport and render are excluded
    because they cannot explain a visible/exported appearance mismatch.
    """

    if isinstance(objects, (str, bytes)):
        raise TypeError("objects must be an iterable of Blender-like objects")
    try:
        resolved_objects = tuple(objects)
    except TypeError as exc:
        raise TypeError("objects must be iterable") from exc

    if _texture_mode_value(texture_export_mode) != (
        A1TextureExportMode.NORMAL_UV_SEGMENTS.value
    ):
        return ()

    result: list[IgnoredNormalUvModifier] = []
    for obj in resolved_objects:
        if obj is None or str(getattr(obj, "type", "")).upper() != "MESH":
            continue
        object_name = _object_name(obj)
        try:
            modifiers = tuple(getattr(obj, "modifiers", ()) or ())
        except TypeError as exc:
            raise TypeError(
                f"Object {object_name!r} modifiers must be iterable"
            ) from exc

        for modifier in modifiers:
            show_viewport = bool(getattr(modifier, "show_viewport", True))
            show_render = bool(getattr(modifier, "show_render", True))
            if not (show_viewport or show_render):
                continue
            result.append(
                IgnoredNormalUvModifier(
                    object_name=object_name,
                    modifier_name=_modifier_name(modifier),
                    modifier_type=_modifier_type(modifier),
                    show_viewport=show_viewport,
                    show_render=show_render,
                )
            )
    return tuple(result)


def group_ignored_modifiers_by_object(
    values: Iterable[IgnoredNormalUvModifier],
) -> Tuple[Tuple[str, Tuple[IgnoredNormalUvModifier, ...]], ...]:
    """Group descriptors without losing object or modifier-stack order."""

    if isinstance(values, (str, bytes)):
        raise TypeError("values must contain IgnoredNormalUvModifier instances")
    try:
        resolved = tuple(values)
    except TypeError as exc:
        raise TypeError("values must be iterable") from exc
    if not all(isinstance(value, IgnoredNormalUvModifier) for value in resolved):
        raise TypeError("values must contain IgnoredNormalUvModifier instances")

    grouped: dict[str, list[IgnoredNormalUvModifier]] = {}
    for value in resolved:
        grouped.setdefault(value.object_name, []).append(value)
    return tuple(
        (object_name, tuple(modifiers))
        for object_name, modifiers in grouped.items()
    )


__all__ = [
    "IgnoredNormalUvModifier",
    "collect_normal_uv_ignored_modifiers",
    "group_ignored_modifiers_by_object",
]
