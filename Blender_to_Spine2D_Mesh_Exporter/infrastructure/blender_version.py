"""Blender runtime version contract for the Rewrite add-on."""

from __future__ import annotations

from typing import Any, Iterable


MINIMUM_BLENDER_VERSION = (5, 2, 0)
MINIMUM_BLENDER_VERSION_TEXT = ".".join(str(part) for part in MINIMUM_BLENDER_VERSION)


class UnsupportedBlenderVersionError(RuntimeError):
    """Raised when the add-on is loaded in an unsupported Blender runtime."""


def normalize_blender_version(value: Iterable[Any]) -> tuple[int, int, int]:
    """Normalize one Blender version-like iterable to a strict three-part tuple."""

    try:
        resolved = tuple(int(part) for part in value)
    except Exception as exc:
        raise UnsupportedBlenderVersionError(
            f"Unable to read Blender runtime version from {value!r}"
        ) from exc
    if len(resolved) < 3:
        raise UnsupportedBlenderVersionError(
            f"Blender runtime version must contain at least three parts, got {resolved!r}"
        )
    normalized = resolved[:3]
    if any(part < 0 for part in normalized):
        raise UnsupportedBlenderVersionError(
            f"Blender runtime version cannot contain negative values: {normalized!r}"
        )
    return normalized


def require_supported_blender_runtime(bpy_module: Any) -> tuple[int, int, int]:
    """Validate Blender 5.2+ before any add-on registration mutates runtime state."""

    app = getattr(bpy_module, "app", None)
    raw_version = getattr(app, "version", None)
    if raw_version is None:
        raise UnsupportedBlenderVersionError(
            "Blender bpy.app.version is unavailable; Blender 5.2 or newer is required"
        )
    version = normalize_blender_version(raw_version)
    if version < MINIMUM_BLENDER_VERSION:
        actual = ".".join(str(part) for part in version)
        raise UnsupportedBlenderVersionError(
            "Blender to Spine2D Mesh Exporter requires Blender "
            f"{MINIMUM_BLENDER_VERSION_TEXT} or newer; detected {actual}"
        )
    return version


__all__ = [
    "MINIMUM_BLENDER_VERSION",
    "MINIMUM_BLENDER_VERSION_TEXT",
    "UnsupportedBlenderVersionError",
    "normalize_blender_version",
    "require_supported_blender_runtime",
]
