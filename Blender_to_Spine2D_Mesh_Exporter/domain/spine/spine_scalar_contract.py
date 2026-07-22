"""Shared strict scalar predicates for Blender-independent Spine contracts."""

from __future__ import annotations

from math import isfinite


def require_name(value: object, field_name: str = "name") -> str:
    """Return one non-empty string without normalizing its spelling."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if not value.strip():
        raise ValueError(f"{field_name} cannot be empty")
    return value


def is_finite_number(value: object) -> bool:
    """Return whether value is a non-boolean finite int or float."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return isinstance(value, int) or isfinite(value)


def require_finite_number(value: object, field_name: str) -> int | float:
    """Return one strict finite number while preserving its exact numeric type."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    if not is_finite_number(value):
        raise ValueError(f"{field_name} must be finite")
    return value


__all__ = ["is_finite_number", "require_finite_number", "require_name"]
