"""Application-local scalar and identity validation for A1 rewrite contracts."""

from __future__ import annotations

from math import isfinite
from typing import Any


def require_integer(
    value: Any,
    field_name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return one strict integer while rejecting ``bool`` and invalid ranges."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field_name} must be at most {maximum}")
    return value


def require_finite_number(
    value: Any,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    """Return one finite number while rejecting ``bool`` and invalid ranges."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    if minimum is not None:
        invalid = resolved < minimum if minimum_inclusive else resolved <= minimum
        if invalid:
            operator = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{field_name} must be {operator} {minimum}")
    if maximum is not None:
        invalid = resolved > maximum if maximum_inclusive else resolved >= maximum
        if invalid:
            operator = "<=" if maximum_inclusive else "<"
            raise ValueError(f"{field_name} must be {operator} {maximum}")
    return resolved


def require_non_empty_string(value: Any, field_name: str) -> str:
    """Return one non-empty string without changing its contents."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def require_identity(value: Any, field_name: str) -> str:
    """Return one canonical identity with no boundary whitespace."""

    resolved = require_non_empty_string(value, field_name)
    if resolved != resolved.strip():
        raise ValueError(
            f"{field_name} cannot contain leading or trailing whitespace"
        )
    return resolved


__all__ = [
    "require_finite_number",
    "require_identity",
    "require_integer",
    "require_non_empty_string",
]
