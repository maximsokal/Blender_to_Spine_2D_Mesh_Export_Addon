"""Local scalar, identity, tuple, and vector contracts for geometry domain models."""

from __future__ import annotations

from math import isfinite
from typing import Any, Iterable, Tuple, TypeVar


_T = TypeVar("_T")


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


def require_finite_number(value: Any, field_name: str) -> float:
    """Return one finite number while rejecting ``bool``."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    return resolved


def require_non_empty_string(value: Any, field_name: str) -> str:
    """Return one non-empty string without normalizing caller-owned display text."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def require_identity(value: Any, field_name: str) -> str:
    """Return one canonical identity with no leading or trailing whitespace."""

    resolved = require_non_empty_string(value, field_name)
    if resolved != resolved.strip():
        raise ValueError(
            f"{field_name} cannot contain leading or trailing whitespace"
        )
    return resolved


def require_exact_type(value: Any, expected_type: type[_T], field_name: str) -> _T:
    """Return ``value`` only when its concrete type is exactly ``expected_type``."""

    if not isinstance(expected_type, type):
        raise TypeError("expected_type must be type")
    if type(value) is not expected_type:
        raise TypeError(f"{field_name} must be {expected_type.__name__}")
    return value


def require_optional_exact_type(
    value: Any,
    expected_type: type[_T],
    field_name: str,
) -> _T | None:
    """Validate an optional value without accepting a different ID class."""

    if value is None:
        return None
    return require_exact_type(value, expected_type, field_name)


def require_tuple_items(
    value: Any,
    item_type: type[_T],
    field_name: str,
    *,
    minimum_length: int | None = None,
    exact_length: int | None = None,
) -> Tuple[_T, ...]:
    """Validate one immutable homogeneous tuple using exact item types."""

    if not isinstance(item_type, type):
        raise TypeError("item_type must be type")
    if not isinstance(value, tuple):
        raise TypeError(f"{field_name} must be tuple")
    if exact_length is not None and len(value) != exact_length:
        raise ValueError(
            f"{field_name} must contain exactly {exact_length} {item_type.__name__} values"
        )
    if minimum_length is not None and len(value) < minimum_length:
        raise ValueError(
            f"{field_name} must contain at least {minimum_length} {item_type.__name__} values"
        )
    for index, item in enumerate(value):
        require_exact_type(item, item_type, f"{field_name}[{index}]")
    return value


def require_finite_vector(
    value: Any,
    size: int,
    field_name: str,
) -> tuple[float | int, ...]:
    """Validate an immutable finite vector while preserving caller numeric values."""

    require_integer(size, "size", minimum=1)
    if not isinstance(value, tuple):
        raise TypeError(f"{field_name} must be tuple")
    if len(value) != size:
        raise ValueError(f"{field_name} must contain {size} components")
    for index, component in enumerate(value):
        require_finite_number(component, f"{field_name}[{index}]")
    return value


def vector_is_zero(value: Iterable[int | float]) -> bool:
    """Return whether every already-validated vector component is exactly zero."""

    return all(component == 0 for component in value)


__all__ = [
    "require_exact_type",
    "require_finite_number",
    "require_finite_vector",
    "require_identity",
    "require_integer",
    "require_non_empty_string",
    "require_optional_exact_type",
    "require_tuple_items",
    "vector_is_zero",
]
