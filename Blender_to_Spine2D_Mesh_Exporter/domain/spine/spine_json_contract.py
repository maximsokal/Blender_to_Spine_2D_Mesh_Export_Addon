"""Recursive JSON-safety contract for generated Spine payloads."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any


class SpineJsonContractError(ValueError):
    """Raised when a value cannot be represented as strict, interoperable JSON."""

    def __init__(self, path: str, message: str):
        self.path = path
        self.reason = message
        super().__init__(f"{path}: {message}")


def json_path_key(path: str, key: str) -> str:
    """Append one mapping key to a deterministic JSON-style diagnostic path."""

    if key.isidentifier():
        return f"{path}.{key}"
    escaped = key.replace("\\", "\\\\").replace('"', '\\"')
    return f'{path}["{escaped}"]'


def validate_json_value(value: Any, *, path: str = "$") -> None:
    """Validate one recursively JSON-safe value.

    Accepted containers are mappings with string keys plus ``list`` and ``tuple``.
    Numbers must be finite. ``bool`` is accepted as a JSON boolean but is handled
    before integer checks so callers can distinguish booleans from numeric fields.
    Circular container graphs are rejected with the path where the cycle re-enters.
    """

    _validate_json_value(value, path=path, active_container_ids=set())


def validate_json_mapping(value: Any, *, path: str = "$") -> None:
    """Validate that ``value`` is a JSON-safe mapping with string keys."""

    if not isinstance(value, Mapping):
        raise SpineJsonContractError(path, "must be a mapping")
    validate_json_value(value, path=path)


def _validate_json_value(
    value: Any,
    *,
    path: str,
    active_container_ids: set[int],
) -> None:
    if value is None or isinstance(value, (str, bool)):
        return

    if isinstance(value, int):
        return

    if isinstance(value, float):
        if not isfinite(value):
            raise SpineJsonContractError(path, "number must be finite")
        return

    if isinstance(value, Mapping):
        container_id = id(value)
        if container_id in active_container_ids:
            raise SpineJsonContractError(
                path,
                "circular mapping reference is not JSON-safe",
            )
        active_container_ids.add(container_id)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise SpineJsonContractError(
                        path,
                        f"mapping keys must be str, got {type(key).__name__}",
                    )
                _validate_json_value(
                    item,
                    path=json_path_key(path, key),
                    active_container_ids=active_container_ids,
                )
        finally:
            active_container_ids.remove(container_id)
        return

    if isinstance(value, (list, tuple)):
        container_id = id(value)
        if container_id in active_container_ids:
            raise SpineJsonContractError(
                path,
                "circular sequence reference is not JSON-safe",
            )
        active_container_ids.add(container_id)
        try:
            for index, item in enumerate(value):
                _validate_json_value(
                    item,
                    path=f"{path}[{index}]",
                    active_container_ids=active_container_ids,
                )
        finally:
            active_container_ids.remove(container_id)
        return

    raise SpineJsonContractError(
        path,
        "unsupported JSON value type "
        f"{type(value).__name__}; expected null, bool, str, finite number, "
        "list, tuple, or mapping",
    )


__all__ = [
    "SpineJsonContractError",
    "json_path_key",
    "validate_json_mapping",
    "validate_json_value",
]
