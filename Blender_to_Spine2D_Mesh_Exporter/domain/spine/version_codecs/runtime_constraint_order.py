"""Normalize ordered constraint collections for legacy Spine runtimes.

Spine 3.8-4.2 runtimes build their update cache by iterating integer phases
``0..constraint_count-1`` and selecting one constraint whose authored ``order`` equals
the current phase. Missing phases or duplicate values therefore leave constraints out of
the runtime cache. Canonical rig builders may preserve historical order values for
cross-version parity, so target codecs use this module to create a detached runtime-safe
JSON view without mutating the typed :class:`SpineDocument`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class RuntimeConstraintOrderAssignment:
    """Diagnostic mapping from one authored order to its runtime-safe order."""

    collection: str
    index: int
    name: str
    authored_order: int
    runtime_order: int

    def __post_init__(self) -> None:
        if not isinstance(self.collection, str) or not self.collection.strip():
            raise ValueError("collection must be a non-empty string")
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("index must be a non-negative integer")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        for field_name in ("authored_order", "runtime_order"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")


def _require_mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a JSON object")
    return value


def _require_sequence(value: Any, *, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{path} must be a JSON array")
    return value


def _require_name(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{path} must be a non-empty string")
    return value


def _require_order(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{path} must be a non-negative integer")
    return value


def normalize_runtime_constraint_orders(
    output: dict[str, Any],
    *,
    collections: Sequence[str],
) -> tuple[RuntimeConstraintOrderAssignment, ...]:
    """Rewrite legacy runtime constraint orders to one global ``0..N-1`` schedule.

    The authored dependency relation is preserved by sorting first by authored order,
    then by runtime collection precedence, then by the original collection index. This
    tie-break matches Spine's legacy update-cache search order (IK, transform, path,
    physics) and gives intentionally tied independent constraints a deterministic phase.

    ``output`` is expected to be a detached JSON mapping produced by a serializer. The
    function mutates only the constraint dictionaries inside that detached mapping and
    returns immutable diagnostic assignments. Empty or absent collections are allowed.
    Constraint names must remain globally unique so metadata and animation references
    cannot become ambiguous.
    """

    if not isinstance(output, dict):
        raise TypeError("output must be a JSON object")
    if not isinstance(collections, Sequence) or isinstance(collections, (str, bytes)):
        raise TypeError("collections must be a sequence of collection names")

    resolved_collections = tuple(collections)
    if not resolved_collections:
        raise ValueError("collections must not be empty")
    if not all(isinstance(name, str) and name.strip() for name in resolved_collections):
        raise TypeError("collections must contain non-empty strings")
    if len(resolved_collections) != len(set(resolved_collections)):
        raise ValueError("collections must not contain duplicates")

    records: list[tuple[int, int, int, str, str, dict[str, Any]]] = []
    seen_names: set[str] = set()
    for collection_rank, collection_name in enumerate(resolved_collections):
        raw_collection = output.get(collection_name)
        if raw_collection is None:
            continue
        collection = _require_sequence(
            raw_collection,
            path=f"document.{collection_name}",
        )
        for index, raw_constraint in enumerate(collection):
            path = f"document.{collection_name}[{index}]"
            constraint = _require_mapping(raw_constraint, path=path)
            name = _require_name(constraint.get("name"), path=f"{path}.name")
            if name in seen_names:
                raise ValueError(
                    "Constraint names must be globally unique before runtime order "
                    f"normalization: {name!r}"
                )
            seen_names.add(name)
            authored_order = _require_order(
                constraint.get("order", 0),
                path=f"{path}.order",
            )
            records.append(
                (
                    authored_order,
                    collection_rank,
                    index,
                    collection_name,
                    name,
                    constraint,
                )
            )

    records.sort(key=lambda item: (item[0], item[1], item[2]))
    assignments: list[RuntimeConstraintOrderAssignment] = []
    for runtime_order, record in enumerate(records):
        authored_order, _rank, index, collection_name, name, constraint = record
        constraint["order"] = runtime_order
        assignments.append(
            RuntimeConstraintOrderAssignment(
                collection=collection_name,
                index=index,
                name=name,
                authored_order=authored_order,
                runtime_order=runtime_order,
            )
        )

    return tuple(assignments)


__all__ = [
    "RuntimeConstraintOrderAssignment",
    "normalize_runtime_constraint_orders",
]
