"""Normalize ordered constraint collections for legacy Spine runtimes.

Spine 3.8-4.2 runtimes build their update cache by iterating integer phases
``0..constraint_count-1`` and selecting one constraint whose authored ``order`` equals
the current phase. Missing phases or duplicate values therefore leave constraints out of
the runtime cache. Canonical rig builders may preserve historical order values for
cross-version parity, so target codecs use this module to create a detached runtime-safe
JSON view without mutating the typed :class:`SpineDocument`.

Connected documents add one more requirement: a constraint that changes an ancestor
bone must run before constraints that depend on descendants of that bone. Otherwise the
runtime may reset a descendant after a world-space constraint has derived its applied
transform. This is especially unsafe when the descendant has a singular parent matrix,
such as the historical two-axis ``scaleX == 0`` helper chain.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
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


@dataclass(frozen=True, slots=True)
class _RuntimeConstraintRecord:
    authored_order: int
    collection_rank: int
    index: int
    collection: str
    name: str
    constraint: dict[str, Any]
    constrained_bones: tuple[str, ...]
    dependency_bones: tuple[str, ...]

    @property
    def stable_key(self) -> tuple[int, int, int]:
        return (self.authored_order, self.collection_rank, self.index)


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


def _build_bone_parent_map(output: dict[str, Any]) -> dict[str, str | None]:
    """Return and validate the serialized bone hierarchy when it is available."""

    raw_bones = output.get("bones")
    if raw_bones is None:
        return {}
    bones = _require_sequence(raw_bones, path="document.bones")
    parent_by_name: dict[str, str | None] = {}
    for index, raw_bone in enumerate(bones):
        path = f"document.bones[{index}]"
        bone = _require_mapping(raw_bone, path=path)
        name = _require_name(bone.get("name"), path=f"{path}.name")
        if name in parent_by_name:
            raise ValueError(f"Bone names must be unique: {name!r}")
        raw_parent = bone.get("parent")
        if raw_parent is None:
            parent = None
        else:
            parent = _require_name(raw_parent, path=f"{path}.parent")
        parent_by_name[name] = parent

    for name, parent in parent_by_name.items():
        if parent is not None and parent not in parent_by_name:
            raise ValueError(f"Bone {name!r} references unknown parent {parent!r}")

    for name in parent_by_name:
        visited: set[str] = set()
        current: str | None = name
        while current is not None:
            if current in visited:
                raise ValueError(f"Bone hierarchy contains a cycle at {current!r}")
            visited.add(current)
            current = parent_by_name[current]
    return parent_by_name


def _constraint_bones(
    constraint: dict[str, Any],
    *,
    path: str,
    parent_by_name: dict[str, str | None],
) -> tuple[str, ...]:
    """Read bones modified by IK/transform/path/physics constraints."""

    if not parent_by_name:
        return ()

    result: list[str] = []
    raw_bones = constraint.get("bones")
    if raw_bones is not None:
        bones = _require_sequence(raw_bones, path=f"{path}.bones")
        for index, raw_name in enumerate(bones):
            name = _require_name(raw_name, path=f"{path}.bones[{index}]")
            if name not in parent_by_name:
                raise ValueError(
                    f"{path}.bones[{index}] references unknown bone {name!r}"
                )
            result.append(name)

    raw_bone = constraint.get("bone")
    if raw_bone is not None:
        name = _require_name(raw_bone, path=f"{path}.bone")
        if name not in parent_by_name:
            raise ValueError(f"{path}.bone references unknown bone {name!r}")
        result.append(name)

    return tuple(dict.fromkeys(result))


def _constraint_dependencies(
    constraint: dict[str, Any],
    constrained_bones: tuple[str, ...],
    *,
    path: str,
    parent_by_name: dict[str, str | None],
) -> tuple[str, ...]:
    """Return bones whose current transform may be read by the constraint."""

    dependencies = list(constrained_bones)
    raw_target = constraint.get("target")
    if raw_target is not None and parent_by_name:
        target = _require_name(raw_target, path=f"{path}.target")
        # Path constraints target slots rather than bones. A target participates in the
        # hierarchy graph only when its name resolves to an actual serialized bone.
        if target in parent_by_name:
            dependencies.append(target)
    return tuple(dict.fromkeys(dependencies))


def _is_strict_ancestor(
    ancestor: str,
    descendant: str,
    parent_by_name: dict[str, str | None],
) -> bool:
    current = parent_by_name.get(descendant)
    while current is not None:
        if current == ancestor:
            return True
        current = parent_by_name[current]
    return False


def _requires_ancestor_precedence(
    earlier: _RuntimeConstraintRecord,
    later: _RuntimeConstraintRecord,
    parent_by_name: dict[str, str | None],
) -> bool:
    """Return true when ``earlier`` writes an ancestor used by ``later``."""

    if not parent_by_name:
        return False
    return any(
        _is_strict_ancestor(written, dependency, parent_by_name)
        for written in earlier.constrained_bones
        for dependency in later.dependency_bones
    )


def _topological_runtime_order(
    records: list[_RuntimeConstraintRecord],
    parent_by_name: dict[str, str | None],
) -> tuple[_RuntimeConstraintRecord, ...]:
    """Stable-topologically order constraints by hierarchy, then authored order."""

    count = len(records)
    outgoing: list[set[int]] = [set() for _ in range(count)]
    indegree = [0] * count

    for left_index in range(count):
        left = records[left_index]
        for right_index in range(left_index + 1, count):
            right = records[right_index]
            left_before_right = _requires_ancestor_precedence(
                left,
                right,
                parent_by_name,
            )
            right_before_left = _requires_ancestor_precedence(
                right,
                left,
                parent_by_name,
            )
            if left_before_right:
                outgoing[left_index].add(right_index)
            if right_before_left:
                outgoing[right_index].add(left_index)

    for source_index, targets in enumerate(outgoing):
        for target_index in targets:
            if target_index == source_index:
                raise ValueError(
                    f"Constraint hierarchy produced a self dependency for "
                    f"{records[source_index].name!r}"
                )
            indegree[target_index] += 1

    ready: list[tuple[tuple[int, int, int], int]] = []
    for record_index, record in enumerate(records):
        if indegree[record_index] == 0:
            heapq.heappush(ready, (record.stable_key, record_index))

    ordered: list[_RuntimeConstraintRecord] = []
    while ready:
        _stable_key, record_index = heapq.heappop(ready)
        ordered.append(records[record_index])
        for target_index in sorted(outgoing[record_index]):
            indegree[target_index] -= 1
            if indegree[target_index] == 0:
                heapq.heappush(
                    ready,
                    (records[target_index].stable_key, target_index),
                )

    if len(ordered) != count:
        blocked = tuple(
            sorted(records[index].name for index, degree in enumerate(indegree) if degree)
        )
        raise ValueError(
            "Constraint hierarchy contains a cyclic runtime dependency: "
            f"{blocked}"
        )
    return tuple(ordered)


def normalize_runtime_constraint_orders(
    output: dict[str, Any],
    *,
    collections: Sequence[str],
) -> tuple[RuntimeConstraintOrderAssignment, ...]:
    """Rewrite legacy runtime constraint orders to one global ``0..N-1`` schedule.

    The detached serialized bone hierarchy is used to add dependency edges: constraints
    that modify ancestor bones run before constraints that consume descendant transforms.
    Among constraints not related by hierarchy, authored order, runtime collection
    precedence, and original collection index remain the deterministic stable tie-break.

    ``output`` is expected to be a detached JSON mapping produced by a serializer. The
    function mutates only the constraint dictionaries inside that detached mapping and
    returns immutable diagnostic assignments. Empty or absent collections are allowed.
    Constraint names must remain globally unique so metadata and animation references
    cannot become ambiguous. Cyclic hierarchy dependencies fail closed before any order
    field is changed.
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

    parent_by_name = _build_bone_parent_map(output)
    records: list[_RuntimeConstraintRecord] = []
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
            constrained_bones = _constraint_bones(
                constraint,
                path=path,
                parent_by_name=parent_by_name,
            )
            records.append(
                _RuntimeConstraintRecord(
                    authored_order=authored_order,
                    collection_rank=collection_rank,
                    index=index,
                    collection=collection_name,
                    name=name,
                    constraint=constraint,
                    constrained_bones=constrained_bones,
                    dependency_bones=_constraint_dependencies(
                        constraint,
                        constrained_bones,
                        path=path,
                        parent_by_name=parent_by_name,
                    ),
                )
            )

    ordered = _topological_runtime_order(records, parent_by_name)
    assignments: list[RuntimeConstraintOrderAssignment] = []
    for runtime_order, record in enumerate(ordered):
        record.constraint["order"] = runtime_order
        assignments.append(
            RuntimeConstraintOrderAssignment(
                collection=record.collection,
                index=record.index,
                name=record.name,
                authored_order=record.authored_order,
                runtime_order=runtime_order,
            )
        )

    return tuple(assignments)


__all__ = [
    "RuntimeConstraintOrderAssignment",
    "normalize_runtime_constraint_orders",
]
