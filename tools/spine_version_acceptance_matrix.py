"""Canonical positive export matrix used by Blender/runtime acceptance tooling."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Tuple


@dataclass(frozen=True, slots=True)
class SpineVersionAcceptanceCase:
    """One production target/profile/scope combination that must export successfully."""

    key: str
    target: str
    exact_version: str
    profile: str
    scope: str
    object_count: int

    def __post_init__(self) -> None:
        for field_name in ("key", "target", "exact_version", "profile", "scope"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.object_count, int) or isinstance(self.object_count, bool):
            raise TypeError("object_count must be int")
        if self.object_count <= 0:
            raise ValueError("object_count must be positive")
        expected_count = {
            "SINGLE_OBJECT": 1,
            "STANDALONE_MULTI_OBJECT": 3,
            "CONNECTED_MULTI_OBJECT": 3,
            "MIXED_MULTI_OBJECT": 3,
        }.get(self.scope)
        if expected_count is None:
            raise ValueError(f"Unsupported scope: {self.scope!r}")
        if self.object_count != expected_count:
            raise ValueError(
                f"{self.scope} requires object_count={expected_count}, "
                f"received {self.object_count}"
            )


_TARGETS: Tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "SPINE_3_8",
        "3.8.99",
        ("THREE_AXIS_ROTATION", "TWO_AXIS_ROTATION_SCALE"),
        ("SINGLE_OBJECT", "STANDALONE_MULTI_OBJECT"),
    ),
    (
        "SPINE_4_0",
        "4.0.64",
        ("TWO_AXIS_ROTATION_SCALE",),
        ("SINGLE_OBJECT", "STANDALONE_MULTI_OBJECT"),
    ),
    (
        "SPINE_4_1",
        "4.1.24",
        ("TWO_AXIS_ROTATION_SCALE",),
        ("SINGLE_OBJECT", "STANDALONE_MULTI_OBJECT"),
    ),
    (
        "SPINE_4_2",
        "4.2.43",
        ("THREE_AXIS_ROTATION", "TWO_AXIS_ROTATION_SCALE"),
        (
            "SINGLE_OBJECT",
            "STANDALONE_MULTI_OBJECT",
            "CONNECTED_MULTI_OBJECT",
            "MIXED_MULTI_OBJECT",
        ),
    ),
    (
        "SPINE_4_3",
        "4.3.23",
        ("THREE_AXIS_ROTATION", "TWO_AXIS_ROTATION_SCALE"),
        ("SINGLE_OBJECT", "STANDALONE_MULTI_OBJECT"),
    ),
)


def _case_key(target: str, profile: str, scope: str) -> str:
    return "__".join((target.lower(), profile.lower(), scope.lower()))


def _object_count(scope: str) -> int:
    return 1 if scope == "SINGLE_OBJECT" else 3


POSITIVE_CASES: Tuple[SpineVersionAcceptanceCase, ...] = tuple(
    SpineVersionAcceptanceCase(
        key=_case_key(target, profile, scope),
        target=target,
        exact_version=exact_version,
        profile=profile,
        scope=scope,
        object_count=_object_count(scope),
    )
    for target, exact_version, profiles, scopes in _TARGETS
    for profile in profiles
    for scope in scopes
)

EXACT_VERSION_BY_TARGET: Mapping[str, str] = MappingProxyType(
    {target: exact_version for target, exact_version, _profiles, _scopes in _TARGETS}
)
EXPECTED_CASE_COUNT_BY_TARGET: Mapping[str, int] = MappingProxyType(
    {
        target: sum(1 for case in POSITIVE_CASES if case.target == target)
        for target in EXACT_VERSION_BY_TARGET
    }
)

if len(POSITIVE_CASES) != 20:
    raise RuntimeError(f"Acceptance matrix must contain 20 cases, got {len(POSITIVE_CASES)}")
if len({case.key for case in POSITIVE_CASES}) != len(POSITIVE_CASES):
    raise RuntimeError("Acceptance matrix case keys must be unique")


__all__ = [
    "EXACT_VERSION_BY_TARGET",
    "EXPECTED_CASE_COUNT_BY_TARGET",
    "POSITIVE_CASES",
    "SpineVersionAcceptanceCase",
]
