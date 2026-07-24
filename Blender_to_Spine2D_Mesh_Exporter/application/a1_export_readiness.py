"""Blender-independent contracts for cached export-readiness analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Tuple

from .a1_numeric_contracts import require_non_empty_string
from .contracts import ExportIssue, IssueSeverity


ReadinessStatistic = int | float | str


class A1ReadinessState(str, Enum):
    """UI-visible lifecycle of one export-readiness analysis."""

    NOT_ANALYSED = "NOT_ANALYSED"
    STALE = "STALE"
    READY = "READY"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"


def _freeze_statistics(
    values: Mapping[str, ReadinessStatistic],
    *,
    field_name: str,
) -> Mapping[str, ReadinessStatistic]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    frozen: dict[str, ReadinessStatistic] = {}
    for key, value in values.items():
        require_non_empty_string(key, f"{field_name} key")
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise TypeError(
                f"{field_name}[{key!r}] must be int, finite float, or str"
            )
        if isinstance(value, float) and not isfinite(value):
            raise ValueError(f"{field_name}[{key!r}] must be finite")
        frozen[key] = value
    return MappingProxyType(frozen)


def _validate_issues(values: Tuple[ExportIssue, ...], field_name: str) -> None:
    if not isinstance(values, tuple) or not all(
        isinstance(issue, ExportIssue) for issue in values
    ):
        raise TypeError(f"{field_name} must be a tuple of ExportIssue values")


@dataclass(frozen=True, slots=True)
class A1ObjectReadiness:
    """Readiness result and export metrics for one source object."""

    object_id: str
    issues: Tuple[ExportIssue, ...] = ()
    statistics: Mapping[str, ReadinessStatistic] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty_string(self.object_id, "object_id")
        _validate_issues(self.issues, "issues")
        object.__setattr__(
            self,
            "statistics",
            _freeze_statistics(self.statistics, field_name="statistics"),
        )
        for issue in self.issues:
            if issue.object_id is not None and issue.object_id != self.object_id:
                raise ValueError(
                    "object readiness issues must either omit object_id or match object_id"
                )

    @property
    def blocker_count(self) -> int:
        return sum(issue.severity is IssueSeverity.ERROR for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity is IssueSeverity.WARNING for issue in self.issues)

    @property
    def state(self) -> A1ReadinessState:
        if self.blocker_count:
            return A1ReadinessState.BLOCKED
        if self.warning_count:
            return A1ReadinessState.WARNING
        return A1ReadinessState.READY


@dataclass(frozen=True, slots=True)
class A1ExportReadinessReport:
    """Complete deep-preflight result for the current UI export request."""

    signature: str
    objects: Tuple[A1ObjectReadiness, ...]
    issues: Tuple[ExportIssue, ...] = ()
    statistics: Mapping[str, ReadinessStatistic] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty_string(self.signature, "signature")
        if not isinstance(self.objects, tuple):
            raise TypeError("objects must be tuple")
        if not all(isinstance(item, A1ObjectReadiness) for item in self.objects):
            raise TypeError("objects must contain A1ObjectReadiness values")
        object_ids = tuple(item.object_id for item in self.objects)
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("objects must contain unique object_id values")
        _validate_issues(self.issues, "issues")
        object.__setattr__(
            self,
            "statistics",
            _freeze_statistics(self.statistics, field_name="statistics"),
        )

    @property
    def all_issues(self) -> Tuple[ExportIssue, ...]:
        return self.issues + tuple(
            issue for item in self.objects for issue in item.issues
        )

    @property
    def blocker_count(self) -> int:
        return sum(
            issue.severity is IssueSeverity.ERROR for issue in self.all_issues
        )

    @property
    def warning_count(self) -> int:
        return sum(
            issue.severity is IssueSeverity.WARNING for issue in self.all_issues
        )

    @property
    def state(self) -> A1ReadinessState:
        if self.blocker_count:
            return A1ReadinessState.BLOCKED
        if self.warning_count:
            return A1ReadinessState.WARNING
        return A1ReadinessState.READY

    @property
    def can_export(self) -> bool:
        return self.state in {A1ReadinessState.READY, A1ReadinessState.WARNING}


__all__ = [
    "A1ExportReadinessReport",
    "A1ObjectReadiness",
    "A1ReadinessState",
    "ReadinessStatistic",
]
