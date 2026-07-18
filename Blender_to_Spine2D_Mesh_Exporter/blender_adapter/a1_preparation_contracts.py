"""Shared contracts for staged A1 object preparation."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Tuple

from ..application import A1SingleObjectStage, ExportIssue, IssueSeverity


StatisticsValue = int | float | str


class A1ObjectPreparationError(RuntimeError):
    """Wrap one failed preparation stage without hiding the original exception."""

    def __init__(
        self,
        *,
        stage: A1SingleObjectStage,
        object_id: str | None,
        cause: Exception,
        statistics: Mapping[str, StatisticsValue],
        warnings: Tuple[ExportIssue, ...],
    ) -> None:
        if not isinstance(stage, A1SingleObjectStage):
            raise TypeError("stage must be A1SingleObjectStage")
        if object_id is not None and (
            not isinstance(object_id, str) or not object_id.strip()
        ):
            raise ValueError("object_id must be a non-empty string or None")
        if not isinstance(cause, Exception):
            raise TypeError("cause must be Exception")
        if not isinstance(statistics, Mapping):
            raise TypeError("statistics must be a mapping")
        if not isinstance(warnings, tuple) or not all(
            isinstance(issue, ExportIssue) for issue in warnings
        ):
            raise TypeError("warnings must be a tuple of ExportIssue values")

        self.stage = stage
        self.object_id = object_id
        self.cause = cause
        self.statistics = freeze_statistics(statistics)
        self.warnings = warnings
        message = str(cause) or type(cause).__name__
        super().__init__(
            f"A1 object preparation failed at {stage.value}"
            + ("" if object_id is None else f" for '{object_id}'")
            + f": {message}"
        )


def freeze_statistics(
    *values: Mapping[str, StatisticsValue],
) -> Mapping[str, StatisticsValue]:
    """Merge statistics into one immutable mapping, with later stages taking precedence."""

    merged: dict[str, StatisticsValue] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise TypeError("statistics values must be mappings")
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("statistics keys must be non-empty strings")
            if isinstance(item, bool) or not isinstance(item, (int, float, str)):
                raise TypeError(
                    "statistics values must be int, float, or str; "
                    f"got {type(item).__name__} for {key!r}"
                )
            merged[key] = item
    return MappingProxyType(merged)


def warning_issue(
    *,
    stage: A1SingleObjectStage,
    code: str,
    message: str,
    object_id: str,
    context: Mapping[str, object] | None = None,
) -> ExportIssue:
    """Build one normalized preparation warning."""

    if not isinstance(stage, A1SingleObjectStage):
        raise TypeError("stage must be A1SingleObjectStage")
    for field_name, value in (
        ("code", code),
        ("message", message),
        ("object_id", object_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
    if context is not None and not isinstance(context, Mapping):
        raise TypeError("context must be a mapping or None")
    return ExportIssue(
        severity=IssueSeverity.WARNING,
        stage=stage.value,
        code=code.strip(),
        message=message,
        object_id=object_id,
        context={} if context is None else dict(context),
    )


__all__ = [
    "A1ObjectPreparationError",
    "StatisticsValue",
    "freeze_statistics",
    "warning_issue",
]
