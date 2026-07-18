"""Shared structured failure-result construction for every A1 output route."""

from __future__ import annotations

import logging
from typing import Mapping, Tuple

from ..application import (
    A1MultiObjectStage,
    A1SingleObjectStage,
    ExportIssue,
    ExportResult,
    IssueSeverity,
)
StatisticsValue = int | float | str


A1OutputStage = A1SingleObjectStage | A1MultiObjectStage


def build_a1_failure_result(
    *,
    logger: logging.Logger,
    operation: str,
    stage: A1OutputStage,
    exc: Exception,
    statistics: Mapping[str, StatisticsValue],
    warnings: Tuple[ExportIssue, ...] = (),
    object_id: str | None = None,
    component_id: str | None = None,
    object_stage: str | None = None,
) -> ExportResult:
    """Log one failure and build the normalized public ``ExportResult``."""

    if not isinstance(logger, logging.Logger):
        raise TypeError("logger must be logging.Logger")
    if not isinstance(operation, str) or not operation.strip():
        raise ValueError("operation must be a non-empty string")
    if not isinstance(stage, (A1SingleObjectStage, A1MultiObjectStage)):
        raise TypeError("stage must be A1SingleObjectStage or A1MultiObjectStage")
    if not isinstance(exc, Exception):
        raise TypeError("exc must be Exception")
    if not isinstance(statistics, Mapping):
        raise TypeError("statistics must be a mapping")
    if not isinstance(warnings, tuple) or not all(
        isinstance(issue, ExportIssue) for issue in warnings
    ):
        raise TypeError("warnings must be a tuple of ExportIssue values")
    for field_name, value in (
        ("object_id", object_id),
        ("component_id", component_id),
        ("object_stage", object_stage),
    ):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{field_name} must be a non-empty string or None")

    issue_context: dict[str, object] = {
        "exception_type": type(exc).__name__,
        "operation": operation.strip(),
    }
    if component_id is not None:
        issue_context["component_id"] = component_id
    if object_stage is not None:
        issue_context["object_stage"] = object_stage

    logger.exception(
        "%s failed at %s (component=%s, object=%s)",
        operation,
        stage.value,
        component_id,
        object_id,
    )
    error = ExportIssue(
        severity=IssueSeverity.ERROR,
        stage=stage.value,
        code=stage.error_code,
        message=str(exc) or type(exc).__name__,
        object_id=object_id,
        technical_details=f"{type(exc).__name__}: {exc}",
        context=issue_context,
    )
    return ExportResult(
        success=False,
        issues=warnings + (error,),
        statistics=dict(statistics),
    )


__all__ = ["A1OutputStage", "StatisticsValue", "build_a1_failure_result"]
