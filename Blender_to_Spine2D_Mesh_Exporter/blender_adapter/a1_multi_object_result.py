"""Shared structured result helpers for multi-object output services."""

from __future__ import annotations

import logging
from typing import Mapping, Tuple

from ..application import (
    A1MultiObjectStage,
    ExportIssue,
    ExportResult,
    IssueSeverity,
)
from .a1_object_preparation import StatisticsValue


def build_multi_object_failure_result(
    *,
    logger: logging.Logger,
    operation: str,
    stage: A1MultiObjectStage,
    exc: Exception,
    statistics: Mapping[str, StatisticsValue],
    warnings: Tuple[ExportIssue, ...],
    component_id: str | None = None,
    object_id: str | None = None,
    object_stage: str | None = None,
) -> ExportResult:
    """Log one failure and return the normalized public ``ExportResult``."""

    if not isinstance(logger, logging.Logger):
        raise TypeError("logger must be logging.Logger")
    if not isinstance(operation, str) or not operation.strip():
        raise ValueError("operation must be a non-empty string")
    if not isinstance(stage, A1MultiObjectStage):
        raise TypeError("stage must be A1MultiObjectStage")
    if not isinstance(exc, Exception):
        raise TypeError("exc must be Exception")
    if not isinstance(statistics, Mapping):
        raise TypeError("statistics must be a mapping")
    if not isinstance(warnings, tuple) or not all(
        isinstance(issue, ExportIssue) for issue in warnings
    ):
        raise TypeError("warnings must be a tuple of ExportIssue values")

    issue_context: dict[str, object] = {
        "exception_type": type(exc).__name__,
        "operation": operation,
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


__all__ = ["build_multi_object_failure_result"]
