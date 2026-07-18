"""Compatibility wrapper for normalized multi-object failure results."""

from __future__ import annotations

import logging
from typing import Mapping, Tuple

from ..application import A1MultiObjectStage, ExportIssue, ExportResult
from .a1_export_result import StatisticsValue, build_a1_failure_result


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
    """Delegate the historical multi-object API to the shared A1 result builder."""

    if not isinstance(stage, A1MultiObjectStage):
        raise TypeError("stage must be A1MultiObjectStage")
    return build_a1_failure_result(
        logger=logger,
        operation=operation,
        stage=stage,
        exc=exc,
        statistics=statistics,
        warnings=warnings,
        component_id=component_id,
        object_id=object_id,
        object_stage=object_stage,
    )


__all__ = ["build_multi_object_failure_result"]
