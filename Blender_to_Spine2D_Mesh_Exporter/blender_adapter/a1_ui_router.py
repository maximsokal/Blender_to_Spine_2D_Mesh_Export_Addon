"""Route captured Blender UI requests to typed A1 output services."""

from __future__ import annotations

from dataclasses import replace
import logging

from ..application import ExportIssue, ExportResult
from .a1_mixed_object_output import export_a1_mixed_object
from .a1_multi_object_output import export_a1_multi_object
from .a1_single_object_export import export_a1_single_object
from .a1_ui_export_plan import (
    build_active_ui_export_plan,
    build_selected_ui_export_plan,
)


logger = logging.getLogger(__name__)


def _append_issues(
    result: ExportResult,
    issues: tuple[ExportIssue, ...],
    *,
    statistics: dict[str, int | float | str] | None = None,
) -> ExportResult:
    if not isinstance(result, ExportResult):
        raise TypeError("result must be ExportResult")
    if not isinstance(issues, tuple) or not all(
        isinstance(issue, ExportIssue) for issue in issues
    ):
        raise TypeError("issues must be a tuple of ExportIssue values")
    resolved_statistics = dict(result.statistics)
    if statistics is not None:
        if not isinstance(statistics, dict):
            raise TypeError("statistics must be dict or None")
        resolved_statistics.update(statistics)
    return replace(
        result,
        issues=issues + result.issues,
        statistics=resolved_statistics,
    )


def export_active_object_a1(context) -> ExportResult:
    """Export the active Mesh through the complete single-object A1 output service."""

    plan = build_active_ui_export_plan(context)
    scene = context.scene
    return export_a1_single_object(
        plan.source_object,
        plan.settings,
        context=context,
        scene=scene,
    )


def export_selected_objects_a1(context) -> ExportResult:
    """Export selected meshes through standalone, connected, or mixed A1 output."""

    plan = build_selected_ui_export_plan(context)
    scene = context.scene
    if plan.settings.mode.value == "MIXED":
        result = export_a1_mixed_object(
            plan.connected_sources,
            plan.standalone_sources,
            plan.settings,
            context=context,
            scene=scene,
        )
    else:
        result = export_a1_multi_object(
            plan.all_sources,
            plan.settings,
            context=context,
            scene=scene,
        )

    if not plan.issues:
        return result
    logger.warning(
        "Rewrite UI export plan produced %d request warning(s)",
        len(plan.issues),
    )
    return _append_issues(
        result,
        plan.issues,
        statistics={"ui_request_warning_count": len(plan.issues)},
    )


__all__ = ["export_active_object_a1", "export_selected_objects_a1"]
