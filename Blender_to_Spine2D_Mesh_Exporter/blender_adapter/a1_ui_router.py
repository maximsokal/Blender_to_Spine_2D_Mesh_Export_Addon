"""Route captured Blender UI requests to typed A1 output services."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import logging

from ..application import (
    A1ExportProgressCallback,
    A1MultiObjectMode,
    ExportIssue,
    ExportResult,
)
from .a1_blender_progress import blender_a1_progress_session
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


def export_active_object_a1(
    context,
    *,
    progress_callback: A1ExportProgressCallback | None = None,
) -> ExportResult:
    """Export the active Mesh through the complete single-object A1 output service."""

    progress_owner = (
        blender_a1_progress_session(context, operation_name="Spine2D export")
        if progress_callback is None
        else nullcontext(progress_callback)
    )
    with progress_owner as resolved_progress:
        plan = build_active_ui_export_plan(context)
        scene = context.scene
        return export_a1_single_object(
            plan.source_object,
            plan.settings,
            context=context,
            scene=scene,
            progress_callback=resolved_progress,
        )


def export_selected_objects_a1(
    context,
    *,
    progress_callback: A1ExportProgressCallback | None = None,
) -> ExportResult:
    """Export selected meshes through standalone, connected, or mixed A1 output."""

    progress_owner = (
        blender_a1_progress_session(context, operation_name="Spine2D multi-export")
        if progress_callback is None
        else nullcontext(progress_callback)
    )
    with progress_owner as resolved_progress:
        plan = build_selected_ui_export_plan(context)
        scene = context.scene
        if plan.settings.mode is A1MultiObjectMode.MIXED:
            result = export_a1_mixed_object(
                plan.connected_sources,
                plan.standalone_sources,
                plan.settings,
                context=context,
                scene=scene,
                progress_callback=resolved_progress,
            )
        else:
            result = export_a1_multi_object(
                plan.all_sources,
                plan.settings,
                context=context,
                scene=scene,
                progress_callback=resolved_progress,
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
