"""Build texturing topology and UV state for one prepared A1 source mesh."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Mapping, Tuple

from ..application import (
    A1SingleObjectStage,
    A1TexturingTopology,
    A1UvPropagationResult,
    ExportIssue,
    build_a1_texturing_topology,
    propagate_texturing_uv_to_regions,
)
from ..domain.uv import UvRangePolicy, UvUnwrapResult, enforce_uv_range
from .a1_preparation_contracts import (
    A1ObjectPreparationError,
    StatisticsValue,
    freeze_statistics,
    warning_issue,
)
from .a1_source_geometry_preparation import A1SourceGeometryPreparationResult
from .uv_unwrap import unwrap_snapshot_uv


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class A1UvPreparationResult:
    """UV products plus the source geometry state they were derived from."""

    source: A1SourceGeometryPreparationResult
    texturing_topology: A1TexturingTopology
    unwrap_result: UvUnwrapResult
    uv_regions: A1UvPropagationResult
    warnings: Tuple[ExportIssue, ...]
    statistics: Mapping[str, StatisticsValue]

    def __post_init__(self) -> None:
        if not isinstance(self.source, A1SourceGeometryPreparationResult):
            raise TypeError("source must be A1SourceGeometryPreparationResult")
        expected = (
            ("texturing_topology", A1TexturingTopology),
            ("unwrap_result", UvUnwrapResult),
            ("uv_regions", A1UvPropagationResult),
        )
        for field_name, expected_type in expected:
            if not isinstance(getattr(self, field_name), expected_type):
                raise TypeError(f"{field_name} must be {expected_type.__name__}")
        if not isinstance(self.warnings, tuple) or not all(
            isinstance(issue, ExportIssue) for issue in self.warnings
        ):
            raise TypeError("warnings must be a tuple of ExportIssue values")
        if not isinstance(self.statistics, Mapping):
            raise TypeError("statistics must be a mapping")


def prepare_a1_uv(
    source: A1SourceGeometryPreparationResult,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> A1UvPreparationResult:
    """Build seam topology, unwrap it, validate its range, and propagate region UVs."""

    if not isinstance(source, A1SourceGeometryPreparationResult):
        raise TypeError("source must be A1SourceGeometryPreparationResult")
    stage = A1SingleObjectStage.BUILD_TEXTURING_TOPOLOGY
    warnings = source.warnings
    statistics = source.statistics
    try:
        texturing_topology = build_a1_texturing_topology(
            source.source_snapshot,
            source.geometry,
        )
        statistics = freeze_statistics(
            statistics,
            {"texturing_seam_count": len(texturing_topology.all_seam_edge_ids)},
        )

        stage = A1SingleObjectStage.UNWRAP_TEXTURE_UV
        unwrap_result = unwrap_snapshot_uv(
            texturing_topology.snapshot,
            source.settings.uv,
            context=context,
            scene=scene,
        )
        raw_outside_count = unwrap_result.statistics.outside_unit_square_count
        range_report = enforce_uv_range(
            unwrap_result.snapshot,
            source.settings.uv.layer_name,
            policy=source.settings.uv.range_policy,
            epsilon=source.settings.uv.range_epsilon,
        )
        statistics = freeze_statistics(
            statistics,
            {
                "uv_loop_count": unwrap_result.statistics.loop_count,
                "uv_outside_unit_square": raw_outside_count,
                "uv_outside_range_tolerance": range_report.outside_loop_count,
                "uv_range_policy": source.settings.uv.range_policy.value,
                "uv_range_epsilon": source.settings.uv.range_epsilon,
            },
        )
        if (
            range_report.violations
            and source.settings.uv.range_policy is UvRangePolicy.WARN_ONLY
        ):
            warnings = warnings + (
                warning_issue(
                    stage=stage,
                    code="UV_OUTSIDE_UNIT_SQUARE",
                    message=(
                        f"{range_report.outside_loop_count} UV loops are outside "
                        "the unit square beyond epsilon "
                        f"{range_report.epsilon}; export continues because "
                        "uv.range_policy is WARN_ONLY"
                    ),
                    object_id=source.object_id,
                    context={
                        "layer_name": range_report.layer_name,
                        "epsilon": range_report.epsilon,
                        "outside_loop_count": range_report.outside_loop_count,
                    },
                ),
            )

        stage = A1SingleObjectStage.PROPAGATE_REGION_UV
        uv_regions = propagate_texturing_uv_to_regions(
            unwrap_result.snapshot,
            source.geometry,
            source_layer_name=source.settings.uv.layer_name,
            target_layer_name=source.settings.uv.layer_name,
        )
        logger.debug(
            "Prepared UVs for %s: loops=%d regions=%d raw_outside=%d "
            "outside_tolerance=%d policy=%s epsilon=%s",
            source.object_id,
            unwrap_result.statistics.loop_count,
            len(uv_regions.snapshots),
            raw_outside_count,
            range_report.outside_loop_count,
            source.settings.uv.range_policy.value,
            source.settings.uv.range_epsilon,
        )
        return A1UvPreparationResult(
            source=source,
            texturing_topology=texturing_topology,
            unwrap_result=unwrap_result,
            uv_regions=uv_regions,
            warnings=warnings,
            statistics=statistics,
        )
    except A1ObjectPreparationError:
        raise
    except Exception as exc:
        raise A1ObjectPreparationError(
            stage=stage,
            object_id=source.object_id,
            cause=exc,
            statistics=statistics,
            warnings=warnings,
        ) from exc


__all__ = ["A1UvPreparationResult", "prepare_a1_uv"]
