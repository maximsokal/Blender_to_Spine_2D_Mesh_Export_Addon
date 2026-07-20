"""Analyse shading and build one capability-checked A1 texture plan."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Mapping, Tuple

from ..application import (
    A1SingleObjectStage,
    ExportIssue,
    build_a1_bake_settings,
)
from ..domain.baking import BakePlan, CameraProjectionPlan, ObjectMaterialAnalysis
from .a1_preparation_contracts import (
    A1ObjectPreparationError,
    StatisticsValue,
    freeze_statistics,
    warning_issue,
)
from .a1_uv_preparation import A1UvPreparationResult
from .material_object_analysis import analyse_object_materials
from .production_shader_capability_object_audit import (
    audit_object_material_capabilities,
)
from .production_shader_capability_routing import (
    build_capability_checked_texture_plan,
    strongest_object_capability,
)
from .scene_bake_analyzer import analyse_bake_contexts


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class A1TexturePlanningResult:
    """Material analysis and the selected bake or camera-projection plan."""

    uv: A1UvPreparationResult
    material_analysis: ObjectMaterialAnalysis
    bake_plan: BakePlan
    warnings: Tuple[ExportIssue, ...]
    statistics: Mapping[str, StatisticsValue]

    def __post_init__(self) -> None:
        if not isinstance(self.uv, A1UvPreparationResult):
            raise TypeError("uv must be A1UvPreparationResult")
        if not isinstance(self.material_analysis, ObjectMaterialAnalysis):
            raise TypeError("material_analysis must be ObjectMaterialAnalysis")
        if not isinstance(self.bake_plan, BakePlan):
            raise TypeError("bake_plan must be BakePlan")
        if self.bake_plan.source_object_id != self.uv.source.object_id:
            raise ValueError("bake_plan.source_object_id must match source object_id")
        if not isinstance(self.warnings, tuple) or not all(
            isinstance(issue, ExportIssue) for issue in self.warnings
        ):
            raise TypeError("warnings must be a tuple of ExportIssue values")
        if not isinstance(self.statistics, Mapping):
            raise TypeError("statistics must be a mapping")


def _material_warnings(
    analysis: ObjectMaterialAnalysis,
    *,
    object_id: str,
) -> Tuple[ExportIssue, ...]:
    stage = A1SingleObjectStage.ANALYZE_MATERIALS
    result: list[ExportIssue] = []
    for slot in analysis.slots:
        for issue_index, message in enumerate(slot.issues):
            result.append(
                warning_issue(
                    stage=stage,
                    code="MATERIAL_ANALYSIS_NOTE",
                    message=message,
                    object_id=object_id,
                    context={
                        "slot_index": slot.slot_index,
                        "issue_index": issue_index,
                        "material_kind": slot.kind.value,
                    },
                )
            )
    return tuple(result)


def prepare_a1_texture_plan(
    uv: A1UvPreparationResult,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> A1TexturePlanningResult:
    """Analyse materials and select a renderer-compatible texture strategy."""

    if not isinstance(uv, A1UvPreparationResult):
        raise TypeError("uv must be A1UvPreparationResult")
    source = uv.source
    stage = A1SingleObjectStage.ANALYZE_MATERIALS
    warnings = uv.warnings
    statistics = uv.statistics
    try:
        material_analysis = analyse_object_materials(
            source.source_object,
            source_object_id=source.source_snapshot.source_object_id,
            render_target=source.renderer.shader_target,
        )
        warnings = warnings + _material_warnings(
            material_analysis,
            object_id=source.object_id,
        )
        statistics = freeze_statistics(
            statistics,
            {"material_slot_count": len(material_analysis.slots)},
        )

        stage = A1SingleObjectStage.PLAN_BAKE
        object_bake_context, scene_bake_context = analyse_bake_contexts(
            source.source_object,
            scene=scene,
            context=context,
        )
        source.renderer.validate_scene(scene_bake_context)
        capability_audits = audit_object_material_capabilities(
            source.source_object,
            material_analysis,
            render_target=source.renderer.shader_target,
        )
        required_capability = strongest_object_capability(capability_audits)
        bake_plan = build_capability_checked_texture_plan(
            material_analysis,
            build_a1_bake_settings(source.object_id, source.settings),
            capability_audits,
            source.renderer,
            object_context=object_bake_context,
            scene_context=scene_bake_context,
        )
        camera_projection = isinstance(bake_plan, CameraProjectionPlan)
        statistics = freeze_statistics(
            statistics,
            {
                "shader_capability": required_capability.value,
                "shader_capability_audit_count": len(capability_audits),
                "texture_pipeline": (
                    "CAMERA_RENDER_PROJECTION" if camera_projection else "OBJECT_BAKE"
                ),
                "bake_mode": bake_plan.bake_mode.value,
                "bake_frame_count": len(bake_plan.frame_tasks),
                "bake_pass_count": len(bake_plan.passes),
                "bake_scene_aware": int(bake_plan.scene_aware),
                "bake_strategy_ids": ",".join(
                    pass_plan.strategy_id.value for pass_plan in bake_plan.passes
                ),
                "bake_evaluation_scopes": ",".join(
                    pass_plan.evaluation_scope.value for pass_plan in bake_plan.passes
                ),
                "scene_light_count": len(scene_bake_context.lights),
                "scene_has_camera": int(scene_bake_context.has_camera),
            },
        )
        logger.debug(
            "Planned texture pipeline for %s: pipeline=%s passes=%d frames=%d",
            source.object_id,
            statistics["texture_pipeline"],
            len(bake_plan.passes),
            len(bake_plan.frame_tasks),
        )
        return A1TexturePlanningResult(
            uv=uv,
            material_analysis=material_analysis,
            bake_plan=bake_plan,
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


__all__ = ["A1TexturePlanningResult", "prepare_a1_texture_plan"]
