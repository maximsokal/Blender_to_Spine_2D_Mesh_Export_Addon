"""Analyse shading and build one capability-checked A1 texture plan."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import Any, Mapping, Tuple

from ..application import (
    A1SingleObjectStage,
    ExportIssue,
    build_a1_bake_settings,
)
from ..application.a1_generated_materials import build_generated_material_plan
from ..domain.baking import (
    A1TextureExportMode,
    BakeMode,
    BakePlan,
    BakePlanError,
    CameraProjectionPlan,
    MaterialAnalysis,
    MaterialKind,
    ObjectMaterialAnalysis,
    build_bake_plan,
)
from ..domain.baking.generated_materials import (
    A1MaterialSourcePolicy,
    GeneratedBakePlan,
)
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
from .scene_bake_capture import analyse_bake_contexts


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
        if self.bake_plan.material_analysis != self.material_analysis:
            raise ValueError("material_analysis must match bake_plan.material_analysis")
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


def _used_empty_material_slots(
    uv: A1UvPreparationResult,
    analysis: ObjectMaterialAnalysis,
) -> Tuple[int, ...]:
    used_indices = tuple(
        sorted({face.material_index for face in uv.texturing_topology.snapshot.faces})
    )
    return tuple(
        slot_index
        for slot_index in used_indices
        if slot_index >= len(analysis.slots)
        or analysis.slots[slot_index].kind is MaterialKind.EMPTY
    )


def _should_generate_material(
    uv: A1UvPreparationResult,
    analysis: ObjectMaterialAnalysis,
) -> tuple[bool, str]:
    policy = uv.source.settings.material_source_policy
    if policy is A1MaterialSourcePolicy.FORCE_GENERATED:
        return True, "forced by Rewrite material source policy"
    if policy is A1MaterialSourcePolicy.REQUIRE_SOURCE:
        return False, ""

    usable = tuple(
        slot for slot in analysis.slots if slot.kind is not MaterialKind.EMPTY
    )
    missing_used_slots = _used_empty_material_slots(uv, analysis)
    if not usable:
        return True, "source object has no usable materials"
    if missing_used_slots:
        return (
            True,
            "source geometry uses missing material slots "
            + str(missing_used_slots),
        )
    return False, ""


def _build_generated_bake_plan(
    uv: A1UvPreparationResult,
) -> GeneratedBakePlan:
    source = uv.source
    settings = source.settings
    generated_material = build_generated_material_plan(
        uv.uv_regions,
        source_policy=settings.material_source_policy,
        pattern=settings.generated_material_pattern,
        gray_color=settings.generated_gray_color,
    )
    synthetic_analysis = ObjectMaterialAnalysis(
        source_object_id=source.source_snapshot.source_object_id,
        slots=(
            MaterialAnalysis(
                slot_index=0,
                material_name=generated_material.material_name,
                kind=MaterialKind.SOLID_COLOR,
                node_types=("EMISSION",),
                issues=("Temporary generated Rewrite material",),
            ),
        ),
    )
    bake_settings = replace(
        build_a1_bake_settings(source.object_id, settings),
        diffuse_mode=BakeMode.EMIT,
        procedural_mode=BakeMode.EMIT,
        selected_to_active=False,
    )
    base_plan = build_bake_plan(synthetic_analysis, bake_settings)
    return GeneratedBakePlan.from_bake_plan(base_plan, generated_material)


def prepare_a1_texture_plan(
    uv: A1UvPreparationResult,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> A1TexturePlanningResult:
    """Analyse materials and apply the explicit user-selected texture mode."""

    if not isinstance(uv, A1UvPreparationResult):
        raise TypeError("uv must be A1UvPreparationResult")
    source = uv.source
    stage = A1SingleObjectStage.ANALYZE_MATERIALS
    warnings = uv.warnings
    statistics = uv.statistics
    texture_export_mode = source.settings.bake_execution.texture_export_mode
    if not isinstance(texture_export_mode, A1TextureExportMode):
        raise TypeError("texture_export_mode must be A1TextureExportMode")
    analysis_render_target = (
        "CYCLES"
        if texture_export_mode is A1TextureExportMode.NORMAL_UV_SEGMENTS
        else source.renderer.shader_target
    )
    try:
        source_analysis = analyse_object_materials(
            source.source_object,
            source_object_id=source.source_snapshot.source_object_id,
            render_target=analysis_render_target,
        )
        warnings = warnings + _material_warnings(
            source_analysis,
            object_id=source.object_id,
        )
        statistics = freeze_statistics(
            statistics,
            {
                "material_slot_count": len(source_analysis.slots),
                "texture_export_mode": texture_export_mode.value,
                "shader_analysis_target": analysis_render_target,
            },
        )

        use_generated, generated_reason = _should_generate_material(
            uv,
            source_analysis,
        )
        if use_generated:
            if texture_export_mode is A1TextureExportMode.CAMERA_PROJECTION:
                raise BakePlanError(
                    "Camera Projection requires renderable source materials. "
                    "Set Rewrite Generated Materials to Require Source, or switch "
                    "Export Mode to Normal — UV Segments."
                )
            stage = A1SingleObjectStage.PLAN_BAKE
            bake_plan = _build_generated_bake_plan(uv)
            material_analysis = bake_plan.material_analysis
            warnings = warnings + (
                warning_issue(
                    stage=stage,
                    code="GENERATED_MATERIAL_ACTIVE",
                    message=(
                        f"Using {source.settings.generated_material_pattern.value}: "
                        f"{generated_reason}"
                    ),
                    object_id=source.object_id,
                    context={
                        "source_policy": source.settings.material_source_policy.value,
                        "pattern": source.settings.generated_material_pattern.value,
                    },
                ),
            )
            statistics = freeze_statistics(
                statistics,
                {
                    "generated_material_active": 1,
                    "generated_material_policy": (
                        source.settings.material_source_policy.value
                    ),
                    "generated_material_pattern": (
                        source.settings.generated_material_pattern.value
                    ),
                    "generated_material_face_count": len(
                        bake_plan.generated_material.target_snapshot.faces
                    ),
                    "shader_capability": "GENERATED_LOCAL_EMISSION",
                    "shader_capability_audit_count": 0,
                    "texture_export_mode": texture_export_mode.value,
                    "shader_analysis_target": analysis_render_target,
                    "texture_pipeline": "OBJECT_BAKE",
                    "bake_mode": bake_plan.bake_mode.value,
                    "bake_frame_count": len(bake_plan.frame_tasks),
                    "bake_pass_count": len(bake_plan.passes),
                    "bake_scene_aware": 0,
                    "bake_strategy_ids": ",".join(
                        pass_plan.strategy_id.value
                        for pass_plan in bake_plan.passes
                    ),
                    "bake_evaluation_scopes": ",".join(
                        pass_plan.evaluation_scope.value
                        for pass_plan in bake_plan.passes
                    ),
                    "scene_light_count": 0,
                    "scene_has_camera": 0,
                },
            )
            return A1TexturePlanningResult(
                uv=uv,
                material_analysis=material_analysis,
                bake_plan=bake_plan,
                warnings=warnings,
                statistics=statistics,
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
            source_analysis,
            render_target=analysis_render_target,
        )
        required_capability = strongest_object_capability(capability_audits)
        bake_plan = build_capability_checked_texture_plan(
            source_analysis,
            build_a1_bake_settings(source.object_id, source.settings),
            capability_audits,
            source.renderer,
            object_context=object_bake_context,
            scene_context=scene_bake_context,
            texture_export_mode=texture_export_mode,
        )
        camera_projection = isinstance(bake_plan, CameraProjectionPlan)
        statistics = freeze_statistics(
            statistics,
            {
                "generated_material_active": 0,
                "generated_material_policy": (
                    source.settings.material_source_policy.value
                ),
                "generated_material_pattern": (
                    source.settings.generated_material_pattern.value
                ),
                "shader_capability": required_capability.value,
                "shader_capability_audit_count": len(capability_audits),
                "texture_export_mode": texture_export_mode.value,
                "shader_analysis_target": analysis_render_target,
                "texture_pipeline": (
                    "CAMERA_RENDER_PROJECTION"
                    if camera_projection
                    else "OBJECT_BAKE"
                ),
                "bake_mode": bake_plan.bake_mode.value,
                "bake_frame_count": len(bake_plan.frame_tasks),
                "bake_pass_count": len(bake_plan.passes),
                "bake_scene_aware": int(bake_plan.scene_aware),
                "bake_strategy_ids": ",".join(
                    pass_plan.strategy_id.value for pass_plan in bake_plan.passes
                ),
                "bake_evaluation_scopes": ",".join(
                    pass_plan.evaluation_scope.value
                    for pass_plan in bake_plan.passes
                ),
                "scene_light_count": len(scene_bake_context.lights),
                "scene_has_camera": int(scene_bake_context.has_camera),
            },
        )
        logger.debug(
            "Planned texture pipeline for %s: mode=%s target=%s pipeline=%s "
            "passes=%d frames=%d",
            source.object_id,
            texture_export_mode.value,
            analysis_render_target,
            statistics["texture_pipeline"],
            len(bake_plan.passes),
            len(bake_plan.frame_tasks),
        )
        return A1TexturePlanningResult(
            uv=uv,
            material_analysis=source_analysis,
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
