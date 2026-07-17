"""Prepare one Blender mesh for A1 export without writing output files.

The preparation service is the reusable boundary shared by single- and multi-object
orchestration. It reads/evaluates the live source object, produces immutable domain and
application values, and returns after temporary Blender datablocks and context mutations
have already been cleaned up.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Tuple

from ..application import (
    A1DocumentAssemblyResult,
    A1DocumentAssemblySettings,
    A1GeometryPreparationResult,
    A1ResolvedOutputPaths,
    A1SingleObjectExportSettings,
    A1SingleObjectStage,
    A1SourceGeometryMode,
    A1TexturingTopology,
    A1UvPropagationResult,
    A1ZGroupAssignmentPlan,
    ExportIssue,
    IssueSeverity,
    assemble_a1_camera_projection_document,
    assemble_a1_document,
    build_a1_attachment_path,
    build_a1_attachment_sequence,
    build_a1_bake_settings,
    build_a1_texturing_topology,
    build_a1_z_group_assignment,
    calculate_a1_main_position_pixels,
    calculate_a1_mesh_bounds,
    prepare_a1_geometry_regions,
    propagate_texturing_uv_to_regions,
    resolve_a1_names,
    resolve_a1_output_paths,
)
from ..domain.baking import BakePlan, CameraProjectionPlan, ObjectMaterialAnalysis
from ..domain.geometry import LineageSeverity, MeshSnapshot
from ..domain.spine import (
    LegacyRigBuildRequest,
    LegacyRigBuildResult,
    SpineDocument,
    build_legacy_rig,
)
from ..domain.uv import UvUnwrapResult
from .evaluated_mesh_reader import read_evaluated_mesh_snapshot
from .material_analyzer import analyse_object_materials
from .mesh_reader import read_source_mesh_snapshot
from .production_shader_capabilities import (
    audit_object_material_capabilities,
    build_capability_checked_texture_plan,
    strongest_object_capability,
)
from .render_engine_contract import render_engine_contract_from_execution
from .scene_bake_analyzer import analyse_bake_contexts
from .uv_unwrap import unwrap_snapshot_uv


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
        if not isinstance(warnings, tuple) or not all(
            isinstance(issue, ExportIssue) for issue in warnings
        ):
            raise TypeError("warnings must be a tuple of ExportIssue values")

        self.stage = stage
        self.object_id = object_id
        self.cause = cause
        self.statistics = MappingProxyType(dict(statistics))
        self.warnings = warnings
        message = str(cause) or type(cause).__name__
        super().__init__(
            f"A1 object preparation failed at {stage.value}"
            + ("" if object_id is None else f" for '{object_id}'")
            + f": {message}"
        )


@dataclass(frozen=True, slots=True)
class PreparedA1Object:
    """Complete in-memory product of one A1 object preparation pipeline."""

    source_object: Any
    object_id: str
    prefix: str
    settings: A1SingleObjectExportSettings
    output_paths: A1ResolvedOutputPaths
    source_snapshot: MeshSnapshot
    z_groups: A1ZGroupAssignmentPlan
    geometry: A1GeometryPreparationResult
    texturing_topology: A1TexturingTopology
    unwrap_result: UvUnwrapResult
    uv_regions: A1UvPropagationResult
    material_analysis: ObjectMaterialAnalysis
    bake_plan: BakePlan
    rig: LegacyRigBuildResult
    document_assembly: A1DocumentAssemblyResult
    warnings: Tuple[ExportIssue, ...]
    statistics: Mapping[str, StatisticsValue]

    def __post_init__(self) -> None:
        if self.source_object is None:
            raise ValueError("source_object cannot be None")
        for field_name in ("object_id", "prefix"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        expected_types = (
            ("settings", A1SingleObjectExportSettings),
            ("output_paths", A1ResolvedOutputPaths),
            ("source_snapshot", MeshSnapshot),
            ("z_groups", A1ZGroupAssignmentPlan),
            ("geometry", A1GeometryPreparationResult),
            ("texturing_topology", A1TexturingTopology),
            ("unwrap_result", UvUnwrapResult),
            ("uv_regions", A1UvPropagationResult),
            ("material_analysis", ObjectMaterialAnalysis),
            ("bake_plan", BakePlan),
            ("rig", LegacyRigBuildResult),
            ("document_assembly", A1DocumentAssemblyResult),
        )
        for field_name, expected_type in expected_types:
            if not isinstance(getattr(self, field_name), expected_type):
                raise TypeError(f"{field_name} must be {expected_type.__name__}")
        if not isinstance(self.warnings, tuple) or not all(
            isinstance(issue, ExportIssue) for issue in self.warnings
        ):
            raise TypeError("warnings must be a tuple of ExportIssue values")
        if not isinstance(self.statistics, Mapping):
            raise TypeError("statistics must be a mapping")
        if self.source_snapshot.source_object_id != self.object_id:
            raise ValueError("source_snapshot.source_object_id must match object_id")
        if self.bake_plan.source_object_id != self.object_id:
            raise ValueError("bake_plan.source_object_id must match object_id")
        if self.rig.request.prefix != self.prefix:
            raise ValueError("rig prefix must match prepared prefix")

    @property
    def document(self) -> SpineDocument:
        return self.document_assembly.document

    @property
    def bake_target_snapshot(self) -> MeshSnapshot:
        return self.unwrap_result.snapshot

    @property
    def world_position(self) -> Tuple[float, float, float]:
        matrix = self.source_snapshot.world_matrix
        if len(matrix) != 16:
            raise ValueError("source_snapshot.world_matrix must contain 16 values")
        return float(matrix[3]), float(matrix[7]), float(matrix[11])


def _object_name(obj: Any) -> str:
    if obj is None or getattr(obj, "type", None) != "MESH":
        raise ValueError("source_obj must be a Blender MESH object")
    value = str(
        getattr(obj, "name_full", None)
        or getattr(obj, "name", None)
        or ""
    ).strip()
    if not value:
        raise ValueError("source_obj name is empty")
    if getattr(obj, "data", None) is None:
        raise ValueError("source_obj.data is missing")
    return value


def _warning_issue(
    *,
    stage: A1SingleObjectStage,
    code: str,
    message: str,
    object_id: str,
    context: Mapping[str, object] | None = None,
) -> ExportIssue:
    return ExportIssue(
        severity=IssueSeverity.WARNING,
        stage=stage.value,
        code=code,
        message=message,
        object_id=object_id,
        context={} if context is None else dict(context),
    )


def prepare_a1_object(
    source_obj: Any,
    settings: A1SingleObjectExportSettings,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> PreparedA1Object:
    """Run every A1 object stage up to a validated in-memory Spine document."""

    stage = A1SingleObjectStage.VALIDATE_REQUEST
    object_id: str | None = None
    statistics: dict[str, StatisticsValue] = {}
    warnings: list[ExportIssue] = []

    try:
        if not isinstance(settings, A1SingleObjectExportSettings):
            raise TypeError("settings must be A1SingleObjectExportSettings")
        object_id = _object_name(source_obj)
        prefix, _ = resolve_a1_names(object_id, settings)
        output_paths = resolve_a1_output_paths(object_id, settings)
        renderer = render_engine_contract_from_execution(settings.bake_execution)
        statistics.update(
            {
                "source_object": object_id,
                "rig_prefix": prefix,
                "source_geometry_mode": settings.source_geometry_mode.value,
                "include_control_icons": int(settings.include_control_icons),
                "include_preview_animation": int(
                    settings.include_preview_animation
                ),
                "render_engine": renderer.blender_engine,
                "shader_render_target": renderer.shader_target,
            }
        )

        stage = A1SingleObjectStage.READ_GEOMETRY
        if settings.source_geometry_mode is A1SourceGeometryMode.EVALUATED:
            evaluated = read_evaluated_mesh_snapshot(
                source_obj,
                scene=scene,
                source_object_id=object_id,
                snapshot_id=f"{object_id}:a1-evaluated",
                lineage_policy=settings.modifier_lineage_policy,
            )
            source_snapshot = evaluated.snapshot
            statistics["modifier_count"] = len(evaluated.modifier_stack)
            for issue in evaluated.lineage_report.issues:
                if issue.severity is not LineageSeverity.WARNING:
                    continue
                warnings.append(
                    _warning_issue(
                        stage=stage,
                        code=f"MODIFIER_{issue.code}",
                        message=issue.message,
                        object_id=object_id,
                        context={"channel": issue.channel},
                    )
                )
        else:
            source_snapshot = read_source_mesh_snapshot(
                source_obj,
                source_object_id=object_id,
                snapshot_id=f"{object_id}:a1-source",
            )
            statistics["modifier_count"] = 0
        statistics.update(
            {
                "source_vertices": len(source_snapshot.vertices),
                "source_edges": len(source_snapshot.edges),
                "source_faces": len(source_snapshot.faces),
            }
        )

        stage = A1SingleObjectStage.ASSIGN_Z_GROUPS
        z_groups = build_a1_z_group_assignment(source_snapshot)
        statistics["z_group_count"] = len(z_groups.groups)

        stage = A1SingleObjectStage.PREPARE_GEOMETRY
        geometry = prepare_a1_geometry_regions(
            source_snapshot,
            settings.resolved_geometry_settings(),
        )
        statistics.update(
            {
                "segment_count": len(geometry.segmentation.segments),
                "region_count": len(geometry.regions),
                "decomposition_cut_count": len(geometry.decomposition.cuts),
            }
        )

        stage = A1SingleObjectStage.BUILD_TEXTURING_TOPOLOGY
        texturing_topology = build_a1_texturing_topology(
            source_snapshot,
            geometry,
        )
        statistics["texturing_seam_count"] = len(
            texturing_topology.all_seam_edge_ids
        )

        stage = A1SingleObjectStage.UNWRAP_TEXTURE_UV
        unwrap_result = unwrap_snapshot_uv(
            texturing_topology.snapshot,
            settings.uv,
            context=context,
            scene=scene,
        )
        statistics["uv_loop_count"] = unwrap_result.statistics.loop_count
        statistics["uv_outside_unit_square"] = (
            unwrap_result.statistics.outside_unit_square_count
        )
        if unwrap_result.statistics.outside_unit_square_count:
            warnings.append(
                _warning_issue(
                    stage=stage,
                    code="UV_OUTSIDE_UNIT_SQUARE",
                    message=(
                        f"{unwrap_result.statistics.outside_unit_square_count} UV "
                        "loops are outside the unit square"
                    ),
                    object_id=object_id,
                )
            )

        stage = A1SingleObjectStage.PROPAGATE_REGION_UV
        uv_regions = propagate_texturing_uv_to_regions(
            unwrap_result.snapshot,
            geometry,
            source_layer_name=settings.uv.layer_name,
            target_layer_name=settings.uv.layer_name,
        )

        stage = A1SingleObjectStage.ANALYZE_MATERIALS
        material_analysis = analyse_object_materials(
            source_obj,
            source_object_id=source_snapshot.source_object_id,
            render_target=renderer.shader_target,
        )
        statistics["material_slot_count"] = len(material_analysis.slots)
        for slot in material_analysis.slots:
            for issue_index, message in enumerate(slot.issues):
                warnings.append(
                    _warning_issue(
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

        stage = A1SingleObjectStage.PLAN_BAKE
        object_bake_context, scene_bake_context = analyse_bake_contexts(
            source_obj,
            scene=scene,
            context=context,
        )
        renderer.validate_scene(scene_bake_context)
        capability_audits = audit_object_material_capabilities(
            source_obj,
            material_analysis,
            render_target=renderer.shader_target,
        )
        required_capability = strongest_object_capability(capability_audits)
        statistics["shader_capability"] = required_capability.value
        statistics["shader_capability_audit_count"] = len(capability_audits)
        bake_plan = build_capability_checked_texture_plan(
            material_analysis,
            build_a1_bake_settings(object_id, settings),
            capability_audits,
            renderer,
            object_context=object_bake_context,
            scene_context=scene_bake_context,
        )
        camera_projection = isinstance(bake_plan, CameraProjectionPlan)
        statistics.update(
            {
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
            }
        )

        bounds = calculate_a1_mesh_bounds(source_snapshot)
        stage = A1SingleObjectStage.BUILD_RIG
        rig = build_legacy_rig(
            LegacyRigBuildRequest(
                prefix=prefix,
                texture_width=settings.export.texture_width,
                texture_height=settings.export.texture_height,
                z_groups=z_groups.groups,
                main_position_pixels=(
                    None
                    if camera_projection
                    else calculate_a1_main_position_pixels(source_snapshot, settings)
                ),
                scale_mode=settings.rig_scale_mode,
            )
        )
        statistics["base_rig_bone_count"] = len(rig.bones)

        stage = A1SingleObjectStage.ASSEMBLE_DOCUMENT
        attachment_path = build_a1_attachment_path(bake_plan, output_paths)
        assembly_settings = A1DocumentAssemblySettings(
            prefix=prefix,
            uv_layer_name=settings.uv.layer_name,
            image_path=attachment_path,
            attachment_width=settings.export.texture_width,
            attachment_height=settings.export.texture_height,
            center_x=0.0 if camera_projection else bounds.center_x,
            center_y=0.0 if camera_projection else bounds.center_y,
            sequence=build_a1_attachment_sequence(bake_plan),
            include_control_icons=settings.include_control_icons,
            include_preview_animation=settings.include_preview_animation,
        )
        skeleton_metadata = {
            "hash": "hash_value_placeholder",
            "spine": settings.export.spine_version,
            "x": 0,
            "y": 0,
            "width": settings.export.texture_width,
            "height": settings.export.texture_height,
            "images": "",
            "audio": "./audio",
        }
        if camera_projection:
            assert isinstance(bake_plan, CameraProjectionPlan)
            document_assembly = assemble_a1_camera_projection_document(
                rig,
                z_groups,
                bake_plan,
                assembly_settings,
                skeleton_metadata=skeleton_metadata,
            )
        else:
            document_assembly = assemble_a1_document(
                rig,
                z_groups,
                uv_regions.snapshots,
                assembly_settings,
                skeleton_metadata=skeleton_metadata,
            )
        document = document_assembly.document
        statistics.update(
            {
                "final_bone_count": len(document.bones),
                "slot_count": len(document.slots),
                "attachment_count": sum(
                    len(attachments)
                    for skin in document.skins
                    for attachments in skin.attachments.values()
                ),
            }
        )

        return PreparedA1Object(
            source_object=source_obj,
            object_id=object_id,
            prefix=prefix,
            settings=settings,
            output_paths=output_paths,
            source_snapshot=source_snapshot,
            z_groups=z_groups,
            geometry=geometry,
            texturing_topology=texturing_topology,
            unwrap_result=unwrap_result,
            uv_regions=uv_regions,
            material_analysis=material_analysis,
            bake_plan=bake_plan,
            rig=rig,
            document_assembly=document_assembly,
            warnings=tuple(warnings),
            statistics=MappingProxyType(dict(statistics)),
        )
    except A1ObjectPreparationError:
        raise
    except Exception as exc:
        raise A1ObjectPreparationError(
            stage=stage,
            object_id=object_id,
            cause=exc,
            statistics=statistics,
            warnings=tuple(warnings),
        ) from exc
