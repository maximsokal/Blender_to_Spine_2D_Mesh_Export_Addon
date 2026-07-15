"""Blender-facing orchestration for one complete A1 export transaction.

The function in this module is intentionally not a Blender operator. It translates
one live Mesh object into immutable snapshots, runs every validated application
stage, stages texture and JSON files in one atomic filesystem transaction, and
returns a structured :class:`ExportResult` suitable for the future UI adapter.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..application import (
    A1DocumentAssemblySettings,
    A1SingleObjectExportSettings,
    A1SingleObjectStage,
    A1SourceGeometryMode,
    ExportIssue,
    ExportResult,
    IssueSeverity,
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
from ..domain.baking import build_bake_plan
from ..domain.geometry import LineageSeverity
from ..domain.spine import (
    LegacyRigBuildRequest,
    SpineSerializer,
    build_legacy_rig,
)
from ..infrastructure import AtomicFileCommitError, atomic_file_transaction
from .bake_executor import BakeExecutionError, stage_bake_plan_outputs
from .evaluated_mesh_reader import read_evaluated_mesh_snapshot
from .material_analyzer import analyse_object_materials
from .mesh_reader import read_source_mesh_snapshot
from .uv_unwrap import unwrap_snapshot_uv

logger = logging.getLogger(__name__)


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
    context: dict[str, object] | None = None,
) -> ExportIssue:
    return ExportIssue(
        severity=IssueSeverity.WARNING,
        stage=stage.value,
        code=code,
        message=message,
        object_id=object_id,
        context={} if context is None else context,
    )


def _failure_result(
    *,
    stage: A1SingleObjectStage,
    exc: Exception,
    object_id: str | None,
    statistics: dict[str, int | float | str],
) -> ExportResult:
    technical_details = f"{type(exc).__name__}: {exc}"
    logger.exception(
        "A1 single-object export failed at stage %s for '%s'",
        stage.value,
        object_id,
    )
    return ExportResult(
        success=False,
        issues=(
            ExportIssue(
                severity=IssueSeverity.ERROR,
                stage=stage.value,
                code=stage.error_code,
                message=str(exc) or type(exc).__name__,
                object_id=object_id,
                technical_details=technical_details,
                context={"exception_type": type(exc).__name__},
            ),
        ),
        statistics=dict(statistics),
    )


def _write_staged_json(
    path: Path,
    json_text: str,
) -> None:
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if not isinstance(json_text, str) or not json_text:
        raise ValueError("json_text must be a non-empty string")
    path.write_text(json_text, encoding="utf-8")
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Staged Spine JSON was not written: {path}")


def export_a1_single_object(
    source_obj: Any,
    settings: A1SingleObjectExportSettings,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> ExportResult:
    """Export one Blender mesh through the complete deterministic A1 pipeline.

    No production operator is changed by this function. Source Object, Mesh,
    materials, selection, active object, mode, frame, and render settings are
    restored by the lower-level adapters on both success and failure.
    """

    stage = A1SingleObjectStage.VALIDATE_REQUEST
    object_id: str | None = None
    statistics: dict[str, int | float | str] = {}
    warnings: list[ExportIssue] = []

    try:
        if not isinstance(settings, A1SingleObjectExportSettings):
            raise TypeError("settings must be A1SingleObjectExportSettings")
        object_id = _object_name(source_obj)
        prefix, _ = resolve_a1_names(object_id, settings)
        output_paths = resolve_a1_output_paths(object_id, settings)
        statistics.update(
            {
                "source_object": object_id,
                "rig_prefix": prefix,
                "source_geometry_mode": settings.source_geometry_mode.value,
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
                        f"{unwrap_result.statistics.outside_unit_square_count} UV loops "
                        "are outside the unit square"
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
        bake_plan = build_bake_plan(
            material_analysis,
            build_a1_bake_settings(object_id, settings),
        )
        statistics.update(
            {
                "bake_mode": bake_plan.bake_mode.value,
                "bake_frame_count": len(bake_plan.frame_tasks),
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
                main_position_pixels=calculate_a1_main_position_pixels(
                    source_snapshot,
                    settings,
                ),
                scale_mode=settings.rig_scale_mode,
            )
        )
        statistics["base_rig_bone_count"] = len(rig.bones)

        stage = A1SingleObjectStage.ASSEMBLE_DOCUMENT
        attachment_path = build_a1_attachment_path(bake_plan, output_paths)
        document_assembly = assemble_a1_document(
            rig,
            z_groups,
            uv_regions.snapshots,
            A1DocumentAssemblySettings(
                prefix=prefix,
                uv_layer_name=settings.uv.layer_name,
                image_path=attachment_path,
                attachment_width=settings.export.texture_width,
                attachment_height=settings.export.texture_height,
                center_x=bounds.center_x,
                center_y=bounds.center_y,
                sequence=build_a1_attachment_sequence(bake_plan),
            ),
            skeleton_metadata={
                "hash": "hash_value_placeholder",
                "spine": settings.export.spine_version,
                "x": 0,
                "y": 0,
                "width": settings.export.texture_width,
                "height": settings.export.texture_height,
                "images": "",
                "audio": "./audio",
            },
        )
        document = document_assembly.document
        json_text = SpineSerializer().to_json(
            document,
            indent=settings.json_indent,
        )
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

        stage = A1SingleObjectStage.STAGE_OUTPUTS
        with atomic_file_transaction() as output_transaction:
            json_reservation = output_transaction.reserve(output_paths.json_path)
            _write_staged_json(json_reservation.staged_path, json_text)
            bake_reservations = stage_bake_plan_outputs(
                source_obj,
                unwrap_result.snapshot,
                bake_plan,
                output_transaction,
                settings.bake_execution,
                context=context,
                scene=scene,
            )

            stage = A1SingleObjectStage.COMMIT_OUTPUTS
            committed_paths = output_transaction.commit()

        expected_paths = (
            json_reservation.final_path,
            *(reservation.final_path for reservation in bake_reservations),
        )
        if tuple(committed_paths) != expected_paths:
            raise AtomicFileCommitError(
                "Committed output order does not match reserved JSON and bake files"
            )
        statistics["output_file_count"] = len(committed_paths)
        logger.info(
            "A1 single-object export completed for '%s': %s",
            object_id,
            tuple(str(path) for path in committed_paths),
        )
        return ExportResult(
            success=True,
            output_files=tuple(committed_paths),
            issues=tuple(warnings),
            statistics=statistics,
        )
    except Exception as exc:
        return _failure_result(
            stage=stage,
            exc=exc,
            object_id=object_id,
            statistics=statistics,
        )
