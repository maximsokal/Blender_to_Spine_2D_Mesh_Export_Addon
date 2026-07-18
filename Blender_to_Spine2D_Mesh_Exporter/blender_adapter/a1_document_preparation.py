"""Build the legacy-compatible Spine rig and typed A1 document."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Mapping, Tuple

from ..application import (
    A1DocumentAssemblyResult,
    A1DocumentAssemblySettings,
    A1SingleObjectStage,
    ExportIssue,
    assemble_a1_camera_projection_document,
    assemble_a1_document,
    build_a1_attachment_path,
    build_a1_attachment_sequence,
    calculate_a1_main_position_pixels,
    calculate_a1_mesh_bounds,
)
from ..domain.baking import CameraProjectionPlan
from ..domain.spine import LegacyRigBuildRequest, LegacyRigBuildResult, build_legacy_rig
from .a1_preparation_contracts import (
    A1ObjectPreparationError,
    StatisticsValue,
    build_skeleton_metadata,
    freeze_statistics,
)
from .a1_texture_planning import A1TexturePlanningResult


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class A1DocumentPreparationResult:
    """Final in-memory rig and document produced before output staging."""

    texture: A1TexturePlanningResult
    rig: LegacyRigBuildResult
    document_assembly: A1DocumentAssemblyResult
    warnings: Tuple[ExportIssue, ...]
    statistics: Mapping[str, StatisticsValue]

    def __post_init__(self) -> None:
        if not isinstance(self.texture, A1TexturePlanningResult):
            raise TypeError("texture must be A1TexturePlanningResult")
        if not isinstance(self.rig, LegacyRigBuildResult):
            raise TypeError("rig must be LegacyRigBuildResult")
        if not isinstance(self.document_assembly, A1DocumentAssemblyResult):
            raise TypeError("document_assembly must be A1DocumentAssemblyResult")
        if self.rig.request.prefix != self.texture.uv.source.prefix:
            raise ValueError("rig prefix must match prepared source prefix")
        if not isinstance(self.warnings, tuple) or not all(
            isinstance(issue, ExportIssue) for issue in self.warnings
        ):
            raise TypeError("warnings must be a tuple of ExportIssue values")
        if not isinstance(self.statistics, Mapping):
            raise TypeError("statistics must be a mapping")


def prepare_a1_document(
    texture: A1TexturePlanningResult,
) -> A1DocumentPreparationResult:
    """Build the A1 rig and document from fully analysed geometry and shading."""

    if not isinstance(texture, A1TexturePlanningResult):
        raise TypeError("texture must be A1TexturePlanningResult")
    uv = texture.uv
    source = uv.source
    stage = A1SingleObjectStage.BUILD_RIG
    statistics = texture.statistics
    try:
        camera_projection = isinstance(texture.bake_plan, CameraProjectionPlan)
        bounds = calculate_a1_mesh_bounds(source.source_snapshot)
        rig = build_legacy_rig(
            LegacyRigBuildRequest(
                prefix=source.prefix,
                texture_width=source.settings.export.texture_width,
                texture_height=source.settings.export.texture_height,
                z_groups=source.z_groups.groups,
                main_position_pixels=(
                    None
                    if camera_projection
                    else calculate_a1_main_position_pixels(
                        source.source_snapshot,
                        source.settings,
                    )
                ),
                scale_mode=source.settings.rig_scale_mode,
            )
        )
        statistics = freeze_statistics(
            statistics,
            {"base_rig_bone_count": len(rig.bones)},
        )

        stage = A1SingleObjectStage.ASSEMBLE_DOCUMENT
        assembly_settings = A1DocumentAssemblySettings(
            prefix=source.prefix,
            uv_layer_name=source.settings.uv.layer_name,
            image_path=build_a1_attachment_path(
                texture.bake_plan,
                source.output_paths,
            ),
            attachment_width=source.settings.export.texture_width,
            attachment_height=source.settings.export.texture_height,
            center_x=0.0 if camera_projection else bounds.center_x,
            center_y=0.0 if camera_projection else bounds.center_y,
            sequence=build_a1_attachment_sequence(texture.bake_plan),
            include_control_icons=source.settings.include_control_icons,
            include_preview_animation=source.settings.include_preview_animation,
        )
        skeleton_metadata = build_skeleton_metadata(source.settings)
        if camera_projection:
            if not isinstance(texture.bake_plan, CameraProjectionPlan):
                raise TypeError("camera projection plan type was lost")
            document_assembly = assemble_a1_camera_projection_document(
                rig,
                source.z_groups,
                texture.bake_plan,
                assembly_settings,
                skeleton_metadata=skeleton_metadata,
            )
        else:
            document_assembly = assemble_a1_document(
                rig,
                source.z_groups,
                uv.uv_regions.snapshots,
                assembly_settings,
                skeleton_metadata=skeleton_metadata,
            )
        document = document_assembly.document
        statistics = freeze_statistics(
            statistics,
            {
                "final_bone_count": len(document.bones),
                "slot_count": len(document.slots),
                "attachment_count": sum(
                    len(attachments)
                    for skin in document.skins
                    for attachments in skin.attachments.values()
                ),
            },
        )
        logger.debug(
            "Prepared Spine document for %s: bones=%d slots=%d attachments=%d",
            source.object_id,
            len(document.bones),
            len(document.slots),
            statistics["attachment_count"],
        )
        return A1DocumentPreparationResult(
            texture=texture,
            rig=rig,
            document_assembly=document_assembly,
            warnings=texture.warnings,
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
            warnings=texture.warnings,
        ) from exc


__all__ = ["A1DocumentPreparationResult", "prepare_a1_document"]
