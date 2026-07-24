"""Build the legacy-compatible Spine rig and typed A1 document."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from math import isfinite
from typing import Mapping, Tuple

from ..application import (
    A1DocumentAssemblyResult,
    A1DocumentAssemblySettings,
    A1MeshBounds,
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
from ..domain.spine.legacy_rig_assembly import build_legacy_rig
from ..domain.spine.legacy_rig_contracts import (
    LegacyRigBuildRequest,
    LegacyRigBuildResult,
)
from ..domain.spine.legacy_rig_scale import calculate_uniform_scale
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


def _combine_object_bake_main_position_pixels(
    world_position_pixels: tuple[float, float] | None,
    bounds: A1MeshBounds,
    uniform_scale: float,
) -> tuple[float, float]:
    """Combine Object-origin placement with the centered attachment's local offset.

    Object-bake attachment vertices are intentionally centered around the source XY
    bounding-box midpoint. The inverse center translation therefore belongs on the main
    bone. Blender Y is inverted by attachment projection, so the matching main-bone
    offset is ``(center_x, -center_y) * uniform_scale``.

    Connected preparation disables absolute world placement but still uses this helper;
    its generated document then carries only the local geometry-center offset. Connected
    composition can safely add the anchor-relative Object translation later.
    """

    if world_position_pixels is not None:
        if not isinstance(world_position_pixels, tuple) or len(world_position_pixels) != 2:
            raise ValueError("world_position_pixels must contain two finite values or None")
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(float(value))
            for value in world_position_pixels
        ):
            raise ValueError("world_position_pixels must contain two finite values or None")
        base_x = float(world_position_pixels[0])
        base_y = float(world_position_pixels[1])
    else:
        base_x = 0.0
        base_y = 0.0

    if not isinstance(bounds, A1MeshBounds):
        raise TypeError("bounds must be A1MeshBounds")
    if (
        isinstance(uniform_scale, bool)
        or not isinstance(uniform_scale, (int, float))
        or not isfinite(float(uniform_scale))
        or float(uniform_scale) <= 0.0
    ):
        raise ValueError("uniform_scale must be a finite positive number")

    resolved_scale = float(uniform_scale)
    return (
        base_x + float(bounds.center_x) * resolved_scale,
        base_y - float(bounds.center_y) * resolved_scale,
    )


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
        if camera_projection:
            main_position_pixels = None
        else:
            world_position_pixels = calculate_a1_main_position_pixels(
                source.source_snapshot,
                source.settings,
            )
            uniform_scale = calculate_uniform_scale(
                source.settings.export.texture_width,
                source.settings.export.texture_height,
                source.settings.rig_scale_mode,
            )
            main_position_pixels = _combine_object_bake_main_position_pixels(
                world_position_pixels,
                bounds,
                uniform_scale,
            )

        rig = build_legacy_rig(
            LegacyRigBuildRequest(
                prefix=source.prefix,
                texture_width=source.settings.export.texture_width,
                texture_height=source.settings.export.texture_height,
                z_groups=source.z_groups.groups,
                main_position_pixels=main_position_pixels,
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
            uv_range_policy=source.settings.uv.range_policy,
            uv_range_epsilon=source.settings.uv.range_epsilon,
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
