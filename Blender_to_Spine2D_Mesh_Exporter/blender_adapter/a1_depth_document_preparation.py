"""Build a Normal-style weighted rig over a camera-depth relief surface."""

from __future__ import annotations

import logging

from ..application import (
    A1SingleObjectStage,
    calculate_a1_object_bake_main_position_pixels,
)
from ..domain.baking import A1TextureExportMode, CameraProjectionPlan
from ..domain.geometry import DepthProjectionBaseMode
from ..domain.spine.legacy_rig_contracts import (
    LegacyRigBuildRequest,
    LegacyZGroupOriginMode,
)
from ..domain.spine.rig_builder import build_rig
from ..domain.spine.rig_profiles import resolve_a1_rig_profile
from .a1_document_preparation import (
    A1DocumentPreparationResult,
    _assemble_document_for_texture,
)
from .a1_preparation_contracts import (
    A1ObjectPreparationError,
    freeze_statistics,
)
from .a1_texture_planning import A1TexturePlanningResult


logger = logging.getLogger(__name__)


def _resolve_depth_z_group_origin_mode(
    base_mode: DepthProjectionBaseMode,
) -> LegacyZGroupOriginMode:
    """Map the depth-surface base policy to the existing Normal rig math.

    ``FARTHEST_VISIBLE`` uses the minimum camera-local Z as zero. Every other visible
    point then receives a non-negative offset toward the camera. ``OBJECT_ORIGIN``
    retains absolute camera-local Z relative to camera-space zero and is accepted only
    after the geometry stage proved that the origin lies behind all visible points.
    """

    if not isinstance(base_mode, DepthProjectionBaseMode):
        raise TypeError("base_mode must be DepthProjectionBaseMode")
    if base_mode is DepthProjectionBaseMode.FARTHEST_VISIBLE:
        return LegacyZGroupOriginMode.MINIMUM_Z
    if base_mode is DepthProjectionBaseMode.OBJECT_ORIGIN:
        return LegacyZGroupOriginMode.OBJECT_ORIGIN
    raise AssertionError(f"Unhandled depth base mode: {base_mode}")


def prepare_a1_depth_document(
    texture: A1TexturePlanningResult,
) -> A1DocumentPreparationResult:
    """Build generated vertex bones from depth points while retaining camera texture."""

    if not isinstance(texture, A1TexturePlanningResult):
        raise TypeError("texture must be A1TexturePlanningResult")
    source = texture.uv.source
    mode = source.settings.bake_execution.texture_export_mode
    if mode is not A1TextureExportMode.DEPTH_CAMERA_PROJECTION:
        raise ValueError(
            "prepare_a1_depth_document requires DEPTH_CAMERA_PROJECTION mode"
        )
    if not isinstance(texture.bake_plan, CameraProjectionPlan):
        raise TypeError(
            "Depth Camera Projection requires a CameraProjectionPlan texture route"
        )

    stage = A1SingleObjectStage.BUILD_RIG
    statistics = texture.statistics
    try:
        resolved_profile = resolve_a1_rig_profile(
            source.settings.export.rig_profile
        )
        depth_settings = source.settings.bake_execution.depth_projection
        z_origin_mode = _resolve_depth_z_group_origin_mode(
            depth_settings.base_mode
        )
        main_position = calculate_a1_object_bake_main_position_pixels(
            source.source_snapshot,
            source.settings,
        )
        rig = build_rig(
            LegacyRigBuildRequest(
                prefix=source.prefix,
                texture_width=source.settings.export.texture_width,
                texture_height=source.settings.export.texture_height,
                z_groups=source.z_groups.groups,
                main_position_pixels=main_position,
                scale_mode=source.settings.rig_scale_mode,
                setup_pose_mode=source.settings.rig_setup_pose_mode,
                z_group_origin_mode=z_origin_mode,
            ),
            resolved_profile,
            spine_target=source.settings.export.spine_target,
        )
        offsets = tuple(group.y_offset_pixels for group in rig.info.z_groups)
        if depth_settings.base_mode is DepthProjectionBaseMode.FARTHEST_VISIBLE:
            if not offsets or min(offsets) != 0.0 or any(value < 0.0 for value in offsets):
                raise ValueError(
                    "FARTHEST_VISIBLE depth rig must start at zero and extend only "
                    f"toward the camera; offsets={offsets}"
                )
        statistics = freeze_statistics(
            statistics,
            {
                "base_rig_bone_count": len(rig.bones),
                "rig_profile": rig.profile.profile_id,
                "rig_setup_pose_mode": rig.request.setup_pose_mode.value,
                "z_group_origin_mode": rig.request.z_group_origin_mode.value,
                "depth_camera_vertex_rig": 1,
                "depth_camera_absolute_z_retained": 1,
                "depth_camera_relief_base_mode": depth_settings.base_mode.value,
                "depth_camera_minimum_rig_offset": min(offsets),
                "depth_camera_maximum_rig_offset": max(offsets),
                "depth_camera_offsets_toward_camera_only": int(
                    depth_settings.base_mode
                    is DepthProjectionBaseMode.FARTHEST_VISIBLE
                ),
                "camera_layer_projection_kind": (
                    ""
                    if source.camera_projection_kind is None
                    else source.camera_projection_kind.value
                ),
            },
        )

        stage = A1SingleObjectStage.ASSEMBLE_DOCUMENT
        # The texture is rendered by Camera Projection, but attachment topology and
        # vertex bones are assembled by the Normal / UV Segments document path.
        document_assembly = _assemble_document_for_texture(
            texture,
            rig,
            camera_projection=False,
            active_camera_layout=False,
        )
        final_rig = document_assembly.rig
        document = document_assembly.document
        sequence_enabled = source.settings.export.sequence_frame_count > 0
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
                "final_rig_setup_pose_mode": final_rig.request.setup_pose_mode.value,
                "texture_sequence_enabled": int(sequence_enabled),
                "texture_sequence_fps": (
                    source.settings.export.sequence_timing.resolved_fps
                    if sequence_enabled
                    else 0.0
                ),
                "texture_sequence_encoding": (
                    source.settings.export.spine_target.texture_animation_encoding.value
                    if sequence_enabled
                    else "STATIC"
                ),
            },
        )
        logger.debug(
            "Prepared depth relief document for %s: profile=%s setup=%s base=%s "
            "z_groups=%d offsets=%s bones=%d slots=%d",
            source.object_id,
            final_rig.profile.profile_id,
            final_rig.request.setup_pose_mode.value,
            depth_settings.base_mode.value,
            len(final_rig.info.z_groups),
            offsets,
            len(document.bones),
            len(document.slots),
        )
        return A1DocumentPreparationResult(
            texture=texture,
            rig=final_rig,
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


__all__ = [
    "_resolve_depth_z_group_origin_mode",
    "prepare_a1_depth_document",
]
