"""Build one compensated camera-distance rig over front and reserve surfaces."""

from __future__ import annotations

import logging
from typing import Sequence

from ..application import (
    A1SingleObjectStage,
    calculate_a1_object_bake_main_position_pixels,
)
from ..domain.baking import A1TextureExportMode, CameraProjectionPlan
from ..domain.geometry import DepthParallaxGeometryPackage, DepthProjectionBaseMode
from ..domain.spine.legacy_rig_contracts import (
    LegacyRigBuildRequest,
    LegacyZGroupOriginMode,
)
from ..domain.spine.rig_builder import build_rig
from ..domain.spine.rig_profiles import resolve_a1_rig_profile
from .a1_depth_document_assembly import (
    assemble_and_finalize_a1_depth_document,
)
from .a1_document_preparation import A1DocumentPreparationResult
from .a1_preparation_contracts import (
    A1ObjectPreparationError,
    freeze_statistics,
)
from .a1_texture_planning import A1TexturePlanningResult


logger = logging.getLogger(__name__)


def _resolve_depth_z_group_origin_mode(
    base_mode: DepthProjectionBaseMode,
) -> LegacyZGroupOriginMode:
    """Keep the active camera as the single depth zero for every object."""

    if not isinstance(base_mode, DepthProjectionBaseMode):
        raise TypeError("base_mode must be DepthProjectionBaseMode")
    if base_mode in {
        DepthProjectionBaseMode.FARTHEST_VISIBLE,
        DepthProjectionBaseMode.OBJECT_ORIGIN,
    }:
        return LegacyZGroupOriginMode.OBJECT_ORIGIN
    raise AssertionError(f"Unhandled depth base mode: {base_mode}")


def prepare_a1_depth_document(
    texture: A1TexturePlanningResult,
    reserve_plans: Sequence[CameraProjectionPlan] = (),
) -> A1DocumentPreparationResult:
    """Build one shared vertex rig and all camera-textured depth attachments."""

    if not isinstance(texture, A1TexturePlanningResult):
        raise TypeError("texture must be A1TexturePlanningResult")
    if not isinstance(reserve_plans, (tuple, list)) or not all(
        isinstance(plan, CameraProjectionPlan) for plan in reserve_plans
    ):
        raise TypeError("reserve_plans must contain CameraProjectionPlan values")
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
    package = getattr(source, "parallax_package", None)
    if not isinstance(package, DepthParallaxGeometryPackage):
        raise TypeError(
            "Depth document preparation requires DepthParallaxGeometryPackage"
        )
    expected_attachment_count = package.attachment_count
    if expected_attachment_count != 1 + len(tuple(reserve_plans)):
        raise ValueError(
            "Parallax package attachment count does not match reserve plans"
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
        if source.settings.use_world_location_for_main_bone and main_position is None:
            raise ValueError(
                "Depth Camera Projection lost projected Object Origin placement"
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
        offsets = tuple(float(group.y_offset_pixels) for group in rig.info.z_groups)
        if not offsets or any(value <= 0.0 for value in offsets):
            raise ValueError(
                "Depth rig offsets must be positive distances from shared camera zero; "
                f"offsets={offsets}"
            )
        if len(offsets) > 1 and max(offsets) <= min(offsets):
            raise ValueError(
                f"Depth rig lost camera-distance ordering: offsets={offsets}"
            )

        statistics = freeze_statistics(
            statistics,
            {
                "base_rig_bone_count": len(rig.bones),
                "rig_profile": rig.profile.profile_id,
                "rig_setup_pose_mode": rig.request.setup_pose_mode.value,
                "z_group_origin_mode": rig.request.z_group_origin_mode.value,
                "depth_camera_vertex_rig": 1,
                "depth_camera_global_camera_zero": 1,
                "depth_camera_absolute_distance_retained": 1,
                "depth_camera_relief_base_mode": depth_settings.base_mode.value,
                "depth_camera_minimum_rig_offset": min(offsets),
                "depth_camera_maximum_rig_offset": max(offsets),
                "depth_camera_parent_y_compensated": 1,
                "depth_camera_single_attachment": int(
                    expected_attachment_count == 1
                ),
                "depth_camera_expected_attachment_count": (
                    expected_attachment_count
                ),
                "depth_camera_reserve_attachment_count": len(
                    tuple(reserve_plans)
                ),
                "depth_camera_projected_main_x": (
                    0.0 if main_position is None else float(main_position[0])
                ),
                "depth_camera_projected_main_y": (
                    0.0 if main_position is None else float(main_position[1])
                ),
                "camera_layer_projection_kind": (
                    ""
                    if source.camera_projection_kind is None
                    else source.camera_projection_kind.value
                ),
            },
        )

        stage = A1SingleObjectStage.ASSEMBLE_DOCUMENT
        document_assembly = assemble_and_finalize_a1_depth_document(
            texture,
            rig,
            reserve_plans,
        )
        final_rig = document_assembly.rig
        document = document_assembly.document
        component_count = len(document_assembly.document_build.components)
        if component_count != expected_attachment_count:
            raise ValueError(
                "Depth Camera Projection component count differs from parallax package; "
                f"expected={expected_attachment_count}, got={component_count}"
            )
        sequence_enabled = source.settings.export.sequence_frame_count > 0
        actual_attachment_count = sum(
            len(attachments)
            for skin in document.skins
            for attachments in skin.attachments.values()
        )
        if actual_attachment_count != expected_attachment_count:
            raise ValueError(
                "Depth Camera Projection serialized attachment count differs from "
                f"package; expected={expected_attachment_count}, "
                f"got={actual_attachment_count}"
            )
        statistics = freeze_statistics(
            statistics,
            {
                "final_bone_count": len(document.bones),
                "slot_count": len(document.slots),
                "attachment_count": actual_attachment_count,
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
        logger.info(
            "Prepared depth parallax document for %s: profile=%s setup=%s "
            "base_policy=%s camera_zero=0 main=%s distance_offsets=%s "
            "bones=%d slots=%d attachments=%d reserve=%d",
            source.object_id,
            final_rig.profile.profile_id,
            final_rig.request.setup_pose_mode.value,
            depth_settings.base_mode.value,
            main_position,
            offsets,
            len(document.bones),
            len(document.slots),
            actual_attachment_count,
            len(tuple(reserve_plans)),
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
