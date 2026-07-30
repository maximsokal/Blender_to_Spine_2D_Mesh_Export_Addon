"""Build the selected Spine rig and typed A1 document."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    calculate_a1_object_bake_main_position_pixels,
)
from ..domain.baking import CameraProjectionPlan
from ..domain.spine.legacy_rig_contracts import (
    LegacyRigBuildRequest,
    LegacyRigBuildResult,
)
from ..domain.spine.rig_builder import build_rig
from ..domain.spine.rig_profiles import A1RigProfile, resolve_a1_rig_profile
from ..domain.spine.two_axis_scale_profile import TwoAxisScaleRigProfile
from ..domain.spine.two_axis_scale_spine41 import (
    adapt_two_axis_document_for_spine41,
)
from ..domain.spine.version_target import (
    SpineJsonTarget,
    resolve_spine_json_target,
)
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


def finalize_a1_document_assembly_for_target(
    document_assembly: A1DocumentAssemblyResult,
    *,
    spine_target: object,
    prefix: str,
) -> A1DocumentAssemblyResult:
    """Apply target-specific rig semantics only after canonical document assembly.

    Projection and attachment builders validate the canonical rig against its exact
    deterministic plan. Spine 4.1 changes two transform constraints, so applying those
    changes before projection makes the otherwise valid rig fail its own profile
    validator. The final immutable document is the correct target boundary: attachments,
    weighted vertices, slots, visuals, and animations already exist, while constraint
    topology can still be replaced without mutating the canonical rig result.
    """

    if not isinstance(document_assembly, A1DocumentAssemblyResult):
        raise TypeError("document_assembly must be A1DocumentAssemblyResult")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string")

    resolved_target = resolve_spine_json_target(spine_target)
    if resolved_target is SpineJsonTarget.SPINE_4_2:
        return document_assembly
    if resolved_target is not SpineJsonTarget.SPINE_4_1:
        raise ValueError(
            "A1 document finalization is not implemented for "
            f"{resolved_target.label} ({resolved_target.exact_version})"
        )

    rig = document_assembly.rig
    resolved_profile = resolve_a1_rig_profile(rig.profile.profile_id)
    if (
        resolved_profile is not A1RigProfile.TWO_AXIS_ROTATION_SCALE
        or not isinstance(rig.profile, TwoAxisScaleRigProfile)
    ):
        raise ValueError(
            "Spine 4.1 document finalization currently requires "
            "TWO_AXIS_ROTATION_SCALE"
        )
    if rig.request.prefix.strip() != prefix.strip():
        raise ValueError(
            f"Document finalization prefix {prefix!r} does not match rig prefix "
            f"{rig.request.prefix!r}"
        )

    adapted_document = adapt_two_axis_document_for_spine41(
        document_assembly.document,
        profile=rig.profile,
        prefix=prefix,
    )
    adapted_build = replace(
        document_assembly.document_build,
        document=adapted_document,
    )
    return replace(
        document_assembly,
        document_build=adapted_build,
    )


def prepare_a1_document(
    texture: A1TexturePlanningResult,
) -> A1DocumentPreparationResult:
    """Build the selected A1 rig and document from analysed geometry and shading."""

    if not isinstance(texture, A1TexturePlanningResult):
        raise TypeError("texture must be A1TexturePlanningResult")
    uv = texture.uv
    source = uv.source
    stage = A1SingleObjectStage.BUILD_RIG
    statistics = texture.statistics
    try:
        camera_projection = isinstance(texture.bake_plan, CameraProjectionPlan)
        main_position_pixels = (
            None
            if camera_projection
            else calculate_a1_object_bake_main_position_pixels(
                source.source_snapshot,
                source.settings,
            )
        )

        rig = build_rig(
            LegacyRigBuildRequest(
                prefix=source.prefix,
                texture_width=source.settings.export.texture_width,
                texture_height=source.settings.export.texture_height,
                z_groups=source.z_groups.groups,
                main_position_pixels=main_position_pixels,
                scale_mode=source.settings.rig_scale_mode,
                setup_pose_mode=source.settings.rig_setup_pose_mode,
            ),
            source.settings.export.rig_profile,
            spine_target=source.settings.export.spine_target,
        )
        statistics = freeze_statistics(
            statistics,
            {
                "base_rig_bone_count": len(rig.bones),
                "rig_profile": rig.profile.profile_id,
                "rig_setup_pose_mode": rig.request.setup_pose_mode.value,
            },
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
            # Object-bake mesh coordinates are already relative to Blender Object Origin.
            # Never recenter them around the geometry bounds or the Spine pivot changes.
            center_x=0.0,
            center_y=0.0,
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

        document_assembly = finalize_a1_document_assembly_for_target(
            document_assembly,
            spine_target=source.settings.export.spine_target,
            prefix=source.prefix,
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
            "Prepared Spine document for %s: target=%s profile=%s setup=%s "
            "bones=%d slots=%d attachments=%d",
            source.object_id,
            source.settings.export.spine_version,
            rig.profile.profile_id,
            rig.request.setup_pose_mode.value,
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


__all__ = [
    "A1DocumentPreparationResult",
    "finalize_a1_document_assembly_for_target",
    "prepare_a1_document",
]
