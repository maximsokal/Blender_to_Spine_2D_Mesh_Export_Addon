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
from ..domain.camera_projection import A1CameraProjectionKind
from ..domain.projection import A1ProjectionDirection
from ..domain.spine.legacy_attachment_builder import (
    LegacyMeshDocumentBuildResult,
)
from ..domain.spine.legacy_rig_contracts import (
    LegacyRigBuildRequest,
    LegacyRigBuildResult,
    LegacyZGroupOriginMode,
)
from ..domain.spine.model import MeshAttachment, Skin, SpineDocument
from ..domain.spine.rig_builder import build_rig
from ..domain.spine.rig_profiles import (
    A1CameraLayerProjectionKind,
    A1RigProfile,
    A1RigSetupPoseMode,
    resolve_a1_rig_profile,
)
from ..domain.spine.two_axis_scale_profile import TwoAxisScaleRigProfile
from ..domain.spine.two_axis_scale_spine38 import (
    Spine38TwoAxisDocumentAdaptation,
    adapt_two_axis_document_for_spine38_with_report,
)
from ..domain.spine.two_axis_scale_spine41 import (
    Spine41TwoAxisDocumentAdaptation,
    adapt_two_axis_document_for_spine41_with_report,
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


def _skin_by_name(document: SpineDocument) -> dict[str, Skin]:
    """Index final document skins while rejecting ambiguous names."""

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    result: dict[str, Skin] = {}
    for skin in document.skins:
        if skin.name in result:
            raise ValueError(f"Duplicate skin name in finalized document: {skin.name!r}")
        result[skin.name] = skin
    return result


def _adapted_mesh_attachment(
    skins_by_name: Mapping[str, Skin],
    *,
    skin_name: str,
    slot_name: str,
    attachment_name: str,
) -> MeshAttachment:
    """Resolve one remapped typed mesh attachment by its stable document path."""

    for field_name, value in (
        ("skin_name", skin_name),
        ("slot_name", slot_name),
        ("attachment_name", attachment_name),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")

    skin = skins_by_name.get(skin_name)
    if skin is None:
        raise ValueError(f"Finalized document is missing skin {skin_name!r}")
    slot_attachments = skin.attachments.get(slot_name)
    if not isinstance(slot_attachments, Mapping):
        raise ValueError(
            f"Finalized skin {skin_name!r} is missing slot attachment table "
            f"{slot_name!r}"
        )
    attachment = slot_attachments.get(attachment_name)
    if not isinstance(attachment, MeshAttachment):
        raise TypeError(
            f"Finalized attachment {skin_name!r}/{slot_name!r}/{attachment_name!r} "
            "must be MeshAttachment"
        )
    return attachment


def _synchronize_document_build_for_spine41(
    document_build: LegacyMeshDocumentBuildResult,
    adaptation: (
        Spine38TwoAxisDocumentAdaptation | Spine41TwoAxisDocumentAdaptation
    ),
) -> LegacyMeshDocumentBuildResult:
    """Synchronize builder metadata after legacy bridge insertion.

    The legacy 3.8/4.0/4.1 adapters insert parent bones before existing weighted vertex
    bones. The serialized attachment streams are remapped by the domain adapter, so the
    immutable ``LegacyMeshDocumentBuildResult`` must point at those same remapped
    attachments and expose the corresponding new vertex-bone start indices. The
    canonical rig remains untouched because projection and deterministic rig validation
    already completed.
    """

    if not isinstance(document_build, LegacyMeshDocumentBuildResult):
        raise TypeError("document_build must be LegacyMeshDocumentBuildResult")
    if not isinstance(
        adaptation,
        (Spine38TwoAxisDocumentAdaptation, Spine41TwoAxisDocumentAdaptation),
    ):
        raise TypeError(
            "adaptation must be a Spine 3.8 or Spine 4.1 two-axis adaptation"
        )

    index_map = adaptation.old_to_new_bone_indices
    skins_by_name = _skin_by_name(adaptation.document)

    adapted_components = []
    for component_index, component in enumerate(document_build.components):
        new_start_index = index_map.get(component.vertex_bone_start_index)
        if new_start_index is None:
            raise ValueError(
                "Legacy 3.8/4.x bone remap does not contain component vertex-bone start "
                f"index {component.vertex_bone_start_index} at component "
                f"{component_index}"
            )
        adapted_attachment = _adapted_mesh_attachment(
            skins_by_name,
            skin_name=component.request.skin_name,
            slot_name=component.request.slot_name,
            attachment_name=component.request.attachment_name,
        )
        adapted_components.append(
            replace(
                component,
                vertex_bone_start_index=new_start_index,
                attachment=adapted_attachment,
            )
        )

    adapted_build_skins: list[Skin] = []
    for source_skin in document_build.skins:
        final_skin = skins_by_name.get(source_skin.name)
        if final_skin is None:
            raise ValueError(
                f"Finalized document is missing builder skin {source_skin.name!r}"
            )
        attachment_groups: dict[
            str,
            dict[str, MeshAttachment | Mapping[str, object]],
        ] = {}
        for slot_name, source_attachments in source_skin.attachments.items():
            final_attachments = final_skin.attachments.get(slot_name)
            if not isinstance(final_attachments, Mapping):
                raise ValueError(
                    f"Finalized skin {source_skin.name!r} is missing slot "
                    f"{slot_name!r}"
                )
            resolved_group: dict[str, MeshAttachment | Mapping[str, object]] = {}
            for attachment_name in source_attachments:
                if attachment_name not in final_attachments:
                    raise ValueError(
                        f"Finalized skin {source_skin.name!r} slot {slot_name!r} "
                        f"is missing attachment {attachment_name!r}"
                    )
                resolved_group[attachment_name] = final_attachments[attachment_name]
            attachment_groups[slot_name] = resolved_group
        adapted_build_skins.append(
            replace(source_skin, attachments=attachment_groups)
        )

    return replace(
        document_build,
        components=tuple(adapted_components),
        skins=tuple(adapted_build_skins),
        document=adaptation.document,
    )


def _resolve_z_group_origin_mode(
    *,
    camera_projection: bool,
    rig_profile: A1RigProfile,
) -> LegacyZGroupOriginMode:
    """Select the approved depth-reference policy for one preparation route."""

    if not isinstance(camera_projection, bool):
        raise TypeError("camera_projection must be bool")
    if not isinstance(rig_profile, A1RigProfile):
        raise TypeError("rig_profile must be A1RigProfile")
    if (
        not camera_projection
        and rig_profile is A1RigProfile.TWO_AXIS_ROTATION_SCALE
    ):
        return LegacyZGroupOriginMode.OBJECT_ORIGIN
    return LegacyZGroupOriginMode.MINIMUM_Z


def _active_camera_layer_kind(
    value: A1CameraProjectionKind | None,
) -> A1CameraLayerProjectionKind:
    """Map typed projection-domain camera kind to rig-domain semantics."""

    if value is None:
        raise ValueError(
            "Active Camera preparation did not provide camera_projection_kind"
        )
    if not isinstance(value, A1CameraProjectionKind):
        raise TypeError("value must be A1CameraProjectionKind or None")
    if value is A1CameraProjectionKind.PERSPECTIVE:
        return A1CameraLayerProjectionKind.PERSPECTIVE
    if value is A1CameraProjectionKind.ORTHOGRAPHIC:
        return A1CameraLayerProjectionKind.ORTHOGRAPHIC
    raise AssertionError(f"Unhandled Active Camera projection kind: {value}")


def finalize_a1_document_assembly_for_target(
    document_assembly: A1DocumentAssemblyResult,
    *,
    spine_target: object,
    prefix: str,
) -> A1DocumentAssemblyResult:
    """Apply target-specific rig semantics only after canonical document assembly."""

    if not isinstance(document_assembly, A1DocumentAssemblyResult):
        raise TypeError("document_assembly must be A1DocumentAssemblyResult")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string")

    resolved_target = resolve_spine_json_target(spine_target)
    rig = document_assembly.rig
    resolved_profile = resolve_a1_rig_profile(rig.profile.profile_id)

    if resolved_target in {
        SpineJsonTarget.SPINE_4_2,
        SpineJsonTarget.SPINE_4_3,
    } or (
        resolved_target is SpineJsonTarget.SPINE_3_8
        and resolved_profile is A1RigProfile.THREE_AXIS_ROTATION
    ):
        return document_assembly

    if resolved_target not in {
        SpineJsonTarget.SPINE_3_8,
        SpineJsonTarget.SPINE_4_0,
        SpineJsonTarget.SPINE_4_1,
    }:
        raise ValueError(
            "A1 document finalization is not implemented for "
            f"{resolved_target.label} ({resolved_target.exact_version})"
        )

    if (
        resolved_profile is not A1RigProfile.TWO_AXIS_ROTATION_SCALE
        or not isinstance(rig.profile, TwoAxisScaleRigProfile)
    ):
        raise ValueError(
            f"{resolved_target.label} legacy safety finalization requires "
            "TWO_AXIS_ROTATION_SCALE"
        )
    if rig.request.prefix.strip() != prefix.strip():
        raise ValueError(
            f"Document finalization prefix {prefix!r} does not match rig prefix "
            f"{rig.request.prefix!r}"
        )

    if resolved_target is SpineJsonTarget.SPINE_3_8:
        adaptation = adapt_two_axis_document_for_spine38_with_report(
            document_assembly.document,
            profile=rig.profile,
            prefix=prefix,
        )
    else:
        adaptation = adapt_two_axis_document_for_spine41_with_report(
            document_assembly.document,
            profile=rig.profile,
            prefix=prefix,
        )
    adapted_build = _synchronize_document_build_for_spine41(
        document_assembly.document_build,
        adaptation,
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
        active_camera_layout = (
            not camera_projection
            and source.settings.projection_direction
            is A1ProjectionDirection.ACTIVE_CAMERA
        )
        resolved_rig_profile = resolve_a1_rig_profile(
            source.settings.export.rig_profile
        )
        if (
            active_camera_layout
            and resolved_rig_profile is not A1RigProfile.TWO_AXIS_ROTATION_SCALE
        ):
            raise ValueError(
                "Active Camera rigid layers require TWO_AXIS_ROTATION_SCALE"
            )

        camera_layer_kind = (
            _active_camera_layer_kind(source.camera_projection_kind)
            if active_camera_layout
            else None
        )
        resolved_setup_pose_mode = (
            A1RigSetupPoseMode.PREPROJECTED_SCREEN
            if active_camera_layout
            else source.settings.rig_setup_pose_mode
        )
        z_group_origin_mode = _resolve_z_group_origin_mode(
            camera_projection=camera_projection,
            rig_profile=resolved_rig_profile,
        )
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
                setup_pose_mode=resolved_setup_pose_mode,
                z_group_origin_mode=z_group_origin_mode,
                camera_layer_projection_kind=camera_layer_kind,
            ),
            resolved_rig_profile,
            spine_target=source.settings.export.spine_target,
        )
        statistics = freeze_statistics(
            statistics,
            {
                "base_rig_bone_count": len(rig.bones),
                "rig_profile": rig.profile.profile_id,
                "rig_setup_pose_mode": rig.request.setup_pose_mode.value,
                "z_group_origin_mode": rig.request.z_group_origin_mode.value,
                "camera_layer_projection_kind": (
                    "" if camera_layer_kind is None else camera_layer_kind.value
                ),
                "camera_relative_depth_group_count": (
                    len(rig.info.z_groups) if active_camera_layout else 0
                ),
                "depth_setup_y_compensated": int(active_camera_layout),
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
            center_x=0.0,
            center_y=0.0,
            sequence=build_a1_attachment_sequence(texture.bake_plan),
            include_control_icons=source.settings.include_control_icons,
            include_preview_animation=source.settings.include_preview_animation,
            uv_range_policy=source.settings.uv.range_policy,
            uv_range_epsilon=source.settings.uv.range_epsilon,
            compensate_depth_setup_y=active_camera_layout,
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
        final_rig = document_assembly.rig
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
                "final_rig_setup_pose_mode": final_rig.request.setup_pose_mode.value,
            },
        )
        logger.debug(
            "Prepared Spine document for %s: target=%s profile=%s setup=%s "
            "camera_layer=%s z_origin=%s depth_y_compensation=%s "
            "bones=%d slots=%d attachments=%d",
            source.object_id,
            source.settings.export.spine_version,
            final_rig.profile.profile_id,
            final_rig.request.setup_pose_mode.value,
            (
                ""
                if final_rig.request.camera_layer_projection_kind is None
                else final_rig.request.camera_layer_projection_kind.value
            ),
            final_rig.request.z_group_origin_mode.value,
            active_camera_layout,
            len(document.bones),
            len(document.slots),
            statistics["attachment_count"],
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
    "A1DocumentPreparationResult",
    "finalize_a1_document_assembly_for_target",
    "prepare_a1_document",
]
