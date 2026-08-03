"""Assemble one complete Depth Camera Projection surface into one Spine attachment."""

from __future__ import annotations

from dataclasses import replace

from ..application import (
    A1AttachmentProjectionSettings,
    A1DocumentAssemblyResult,
    A1DocumentAssemblySettings,
    build_a1_attachment_path,
    build_a1_attachment_sequence,
    validate_document_material_correspondence,
)
from ..application.a1_depth_attachment_projection import (
    project_depth_camera_attachment,
)
from ..domain.baking import A1TextureExportMode, CameraProjectionPlan
from ..domain.geometry import MeshSnapshot, MeshSnapshotValidator
from ..domain.spine import (
    LegacyMeshDocumentBuildResult,
    LegacyRigBuildResult,
    apply_attachment_sequence_animations,
    apply_rig_visual_options,
    build_legacy_mesh_document,
)
from ..domain.spine.vertex_bone_optimizer import optimize_shared_vertex_bones
from .a1_document_preparation import (
    finalize_a1_document_assembly_for_target,
)
from .a1_preparation_contracts import build_skeleton_metadata
from .a1_texture_planning import A1TexturePlanningResult


def _build_depth_document(
    texture: A1TexturePlanningResult,
    rig: LegacyRigBuildResult,
) -> A1DocumentAssemblyResult:
    """Build canonical one-attachment depth document before target adaptation."""

    if not isinstance(texture, A1TexturePlanningResult):
        raise TypeError("texture must be A1TexturePlanningResult")
    if not isinstance(rig, LegacyRigBuildResult):
        raise TypeError("rig must be LegacyRigBuildResult")

    source = texture.uv.source
    if (
        source.settings.bake_execution.texture_export_mode
        is not A1TextureExportMode.DEPTH_CAMERA_PROJECTION
    ):
        raise ValueError(
            "Depth document assembly requires DEPTH_CAMERA_PROJECTION mode"
        )
    if not isinstance(texture.bake_plan, CameraProjectionPlan):
        raise TypeError(
            "Depth document assembly requires CameraProjectionPlan texture output"
        )

    snapshot = texture.uv.unwrap_result.snapshot
    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("Depth UV result must retain a MeshSnapshot")
    MeshSnapshotValidator().validate_or_raise(snapshot)
    if snapshot.snapshot_id != texture.uv.texturing_topology.snapshot.snapshot_id:
        raise ValueError(
            "Depth UV snapshot identity differs from its complete texturing topology"
        )

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
        sequence_timing=source.settings.export.sequence_timing,
        uv_range_policy=source.settings.uv.range_policy,
        uv_range_epsilon=source.settings.uv.range_epsilon,
        compensate_depth_setup_y=False,
    )
    segment_name = rig.profile.segment_slot(source.prefix, 0)
    projection = project_depth_camera_attachment(
        snapshot,
        rig,
        A1AttachmentProjectionSettings(
            slot_name=segment_name,
            attachment_name=segment_name,
            vertex_prefix=segment_name,
            image_path=assembly_settings.image_path,
            uv_layer_name=assembly_settings.uv_layer_name,
            attachment_width=assembly_settings.attachment_width,
            attachment_height=assembly_settings.attachment_height,
            center_x=assembly_settings.center_x,
            center_y=assembly_settings.center_y,
            z_bindings=source.z_groups.projection_bindings(snapshot),
            sequence=assembly_settings.sequence,
            skin_name=assembly_settings.skin_name,
        ),
    )

    document_build: LegacyMeshDocumentBuildResult = build_legacy_mesh_document(
        rig,
        (projection.request,),
        skeleton_metadata=build_skeleton_metadata(source.settings),
    )
    document_build = optimize_shared_vertex_bones(document_build)
    validate_document_material_correspondence((projection,), document_build)

    document = apply_rig_visual_options(
        document_build.document,
        prefix=source.prefix,
        rig_profile=rig.profile.profile_id,
        include_control_icons=assembly_settings.include_control_icons,
        include_preview_animation=assembly_settings.include_preview_animation,
    )
    document = apply_attachment_sequence_animations(
        document,
        frame_delay=assembly_settings.sequence_timing.frame_duration,
        legacy_per_frame=True,
    )
    document_build = replace(document_build, document=document)
    return A1DocumentAssemblyResult(
        settings=assembly_settings,
        rig=rig,
        z_groups=source.z_groups,
        projections=(projection,),
        document_build=document_build,
    )


def assemble_and_finalize_a1_depth_document(
    texture: A1TexturePlanningResult,
    rig: LegacyRigBuildResult,
) -> A1DocumentAssemblyResult:
    """Build one depth attachment and apply the selected Spine target exactly once."""

    canonical = _build_depth_document(texture, rig)
    source = texture.uv.source
    finalized = finalize_a1_document_assembly_for_target(
        canonical,
        spine_target=source.settings.export.spine_target,
        prefix=source.prefix,
    )
    if len(finalized.document_build.components) != 1:
        raise ValueError(
            "Depth Camera Projection target finalization must retain one mesh component; "
            f"got {len(finalized.document_build.components)}"
        )
    if len(finalized.projections) != 1:
        raise ValueError(
            "Depth Camera Projection target finalization must retain one projection"
        )
    return finalized


__all__ = ["assemble_and_finalize_a1_depth_document"]
