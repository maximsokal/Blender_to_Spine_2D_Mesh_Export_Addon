"""Assemble front and textured reserve Depth Camera Projection attachments."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

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
from ..domain.geometry import (
    DepthParallaxGeometryPackage,
    DepthParallaxReserveSurface,
    MeshSnapshot,
    MeshSnapshotValidator,
)
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


def _parallax_package(texture: A1TexturePlanningResult) -> DepthParallaxGeometryPackage:
    source = texture.uv.source
    package = getattr(source, "parallax_package", None)
    if not isinstance(package, DepthParallaxGeometryPackage):
        raise TypeError(
            "Depth document assembly requires DepthParallaxGeometryPackage"
        )
    return package


def _reserve_plan_by_view(
    reserve_plans: Sequence[CameraProjectionPlan],
) -> dict[str, CameraProjectionPlan]:
    if not isinstance(reserve_plans, (tuple, list)):
        raise TypeError("reserve_plans must be a sequence")
    resolved: dict[str, CameraProjectionPlan] = {}
    for plan in reserve_plans:
        if not isinstance(plan, CameraProjectionPlan):
            raise TypeError("reserve_plans must contain CameraProjectionPlan values")
        if not plan.virtual_view:
            raise ValueError("reserve_plans cannot contain the FRONT plan")
        if plan.view_id in resolved:
            raise ValueError(f"duplicate reserve plan view_id {plan.view_id!r}")
        resolved[plan.view_id] = plan
    return resolved


def _project_depth_surface(
    snapshot: MeshSnapshot,
    plan: CameraProjectionPlan,
    *,
    slot_name: str,
    texture: A1TexturePlanningResult,
    rig: LegacyRigBuildResult,
    assembly_settings: A1DocumentAssemblySettings,
):
    source = texture.uv.source
    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    MeshSnapshotValidator().validate_or_raise(snapshot)
    return project_depth_camera_attachment(
        snapshot,
        rig,
        A1AttachmentProjectionSettings(
            slot_name=slot_name,
            attachment_name=slot_name,
            vertex_prefix=slot_name,
            image_path=build_a1_attachment_path(plan, source.output_paths),
            uv_layer_name=assembly_settings.uv_layer_name,
            attachment_width=assembly_settings.attachment_width,
            attachment_height=assembly_settings.attachment_height,
            center_x=assembly_settings.center_x,
            center_y=assembly_settings.center_y,
            z_bindings=source.z_groups.projection_bindings(snapshot),
            sequence=build_a1_attachment_sequence(plan),
            skin_name=assembly_settings.skin_name,
        ),
    )


def _reserve_slot_name(prefix: str, surface: DepthParallaxReserveSurface) -> str:
    return f"{prefix}_Parallax_{surface.view.view_id.value}"


def _build_depth_document(
    texture: A1TexturePlanningResult,
    rig: LegacyRigBuildResult,
    reserve_plans: Sequence[CameraProjectionPlan] = (),
) -> A1DocumentAssemblyResult:
    """Build canonical front plus optional reserve attachments before adaptation."""

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
    if texture.bake_plan.virtual_view:
        raise ValueError("texture.bake_plan must be the FRONT camera plan")

    package = _parallax_package(texture)
    plans_by_view = _reserve_plan_by_view(reserve_plans)
    package_view_ids = tuple(
        surface.view.view_id.value for surface in package.reserve_surfaces
    )
    if tuple(sorted(plans_by_view)) != tuple(sorted(package_view_ids)):
        raise ValueError(
            "reserve plan views must match parallax package surfaces exactly; "
            f"plans={tuple(sorted(plans_by_view))}, "
            f"surfaces={tuple(sorted(package_view_ids))}"
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

    # Spine slot order is bottom-to-top. Reserve layers are deliberately emitted first;
    # the established front Segment_0 is emitted last and remains visually authoritative.
    projections = []
    for surface in package.reserve_surfaces:
        projections.append(
            _project_depth_surface(
                surface.snapshot,
                plans_by_view[surface.view.view_id.value],
                slot_name=_reserve_slot_name(source.prefix, surface),
                texture=texture,
                rig=rig,
                assembly_settings=assembly_settings,
            )
        )
    front_name = rig.profile.segment_slot(source.prefix, 0)
    projections.append(
        _project_depth_surface(
            package.front_snapshot,
            texture.bake_plan,
            slot_name=front_name,
            texture=texture,
            rig=rig,
            assembly_settings=assembly_settings,
        )
    )
    projection_tuple = tuple(projections)

    document_build: LegacyMeshDocumentBuildResult = build_legacy_mesh_document(
        rig,
        tuple(projection.request for projection in projection_tuple),
        skeleton_metadata=build_skeleton_metadata(source.settings),
    )
    document_build = optimize_shared_vertex_bones(document_build)
    validate_document_material_correspondence(projection_tuple, document_build)

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
        projections=projection_tuple,
        document_build=document_build,
    )


def assemble_and_finalize_a1_depth_document(
    texture: A1TexturePlanningResult,
    rig: LegacyRigBuildResult,
    reserve_plans: Sequence[CameraProjectionPlan] = (),
) -> A1DocumentAssemblyResult:
    """Build all depth views and apply the selected Spine target exactly once."""

    canonical = _build_depth_document(texture, rig, reserve_plans)
    source = texture.uv.source
    finalized = finalize_a1_document_assembly_for_target(
        canonical,
        spine_target=source.settings.export.spine_target,
        prefix=source.prefix,
    )
    expected = 1 + len(tuple(reserve_plans))
    if len(finalized.document_build.components) != expected:
        raise ValueError(
            "Depth Camera Projection target finalization changed component count; "
            f"expected={expected}, got={len(finalized.document_build.components)}"
        )
    if len(finalized.projections) != expected:
        raise ValueError(
            "Depth Camera Projection target finalization changed projection count; "
            f"expected={expected}, got={len(finalized.projections)}"
        )
    return finalized


__all__ = ["assemble_and_finalize_a1_depth_document"]
