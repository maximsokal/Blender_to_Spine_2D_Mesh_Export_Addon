"""Rebuild prepared B4 Spine attachments after render-derived layout analysis."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import Any

from ..application import (
    A1ZGroupAssignmentPlan,
    assemble_a1_camera_projection_document,
    recenter_a1_camera_projection_document,
)
from ..domain.baking import CameraProjectionPlan, resolve_projection_output_policy
from ..domain.baking.projection_layout import CameraProjectionLayout
from ..domain.camera_projection import A1CameraProjectionKind
from ..domain.spine import LegacyRigBuildResult, build_legacy_rig
from ..domain.spine.preprojected_setup import ensure_preprojected_screen_rig
from ..domain.spine.rig_profiles import (
    A1CameraLayerProjectionKind,
    A1RigProfile,
    resolve_a1_rig_profile,
)
from .a1_document_preparation import finalize_a1_document_assembly_for_target
from .a1_preparation_contracts import (
    A1BlenderFinalizationContext,
    PreparedA1Object,
    build_skeleton_metadata,
    freeze_statistics,
)
from .active_camera_projection import resolve_a1_active_camera_projection_frame
from .scene_context_contract import (
    BlenderSceneContextError,
    require_depsgraph_scene_consistency,
)


def _resolved_scene_and_depsgraph(
    context: Any | None,
    scene: Any | None,
) -> tuple[Any, Any]:
    """Resolve the exact Scene/depsgraph pair used for final pivot projection."""

    try:
        import bpy
    except Exception as exc:
        raise ValueError(
            "Blender bpy module is unavailable for Camera Projection finalization"
        ) from exc

    resolved_context = context or getattr(bpy, "context", None)
    resolved_scene = scene or getattr(resolved_context, "scene", None)
    if resolved_scene is None:
        raise ValueError("Camera Projection finalization requires a Blender Scene")

    evaluated_depsgraph_get = getattr(
        resolved_context,
        "evaluated_depsgraph_get",
        None,
    )
    if not callable(evaluated_depsgraph_get):
        raise ValueError(
            "Camera Projection finalization requires evaluated_depsgraph_get()"
        )
    try:
        depsgraph = evaluated_depsgraph_get()
    except Exception as exc:
        raise ValueError(
            "Unable to acquire evaluated dependency graph for Camera Projection pivot"
        ) from exc
    if depsgraph is None:
        raise ValueError(
            "Blender returned no dependency graph for Camera Projection pivot"
        )
    try:
        require_depsgraph_scene_consistency(depsgraph, resolved_scene)
    except BlenderSceneContextError as exc:
        raise ValueError(
            f"Camera Projection Scene and dependency graph disagree: {exc}"
        ) from exc
    return resolved_scene, depsgraph


def _evaluated_object_origin(
    source_object: Any,
    scene: Any,
    depsgraph: Any,
) -> tuple[float, float, float]:
    """Read Blender Object Origin from the same evaluated Scene used by the camera."""

    if source_object is None:
        raise ValueError("source_object cannot be None")
    try:
        scene_objects = tuple(scene.objects)
    except Exception as exc:
        raise ValueError(
            "Unable to inspect Scene objects for Camera Projection pivot"
        ) from exc
    if source_object not in scene_objects:
        raise ValueError(
            "Camera Projection source object is not linked to the finalization Scene"
        )

    evaluated_get = getattr(source_object, "evaluated_get", None)
    if not callable(evaluated_get):
        raise ValueError("Camera Projection source object has no evaluated_get()")
    try:
        evaluated_source = evaluated_get(depsgraph)
    except Exception as exc:
        raise ValueError(
            "Unable to evaluate Camera Projection source object"
        ) from exc
    if evaluated_source is None:
        raise ValueError("Camera Projection source evaluation returned None")

    matrix_world = getattr(evaluated_source, "matrix_world", None)
    if matrix_world is None:
        raise ValueError(
            "Evaluated Camera Projection source has no matrix_world"
        )
    try:
        origin = tuple(float(matrix_world[index][3]) for index in range(3))
    except Exception as exc:
        raise ValueError(
            "Unable to read evaluated Camera Projection Object Origin"
        ) from exc
    if len(origin) != 3 or not all(isfinite(value) for value in origin):
        raise ValueError(
            "Evaluated Camera Projection Object Origin must contain finite XYZ"
        )
    return origin[0], origin[1], origin[2]


def _camera_layer_kind(
    value: A1CameraProjectionKind,
) -> A1CameraLayerProjectionKind:
    """Map projection-domain camera kind to rig-domain layer semantics."""

    if not isinstance(value, A1CameraProjectionKind):
        raise TypeError("value must be A1CameraProjectionKind")
    if value is A1CameraProjectionKind.PERSPECTIVE:
        return A1CameraLayerProjectionKind.PERSPECTIVE
    if value is A1CameraProjectionKind.ORTHOGRAPHIC:
        return A1CameraLayerProjectionKind.ORTHOGRAPHIC
    raise AssertionError(f"Unhandled camera projection kind: {value}")


def _rendered_camera_main_position(
    prepared: PreparedA1Object,
    plan: CameraProjectionPlan,
    *,
    context: Any | None,
    scene: Any | None,
) -> tuple[tuple[float, float], float, A1CameraLayerProjectionKind]:
    """Project Object Origin and capture exact Perspective/Orthographic semantics."""

    resolved_scene, depsgraph = _resolved_scene_and_depsgraph(context, scene)
    frame = resolve_a1_active_camera_projection_frame(
        resolved_scene,
        texture_width=plan.settings.width,
        texture_height=plan.settings.height,
        depsgraph=depsgraph,
    )
    origin = _evaluated_object_origin(
        prepared.source_object,
        resolved_scene,
        depsgraph,
    )
    projected = frame.project_world_point(
        origin,
        field_name="rendered_camera_projection_object_origin",
    )

    # Render contour pixels use image-row Y and are negated by the established
    # attachment projector. Match that final Spine convention for the base placement.
    return (
        (float(projected.u), -float(projected.v)),
        float(projected.depth),
        _camera_layer_kind(frame.kind),
    )


def _positioned_projection_rig(
    rig: LegacyRigBuildResult,
    main_position: tuple[float, float],
    camera_depth: float,
    camera_projection_kind: A1CameraLayerProjectionKind,
) -> LegacyRigBuildResult:
    """Rebuild the selected profile around camera zero and Blender Object Origin."""

    if not isinstance(rig, LegacyRigBuildResult):
        raise TypeError("rig must be LegacyRigBuildResult")
    if not isinstance(camera_depth, (int, float)) or isinstance(camera_depth, bool):
        raise TypeError("camera_depth must be a finite number")
    if not isfinite(float(camera_depth)):
        raise ValueError("camera_depth must be finite")
    if not isinstance(camera_projection_kind, A1CameraLayerProjectionKind):
        raise TypeError(
            "camera_projection_kind must be A1CameraLayerProjectionKind"
        )

    resolved_profile = resolve_a1_rig_profile(rig.profile.profile_id)

    if resolved_profile is A1RigProfile.TWO_AXIS_ROTATION_SCALE:
        positioned = ensure_preprojected_screen_rig(
            rig,
            main_position_pixels=main_position,
            camera_depth=float(camera_depth),
            camera_projection_kind=camera_projection_kind,
        )
    elif resolved_profile is A1RigProfile.THREE_AXIS_ROTATION:
        # Preserve the historical three-axis payload. Camera-relative rigid layering is
        # a two-axis contract and must not silently rewrite Legacy 3-Axis semantics.
        positioned = build_legacy_rig(
            replace(
                rig.request,
                main_position_pixels=main_position,
            )
        )
    else:
        raise AssertionError(f"Unhandled rig profile: {resolved_profile}")

    positioned.validate()
    return positioned


def _camera_relative_z_groups(
    source: A1ZGroupAssignmentPlan,
    rig: LegacyRigBuildResult,
) -> A1ZGroupAssignmentPlan:
    """Bind every source vertex to the one rigid Object-Origin depth layer."""

    if not isinstance(source, A1ZGroupAssignmentPlan):
        raise TypeError("source must be A1ZGroupAssignmentPlan")
    if not isinstance(rig, LegacyRigBuildResult):
        raise TypeError("rig must be LegacyRigBuildResult")
    if len(rig.request.z_groups) != 1 or len(rig.info.z_groups) != 1:
        raise ValueError(
            "camera-relative rendered projection requires exactly one rig depth group"
        )

    target_index = rig.info.z_groups[0].index
    return replace(
        source,
        z_index_base=target_index,
        groups=rig.request.z_groups,
        source_bindings=tuple(
            replace(binding, z_group_index=target_index)
            for binding in source.source_bindings
        ),
    )


def finalize_prepared_camera_projection(
    prepared: PreparedA1Object,
    layout: CameraProjectionLayout | None,
) -> PreparedA1Object:
    """Return a prepared object whose document matches render, crop, and camera origin.

    The historical two-argument call contract is preserved. Blender runtime references
    captured by ``prepare_a1_object`` travel inside ``PreparedA1Object`` so fault-matrix,
    rollback, and monkeypatch contracts remain stable while real exports still use the
    caller-selected Scene and dependency graph.
    """

    if not isinstance(prepared, PreparedA1Object):
        raise TypeError("prepared must be PreparedA1Object")
    if not isinstance(prepared.finalization_context, A1BlenderFinalizationContext):
        raise TypeError(
            "prepared.finalization_context must be A1BlenderFinalizationContext"
        )

    camera_projection = isinstance(prepared.bake_plan, CameraProjectionPlan)
    if not camera_projection:
        if layout is not None:
            raise ValueError("object-bake preparation cannot accept a projection layout")
        return prepared
    if not isinstance(layout, CameraProjectionLayout):
        raise TypeError("camera projection preparation requires CameraProjectionLayout")

    plan = prepared.bake_plan
    assert isinstance(plan, CameraProjectionPlan)
    output_policy = resolve_projection_output_policy(
        prepared.settings.bake_execution.projection_output_policy,
        plan.settings.texture_format,
    )
    (
        main_position,
        projected_depth,
        camera_projection_kind,
    ) = _rendered_camera_main_position(
        prepared,
        plan,
        context=prepared.finalization_context.context,
        scene=prepared.finalization_context.scene,
    )
    positioned_rig = _positioned_projection_rig(
        prepared.rig,
        main_position,
        projected_depth,
        camera_projection_kind,
    )
    resolved_profile = resolve_a1_rig_profile(positioned_rig.profile.profile_id)
    positioned_z_groups = (
        _camera_relative_z_groups(prepared.z_groups, positioned_rig)
        if resolved_profile is A1RigProfile.TWO_AXIS_ROTATION_SCALE
        else prepared.z_groups
    )

    skeleton_metadata = build_skeleton_metadata(prepared.settings)
    document_assembly = assemble_a1_camera_projection_document(
        positioned_rig,
        positioned_z_groups,
        plan,
        prepared.document_assembly.settings,
        layout=layout,
        skeleton_metadata=skeleton_metadata,
    )
    document_assembly = recenter_a1_camera_projection_document(
        document_assembly,
        positioned_rig,
        main_position,
        skeleton_metadata=skeleton_metadata,
    )
    document_assembly = finalize_a1_document_assembly_for_target(
        document_assembly,
        spine_target=prepared.settings.export.spine_target,
        prefix=prepared.prefix,
    )

    document = document_assembly.document
    offset_x, offset_y = layout.offset_pixels
    statistics = freeze_statistics(
        prepared.statistics,
        {
            "projection_full_width": layout.full_width,
            "projection_full_height": layout.full_height,
            "projection_crop_min_x": layout.crop.minimum_x,
            "projection_crop_min_y": layout.crop.minimum_y,
            "projection_crop_max_x": layout.crop.maximum_x,
            "projection_crop_max_y": layout.crop.maximum_y,
            "projection_crop_width": layout.cropped_width,
            "projection_crop_height": layout.cropped_height,
            "projection_offset_x": offset_x,
            "projection_offset_y": offset_y,
            "projection_hull_vertex_count": len(layout.hull),
            "projection_contour_vertex_count": len(layout.hull),
            "projection_source_contour_vertex_count": layout.source_contour_vertex_count,
            "projection_contour_mode": layout.contour_mode.value,
            "projection_contour_concave": int(layout.concave),
            "projection_outer_component_count": layout.outer_component_count,
            "projection_contour_fallback_reason": layout.contour_fallback_reason or "",
            "projection_contour_simplify_tolerance_pixels": layout.simplify_tolerance_pixels,
            "projection_union_visible_pixels": layout.visible_pixel_count,
            "projection_alpha_threshold": layout.alpha_threshold,
            "projection_padding_pixels": layout.padding_pixels,
            "projection_coverage_mode": layout.coverage_mode.value,
            "projection_coverage_core_alpha_threshold": layout.coverage_core_alpha_threshold,
            "projection_coverage_raw_nonzero_pixels": layout.coverage_raw_nonzero_pixel_count,
            "projection_coverage_strong_pixels": layout.coverage_strong_pixel_count,
            "projection_coverage_components_before_cleanup": layout.coverage_component_count_before_cleanup,
            "projection_coverage_components_after_cleanup": layout.coverage_component_count_after_cleanup,
            "projection_coverage_removed_component_pixels": layout.coverage_removed_component_pixel_count,
            "projection_coverage_filled_hole_pixels": layout.coverage_filled_hole_pixel_count,
            "projection_coverage_used_weak_only_fallback": int(
                layout.coverage_used_weak_only_fallback
            ),
            "projection_output_texture_format": output_policy.texture_format.value,
            "projection_output_dynamic_range": output_policy.dynamic_range.value,
            "projection_output_tone_mapping": output_policy.tone_mapping.value,
            "projection_output_alpha_representation": output_policy.alpha_representation.value,
            "projection_output_color_depth": output_policy.color_depth,
            "projection_output_float_buffer": int(output_policy.float_buffer),
            "projection_object_origin_main_x": main_position[0],
            "projection_object_origin_main_y": main_position[1],
            "projection_object_origin_depth": projected_depth,
            "projection_camera_layer_kind": camera_projection_kind.value,
            "projection_camera_relative_depth_group_count": len(
                positioned_rig.info.z_groups
            ),
            "projection_setup_pose_mode": positioned_rig.request.setup_pose_mode.value,
            "final_bone_count": len(document.bones),
            "slot_count": len(document.slots),
            "attachment_count": sum(
                len(attachments)
                for skin in document.skins
                for attachments in skin.attachments.values()
            ),
        },
    )
    return replace(
        prepared,
        z_groups=positioned_z_groups,
        rig=document_assembly.rig,
        document_assembly=document_assembly,
        statistics=statistics,
    )


__all__ = ["finalize_prepared_camera_projection"]
