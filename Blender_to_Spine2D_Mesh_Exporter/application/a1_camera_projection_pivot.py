"""Recenter one rendered Camera Projection document around Blender Object Origin."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import Mapping

from ..domain.spine import (
    LegacyRigBuildResult,
    apply_attachment_sequence_animations,
    apply_legacy_visual_options,
    build_legacy_mesh_document,
)
from .a1_document_assembly import (
    A1DocumentAssemblyError,
    A1DocumentAssemblyResult,
)
from .a1_material_correspondence import validate_document_material_correspondence


def _finite_pair(value: object, field_name: str) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{field_name} must be a two-value tuple")
    resolved = tuple(float(component) for component in value)
    if not all(isfinite(component) for component in resolved):
        raise ValueError(f"{field_name} must contain finite values")
    return resolved[0], resolved[1]


def recenter_a1_camera_projection_document(
    assembly: A1DocumentAssemblyResult,
    rig: LegacyRigBuildResult,
    main_position_pixels: tuple[float, float],
    *,
    skeleton_metadata: Mapping[str, object] | None = None,
) -> A1DocumentAssemblyResult:
    """Move the rendered attachment pivot without changing its setup screen position.

    Camera Projection initially builds its contour in absolute full-frame pixel space.
    Once Blender Object Origin has been projected through the exact render camera, the
    rig main bone owns that screen position and every generated vertex bone becomes
    relative to it. UVs, triangles, hull, edges, depth binding, and image path remain
    byte-for-byte unchanged.
    """

    if not isinstance(assembly, A1DocumentAssemblyResult):
        raise TypeError("assembly must be A1DocumentAssemblyResult")
    if not isinstance(rig, LegacyRigBuildResult):
        raise TypeError("rig must be LegacyRigBuildResult")
    main_x, main_y = _finite_pair(
        main_position_pixels,
        "main_position_pixels",
    )
    if rig.request.main_position_pixels is None:
        raise A1DocumentAssemblyError(
            "Rendered Camera Projection rig must define main_position_pixels"
        )
    rig_main_x, rig_main_y = _finite_pair(
        rig.request.main_position_pixels,
        "rig.request.main_position_pixels",
    )
    if (rig_main_x, rig_main_y) != (main_x, main_y):
        raise A1DocumentAssemblyError(
            "Rendered Camera Projection rig main position does not match projected "
            "Blender Object Origin"
        )
    if len(assembly.projections) != 1:
        raise A1DocumentAssemblyError(
            "Rendered Camera Projection must contain exactly one attachment projection"
        )
    if assembly.settings.prefix.strip() != rig.request.prefix.strip():
        raise A1DocumentAssemblyError(
            "Rendered Camera Projection assembly prefix does not match rig"
        )

    source_projection = assembly.projections[0]
    adjusted_vertices = tuple(
        replace(
            vertex,
            bone_position_pixels=(
                float(vertex.bone_position_pixels[0]) - main_x,
                float(vertex.bone_position_pixels[1]) - main_y,
            ),
        )
        for vertex in source_projection.request.vertices
    )
    adjusted_projection = replace(
        source_projection,
        request=replace(
            source_projection.request,
            vertices=adjusted_vertices,
        ),
    )

    try:
        document_build = build_legacy_mesh_document(
            rig,
            (adjusted_projection.request,),
            skeleton_metadata=skeleton_metadata,
        )
        validate_document_material_correspondence(
            (adjusted_projection,),
            document_build,
        )
        document = apply_legacy_visual_options(
            document_build.document,
            prefix=assembly.settings.prefix,
            include_control_icons=assembly.settings.include_control_icons,
            include_preview_animation=assembly.settings.include_preview_animation,
        )
        document = apply_attachment_sequence_animations(document)
        document_build = replace(document_build, document=document)
    except Exception as exc:
        raise A1DocumentAssemblyError(
            "Unable to recenter rendered Camera Projection around Blender Object "
            f"Origin for '{assembly.settings.prefix}': {exc}"
        ) from exc

    return replace(
        assembly,
        rig=rig,
        projections=(adjusted_projection,),
        document_build=document_build,
    )


__all__ = ["recenter_a1_camera_projection_document"]
