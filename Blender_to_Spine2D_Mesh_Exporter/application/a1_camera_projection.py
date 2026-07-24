"""Build a cropped screen-space Spine mesh for a camera-rendered texture."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Tuple

from ..domain.baking import CameraProjectionPlan
from ..domain.baking.projection_layout import (
    CameraProjectionLayout,
    ProjectionTriangle,
    build_full_frame_layout,
)
from ..domain.geometry import (
    IDENTITY_MATRIX_4X4,
    EdgeId,
    FaceId,
    LoopId,
    LoopUV,
    MeshEdge,
    MeshFace,
    MeshLoop,
    MeshSnapshot,
    MeshSnapshotValidator,
    MeshVertex,
    SourceEdgeId,
    SourceFaceId,
    SourceLoopId,
    SourceVertexId,
    VertexId,
)
from ..domain.spine import (
    LegacyRigBuildResult,
    apply_legacy_visual_options,
    build_legacy_mesh_document,
)
from ..domain.uv import UvRangePolicy, enforce_uv_range
from .a1_attachment_projection_service import (
    A1AttachmentProjectionSettings,
    A1VertexZBinding,
    project_triangulated_disk_attachment,
)
from .a1_document_assembly import (
    A1DocumentAssemblyError,
    A1DocumentAssemblyResult,
    A1DocumentAssemblySettings,
)
from .a1_z_groups import A1ZGroupAssignmentPlan


def _resolved_layout(
    plan: CameraProjectionPlan,
    layout: CameraProjectionLayout | None,
) -> CameraProjectionLayout:
    if layout is None:
        return build_full_frame_layout(
            plan.settings.width,
            plan.settings.height,
            frame_count=len(plan.frame_tasks),
        )
    if not isinstance(layout, CameraProjectionLayout):
        raise TypeError("layout must be CameraProjectionLayout or None")
    if (
        layout.full_width != plan.settings.width
        or layout.full_height != plan.settings.height
    ):
        raise A1DocumentAssemblyError(
            "camera projection layout full dimensions do not match the render plan"
        )
    if layout.frame_count != len(plan.frame_tasks):
        raise A1DocumentAssemblyError(
            "camera projection layout frame count does not match the render plan"
        )
    return layout


def _normalized_pair(first: int, second: int) -> tuple[int, int]:
    return (first, second) if first < second else (second, first)


def _edge_pairs(
    vertex_count: int,
    triangle_indices: Tuple[ProjectionTriangle, ...],
) -> tuple[tuple[int, int], ...]:
    """Return boundary edges first, then deterministic internal fan diagonals."""

    if vertex_count < 3:
        raise A1DocumentAssemblyError(
            "camera projection hull requires at least 3 vertices"
        )
    if len(triangle_indices) != vertex_count - 2:
        raise A1DocumentAssemblyError(
            "camera projection triangle count does not match convex hull vertex count"
        )

    boundary = tuple(
        (index, (index + 1) % vertex_count) for index in range(vertex_count)
    )
    resolved: list[tuple[int, int]] = list(boundary)
    seen = {_normalized_pair(first, second) for first, second in boundary}

    for face_index, triangle in enumerate(triangle_indices):
        if not isinstance(triangle, tuple) or len(triangle) != 3:
            raise A1DocumentAssemblyError(
                f"camera projection triangle {face_index} must contain three indices"
            )
        if any(
            not isinstance(index, int) or index < 0 or index >= vertex_count
            for index in triangle
        ):
            raise A1DocumentAssemblyError(
                f"camera projection triangle {face_index} references an invalid vertex"
            )
        if len(set(triangle)) != 3:
            raise A1DocumentAssemblyError(
                f"camera projection triangle {face_index} is degenerate"
            )
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            pair = _normalized_pair(first, second)
            if pair in seen:
                continue
            seen.add(pair)
            resolved.append(pair)

    expected_edge_count = vertex_count + max(0, vertex_count - 3)
    if len(resolved) != expected_edge_count:
        raise A1DocumentAssemblyError(
            "camera projection fan produced an unexpected edge count; "
            f"expected={expected_edge_count}, actual={len(resolved)}"
        )
    return tuple(resolved)


def build_camera_projection_mesh_snapshot(
    plan: CameraProjectionPlan,
    rig: LegacyRigBuildResult,
    *,
    uv_layer_name: str,
    layout: CameraProjectionLayout | None = None,
) -> MeshSnapshot:
    """Create a triangulated convex screen-space hull in the current rig units."""

    if not isinstance(plan, CameraProjectionPlan):
        raise TypeError("plan must be CameraProjectionPlan")
    if not isinstance(rig, LegacyRigBuildResult):
        raise TypeError("rig must be LegacyRigBuildResult")
    if not isinstance(uv_layer_name, str) or not uv_layer_name.strip():
        raise ValueError("uv_layer_name must be a non-empty string")
    rig.validate()
    scale = float(rig.info.uniform_scale)
    if scale <= 0.0:
        raise A1DocumentAssemblyError(
            "camera projection rig scale must be positive"
        )

    resolved_layout = _resolved_layout(plan, layout)
    points = resolved_layout.hull
    triangle_indices = resolved_layout.triangle_indices
    lineage_id = plan.source_object_id
    screen_positions = tuple(
        resolved_layout.spine_position_pixels(point) for point in points
    )
    positions = tuple(
        (position_x / scale, position_y / scale, 0.0)
        for position_x, position_y in screen_positions
    )
    uvs = tuple(resolved_layout.spine_uv(point) for point in points)
    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(lineage_id, index),
            position=position,
            normal=(0.0, 0.0, 1.0),
        )
        for index, position in enumerate(positions)
    )

    edge_pairs = _edge_pairs(len(points), triangle_indices)
    edges = tuple(
        MeshEdge(
            id=EdgeId(index),
            source_id=SourceEdgeId(lineage_id, index),
            vertex_ids=(VertexId(first), VertexId(second)),
        )
        for index, (first, second) in enumerate(edge_pairs)
    )
    edge_index_by_pair = {
        _normalized_pair(first, second): index
        for index, (first, second) in enumerate(edge_pairs)
    }

    loops: list[MeshLoop] = []
    faces: list[MeshFace] = []
    loop_index = 0
    for face_index, vertex_indices in enumerate(triangle_indices):
        triangle_pairs = (
            (vertex_indices[0], vertex_indices[1]),
            (vertex_indices[1], vertex_indices[2]),
            (vertex_indices[2], vertex_indices[0]),
        )
        edge_indices = tuple(
            edge_index_by_pair[_normalized_pair(first, second)]
            for first, second in triangle_pairs
        )
        face_loop_ids: list[LoopId] = []
        for corner_index, (vertex_index, edge_index) in enumerate(
            zip(vertex_indices, edge_indices)
        ):
            loop_id = LoopId(loop_index)
            face_loop_ids.append(loop_id)
            loops.append(
                MeshLoop(
                    id=loop_id,
                    source_id=SourceLoopId(lineage_id, face_index, corner_index),
                    vertex_id=VertexId(vertex_index),
                    edge_id=EdgeId(edge_index),
                    uvs=(
                        LoopUV(
                            layer_name=uv_layer_name,
                            coordinate=uvs[vertex_index],
                        ),
                    ),
                )
            )
            loop_index += 1
        faces.append(
            MeshFace(
                id=FaceId(face_index),
                source_id=SourceFaceId(lineage_id, face_index),
                loop_ids=tuple(face_loop_ids),
                material_index=0,
                normal=(0.0, 0.0, 1.0),
            )
        )

    snapshot = MeshSnapshot(
        snapshot_id=f"{plan.source_object_id}:camera-projection-hull",
        source_object_id=plan.source_object_id,
        object_name=f"{plan.source_object_id}_CameraProjectionHull",
        vertices=vertices,
        edges=edges,
        loops=tuple(loops),
        faces=tuple(faces),
        uv_layer_names=(uv_layer_name,),
        active_uv_layer=uv_layer_name,
        world_matrix=IDENTITY_MATRIX_4X4,
    )
    MeshSnapshotValidator().validate_or_raise(snapshot)
    # Camera projection UVs are generated by exporter-owned crop math. They are
    # therefore an internal invariant and remain strict even when authored unwrap
    # UVs use WARN_ONLY policy.
    enforce_uv_range(
        snapshot,
        uv_layer_name,
        policy=UvRangePolicy.REQUIRE_UNIT_SQUARE,
        epsilon=0.0,
    )
    return snapshot


def build_camera_projection_quad_snapshot(
    plan: CameraProjectionPlan,
    rig: LegacyRigBuildResult,
    *,
    uv_layer_name: str,
) -> MeshSnapshot:
    """Compatibility wrapper returning the original full-frame four-vertex mesh."""

    return build_camera_projection_mesh_snapshot(
        plan,
        rig,
        uv_layer_name=uv_layer_name,
        layout=None,
    )


def assemble_a1_camera_projection_document(
    rig: LegacyRigBuildResult,
    z_groups: A1ZGroupAssignmentPlan,
    plan: CameraProjectionPlan,
    settings: A1DocumentAssemblySettings,
    *,
    layout: CameraProjectionLayout | None = None,
    skeleton_metadata: Mapping[str, object] | None = None,
) -> A1DocumentAssemblyResult:
    """Compose one cropped convex projection attachment for a rendered texture."""

    if not isinstance(rig, LegacyRigBuildResult):
        raise TypeError("rig must be LegacyRigBuildResult")
    if not isinstance(z_groups, A1ZGroupAssignmentPlan):
        raise TypeError("z_groups must be A1ZGroupAssignmentPlan")
    if not isinstance(plan, CameraProjectionPlan):
        raise TypeError("plan must be CameraProjectionPlan")
    if not isinstance(settings, A1DocumentAssemblySettings):
        raise TypeError("settings must be A1DocumentAssemblySettings")
    if settings.prefix.strip() != rig.request.prefix.strip():
        raise A1DocumentAssemblyError(
            "camera projection prefix does not match rig"
        )

    resolved_layout = _resolved_layout(plan, layout)
    resolved_settings = replace(
        settings,
        attachment_width=float(resolved_layout.cropped_width),
        attachment_height=float(resolved_layout.cropped_height),
    )
    snapshot = build_camera_projection_mesh_snapshot(
        plan,
        rig,
        uv_layer_name=resolved_settings.uv_layer_name,
        layout=resolved_layout,
    )
    if not rig.info.z_groups:
        raise A1DocumentAssemblyError("camera projection rig has no Z groups")
    target_z_group = rig.info.z_groups[0].index
    segment_name = rig.profile.segment_slot(
        resolved_settings.prefix,
        resolved_settings.segment_index_base,
    )
    projection = project_triangulated_disk_attachment(
        snapshot,
        rig,
        A1AttachmentProjectionSettings(
            slot_name=segment_name,
            attachment_name=segment_name,
            vertex_prefix=segment_name,
            image_path=resolved_settings.image_path,
            uv_layer_name=resolved_settings.uv_layer_name,
            attachment_width=resolved_settings.attachment_width,
            attachment_height=resolved_settings.attachment_height,
            center_x=0.0,
            center_y=0.0,
            z_bindings=tuple(
                A1VertexZBinding(
                    vertex_id=vertex.id,
                    z_group_index=target_z_group,
                )
                for vertex in snapshot.vertices
            ),
            sequence=resolved_settings.sequence,
            skin_name=resolved_settings.skin_name,
        ),
    )
    try:
        document_build = build_legacy_mesh_document(
            rig,
            (projection.request,),
            skeleton_metadata=skeleton_metadata,
        )
        document = apply_legacy_visual_options(
            document_build.document,
            prefix=resolved_settings.prefix,
            include_control_icons=resolved_settings.include_control_icons,
            include_preview_animation=resolved_settings.include_preview_animation,
        )
        document_build = replace(document_build, document=document)
    except Exception as exc:
        raise A1DocumentAssemblyError(
            f"Unable to compose camera projection document for '{settings.prefix}': {exc}"
        ) from exc

    return A1DocumentAssemblyResult(
        settings=resolved_settings,
        rig=rig,
        z_groups=z_groups,
        projections=(projection,),
        document_build=document_build,
    )


__all__ = [
    "assemble_a1_camera_projection_document",
    "build_camera_projection_mesh_snapshot",
    "build_camera_projection_quad_snapshot",
]
