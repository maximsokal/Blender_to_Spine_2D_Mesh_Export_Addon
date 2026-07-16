"""Build one full-frame Spine mesh for a camera-rendered B4 texture."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from ..domain.baking import CameraProjectionPlan
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
from .a1_attachment_projection import (
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


def build_camera_projection_quad_snapshot(
    plan: CameraProjectionPlan,
    rig: LegacyRigBuildResult,
    *,
    uv_layer_name: str,
) -> MeshSnapshot:
    """Create a triangulated full-frame quad in legacy rig coordinate units."""

    if not isinstance(plan, CameraProjectionPlan):
        raise TypeError("plan must be CameraProjectionPlan")
    if not isinstance(rig, LegacyRigBuildResult):
        raise TypeError("rig must be LegacyRigBuildResult")
    if not isinstance(uv_layer_name, str) or not uv_layer_name.strip():
        raise ValueError("uv_layer_name must be a non-empty string")
    rig.validate()
    scale = float(rig.info.uniform_scale)
    if scale <= 0.0:
        raise A1DocumentAssemblyError("camera projection rig scale must be positive")

    half_x = float(plan.settings.width) / (2.0 * scale)
    half_y = float(plan.settings.height) / (2.0 * scale)
    # Synthetic topology still belongs to the exported source object.  Snapshot identity
    # distinguishes it from the source mesh; Source*Id.object_id must remain equal to
    # MeshSnapshot.source_object_id for the global lineage invariant.
    lineage_id = plan.source_object_id
    positions = (
        (-half_x, half_y, 0.0),
        (half_x, half_y, 0.0),
        (half_x, -half_y, 0.0),
        (-half_x, -half_y, 0.0),
    )
    uvs = (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    )
    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(lineage_id, index),
            position=position,
            normal=(0.0, 0.0, 1.0),
        )
        for index, position in enumerate(positions)
    )
    edge_pairs = ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2))
    edges = tuple(
        MeshEdge(
            id=EdgeId(index),
            source_id=SourceEdgeId(lineage_id, index),
            vertex_ids=(VertexId(first), VertexId(second)),
        )
        for index, (first, second) in enumerate(edge_pairs)
    )
    face_vertices = ((0, 1, 2), (0, 2, 3))
    face_edges = ((0, 1, 4), (4, 2, 3))
    loops = []
    faces = []
    loop_index = 0
    for face_index, (vertex_indices, edge_indices) in enumerate(
        zip(face_vertices, face_edges)
    ):
        face_loop_ids = []
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
        snapshot_id=f"{plan.source_object_id}:camera-projection-quad",
        source_object_id=plan.source_object_id,
        object_name=f"{plan.source_object_id}_CameraProjectionQuad",
        vertices=vertices,
        edges=edges,
        loops=tuple(loops),
        faces=tuple(faces),
        uv_layer_names=(uv_layer_name,),
        active_uv_layer=uv_layer_name,
        world_matrix=IDENTITY_MATRIX_4X4,
    )
    MeshSnapshotValidator().validate_or_raise(snapshot)
    return snapshot


def assemble_a1_camera_projection_document(
    rig: LegacyRigBuildResult,
    z_groups: A1ZGroupAssignmentPlan,
    plan: CameraProjectionPlan,
    settings: A1DocumentAssemblySettings,
    *,
    skeleton_metadata: Mapping[str, object] | None = None,
) -> A1DocumentAssemblyResult:
    """Compose one full-frame quad attachment for a camera-rendered texture."""

    if not isinstance(rig, LegacyRigBuildResult):
        raise TypeError("rig must be LegacyRigBuildResult")
    if not isinstance(z_groups, A1ZGroupAssignmentPlan):
        raise TypeError("z_groups must be A1ZGroupAssignmentPlan")
    if not isinstance(plan, CameraProjectionPlan):
        raise TypeError("plan must be CameraProjectionPlan")
    if not isinstance(settings, A1DocumentAssemblySettings):
        raise TypeError("settings must be A1DocumentAssemblySettings")
    if settings.prefix.strip() != rig.request.prefix.strip():
        raise A1DocumentAssemblyError("camera projection prefix does not match rig")

    snapshot = build_camera_projection_quad_snapshot(
        plan,
        rig,
        uv_layer_name=settings.uv_layer_name,
    )
    if not rig.info.z_groups:
        raise A1DocumentAssemblyError("camera projection rig has no Z groups")
    target_z_group = rig.info.z_groups[0].index
    segment_name = rig.profile.segment_slot(settings.prefix, settings.segment_index_base)
    projection = project_triangulated_disk_attachment(
        snapshot,
        rig,
        A1AttachmentProjectionSettings(
            slot_name=segment_name,
            attachment_name=segment_name,
            vertex_prefix=segment_name,
            image_path=settings.image_path,
            uv_layer_name=settings.uv_layer_name,
            attachment_width=settings.attachment_width,
            attachment_height=settings.attachment_height,
            center_x=0.0,
            center_y=0.0,
            z_bindings=tuple(
                A1VertexZBinding(
                    vertex_id=vertex.id,
                    z_group_index=target_z_group,
                )
                for vertex in snapshot.vertices
            ),
            sequence=settings.sequence,
            skin_name=settings.skin_name,
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
            prefix=settings.prefix,
            include_control_icons=settings.include_control_icons,
            include_preview_animation=settings.include_preview_animation,
        )
        document_build = replace(document_build, document=document)
    except Exception as exc:
        raise A1DocumentAssemblyError(
            f"Unable to compose camera projection document for '{settings.prefix}': {exc}"
        ) from exc

    return A1DocumentAssemblyResult(
        settings=settings,
        rig=rig,
        z_groups=z_groups,
        projections=(projection,),
        document_build=document_build,
    )


__all__ = [
    "assemble_a1_camera_projection_document",
    "build_camera_projection_quad_snapshot",
]
