from __future__ import annotations

from dataclasses import replace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1AttachmentProjectionResult,
    A1AttachmentVertexKey,
)
from Blender_to_Spine2D_Mesh_Exporter.application.a1_attachment_projection_service import (
    normalize_a1_attachment_projection_hull,
)
from Blender_to_Spine2D_Mesh_Exporter.application.a1_material_correspondence import (
    A1MaterialCorrespondenceError,
    attachment_setup_positions,
    validate_document_material_correspondence,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import LoopId, VertexId
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    LegacyAttachmentVertex,
    LegacyMeshAttachmentRequest,
    LegacyRigBuildRequest,
    LegacyZGroup,
    build_legacy_mesh_document,
    build_legacy_rig,
)


def _layered_rig():
    return build_legacy_rig(
        LegacyRigBuildRequest(
            prefix="Layered",
            texture_width=100,
            texture_height=100,
            z_groups=(
                LegacyZGroup(0.0, height_real_pixels=0.0),
                LegacyZGroup(1.0, height_real_pixels=100.0),
            ),
        )
    )


def _raw_layered_projection() -> A1AttachmentProjectionResult:
    vertices = (
        LegacyAttachmentVertex(0, (0.0, 0.0), (-1.0, -1.0), 1),
        LegacyAttachmentVertex(1, (1.0, 0.0), (1.0, -1.0), 1),
        LegacyAttachmentVertex(2, (0.0, 1.0), (-1.0, 1.0), 1),
        LegacyAttachmentVertex(3, (0.5, 0.5), (0.0, 0.0), 2),
    )
    keys = tuple(
        A1AttachmentVertexKey(VertexId(index), vertex.uv)
        for index, vertex in enumerate(vertices)
    )
    triangles = (0, 1, 3, 0, 3, 2)
    request = LegacyMeshAttachmentRequest(
        slot_name="Layered_Segment_0",
        attachment_name="Layered_Segment_0",
        vertex_prefix="Layered_Segment_0",
        image_path="images/Layered_Baked",
        width=100.0,
        height=100.0,
        vertices=vertices,
        triangles=triangles,
        hull=3,
        edges=(0, 1, 1, 3, 3, 0, 3, 2, 2, 0),
    )
    return A1AttachmentProjectionResult(
        request=request,
        hull_vertex_keys=keys[:3],
        ordered_vertex_keys=keys,
        loop_to_attachment_index=tuple(
            (LoopId(loop_index), attachment_index)
            for loop_index, attachment_index in enumerate(triangles)
        ),
    )


def test_setup_positions_include_z_group_parent_translation():
    rig = _layered_rig()
    raw = _raw_layered_projection()

    assert attachment_setup_positions(raw.request.vertices, rig) == (
        (-1.0, -1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (0.0, 100.0),
    )


def test_setup_pose_hull_promotion_remaps_every_dependent_index_stream():
    rig = _layered_rig()
    raw = _raw_layered_projection()

    normalized = normalize_a1_attachment_projection_hull(raw, rig=rig)

    assert normalized.request.hull == 4
    assert normalized.ordered_vertex_ids == (
        VertexId(0),
        VertexId(1),
        VertexId(3),
        VertexId(2),
    )
    assert tuple(vertex.uv for vertex in normalized.request.vertices) == (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.5, 0.5),
        (0.0, 1.0),
    )
    assert normalized.request.triangles == (0, 1, 2, 0, 2, 3)
    assert tuple(
        attachment_index
        for _loop_id, attachment_index in normalized.loop_to_attachment_index
    ) == normalized.request.triangles
    assert attachment_setup_positions(normalized.request.vertices, rig) == (
        (-1.0, -1.0),
        (1.0, -1.0),
        (0.0, 100.0),
        (-1.0, 1.0),
    )


def test_final_document_preserves_uv_triangle_and_weighted_bone_correspondence():
    rig = _layered_rig()
    projection = normalize_a1_attachment_projection_hull(
        _raw_layered_projection(),
        rig=rig,
    )
    document_build = build_legacy_mesh_document(rig, (projection.request,))

    validate_document_material_correspondence((projection,), document_build)


def test_correspondence_validator_rejects_shifted_weighted_bone_index():
    rig = _layered_rig()
    projection = normalize_a1_attachment_projection_hull(
        _raw_layered_projection(),
        rig=rig,
    )
    document_build = build_legacy_mesh_document(rig, (projection.request,))
    component = document_build.components[0]
    corrupted_stream = list(component.attachment.vertices)
    corrupted_stream[1] = int(corrupted_stream[1]) + 1
    corrupted_component = replace(
        component,
        attachment=replace(
            component.attachment,
            vertices=tuple(corrupted_stream),
        ),
    )
    corrupted_build = replace(
        document_build,
        components=(corrupted_component,),
    )

    with pytest.raises(A1MaterialCorrespondenceError, match="references bone"):
        validate_document_material_correspondence((projection,), corrupted_build)
