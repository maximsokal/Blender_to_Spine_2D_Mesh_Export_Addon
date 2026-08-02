"""Pure pivot-preservation tests for rendered Camera Projection documents."""

from __future__ import annotations

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1AttachmentProjectionResult,
    A1AttachmentVertexKey,
    A1DocumentAssemblyResult,
    A1DocumentAssemblySettings,
    A1SourceVertexZBinding,
    A1ZGroupAssignmentPlan,
    recenter_a1_camera_projection_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    LoopId,
    SourceVertexId,
    VertexId,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    A1CameraLayerProjectionKind,
    A1RigProfile,
    A1RigSetupPoseMode,
    LegacyAttachmentVertex,
    LegacyMeshAttachmentRequest,
    LegacyRigBuildRequest,
    LegacyZGroup,
    LegacyZGroupOriginMode,
    SpineJsonTarget,
    build_legacy_mesh_document,
    build_rig,
)


def _rig(main_position: tuple[float, float]):
    return build_rig(
        LegacyRigBuildRequest(
            prefix="Rendered",
            texture_width=128,
            texture_height=128,
            z_groups=(LegacyZGroup(0.0),),
            main_position_pixels=main_position,
            setup_pose_mode=A1RigSetupPoseMode.PREPROJECTED_SCREEN,
            z_group_origin_mode=LegacyZGroupOriginMode.OBJECT_ORIGIN,
            camera_layer_projection_kind=(
                A1CameraLayerProjectionKind.PERSPECTIVE
            ),
        ),
        A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        spine_target=SpineJsonTarget.SPINE_4_2,
    )


def _assembly(main_position: tuple[float, float]) -> A1DocumentAssemblyResult:
    rig = _rig(main_position)
    z_index = rig.info.z_groups[0].index
    keys = tuple(
        A1AttachmentVertexKey(
            vertex_id=VertexId(index),
            uv=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))[index],
        )
        for index in range(3)
    )
    absolute_positions = ((-35.0, 18.0), (21.0, 18.0), (-35.0, -22.0))
    request = LegacyMeshAttachmentRequest(
        slot_name="Rendered_Segment_0",
        attachment_name="Rendered_Segment_0",
        vertex_prefix="Rendered_Segment_0",
        image_path="images/Rendered_Baked",
        width=56.0,
        height=40.0,
        vertices=tuple(
            LegacyAttachmentVertex(
                index=index,
                uv=key.uv,
                bone_position_pixels=absolute_positions[index],
                z_group_index=z_index,
            )
            for index, key in enumerate(keys)
        ),
        triangles=(0, 1, 2),
        hull=3,
        edges=(0, 1, 1, 2, 2, 0),
    )
    projection = A1AttachmentProjectionResult(
        request=request,
        hull_vertex_keys=keys,
        ordered_vertex_keys=keys,
        loop_to_attachment_index=tuple(
            (LoopId(index), index) for index in range(3)
        ),
    )
    settings = A1DocumentAssemblySettings(
        prefix="Rendered",
        uv_layer_name="SpineBakeUV",
        image_path="images/Rendered_Baked",
        attachment_width=56.0,
        attachment_height=40.0,
        center_x=0.0,
        center_y=0.0,
    )
    z_groups = A1ZGroupAssignmentPlan(
        source_snapshot_id="Rendered:camera-projection",
        z_index_base=z_index,
        groups=(LegacyZGroup(0.0),),
        source_bindings=tuple(
            A1SourceVertexZBinding(
                source_vertex_id=SourceVertexId("Rendered", index),
                z_group_index=z_index,
            )
            for index in range(3)
        ),
    )
    document_build = build_legacy_mesh_document(
        rig,
        (request,),
        skeleton_metadata={"spine": "4.2.43"},
    )
    return A1DocumentAssemblyResult(
        settings=settings,
        rig=rig,
        z_groups=z_groups,
        projections=(projection,),
        document_build=document_build,
    )


def _bone_by_name(document) -> dict[str, object]:
    return {bone.name: bone for bone in document.bones}


def test_recentered_projection_preserves_absolute_setup_positions() -> None:
    main_position = (17.25, -9.5)
    source = _assembly(main_position)
    source_projection = source.projections[0]
    source_positions = tuple(
        vertex.bone_position_pixels
        for vertex in source_projection.request.vertices
    )

    result = recenter_a1_camera_projection_document(
        source,
        source.rig,
        main_position,
        skeleton_metadata={"spine": "4.2.43"},
    )

    assert result is not source
    assert tuple(
        vertex.bone_position_pixels
        for vertex in source_projection.request.vertices
    ) == source_positions

    adjusted_positions = tuple(
        vertex.bone_position_pixels
        for vertex in result.projections[0].request.vertices
    )
    assert adjusted_positions == tuple(
        (position[0] - main_position[0], position[1] - main_position[1])
        for position in source_positions
    )
    assert tuple(
        (main_position[0] + local[0], main_position[1] + local[1])
        for local in adjusted_positions
    ) == source_positions


def test_recentered_projection_preserves_mesh_and_texture_contracts() -> None:
    source = _assembly((11.0, 7.0))
    result = recenter_a1_camera_projection_document(
        source,
        source.rig,
        (11.0, 7.0),
        skeleton_metadata={"spine": "4.2.43"},
    )

    before = source.projections[0].request
    after = result.projections[0].request
    assert tuple(vertex.uv for vertex in after.vertices) == tuple(
        vertex.uv for vertex in before.vertices
    )
    assert tuple(vertex.z_group_index for vertex in after.vertices) == tuple(
        vertex.z_group_index for vertex in before.vertices
    )
    assert after.triangles == before.triangles
    assert after.edges == before.edges
    assert after.hull == before.hull
    assert after.width == before.width
    assert after.height == before.height
    assert after.image_path == before.image_path


def test_recentered_document_uses_camera_zero_and_object_base() -> None:
    main_position = (23.0, -11.0)
    source = _assembly(main_position)
    result = recenter_a1_camera_projection_document(
        source,
        source.rig,
        main_position,
        skeleton_metadata={"spine": "4.2.43"},
    )
    bones = _bone_by_name(result.document)

    main = bones[result.rig.info.main_bone_name]
    base = bones[result.rig.info.base_bone_name]
    assert (main.x, main.y) == (0.0, 0.0)
    assert (base.x, base.y) == main_position

    for index, source_vertex in enumerate(
        source.projections[0].request.vertices
    ):
        vertex_bone_name = result.rig.profile.vertex_bone(
            "Rendered_Segment_0",
            index,
        )
        vertex_bone = bones[vertex_bone_name]
        assert vertex_bone.parent == result.rig.info.z_groups[0].bone_name
        assert (
            round(float(base.x) + float(vertex_bone.x), 2),
            round(float(base.y) + float(vertex_bone.y), 2),
        ) == source_vertex.bone_position_pixels
