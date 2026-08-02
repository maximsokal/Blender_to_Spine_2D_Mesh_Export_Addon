"""Pure setup-pose contract for one rigid Active Camera depth parent."""

from __future__ import annotations

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application.a1_attachment_projection import (
    A1AttachmentProjectionResult,
    A1AttachmentVertexKey,
)
from Blender_to_Spine2D_Mesh_Exporter.application.a1_document_assembly import (
    A1DocumentAssemblyError,
    _compensate_projection_depth_setup_y,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import LoopId, VertexId
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    A1RigProfile,
    A1RigSetupPoseMode,
    LegacyAttachmentVertex,
    LegacyMeshAttachmentRequest,
    LegacyRigBuildRequest,
    LegacyZGroup,
    LegacyZGroupOriginMode,
    SpineJsonTarget,
    build_rig,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_rig_contracts import (
    LegacyRigBuildResult,
)


def _rig(
    setup_pose_mode: A1RigSetupPoseMode = A1RigSetupPoseMode.PREPROJECTED_SCREEN,
) -> LegacyRigBuildResult:
    """Build one real two-axis rig with one camera-space Object Origin depth."""

    return build_rig(
        LegacyRigBuildRequest(
            prefix="CameraObject",
            texture_width=100,
            texture_height=100,
            z_groups=(LegacyZGroup(z_value=-5.0),),
            main_position_pixels=(17.0, -9.0),
            setup_pose_mode=setup_pose_mode,
            z_group_origin_mode=LegacyZGroupOriginMode.OBJECT_ORIGIN,
        ),
        A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        spine_target=SpineJsonTarget.SPINE_4_2,
    )


def _group_index(rig: LegacyRigBuildResult) -> int:
    if not isinstance(rig, LegacyRigBuildResult):
        raise TypeError("rig must be LegacyRigBuildResult")
    if len(rig.info.z_groups) != 1:
        raise AssertionError("fixture must contain exactly one rigid depth group")
    return rig.info.z_groups[0].index


def _projection(z_group_index: int) -> A1AttachmentProjectionResult:
    keys = tuple(
        A1AttachmentVertexKey(
            vertex_id=VertexId(index),
            uv=((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))[index],
        )
        for index in range(3)
    )
    screen_y = (12.0, -8.0, 5.0)
    request = LegacyMeshAttachmentRequest(
        slot_name="CameraObject_Segment_0",
        attachment_name="CameraObject_Segment_0",
        vertex_prefix="CameraObject_Segment_0",
        image_path="images/camera.png",
        width=100.0,
        height=100.0,
        vertices=tuple(
            LegacyAttachmentVertex(
                index=index,
                uv=keys[index].uv,
                bone_position_pixels=(float(index * 10), screen_y[index]),
                z_group_index=z_group_index,
            )
            for index in range(3)
        ),
        triangles=(0, 1, 2),
        hull=3,
    )
    return A1AttachmentProjectionResult(
        request=request,
        hull_vertex_keys=keys,
        ordered_vertex_keys=keys,
        loop_to_attachment_index=tuple(
            (LoopId(index), index) for index in range(3)
        ),
    )


def test_one_parent_plus_compensated_vertex_y_preserves_screen_y() -> None:
    rig = _rig()
    source = _projection(_group_index(rig))
    source_positions = tuple(
        vertex.bone_position_pixels for vertex in source.request.vertices
    )

    result = _compensate_projection_depth_setup_y(source, rig)

    assert result is not source
    assert tuple(
        vertex.bone_position_pixels for vertex in source.request.vertices
    ) == source_positions

    parent_y = rig.info.z_groups[0].y_offset_pixels
    assert {vertex.z_group_index for vertex in result.request.vertices} == {
        _group_index(rig)
    }
    for original, adjusted in zip(
        source.request.vertices,
        result.request.vertices,
        strict=True,
    ):
        assert parent_y + adjusted.bone_position_pixels[1] == (
            original.bone_position_pixels[1]
        )
        assert adjusted.bone_position_pixels[0] == original.bone_position_pixels[0]
        assert adjusted.uv == original.uv


def test_non_screen_setup_rig_fails_before_partial_compensation() -> None:
    rig = _rig(A1RigSetupPoseMode.PRESERVE_COMPOSITION)

    with pytest.raises(
        A1DocumentAssemblyError,
        match="requires PREPROJECTED_SCREEN setup",
    ):
        _compensate_projection_depth_setup_y(
            _projection(_group_index(rig)),
            rig,
        )


def test_unknown_depth_group_fails_closed() -> None:
    rig = _rig()
    unknown_index = _group_index(rig) + 97

    with pytest.raises(
        A1DocumentAssemblyError,
        match=rf"unknown depth group {unknown_index}",
    ):
        _compensate_projection_depth_setup_y(
            _projection(unknown_index),
            rig,
        )
