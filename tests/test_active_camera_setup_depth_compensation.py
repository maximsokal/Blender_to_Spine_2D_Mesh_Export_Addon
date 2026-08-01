"""Pure setup-pose contract for Active Camera depth-parent compensation."""

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
    LegacyAttachmentVertex,
    LegacyMeshAttachmentRequest,
    LegacyRigBuildRequest,
    LegacyZGroup,
    LegacyZGroupOriginMode,
    SpineJsonTarget,
    build_rig,
)


def _rig():
    return build_rig(
        LegacyRigBuildRequest(
            prefix="CameraObject",
            texture_width=100,
            texture_height=100,
            z_groups=tuple(
                LegacyZGroup(z_value=z_value)
                for z_value in (-1.0, 0.0, 2.0)
            ),
            main_position_pixels=(0.0, 0.0),
            z_group_origin_mode=LegacyZGroupOriginMode.OBJECT_ORIGIN,
        ),
        A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        spine_target=SpineJsonTarget.SPINE_4_2,
    )


def _projection(
    z_group_indices: tuple[int, int, int] = (0, 1, 2),
) -> A1AttachmentProjectionResult:
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
                z_group_index=z_group_indices[index],
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


def test_depth_parent_plus_compensated_vertex_y_equals_camera_screen_y() -> None:
    rig = _rig()
    source = _projection()
    source_positions = tuple(
        vertex.bone_position_pixels for vertex in source.request.vertices
    )

    result = _compensate_projection_depth_setup_y(source, rig)

    assert result is not source
    assert tuple(
        vertex.bone_position_pixels for vertex in source.request.vertices
    ) == source_positions

    offset_by_index = {
        group.index: group.y_offset_pixels for group in rig.info.z_groups
    }
    for original, adjusted in zip(
        source.request.vertices,
        result.request.vertices,
        strict=True,
    ):
        parent_y = offset_by_index[adjusted.z_group_index]
        final_relative_y = parent_y + adjusted.bone_position_pixels[1]

        assert final_relative_y == original.bone_position_pixels[1]
        assert adjusted.bone_position_pixels[0] == original.bone_position_pixels[0]
        assert adjusted.uv == original.uv
        assert adjusted.z_group_index == original.z_group_index


def test_zero_depth_group_keeps_original_vertex_y() -> None:
    rig = _rig()
    source = _projection()

    result = _compensate_projection_depth_setup_y(source, rig)

    assert result.request.vertices[1].z_group_index == 1
    assert result.request.vertices[1].bone_position_pixels[1] == -8.0


def test_unknown_depth_group_fails_closed() -> None:
    with pytest.raises(A1DocumentAssemblyError, match="unknown depth group 99"):
        _compensate_projection_depth_setup_y(
            _projection((0, 1, 99)),
            _rig(),
        )
