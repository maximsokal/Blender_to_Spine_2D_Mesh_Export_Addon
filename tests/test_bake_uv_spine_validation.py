import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1AttachmentProjectionResult,
    A1AttachmentVertexKey,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.bake_uv_spine_validation import (
    BakedUvSpineValidationError,
    RgbaImageBuffer,
    validate_projection_uv_coverage,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import LoopId, VertexId
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    LegacyAttachmentVertex,
    LegacyMeshAttachmentRequest,
)


def _image_with_lower_left_coverage():
    width = height = 8
    pixels = [0.0] * (width * height * 4)
    for y in range(4):
        for x in range(4):
            offset = (y * width + x) * 4
            pixels[offset : offset + 4] = (1.0, 0.0, 0.0, 1.0)
    return RgbaImageBuffer(width, height, tuple(pixels))


def _projection(uvs):
    vertices = tuple(
        LegacyAttachmentVertex(
            index=index,
            uv=tuple(uv),
            bone_position_pixels=(float(index), float(index)),
            z_group_index=0,
        )
        for index, uv in enumerate(uvs)
    )
    request = LegacyMeshAttachmentRequest(
        slot_name="Segment",
        attachment_name="Segment",
        vertex_prefix="Segment",
        image_path="images/Test",
        width=8.0,
        height=8.0,
        vertices=vertices,
        triangles=(0, 1, 2),
        hull=3,
        edges=(0, 1, 1, 2, 2, 0),
    )
    keys = tuple(
        A1AttachmentVertexKey(VertexId(index), tuple(uv))
        for index, uv in enumerate(uvs)
    )
    return A1AttachmentProjectionResult(
        request=request,
        hull_vertex_keys=keys,
        ordered_vertex_keys=keys,
        loop_to_attachment_index=(
            (LoopId(0), 0),
            (LoopId(1), 1),
            (LoopId(2), 2),
        ),
    )


def test_spine_uv_triangle_samples_the_baked_island():
    projection = _projection(((0.1, 0.1), (0.4, 0.1), (0.1, 0.4)))

    samples = validate_projection_uv_coverage(
        _image_with_lower_left_coverage(),
        (projection,),
    )

    assert len(samples) == 1
    assert samples[0].maximum_alpha == 1.0


def test_vertical_uv_flip_is_detected_as_empty_texture_sampling():
    projection = _projection(((0.1, 0.9), (0.4, 0.9), (0.1, 0.6)))

    with pytest.raises(
        BakedUvSpineValidationError,
        match="point only into empty baked pixels",
    ):
        validate_projection_uv_coverage(
            _image_with_lower_left_coverage(),
            (projection,),
        )


def test_out_of_range_spine_uv_is_rejected_before_pixel_lookup():
    projection = _projection(((-0.1, 0.1), (0.4, 0.1), (0.1, 0.4)))

    with pytest.raises(BakedUvSpineValidationError, match="outside the unit square"):
        validate_projection_uv_coverage(
            _image_with_lower_left_coverage(),
            (projection,),
        )
