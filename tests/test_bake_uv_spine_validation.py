import math

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


def _loaded_image_with_spine_lower_left_coverage():
    """Return a Blender-loaded buffer for a PNG whose Spine lower-left is opaque.

    Blender exposes loaded pixels bottom-up. A PNG saved for Spine file-space has its
    lower Spine rows in the upper half of this reloaded buffer.
    """

    width = height = 8
    pixels = [0.0] * (width * height * 4)
    for y in range(4, 8):
        for x in range(4):
            offset = (y * width + x) * 4
            pixels[offset : offset + 4] = (1.0, 0.0, 0.0, 1.0)
    return RgbaImageBuffer(width, height, tuple(pixels))


def _loaded_image_with_single_spine_file_pixel(
    *,
    width: int,
    height: int,
    pixel_x: int,
    pixel_y: int,
):
    """Return a loaded buffer with one opaque texel addressed in Spine file-space."""

    pixels = [0.0] * (width * height * 4)
    loaded_y = height - 1 - pixel_y
    offset = (loaded_y * width + pixel_x) * 4
    pixels[offset : offset + 4] = (0.25, 0.5, 0.75, 1.0)
    return RgbaImageBuffer(width, height, tuple(pixels))


def _transparent_image(*, width: int = 8, height: int = 8):
    return RgbaImageBuffer(
        width,
        height,
        tuple(0.0 for _ in range(width * height * 4)),
    )


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


def test_spine_uv_triangle_samples_the_saved_png_island():
    projection = _projection(((0.1, 0.1), (0.4, 0.1), (0.1, 0.4)))

    samples = validate_projection_uv_coverage(
        _loaded_image_with_spine_lower_left_coverage(),
        (projection,),
    )

    assert len(samples) == 1
    assert samples[0].maximum_alpha == 1.0
    assert samples[0].resolution_representable is True


def test_missing_spine_file_space_vertical_transform_is_detected():
    projection = _projection(((0.1, 0.9), (0.4, 0.9), (0.1, 0.6)))

    with pytest.raises(
        BakedUvSpineValidationError,
        match="point only into empty baked pixels",
    ):
        validate_projection_uv_coverage(
            _loaded_image_with_spine_lower_left_coverage(),
            (projection,),
        )


def test_spine_file_space_sampling_inverts_only_the_loaded_image_v_axis():
    image = RgbaImageBuffer(
        width=2,
        height=2,
        pixels=(
            # Blender-loaded bottom row.
            1.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 1.0,
            # Blender-loaded top row.
            0.0, 0.0, 1.0, 1.0,
            1.0, 1.0, 0.0, 1.0,
        ),
    )

    assert image.rgba(0.0, 0.0) == (1.0, 0.0, 0.0, 1.0)
    assert image.rgba_spine_file_space(0.0, 0.0) == (0.0, 0.0, 1.0, 1.0)
    assert image.rgba_spine_file_space(0.0, 1.0) == (1.0, 0.0, 0.0, 1.0)
    assert image.rgba_spine_file_pixel(0, 0) == (0.0, 0.0, 1.0, 1.0)
    assert image.rgba_spine_file_pixel(0, 1) == (1.0, 0.0, 0.0, 1.0)


def test_subpixel_triangle_without_texel_center_is_resolution_unrepresentable():
    # At 8x8 this finite triangle lives entirely near the top-left corner of pixel
    # cell (3, 2), but it contains neither that cell's centre nor any other texel centre.
    triangle = (
        (0.37510, 0.25010),
        (0.38000, 0.25010),
        (0.37510, 0.25500),
    )
    projection = _projection(triangle)

    samples = validate_projection_uv_coverage(
        _transparent_image(),
        (projection,),
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.maximum_alpha == 0.0
    assert sample.resolution_representable is False
    assert sample.raster_sample_count == 0
    assert sample.triangle_twice_area_pixels > 0.0


def test_rasterizable_triangle_with_empty_texel_center_still_fails_closed():
    # Pixel (3, 2) centre is (0.4375, 0.3125) at 8x8 and lies inside this triangle.
    projection = _projection(
        (
            (0.400, 0.280),
            (0.490, 0.280),
            (0.440, 0.360),
        )
    )

    with pytest.raises(
        BakedUvSpineValidationError,
        match="despite having raster sample centres",
    ):
        validate_projection_uv_coverage(
            _transparent_image(),
            (projection,),
        )


def test_rasterizable_triangle_accepts_its_opaque_texel():
    image = _loaded_image_with_single_spine_file_pixel(
        width=8,
        height=8,
        pixel_x=3,
        pixel_y=2,
    )
    projection = _projection(
        (
            (0.400, 0.280),
            (0.490, 0.280),
            (0.440, 0.360),
        )
    )

    samples = validate_projection_uv_coverage(image, (projection,))

    assert len(samples) == 1
    assert samples[0].maximum_alpha == 1.0
    assert samples[0].resolution_representable is True


def test_unrelated_opaque_texel_does_not_rescue_rasterizable_triangle():
    image = _loaded_image_with_single_spine_file_pixel(
        width=8,
        height=8,
        pixel_x=7,
        pixel_y=7,
    )
    projection = _projection(
        (
            (0.400, 0.280),
            (0.490, 0.280),
            (0.440, 0.360),
        )
    )

    with pytest.raises(
        BakedUvSpineValidationError,
        match="despite having raster sample centres",
    ):
        validate_projection_uv_coverage(image, (projection,))


def test_exactly_degenerate_uv_triangle_remains_a_hard_error():
    projection = _projection(
        (
            (0.2, 0.2),
            (0.3, 0.3),
            (0.4, 0.4),
        )
    )

    with pytest.raises(
        BakedUvSpineValidationError,
        match="degenerate UV area",
    ):
        validate_projection_uv_coverage(
            _transparent_image(),
            (projection,),
        )


@pytest.mark.parametrize(
    "invalid_uv",
    (
        (-0.1, 0.1),
        (1.1, 0.1),
        (0.1, -0.1),
        (0.1, 1.1),
    ),
)
def test_out_of_range_spine_uv_is_rejected_before_pixel_lookup(invalid_uv):
    projection = _projection((invalid_uv, (0.4, 0.1), (0.1, 0.4)))

    with pytest.raises(BakedUvSpineValidationError, match="outside the unit square"):
        validate_projection_uv_coverage(
            _loaded_image_with_spine_lower_left_coverage(),
            (projection,),
        )


@pytest.mark.parametrize("invalid_value", (math.nan, math.inf, -math.inf))
def test_non_finite_pixel_sample_uv_is_rejected(invalid_value):
    with pytest.raises(BakedUvSpineValidationError, match="non-finite"):
        _loaded_image_with_spine_lower_left_coverage().rgba_spine_file_space(
            invalid_value,
            0.5,
        )


def test_uv_values_inside_boundary_epsilon_are_clamped_deterministically():
    image = RgbaImageBuffer(
        width=2,
        height=2,
        pixels=(
            1.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 1.0,
            1.0, 1.0, 0.0, 1.0,
        ),
    )

    assert image.rgba(-5.0e-7, 1.0 + 5.0e-7) == (0.0, 0.0, 1.0, 1.0)
    assert image.rgba_spine_file_space(
        -5.0e-7,
        1.0 + 5.0e-7,
    ) == (1.0, 0.0, 0.0, 1.0)
