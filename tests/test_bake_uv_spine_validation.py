import math

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1AttachmentProjectionResult,
    A1AttachmentVertexKey,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.bake_uv_spine_validation import (
    BakedUvSpineValidationError,
    RgbaImageBuffer,
    _inset_samples,
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


def test_small_triangle_uses_intersecting_raster_footprint_after_point_samples_miss():
    image = _loaded_image_with_single_spine_file_pixel(
        width=8,
        height=8,
        pixel_x=3,
        pixel_y=2,
    )
    triangle = ((0.25, 0.25), (0.40, 0.25), (0.25, 0.40))
    projection = _projection(triangle)

    # The historical four-point validator samples only texel (2, 2) for this triangle.
    # Pixel (3, 2) is nevertheless intersected by the triangle's real raster footprint.
    assert all(
        image.rgba_spine_file_space(u, v)[3] == 0.0
        for u, v in _inset_samples(triangle)
    )

    samples = validate_projection_uv_coverage(image, (projection,))

    assert len(samples) == 1
    assert samples[0].maximum_alpha == 1.0
    assert len(samples[0].rgba_samples) > 4


def test_raster_footprint_does_not_accept_unrelated_opaque_texel():
    image = _loaded_image_with_single_spine_file_pixel(
        width=8,
        height=8,
        pixel_x=7,
        pixel_y=7,
    )
    projection = _projection(((0.25, 0.25), (0.40, 0.25), (0.25, 0.40)))

    with pytest.raises(
        BakedUvSpineValidationError,
        match="point only into empty baked pixels",
    ):
        validate_projection_uv_coverage(image, (projection,))


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
