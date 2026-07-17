import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.baking.projection_layout import (
    CameraProjectionLayoutError,
    ProjectionAlphaUnionAccumulator,
    ProjectionPixelPoint,
    build_full_frame_layout,
    build_sequence_union_layout,
    convex_hull,
)


def _mask(width, height, visible):
    result = bytearray(width * height)
    for x, y in visible:
        result[y * width + x] = 1
    return bytes(result)


def test_sequence_union_crop_contains_all_frames_with_padding():
    first = _mask(10, 8, {(2, 2), (3, 2), (2, 3), (3, 3)})
    second = _mask(10, 8, {(6, 4), (7, 4), (6, 5), (7, 5)})

    layout = build_sequence_union_layout(
        (first, second),
        width=10,
        height=8,
        alpha_threshold=0.01,
        padding_pixels=1,
    )

    assert (layout.crop.minimum_x, layout.crop.minimum_y) == (1, 1)
    assert (layout.crop.maximum_x, layout.crop.maximum_y) == (9, 7)
    assert layout.cropped_width == 8
    assert layout.cropped_height == 6
    assert layout.frame_count == 2
    assert layout.visible_pixel_count == 8
    assert layout.cropped


def test_union_hull_is_stable_convex_and_counter_clockwise():
    mask = _mask(8, 8, {(2, 2), (3, 2), (4, 2), (2, 3), (2, 4)})
    layout = build_sequence_union_layout(
        (mask,),
        width=8,
        height=8,
        alpha_threshold=0.1,
        padding_pixels=0,
    )

    assert len(layout.hull) >= 3
    assert layout.hull[0] == min(layout.hull)
    assert len(layout.hull) < 8
    assert all(
        layout.crop.minimum_x <= point.x <= layout.crop.maximum_x
        and layout.crop.minimum_y <= point.y <= layout.crop.maximum_y
        for point in layout.hull
    )


def test_incremental_accumulator_matches_compatibility_wrapper():
    masks = (
        _mask(12, 9, {(1, 1), (2, 1), (2, 2)}),
        _mask(12, 9, {(7, 5), (8, 5), (8, 6)}),
        _mask(12, 9, {(4, 3), (5, 3)}),
    )
    expected = build_sequence_union_layout(
        masks,
        width=12,
        height=9,
        alpha_threshold=1.0 / 255.0,
        padding_pixels=2,
    )
    accumulator = ProjectionAlphaUnionAccumulator(
        width=12,
        height=9,
        alpha_threshold=1.0 / 255.0,
        padding_pixels=2,
    )
    for index, mask in enumerate(masks):
        accumulator.add_mask(mask, frame_index=index)

    assert accumulator.build_layout() == expected


def test_incremental_union_is_independent_of_frame_order():
    masks = (
        _mask(9, 7, {(1, 1), (2, 1)}),
        _mask(9, 7, {(6, 4), (7, 4)}),
        _mask(9, 7, {(4, 2), (4, 3)}),
    )

    forward = ProjectionAlphaUnionAccumulator(9, 7, 0.01, 1)
    reverse = ProjectionAlphaUnionAccumulator(9, 7, 0.01, 1)
    for index, mask in enumerate(masks):
        forward.add_mask(mask, frame_index=index)
    for index, mask in enumerate(reversed(masks)):
        reverse.add_mask(mask, frame_index=index)

    assert forward.build_layout() == reverse.build_layout()


def test_duplicate_pixels_are_counted_only_once():
    mask = _mask(6, 5, {(2, 2), (3, 2), (3, 3)})
    accumulator = ProjectionAlphaUnionAccumulator(6, 5, 0.01, 0)

    assert accumulator.add_mask(mask, frame_index=0) == 3
    assert accumulator.add_mask(mask, frame_index=1) == 0
    layout = accumulator.build_layout()

    assert layout.frame_count == 2
    assert layout.visible_pixel_count == 3


def test_padding_is_clamped_to_full_frame_edges():
    accumulator = ProjectionAlphaUnionAccumulator(8, 6, 0.01, 5)
    accumulator.add_mask(_mask(8, 6, {(0, 0), (7, 5)}))

    layout = accumulator.build_layout()

    assert (
        layout.crop.minimum_x,
        layout.crop.minimum_y,
        layout.crop.maximum_x,
        layout.crop.maximum_y,
    ) == (0, 0, 8, 6)


def test_long_sequence_keeps_one_fixed_union_buffer():
    width = 32
    height = 24
    accumulator = ProjectionAlphaUnionAccumulator(width, height, 0.01, 1)
    expected_bytes = width * height

    for frame_index in range(250):
        point = (frame_index % width, (frame_index // width) % height)
        accumulator.add_mask(
            _mask(width, height, {point}),
            frame_index=frame_index,
        )
        assert accumulator.allocated_mask_bytes == expected_bytes

    assert accumulator.frame_count == 250
    assert not hasattr(accumulator, "__dict__")
    assert accumulator.build_layout().frame_count == 250


def test_incremental_all_transparent_sequence_is_rejected():
    accumulator = ProjectionAlphaUnionAccumulator(4, 4, 0.01, 0)
    accumulator.add_mask(_mask(4, 4, set()), frame_index=0)
    accumulator.add_mask(_mask(4, 4, set()), frame_index=1)

    with pytest.raises(CameraProjectionLayoutError, match="no pixels"):
        accumulator.build_layout()


def test_incremental_layout_requires_at_least_one_frame():
    accumulator = ProjectionAlphaUnionAccumulator(4, 4, 0.01, 0)

    with pytest.raises(ValueError, match="at least one"):
        accumulator.build_layout()


def test_incremental_mask_size_error_identifies_frame():
    accumulator = ProjectionAlphaUnionAccumulator(4, 4, 0.01, 0)

    with pytest.raises(ValueError, match=r"alpha_masks\[7\].*expected 16"):
        accumulator.add_mask(bytes(15), frame_index=7)


def test_spine_uv_inverts_blender_bottom_left_y_axis():
    layout = build_full_frame_layout(100, 50)

    assert layout.spine_uv(ProjectionPixelPoint(0, 0)) == (0.0, 1.0)
    assert layout.spine_uv(ProjectionPixelPoint(100, 50)) == (1.0, 0.0)
    assert layout.spine_position_pixels(ProjectionPixelPoint(0, 0)) == (-50.0, -25.0)
    assert layout.spine_position_pixels(ProjectionPixelPoint(100, 50)) == (50.0, 25.0)


def test_full_frame_layout_matches_legacy_quad_extent():
    layout = build_full_frame_layout(64, 32, frame_count=3)

    assert not layout.cropped
    assert layout.cropped_width == 64
    assert layout.cropped_height == 32
    assert layout.frame_count == 3
    assert layout.hull == (
        ProjectionPixelPoint(0, 0),
        ProjectionPixelPoint(64, 0),
        ProjectionPixelPoint(64, 32),
        ProjectionPixelPoint(0, 32),
    )


def test_all_transparent_sequence_is_rejected():
    with pytest.raises(CameraProjectionLayoutError, match="no pixels"):
        build_sequence_union_layout(
            (_mask(4, 4, set()), _mask(4, 4, set())),
            width=4,
            height=4,
            alpha_threshold=0.01,
            padding_pixels=0,
        )


def test_convex_hull_removes_collinear_points():
    hull = convex_hull(
        (
            ProjectionPixelPoint(0, 0),
            ProjectionPixelPoint(1, 0),
            ProjectionPixelPoint(2, 0),
            ProjectionPixelPoint(2, 1),
            ProjectionPixelPoint(2, 2),
            ProjectionPixelPoint(0, 2),
        )
    )

    assert hull == (
        ProjectionPixelPoint(0, 0),
        ProjectionPixelPoint(2, 0),
        ProjectionPixelPoint(2, 2),
        ProjectionPixelPoint(0, 2),
    )
