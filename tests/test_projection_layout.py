import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.baking.projection_layout import (
    CameraProjectionLayoutError,
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
