import pytest

from Blender_to_Spine2D_Mesh_Exporter.application.a1_camera_projection import (
    _edge_pairs,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking.projection_layout import (
    CameraProjectionLayout,
    ProjectionCropBounds,
    ProjectionPixelPoint,
    build_sequence_union_layout,
    triangulate_convex_hull,
)


def _mask(width, height, visible):
    result = bytearray(width * height)
    for x, y in visible:
        result[y * width + x] = 1
    return bytes(result)


def _polygon_area2(points):
    return sum(
        first.x * second.y - second.x * first.y
        for first, second in zip(points, points[1:] + points[:1])
    )


def _triangle_area2(points, triangle):
    first, second, third = (points[index] for index in triangle)
    return (second.x - first.x) * (third.y - first.y) - (
        second.y - first.y
    ) * (third.x - first.x)


def _layout(points):
    return CameraProjectionLayout(
        full_width=16,
        full_height=16,
        crop=ProjectionCropBounds(0, 0, 16, 16),
        hull=points,
        alpha_threshold=0.01,
        padding_pixels=0,
        frame_count=1,
        visible_pixel_count=1,
    )


def test_manual_concave_hull_is_rejected():
    with pytest.raises(ValueError, match="strictly convex"):
        _layout(
            (
                ProjectionPixelPoint(0, 0),
                ProjectionPixelPoint(8, 0),
                ProjectionPixelPoint(8, 8),
                ProjectionPixelPoint(0, 8),
                ProjectionPixelPoint(4, 3),
            )
        )


def test_manual_collinear_hull_is_rejected():
    with pytest.raises(ValueError, match="strictly convex"):
        _layout(
            (
                ProjectionPixelPoint(0, 0),
                ProjectionPixelPoint(4, 0),
                ProjectionPixelPoint(8, 0),
                ProjectionPixelPoint(8, 8),
                ProjectionPixelPoint(0, 8),
            )
        )


def test_clockwise_hull_is_rejected():
    with pytest.raises(ValueError, match="counter-clockwise"):
        _layout(
            (
                ProjectionPixelPoint(0, 0),
                ProjectionPixelPoint(0, 8),
                ProjectionPixelPoint(8, 8),
                ProjectionPixelPoint(8, 0),
            )
        )


def test_generated_irregular_hull_has_exact_non_degenerate_fan():
    mask = _mask(12, 10, {(1, 1), (9, 1), (10, 5), (7, 8), (2, 7)})
    layout = build_sequence_union_layout(
        (mask,),
        width=12,
        height=10,
        alpha_threshold=0.01,
        padding_pixels=1,
    )

    triangles = layout.triangle_indices

    assert len(triangles) == len(layout.hull) - 2
    assert all(_triangle_area2(layout.hull, triangle) > 0 for triangle in triangles)
    assert sum(
        _triangle_area2(layout.hull, triangle) for triangle in triangles
    ) == _polygon_area2(layout.hull)


def test_single_visible_pixel_produces_valid_quad_fan():
    layout = build_sequence_union_layout(
        (_mask(4, 4, {(2, 1)}),),
        width=4,
        height=4,
        alpha_threshold=0.01,
        padding_pixels=0,
    )

    assert layout.hull == (
        ProjectionPixelPoint(2, 1),
        ProjectionPixelPoint(3, 1),
        ProjectionPixelPoint(3, 2),
        ProjectionPixelPoint(2, 2),
    )
    assert layout.triangle_indices == ((0, 1, 2), (0, 2, 3))


def test_triangulation_preserves_valid_rotated_hull_order():
    points = (
        ProjectionPixelPoint(8, 0),
        ProjectionPixelPoint(8, 8),
        ProjectionPixelPoint(0, 8),
        ProjectionPixelPoint(0, 0),
    )

    assert triangulate_convex_hull(points) == ((0, 1, 2), (0, 2, 3))


def test_projection_edges_follow_validated_fan_topology():
    triangles = ((0, 1, 2), (0, 2, 3), (0, 3, 4))

    assert _edge_pairs(5, triangles) == (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (0, 2),
        (0, 3),
    )


def test_projection_edges_reject_incomplete_triangle_fan():
    with pytest.raises(ValueError, match="triangle count"):
        _edge_pairs(5, ((0, 1, 2),))
