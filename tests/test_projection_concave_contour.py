import random

from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    ProjectionContourMode,
    ProjectionPixelPoint,
    build_sequence_union_layout,
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


def _assert_exact_triangulation(layout):
    triangles = layout.triangle_indices
    assert len(triangles) == len(layout.hull) - 2
    assert all(_triangle_area2(layout.hull, triangle) > 0 for triangle in triangles)
    assert sum(
        _triangle_area2(layout.hull, triangle) for triangle in triangles
    ) == _polygon_area2(layout.hull)


def test_connected_l_shape_keeps_real_concavity():
    visible = {
        (2, 2),
        (3, 2),
        (4, 2),
        (2, 3),
        (2, 4),
    }
    layout = build_sequence_union_layout(
        (_mask(8, 8, visible),),
        width=8,
        height=8,
        alpha_threshold=0.01,
        padding_pixels=0,
    )

    assert layout.contour_mode is ProjectionContourMode.SIMPLIFIED_CONCAVE
    assert layout.concave
    assert layout.outer_component_count == 1
    assert layout.contour_fallback_reason is None
    assert layout.hull == (
        ProjectionPixelPoint(2, 2),
        ProjectionPixelPoint(5, 2),
        ProjectionPixelPoint(5, 3),
        ProjectionPixelPoint(3, 3),
        ProjectionPixelPoint(3, 5),
        ProjectionPixelPoint(2, 5),
    )
    _assert_exact_triangulation(layout)


def test_shallow_reflex_notch_is_simplified_outward_without_clipping():
    visible = {(0, 0), (1, 0), (0, 1)}
    exact = build_sequence_union_layout(
        (_mask(4, 4, visible),),
        width=4,
        height=4,
        alpha_threshold=0.01,
        padding_pixels=0,
        simplify_tolerance_pixels=0.0,
    )
    simplified = build_sequence_union_layout(
        (_mask(4, 4, visible),),
        width=4,
        height=4,
        alpha_threshold=0.01,
        padding_pixels=0,
        simplify_tolerance_pixels=1.0,
    )

    assert exact.concave
    assert len(exact.hull) == 6
    assert len(simplified.hull) == 5
    assert simplified.source_contour_vertex_count == 6
    assert _polygon_area2(simplified.hull) > _polygon_area2(exact.hull)
    _assert_exact_triangulation(simplified)


def test_deep_concavity_survives_default_tolerance():
    visible = {
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (0, 2),
    }
    layout = build_sequence_union_layout(
        (_mask(5, 5, visible),),
        width=5,
        height=5,
        alpha_threshold=0.01,
        padding_pixels=0,
        simplify_tolerance_pixels=1.0,
    )

    assert layout.concave
    assert len(layout.hull) == 6
    _assert_exact_triangulation(layout)


def test_internal_hole_remains_texture_alpha_inside_one_outer_contour():
    visible = {
        (x, y)
        for x in range(3)
        for y in range(3)
        if (x, y) != (1, 1)
    }
    layout = build_sequence_union_layout(
        (_mask(5, 5, visible),),
        width=5,
        height=5,
        alpha_threshold=0.01,
        padding_pixels=0,
    )

    assert layout.contour_mode is ProjectionContourMode.SIMPLIFIED_CONCAVE
    assert layout.outer_component_count == 1
    assert layout.hull == (
        ProjectionPixelPoint(0, 0),
        ProjectionPixelPoint(3, 0),
        ProjectionPixelPoint(3, 3),
        ProjectionPixelPoint(0, 3),
    )
    _assert_exact_triangulation(layout)


def test_diagonal_touch_is_two_components_and_uses_convex_fallback():
    layout = build_sequence_union_layout(
        (_mask(4, 4, {(0, 0), (1, 1)}),),
        width=4,
        height=4,
        alpha_threshold=0.01,
        padding_pixels=0,
    )

    assert layout.contour_mode is ProjectionContourMode.CONVEX_HULL
    assert layout.outer_component_count == 2
    assert layout.contour_fallback_reason == "MULTIPLE_OUTER_COMPONENTS"
    _assert_exact_triangulation(layout)


def test_explicit_convex_mode_preserves_previous_layout_policy():
    visible = {(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)}
    layout = build_sequence_union_layout(
        (_mask(5, 5, visible),),
        width=5,
        height=5,
        alpha_threshold=0.01,
        padding_pixels=0,
        contour_mode=ProjectionContourMode.CONVEX_HULL,
    )

    assert layout.contour_mode is ProjectionContourMode.CONVEX_HULL
    assert not layout.concave
    assert layout.contour_fallback_reason is None
    _assert_exact_triangulation(layout)


def test_random_binary_masks_always_produce_deterministic_exact_meshes():
    for seed in range(250):
        randomizer = random.Random(seed)
        width = 8
        height = 7
        visible = {
            (x, y)
            for y in range(height)
            for x in range(width)
            if randomizer.random() < 0.32
        }
        if not visible:
            continue
        alpha_mask = _mask(width, height, visible)
        first = build_sequence_union_layout(
            (alpha_mask,),
            width=width,
            height=height,
            alpha_threshold=0.01,
            padding_pixels=1,
        )
        second = build_sequence_union_layout(
            (alpha_mask,),
            width=width,
            height=height,
            alpha_threshold=0.01,
            padding_pixels=1,
        )

        assert first == second
        _assert_exact_triangulation(first)
