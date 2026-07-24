from Blender_to_Spine2D_Mesh_Exporter.application.camera_projection_attachment_topology import (
    build_camera_projection_attachment_topology,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    CameraProjectionLayout,
    ProjectionContourMode,
    ProjectionCropBounds,
    ProjectionPixelPoint,
)


def _concave_layout():
    return CameraProjectionLayout(
        full_width=4,
        full_height=4,
        crop=ProjectionCropBounds(
            minimum_x=0,
            minimum_y=0,
            maximum_x=4,
            maximum_y=4,
        ),
        hull=(
            ProjectionPixelPoint(0, 0),
            ProjectionPixelPoint(4, 0),
            ProjectionPixelPoint(4, 4),
            ProjectionPixelPoint(2, 2),
            ProjectionPixelPoint(0, 4),
        ),
        alpha_threshold=0.01,
        padding_pixels=0,
        frame_count=1,
        visible_pixel_count=8,
        contour_mode=ProjectionContourMode.SIMPLIFIED_CONCAVE,
    )


def _undirected_edges(values):
    return {
        tuple(sorted((values[index], values[index + 1])))
        for index in range(0, len(values), 2)
    }


def test_concave_camera_contour_moves_only_convex_points_to_spine_hull_prefix():
    layout = _concave_layout()
    source_triangles = layout.triangle_indices

    topology = build_camera_projection_attachment_topology(layout)

    assert topology.hull_count == 4
    assert topology.source_indices == (0, 1, 2, 4, 3)
    assert topology.points[: topology.hull_count] == (
        ProjectionPixelPoint(0, 0),
        ProjectionPixelPoint(4, 0),
        ProjectionPixelPoint(4, 4),
        ProjectionPixelPoint(0, 4),
    )
    assert topology.points[topology.hull_count :] == (
        ProjectionPixelPoint(2, 2),
    )

    old_to_new = {
        old_index: new_index
        for new_index, old_index in enumerate(topology.source_indices)
    }
    assert topology.triangles == tuple(
        tuple(old_to_new[index] for index in triangle)
        for triangle in source_triangles
    )


def test_concave_camera_topology_preserves_boundary_and_internal_edge_graph():
    layout = _concave_layout()
    topology = build_camera_projection_attachment_topology(layout)
    old_to_new = {
        old_index: new_index
        for new_index, old_index in enumerate(topology.source_indices)
    }

    source_boundary = {
        tuple(sorted((index, (index + 1) % len(layout.contour))))
        for index in range(len(layout.contour))
    }
    source_triangle_edges = {
        tuple(sorted((first, second)))
        for triangle in layout.triangle_indices
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        )
    }
    expected_edges = {
        tuple(sorted((old_to_new[first], old_to_new[second])))
        for first, second in source_boundary | source_triangle_edges
    }

    assert _undirected_edges(topology.edges) == expected_edges
    assert len(topology.edges) // 2 == 2 * len(layout.contour) - 3


def test_convex_camera_contour_keeps_original_order():
    layout = CameraProjectionLayout(
        full_width=4,
        full_height=4,
        crop=ProjectionCropBounds(0, 0, 4, 4),
        hull=(
            ProjectionPixelPoint(0, 0),
            ProjectionPixelPoint(4, 0),
            ProjectionPixelPoint(4, 4),
            ProjectionPixelPoint(0, 4),
        ),
        alpha_threshold=0.01,
        padding_pixels=0,
        frame_count=1,
        visible_pixel_count=16,
    )

    topology = build_camera_projection_attachment_topology(layout)

    assert topology.source_indices == (0, 1, 2, 3)
    assert topology.points == layout.contour
    assert topology.hull_count == 4
