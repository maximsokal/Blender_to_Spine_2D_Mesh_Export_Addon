import pytest

from Blender_to_Spine2D_Mesh_Exporter.application.a1_single_object import A1MeshBounds
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_document_preparation import (
    _combine_object_bake_main_position_pixels,
)


def _bounds(*, center_x: float, center_y: float) -> A1MeshBounds:
    return A1MeshBounds(
        minimum_x=center_x - 1.0,
        maximum_x=center_x + 1.0,
        minimum_y=center_y - 2.0,
        maximum_y=center_y + 2.0,
        center_x=center_x,
        center_y=center_y,
    )


def test_object_bake_main_position_preserves_offset_geometry_around_object_origin():
    result = _combine_object_bake_main_position_pixels(
        (1000.0, 2000.0),
        _bounds(center_x=3.0, center_y=-4.0),
        100.0,
    )

    assert result == (1300.0, 2400.0)


def test_centered_attachment_and_main_offset_reconstruct_original_world_xy():
    scale = 100.0
    center_x = 3.0
    center_y = -4.0
    vertex_x = 5.0
    vertex_y = -1.0
    world_x_pixels = 1000.0
    world_y_pixels = 2000.0

    main_x, main_y = _combine_object_bake_main_position_pixels(
        (world_x_pixels, world_y_pixels),
        _bounds(center_x=center_x, center_y=center_y),
        scale,
    )
    centered_vertex_x = (vertex_x - center_x) * scale
    centered_vertex_y = -(vertex_y - center_y) * scale

    assert main_x + centered_vertex_x == world_x_pixels + vertex_x * scale
    assert main_y + centered_vertex_y == world_y_pixels - vertex_y * scale


def test_connected_preparation_keeps_only_document_local_geometry_offset():
    result = _combine_object_bake_main_position_pixels(
        None,
        _bounds(center_x=-2.5, center_y=1.25),
        80.0,
    )

    assert result == (-200.0, -100.0)


def test_centered_geometry_does_not_change_existing_world_position():
    result = _combine_object_bake_main_position_pixels(
        (-320.0, 640.0),
        _bounds(center_x=0.0, center_y=0.0),
        256.0,
    )

    assert result == (-320.0, 640.0)


@pytest.mark.parametrize(
    ("world_position", "uniform_scale", "error_type"),
    (
        ((0.0,), 100.0, ValueError),
        ((0.0, float("inf")), 100.0, ValueError),
        ((0.0, 0.0), 0.0, ValueError),
        ((0.0, 0.0), True, ValueError),
    ),
)
def test_object_bake_main_position_rejects_invalid_numeric_contracts(
    world_position,
    uniform_scale,
    error_type,
):
    with pytest.raises(error_type):
        _combine_object_bake_main_position_pixels(
            world_position,
            _bounds(center_x=0.0, center_y=0.0),
            uniform_scale,
        )
