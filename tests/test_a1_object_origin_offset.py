from dataclasses import replace
from pathlib import Path

import pytest
from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1MeshBounds,
    A1SingleObjectExportSettings,
    ExportSettings,
    calculate_a1_object_bake_main_position_pixels,
)

from test_geometry_domain import build_square_snapshot


def _settings(
    *,
    use_world_location_for_main_bone: bool = True,
) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=100,
            texture_height=100,
            output_directory=Path("object-origin-test-output"),
        ),
        use_world_location_for_main_bone=use_world_location_for_main_bone,
    )


def _offset_snapshot(
    *,
    offset_x: float,
    offset_y: float,
    world_x: float,
    world_y: float,
):
    source = build_square_snapshot()
    return replace(
        source,
        vertices=tuple(
            replace(
                vertex,
                position=(
                    float(vertex.position[0]) + offset_x,
                    float(vertex.position[1]) + offset_y,
                    float(vertex.position[2]),
                ),
            )
            for vertex in source.vertices
        ),
        world_matrix=(
            1.0,
            0.0,
            0.0,
            world_x,
            0.0,
            1.0,
            0.0,
            world_y,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    )


def test_object_bake_main_position_uses_world_object_origin_only():
    snapshot = _offset_snapshot(
        offset_x=2.5,
        offset_y=-4.5,
        world_x=10.0,
        world_y=20.0,
    )

    result = calculate_a1_object_bake_main_position_pixels(
        snapshot,
        _settings(),
    )

    assert result == (1000.0, 2000.0)


def test_origin_relative_attachment_and_main_reconstruct_original_world_xy():
    snapshot = _offset_snapshot(
        offset_x=2.5,
        offset_y=-4.5,
        world_x=10.0,
        world_y=20.0,
    )
    main_x, main_y = calculate_a1_object_bake_main_position_pixels(
        snapshot,
        _settings(),
    )
    vertex_x = 5.0
    vertex_y = -1.0
    scale = 100.0
    origin_relative_vertex_x = vertex_x * scale
    origin_relative_vertex_y = -vertex_y * scale

    assert main_x + origin_relative_vertex_x == (10.0 + vertex_x) * scale
    assert main_y + origin_relative_vertex_y == (20.0 - vertex_y) * scale


def test_connected_preparation_keeps_neutral_document_local_origin():
    snapshot = _offset_snapshot(
        offset_x=-3.0,
        offset_y=0.75,
        world_x=10.0,
        world_y=20.0,
    )

    result = calculate_a1_object_bake_main_position_pixels(
        snapshot,
        _settings(use_world_location_for_main_bone=False),
    )

    assert result == (0.0, 0.0)


def test_centered_geometry_does_not_change_existing_world_origin():
    source = build_square_snapshot()
    snapshot = replace(
        source,
        vertices=tuple(
            replace(
                vertex,
                position=(
                    float(vertex.position[0]) - 0.5,
                    float(vertex.position[1]) - 0.5,
                    float(vertex.position[2]),
                ),
            )
            for vertex in source.vertices
        ),
        world_matrix=(
            1.0,
            0.0,
            0.0,
            -3.2,
            0.0,
            1.0,
            0.0,
            6.4,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    )

    result = calculate_a1_object_bake_main_position_pixels(
        snapshot,
        _settings(),
    )

    assert result == (-320.0, 640.0)


def test_valid_cached_bounds_never_move_the_object_origin():
    snapshot = _offset_snapshot(
        offset_x=2.5,
        offset_y=-4.5,
        world_x=10.0,
        world_y=20.0,
    )

    result = calculate_a1_object_bake_main_position_pixels(
        snapshot,
        _settings(),
        bounds=A1MeshBounds(
            minimum_x=100.0,
            maximum_x=200.0,
            minimum_y=-50.0,
            maximum_y=10.0,
            center_x=150.0,
            center_y=-20.0,
        ),
    )

    assert result == (1000.0, 2000.0)


def test_object_bake_main_position_rejects_invalid_contract_types():
    snapshot = build_square_snapshot()
    settings = _settings()

    with pytest.raises(TypeError, match="snapshot"):
        calculate_a1_object_bake_main_position_pixels(object(), settings)
    with pytest.raises(TypeError, match="settings"):
        calculate_a1_object_bake_main_position_pixels(snapshot, object())
    with pytest.raises(TypeError, match="bounds"):
        calculate_a1_object_bake_main_position_pixels(
            snapshot,
            settings,
            bounds=object(),
        )


def test_object_bake_main_position_rejects_non_finite_or_inverted_bounds():
    snapshot = build_square_snapshot()
    settings = _settings()

    with pytest.raises(ValueError, match=r"bounds\.center_x.*finite"):
        calculate_a1_object_bake_main_position_pixels(
            snapshot,
            settings,
            bounds=A1MeshBounds(0.0, 1.0, 0.0, 1.0, float("nan"), 0.5),
        )
    with pytest.raises(ValueError, match="minimum_x cannot exceed"):
        calculate_a1_object_bake_main_position_pixels(
            snapshot,
            settings,
            bounds=A1MeshBounds(2.0, 1.0, 0.0, 1.0, 1.5, 0.5),
        )


def test_object_bake_main_position_rejects_inconsistent_cached_center():
    with pytest.raises(ValueError, match="center_x must be the midpoint"):
        calculate_a1_object_bake_main_position_pixels(
            build_square_snapshot(),
            _settings(),
            bounds=A1MeshBounds(0.0, 1.0, 0.0, 1.0, 0.25, 0.5),
        )


def test_object_bake_main_position_rejects_cached_midpoint_overflow():
    huge = 1.0e308
    with pytest.raises(ValueError, match="bounds.expected_center_x must be finite"):
        calculate_a1_object_bake_main_position_pixels(
            build_square_snapshot(),
            _settings(),
            bounds=A1MeshBounds(huge, huge, 0.0, 0.0, huge, 0.0),
        )


def test_object_bake_main_position_rejects_world_origin_overflow():
    snapshot = _offset_snapshot(
        offset_x=0.0,
        offset_y=0.0,
        world_x=2.0e306,
        world_y=0.0,
    )

    with pytest.raises(ValueError, match="object_origin_main_x must be finite"):
        calculate_a1_object_bake_main_position_pixels(
            snapshot,
            _settings(),
        )


def test_object_bake_placement_math_has_one_blender_independent_owner():
    root = Path(__file__).resolve().parents[1]
    application_source = (
        root
        / "Blender_to_Spine2D_Mesh_Exporter"
        / "application"
        / "a1_object_bake_placement.py"
    ).read_text(encoding="utf-8")
    adapter_source = (
        root
        / "Blender_to_Spine2D_Mesh_Exporter"
        / "blender_adapter"
        / "a1_document_preparation.py"
    ).read_text(encoding="utf-8")

    assert "import bpy" not in application_source
    assert "def _combine_object_bake_main_position_pixels" not in adapter_source
    assert "calculate_a1_object_bake_main_position_pixels(" in adapter_source
    assert "center_x=0.0" in adapter_source
    assert "center_y=0.0" in adapter_source
