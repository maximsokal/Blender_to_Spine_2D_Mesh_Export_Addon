from dataclasses import replace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    MeshWorldTransformError,
    normalize_mesh_snapshot_world_transform,
)

from test_geometry_domain import build_square_snapshot


def _transform_point(matrix, point):
    x, y, z = point
    return (
        matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3],
        matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7],
        matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11],
    )


def test_rotation_non_uniform_scale_and_translation_preserve_world_geometry():
    world_matrix = (
        0.0,
        -3.0,
        0.0,
        10.0,
        2.0,
        0.0,
        0.0,
        20.0,
        0.0,
        0.0,
        4.0,
        30.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    source = replace(build_square_snapshot(), world_matrix=world_matrix)

    result = normalize_mesh_snapshot_world_transform(source)

    assert result.changed is True
    assert result.mirrored is False
    assert result.determinant == pytest.approx(24.0)
    assert result.snapshot.world_matrix == (
        1.0,
        0.0,
        0.0,
        10.0,
        0.0,
        1.0,
        0.0,
        20.0,
        0.0,
        0.0,
        1.0,
        30.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    assert tuple(vertex.position for vertex in result.snapshot.vertices) == (
        (0.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (-3.0, 2.0, 0.0),
        (-3.0, 0.0, 0.0),
    )
    assert all(
        vertex.normal == (0.0, 0.0, 1.0)
        for vertex in result.snapshot.vertices
    )
    assert all(
        face.normal == (0.0, 0.0, 1.0)
        for face in result.snapshot.faces
    )

    for old_vertex, new_vertex in zip(
        source.vertices,
        result.snapshot.vertices,
        strict=True,
    ):
        assert _transform_point(
            source.world_matrix,
            old_vertex.position,
        ) == pytest.approx(
            _transform_point(
                result.snapshot.world_matrix,
                new_vertex.position,
            )
        )


def test_mirrored_transform_preserves_oriented_winding_normals():
    source = replace(
        build_square_snapshot(),
        world_matrix=(
            -2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            3.0,
            0.0,
            0.0,
            0.0,
            0.0,
            4.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    )

    result = normalize_mesh_snapshot_world_transform(source)

    assert result.mirrored is True
    assert result.determinant == pytest.approx(-24.0)
    assert tuple(vertex.position for vertex in result.snapshot.vertices) == (
        (0.0, 0.0, 0.0),
        (-2.0, 0.0, 0.0),
        (-2.0, 3.0, 0.0),
        (0.0, 3.0, 0.0),
    )
    assert all(
        vertex.normal == (0.0, 0.0, -1.0)
        for vertex in result.snapshot.vertices
    )
    assert all(
        face.normal == (0.0, 0.0, -1.0)
        for face in result.snapshot.faces
    )


def test_large_valid_non_uniform_scale_is_not_misclassified_as_singular():
    source = replace(
        build_square_snapshot(),
        world_matrix=(
            1_000_000.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
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

    result = normalize_mesh_snapshot_world_transform(source)

    assert result.changed is True
    assert result.determinant == pytest.approx(1_000_000.0)
    assert result.snapshot.vertices[1].position == (1_000_000.0, 0.0, 0.0)


def test_translation_only_snapshot_is_returned_without_geometry_rebuild():
    source = replace(
        build_square_snapshot(),
        world_matrix=(
            1.0,
            0.0,
            0.0,
            5.0,
            0.0,
            1.0,
            0.0,
            6.0,
            0.0,
            0.0,
            1.0,
            7.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    )

    result = normalize_mesh_snapshot_world_transform(source)

    assert result.changed is False
    assert result.snapshot is source
    assert result.translation == (5.0, 6.0, 7.0)


def test_singular_object_transform_is_rejected_before_segmentation():
    source = replace(
        build_square_snapshot(),
        world_matrix=(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    )

    with pytest.raises(MeshWorldTransformError, match="singular"):
        normalize_mesh_snapshot_world_transform(source)
