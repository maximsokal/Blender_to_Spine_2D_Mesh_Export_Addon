import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.mesh_edge_contract import (
    SpineMeshEdgeContractError,
    encode_spine_mesh_edge_offsets,
    validate_logical_mesh_edges,
    validate_spine_mesh_edge_offsets,
)


def test_logical_vertex_indices_encode_to_spine_coordinate_offsets():
    logical = (0, 1, 1, 2, 2, 0)

    assert encode_spine_mesh_edge_offsets(logical, vertex_count=3) == (
        0,
        2,
        2,
        4,
        4,
        0,
    )


def test_official_spine_edge_offset_shape_is_even_and_round_trippable():
    offsets = (4, 6, 6, 8, 8, 10, 10, 0)

    assert validate_spine_mesh_edge_offsets(offsets, vertex_count=6) == offsets
    assert tuple(value // 2 for value in offsets) == (2, 3, 3, 4, 4, 5, 5, 0)


@pytest.mark.parametrize(
    "edges",
    (
        (0, 1, 2),
        (0, 0),
        (0, 1, 1, 0),
        (0, 3),
    ),
)
def test_logical_edge_contract_rejects_incomplete_self_duplicate_and_range_errors(edges):
    with pytest.raises((TypeError, SpineMeshEdgeContractError)):
        validate_logical_mesh_edges(edges, vertex_count=3)


@pytest.mark.parametrize(
    "offsets",
    (
        (0, 2, 4),
        (0, 1),
        (2, 2),
        (0, 2, 2, 0),
        (0, 6),
    ),
)
def test_serialized_offset_contract_rejects_invalid_spine_edges(offsets):
    with pytest.raises((TypeError, SpineMeshEdgeContractError)):
        validate_spine_mesh_edge_offsets(offsets, vertex_count=3)


def test_boolean_endpoints_are_not_accepted_as_integer_indices():
    with pytest.raises(TypeError):
        encode_spine_mesh_edge_offsets((False, 1), vertex_count=2)
