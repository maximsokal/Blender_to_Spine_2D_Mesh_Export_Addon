import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    WeightedVertex,
    WeightedVertexInfluence,
    decode_weighted_vertices,
    encode_weighted_vertices,
)


def test_weighted_vertices_round_trip():
    vertices = (
        WeightedVertex((WeightedVertexInfluence(0, 1.0, 2.0, 1.0),)),
        WeightedVertex(
            (
                WeightedVertexInfluence(1, -2.0, 5.0, 0.25),
                WeightedVertexInfluence(3, 4.0, -1.0, 0.75),
            )
        ),
    )

    encoded = encode_weighted_vertices(vertices)

    assert encoded == (
        1,
        0,
        1.0,
        2.0,
        1.0,
        2,
        1,
        -2.0,
        5.0,
        0.25,
        3,
        4.0,
        -1.0,
        0.75,
    )
    assert decode_weighted_vertices(encoded, expected_vertex_count=2) == vertices


def test_decode_rejects_truncated_stream():
    with pytest.raises(ValueError, match="truncated"):
        decode_weighted_vertices((2, 0, 0.0, 0.0, 1.0))


def test_decode_rejects_mismatched_vertex_count():
    with pytest.raises(ValueError, match="expected 2"):
        decode_weighted_vertices((1, 0, 0.0, 0.0, 1.0), expected_vertex_count=2)
