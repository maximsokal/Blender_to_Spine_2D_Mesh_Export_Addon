from math import inf, nan

import pytest

import Blender_to_Spine2D_Mesh_Exporter.domain.spine.weighted_vertices as weighted_vertices_contract
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    WeightedVertex,
    WeightedVertexInfluence,
    decode_weighted_vertices,
    encode_weighted_vertices,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.spine_scalar_contract import (
    is_finite_number,
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


def test_weighted_vertices_alias_shared_finite_predicate():
    assert weighted_vertices_contract._is_finite_number is is_finite_number


@pytest.mark.parametrize(
    "values, expected",
    (
        ((True, 0.0, 0.0, 1.0), "bone_index must be a non-negative integer"),
        ((0, True, 0.0, 1.0), "x must be a finite number"),
        ((0, 0.0, True, 1.0), "y must be a finite number"),
        ((0, 0.0, 0.0, True), "weight must be a finite number"),
    ),
)
def test_typed_influence_rejects_bool_in_every_numeric_field(values, expected):
    with pytest.raises(ValueError, match=expected):
        WeightedVertexInfluence(*values)


@pytest.mark.parametrize("value", (nan, inf, -inf))
@pytest.mark.parametrize(
    "values, expected",
    (
        ((0, None, 0.0, 1.0), "x must be a finite number"),
        ((0, 0.0, None, 1.0), "y must be a finite number"),
        ((0, 0.0, 0.0, None), "weight must be a finite number"),
    ),
)
def test_typed_influence_rejects_non_finite_components(value, values, expected):
    mutable = list(values)
    mutable[mutable.index(None)] = value
    with pytest.raises(ValueError, match=expected):
        WeightedVertexInfluence(*mutable)


@pytest.mark.parametrize("value", (True, "1.0", None, ()))
@pytest.mark.parametrize(
    "stream_index, expected",
    (
        (2, "X coordinate"),
        (3, "Y coordinate"),
        (4, "Weight"),
    ),
)
def test_decoder_rejects_non_numeric_influence_components(
    value,
    stream_index,
    expected,
):
    stream = [1, 0, 0.0, 0.0, 1.0]
    stream[stream_index] = value

    with pytest.raises(TypeError, match=rf"{expected}.*is not numeric"):
        decode_weighted_vertices(stream)


@pytest.mark.parametrize("value", (nan, inf, -inf))
@pytest.mark.parametrize(
    "stream_index, expected",
    (
        (2, "X coordinate"),
        (3, "Y coordinate"),
        (4, "Weight"),
    ),
)
def test_decoder_rejects_non_finite_influence_components(
    value,
    stream_index,
    expected,
):
    stream = [1, 0, 0.0, 0.0, 1.0]
    stream[stream_index] = value

    with pytest.raises(ValueError, match=rf"{expected}.*must be finite"):
        decode_weighted_vertices(stream)


@pytest.mark.parametrize("value", (True, "1", None, ()))
def test_decoder_rejects_non_numeric_bone_index(value):
    with pytest.raises(TypeError, match=r"Bone index.*is not numeric"):
        decode_weighted_vertices((1, value, 0.0, 0.0, 1.0))


@pytest.mark.parametrize("value", (nan, inf, -inf))
def test_decoder_rejects_non_finite_count_and_bone_index(value):
    with pytest.raises(ValueError, match=r"Influence count.*must be finite"):
        decode_weighted_vertices((value,))
    with pytest.raises(ValueError, match=r"Bone index.*must be finite"):
        decode_weighted_vertices((1, value, 0.0, 0.0, 1.0))


@pytest.mark.parametrize("value", (True, 1.5, "1"))
def test_expected_vertex_count_requires_strict_integer_or_none(value):
    with pytest.raises(TypeError, match="expected_vertex_count must be int or None"):
        decode_weighted_vertices(
            (1, 0, 0.0, 0.0, 1.0),
            expected_vertex_count=value,
        )


def test_weighted_vertex_requires_tuple_for_immutable_influence_storage():
    influence = WeightedVertexInfluence(0, 0.0, 0.0, 1.0)

    with pytest.raises(TypeError, match="influences must be tuple"):
        WeightedVertex([influence])


@pytest.mark.parametrize("value", ("invalid", 1, None, object()))
def test_weighted_vertex_rejects_non_influence_items(value):
    with pytest.raises(
        TypeError,
        match=r"influences\[0\] must be WeightedVertexInfluence",
    ):
        WeightedVertex((value,))


def test_weighted_vertex_preserves_empty_tuple_diagnostic():
    with pytest.raises(
        ValueError,
        match="WeightedVertex must contain at least one influence",
    ):
        WeightedVertex(())
