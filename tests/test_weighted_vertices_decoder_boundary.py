from array import array
from collections.abc import Sequence

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    WeightedVertex,
    WeightedVertexInfluence,
    decode_weighted_vertices,
)


class IndexOnlyNumericSequence(Sequence):
    """Sequence that supports integer indexing but deliberately rejects slicing."""

    def __init__(self, values):
        self._values = tuple(values)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise AssertionError("decoder must not require sequence slicing")
        return self._values[index]

    def __iter__(self):
        raise AssertionError("decoder must not materialize the sequence")


class FailingNumericSequence(Sequence):
    """Sequence that exposes source access failures without wrapping them."""

    def __len__(self):
        return 1

    def __getitem__(self, index):
        raise RuntimeError("numeric sequence access failed")


class ClaimedLengthNumericSequence(Sequence):
    """Sequence whose reported length intentionally exceeds indexed storage."""

    def __init__(self, values, claimed_length):
        self._values = tuple(values)
        self._claimed_length = claimed_length

    def __len__(self):
        return self._claimed_length

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise AssertionError("decoder must not require sequence slicing")
        return self._values[index]

    def __iter__(self):
        raise AssertionError("decoder must not materialize the sequence")


@pytest.mark.parametrize(
    "stream",
    (
        bytearray((1, 0, 0, 0, 1)),
        memoryview(bytes((1, 0, 0, 0, 1))),
        memoryview(array("d", (1.0, 0.0, 0.0, 0.0, 1.0))),
        bytearray(),
        memoryview(b""),
    ),
)
def test_decoder_rejects_binary_sequence_containers(stream):
    with pytest.raises(TypeError, match="stream must be a numeric sequence"):
        decode_weighted_vertices(stream)


def test_decoder_accepts_custom_indexed_sequence_without_slicing_or_materializing():
    stream = IndexOnlyNumericSequence(
        (
            2,
            0,
            1.0,
            2.0,
            0.25,
            3,
            4.0,
            5.0,
            0.75,
            1,
            2,
            -1.0,
            -2.0,
            1.0,
        )
    )

    assert decode_weighted_vertices(stream, expected_vertex_count=2) == (
        WeightedVertex(
            (
                WeightedVertexInfluence(0, 1.0, 2.0, 0.25),
                WeightedVertexInfluence(3, 4.0, 5.0, 0.75),
            )
        ),
        WeightedVertex((WeightedVertexInfluence(2, -1.0, -2.0, 1.0),)),
    )


def test_decoder_accepts_standard_numeric_array_sequence():
    stream = array("d", (1.0, 0.0, 1.0, 2.0, 1.0))

    assert decode_weighted_vertices(stream) == (
        WeightedVertex((WeightedVertexInfluence(0, 1.0, 2.0, 1.0),)),
    )


@pytest.mark.parametrize(
    "stream",
    (
        (value for value in ()),
        iter((1, 0, 0, 0, 1)),
    ),
)
def test_decoder_rejects_non_sequence_iterables(stream):
    with pytest.raises(TypeError, match="stream must be a numeric sequence"):
        decode_weighted_vertices(stream)


@pytest.mark.parametrize("stream", ([], ()))
def test_decoder_keeps_empty_numeric_sequences_valid(stream):
    assert decode_weighted_vertices(stream) == ()


def test_decoder_does_not_mutate_mutable_numeric_sequence():
    stream = [1, 0, 1.0, 2.0, 1.0]
    before = list(stream)

    decode_weighted_vertices(stream)

    assert stream == before


def test_decoder_preserves_sequence_access_failures():
    with pytest.raises(RuntimeError, match="numeric sequence access failed"):
        decode_weighted_vertices(FailingNumericSequence())


def test_decoder_normalizes_missing_count_after_claimed_length_to_truncation():
    stream = ClaimedLengthNumericSequence((), claimed_length=1)

    with pytest.raises(
        ValueError,
        match=(
            r"Weighted vertex 0 is truncated: expected influence count "
            r"at stream index 0"
        ),
    ) as exc_info:
        decode_weighted_vertices(stream)

    assert isinstance(exc_info.value.__cause__, IndexError)


@pytest.mark.parametrize(
    "values",
    (
        (1,),
        (1, 0),
        (1, 0, 1.0),
        (1, 0, 1.0, 2.0),
    ),
)
def test_decoder_normalizes_missing_influence_items_to_truncation(values):
    stream = ClaimedLengthNumericSequence(values, claimed_length=5)

    with pytest.raises(
        ValueError,
        match=r"Weighted vertex 0 is truncated: expected 4 influence values",
    ) as exc_info:
        decode_weighted_vertices(stream)

    assert isinstance(exc_info.value.__cause__, IndexError)


def test_decoder_reports_missing_next_vertex_count_with_exact_vertex_and_index():
    stream = ClaimedLengthNumericSequence(
        (1, 0, 1.0, 2.0, 1.0),
        claimed_length=6,
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Weighted vertex 1 is truncated: expected influence count "
            r"at stream index 5"
        ),
    ) as exc_info:
        decode_weighted_vertices(stream)

    assert isinstance(exc_info.value.__cause__, IndexError)
