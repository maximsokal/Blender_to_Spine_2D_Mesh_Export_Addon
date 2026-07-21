from math import inf, nan

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    LegacyAttachmentSequence,
    LegacyAttachmentVertex,
    LegacyMeshAttachmentRequest,
)


def build_vertices():
    return (
        LegacyAttachmentVertex(0, (0.0, 0.0), (10.0, 20.0), 0),
        LegacyAttachmentVertex(1, (1.0, 0.0), (30.0, 20.0), 0),
        LegacyAttachmentVertex(2, (0.0, 1.0), (10.0, 40.0), 1),
    )


def build_request(**changes):
    values = {
        "slot_name": "Segment",
        "attachment_name": "Segment",
        "vertex_prefix": "Segment",
        "image_path": "images/Segment.png",
        "width": 128.0,
        "height": 64.0,
        "vertices": build_vertices(),
        "triangles": (0, 1, 2),
        "hull": 3,
        "edges": (0, 1, 1, 2, 2, 0),
        "sequence": LegacyAttachmentSequence(count=3, start=7, digits=4, setup=1),
    }
    values.update(changes)
    return LegacyMeshAttachmentRequest(**values)


@pytest.mark.parametrize(
    "changes, expected",
    (
        ({"index": True}, "index must be int"),
        ({"uv": (True, 0.5)}, r"uv\[0\]"),
        ({"uv": (nan, 0.5)}, r"uv\[0\]"),
        ({"uv": (inf, 0.5)}, r"uv\[0\]"),
        ({"bone_position_pixels": (0.0, False)}, r"bone_position_pixels\[1\]"),
        ({"bone_position_pixels": (0.0, inf)}, r"bone_position_pixels\[1\]"),
        ({"z_group_index": False}, "z_group_index must be int"),
    ),
)
def test_attachment_vertex_rejects_permissive_python_scalars(changes, expected):
    values = {
        "index": 0,
        "uv": (0.0, 0.0),
        "bone_position_pixels": (10.0, 20.0),
        "z_group_index": 0,
    }
    values.update(changes)

    with pytest.raises((TypeError, ValueError), match=expected):
        LegacyAttachmentVertex(**values)


@pytest.mark.parametrize(
    "changes, expected",
    (
        ({"count": True}, "count must be int"),
        ({"start": False}, "start must be int"),
        ({"digits": True}, "digits must be int"),
        ({"setup": False}, "setup must be int"),
        ({"count": 0}, "count must be at least 1"),
        ({"digits": -1}, "digits must be at least 0"),
        ({"count": 2, "setup": 2}, "setup must be at most 1"),
    ),
)
def test_attachment_sequence_rejects_boolean_and_invalid_indices(changes, expected):
    values = {"count": 3, "start": 0, "digits": 4, "setup": 1}
    values.update(changes)

    with pytest.raises((TypeError, ValueError), match=expected):
        LegacyAttachmentSequence(**values)


@pytest.mark.parametrize(
    "start, digits",
    (
        (-7, 0),
        (0, 4),
        (7, 100),
    ),
)
def test_attachment_sequence_accepts_runtime_integer_ranges(start, digits):
    sequence = LegacyAttachmentSequence(
        count=3,
        start=start,
        digits=digits,
        setup=1,
    )

    assert sequence.to_spine_mapping() == {
        "count": 3,
        "start": start,
        "digits": digits,
        "setup": 1,
    }


def test_attachment_sequence_mapping_is_unchanged_for_legacy_defaults():
    sequence = LegacyAttachmentSequence(count=3, start=7, digits=4, setup=1)

    assert sequence.to_spine_mapping() == {
        "count": 3,
        "start": 7,
        "digits": 4,
        "setup": 1,
    }
    assert LegacyAttachmentSequence(count=2, start=0).digits == 4
    assert LegacyAttachmentSequence(count=2, start=0).resolved_setup == 1
    assert LegacyAttachmentSequence(count=1, start=0).resolved_setup == 0


@pytest.mark.parametrize(
    "changes, expected",
    (
        ({"width": True}, "width must be a finite number"),
        ({"height": False}, "height must be a finite number"),
        ({"width": nan}, "width must be finite"),
        ({"height": inf}, "height must be finite"),
        ({"width": 0.0}, "width must be > 0.0"),
        ({"hull": True}, "hull must be int"),
        ({"triangles": (0, True, 2)}, r"triangles\[1\] must be int"),
        ({"edges": (0, False)}, r"edges\[1\] must be int"),
    ),
)
def test_attachment_request_rejects_permissive_scalars(changes, expected):
    with pytest.raises((TypeError, ValueError), match=expected):
        build_request(**changes)


def test_valid_attachment_request_preserves_spine_array_contract():
    request = build_request()

    assert tuple(vertex.index for vertex in request.vertices) == (0, 1, 2)
    assert tuple(component for vertex in request.vertices for component in vertex.uv) == (
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
    )
    assert request.triangles == (0, 1, 2)
    assert request.hull == 3
    assert request.edges == (0, 1, 1, 2, 2, 0)
    assert request.sequence.to_spine_mapping() == {
        "count": 3,
        "start": 7,
        "digits": 4,
        "setup": 1,
    }


def test_request_still_requires_dense_vertex_indices():
    invalid_vertices = (
        LegacyAttachmentVertex(0, (0.0, 0.0), (0.0, 0.0), 0),
        LegacyAttachmentVertex(2, (1.0, 0.0), (1.0, 0.0), 0),
    )

    with pytest.raises(ValueError, match="ordered and dense"):
        build_request(vertices=invalid_vertices, triangles=(), hull=2, edges=(0, 1))
