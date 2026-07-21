import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    LegacyAttachmentVertex,
    LegacyMeshAttachmentRequest,
)


def build_vertices(count: int = 4):
    coordinates = (
        ((0.0, 0.0), (0.0, 0.0)),
        ((1.0, 0.0), (100.0, 0.0)),
        ((1.0, 1.0), (100.0, 100.0)),
        ((0.0, 1.0), (0.0, 100.0)),
    )
    return tuple(
        LegacyAttachmentVertex(
            index=index,
            uv=coordinates[index][0],
            bone_position_pixels=coordinates[index][1],
            z_group_index=0,
        )
        for index in range(count)
    )


def build_request(**changes):
    values = {
        "slot_name": "Segment",
        "attachment_name": "Segment",
        "vertex_prefix": "Segment",
        "image_path": "images/Segment.png",
        "width": 128.0,
        "height": 128.0,
        "vertices": build_vertices(),
        "triangles": (0, 1, 2, 0, 2, 3),
        "hull": 4,
        "edges": (0, 1, 1, 2, 2, 3, 3, 0, 0, 2),
    }
    values.update(changes)
    return LegacyMeshAttachmentRequest(**values)


def test_attachment_request_rejects_empty_triangle_array():
    with pytest.raises(ValueError, match="at least one triangle"):
        build_request(triangles=())


@pytest.mark.parametrize(
    "triangles",
    (
        (0, 0, 1),
        (2, 3, 2),
        (1, 1, 1),
    ),
)
def test_attachment_request_rejects_degenerate_triangles(triangles):
    with pytest.raises(ValueError, match="is degenerate"):
        build_request(
            vertices=build_vertices(3),
            triangles=triangles,
            hull=3,
            edges=(0, 1, 1, 2, 2, 0),
        )


def test_attachment_request_rejects_duplicate_triangle_with_reversed_winding():
    with pytest.raises(ValueError, match="duplicate geometry"):
        build_request(
            vertices=build_vertices(3),
            triangles=(0, 1, 2, 2, 1, 0),
            hull=3,
            edges=(0, 1, 1, 2, 2, 0),
        )


def test_attachment_request_rejects_vertices_not_referenced_by_triangles():
    with pytest.raises(ValueError, match=r"missing=\(3,\)"):
        build_request(triangles=(0, 1, 2))


def test_attachment_request_rejects_self_edges():
    with pytest.raises(ValueError, match="self-edge"):
        build_request(edges=(0, 1, 1, 1, 1, 2, 2, 3, 3, 0, 0, 2))


def test_attachment_request_rejects_duplicate_undirected_edges():
    with pytest.raises(ValueError, match="duplicate undirected pair"):
        build_request(edges=(0, 1, 1, 0, 1, 2, 2, 3, 3, 0, 0, 2))


def test_valid_quad_topology_preserves_spine_arrays():
    request = build_request()

    assert request.triangles == (0, 1, 2, 0, 2, 3)
    assert request.hull == 4
    assert request.edges == (0, 1, 1, 2, 2, 3, 3, 0, 0, 2)
    assert tuple(vertex.index for vertex in request.vertices) == (0, 1, 2, 3)
