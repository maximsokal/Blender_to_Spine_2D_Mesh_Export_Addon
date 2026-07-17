from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    EdgeId,
    MeshEdge,
    VertexId,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.disk_region import (
    _edge_set_is_open_path,
)


def _edge(index, first, second):
    return MeshEdge(
        id=EdgeId(index),
        source_id=None,
        vertex_ids=(VertexId(first), VertexId(second)),
    )


def test_distinct_edge_chain_is_an_open_path():
    edge_map = {
        EdgeId(0): _edge(0, 0, 1),
        EdgeId(1): _edge(1, 1, 2),
        EdgeId(2): _edge(2, 2, 3),
    }

    assert _edge_set_is_open_path(tuple(edge_map), edge_map)


def test_parallel_edges_form_two_edge_cycle_not_open_path():
    edge_map = {
        EdgeId(0): _edge(0, 0, 1),
        EdgeId(1): _edge(1, 0, 1),
    }

    assert not _edge_set_is_open_path(tuple(edge_map), edge_map)


def test_repeated_edge_id_is_rejected():
    edge_map = {EdgeId(0): _edge(0, 0, 1)}

    assert not _edge_set_is_open_path((EdgeId(0), EdgeId(0)), edge_map)


def test_disconnected_edge_set_is_not_open_path():
    edge_map = {
        EdgeId(0): _edge(0, 0, 1),
        EdgeId(1): _edge(1, 2, 3),
    }

    assert not _edge_set_is_open_path(tuple(edge_map), edge_map)
