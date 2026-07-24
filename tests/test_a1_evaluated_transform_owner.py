from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_source_geometry_preparation import (
    _evaluated_source_world_matrix,
)


class _SourceObject:
    def __init__(self, evaluated):
        self._evaluated = evaluated
        self.received_depsgraph = None

    def evaluated_get(self, depsgraph):
        self.received_depsgraph = depsgraph
        return self._evaluated


def test_evaluated_source_world_matrix_uses_same_dependency_graph_proxy():
    original_matrix = (
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 2.0),
        (0.0, 0.0, 1.0, 3.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    evaluated_matrix = (
        (0.0, -2.0, 0.0, 10.0),
        (3.0, 0.0, 0.0, 20.0),
        (0.0, 0.0, 4.0, 30.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    depsgraph = object()
    source = _SourceObject(
        SimpleNamespace(matrix_world=evaluated_matrix),
    )
    source.matrix_world = original_matrix

    result = _evaluated_source_world_matrix(source, depsgraph)

    assert source.received_depsgraph is depsgraph
    assert result == (
        0.0,
        -2.0,
        0.0,
        10.0,
        3.0,
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
    assert result != tuple(value for row in original_matrix for value in row)


def test_evaluated_source_world_matrix_rejects_missing_evaluated_proxy():
    source = _SourceObject(None)

    with pytest.raises(ValueError, match="returned None"):
        _evaluated_source_world_matrix(source, object())


def test_evaluated_source_world_matrix_rejects_missing_matrix():
    source = _SourceObject(SimpleNamespace())

    with pytest.raises(ValueError, match="no matrix_world"):
        _evaluated_source_world_matrix(source, object())
