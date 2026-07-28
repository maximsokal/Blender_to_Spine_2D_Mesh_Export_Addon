from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (
    MeshReadError,
    read_source_mesh_snapshot,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    MeshSnapshotValidator,
    SourceLoopId,
)


class FakeUvLayers(list):
    def __init__(self, layers, active=None):
        super().__init__(layers)
        self.active = active


class FakeAttributes:
    def __init__(self, values):
        self._values = dict(values)

    def get(self, name):
        return self._values.get(name)


def _boolean_edge_attribute(values):
    return SimpleNamespace(
        domain="EDGE",
        data_type="BOOLEAN",
        data=[SimpleNamespace(value=value) for value in values],
    )


class FakeMatrix:
    def __getitem__(self, row):
        return (
            (1.0, 0.0, 0.0, 2.0),
            (0.0, 1.0, 0.0, 3.0),
            (0.0, 0.0, 1.0, 4.0),
            (0.0, 0.0, 0.0, 1.0),
        )[row]


def _uv_layer(name, coordinates, *, active_render=False):
    return SimpleNamespace(
        name=name,
        active_render=active_render,
        uv=[SimpleNamespace(vector=coordinate) for coordinate in coordinates],
    )


def make_fake_quad():
    vertices = [
        SimpleNamespace(index=0, co=(0, 0, 0), normal=(0, 0, 1)),
        SimpleNamespace(index=1, co=(1, 0, 0), normal=(0, 0, 1)),
        SimpleNamespace(index=2, co=(1, 1, 0), normal=(0, 0, 1)),
        SimpleNamespace(index=3, co=(0, 1, 0), normal=(0, 0, 1)),
    ]
    edges = [
        SimpleNamespace(
            index=0,
            vertices=(0, 1),
            use_seam=False,
            use_edge_sharp=False,
        ),
        SimpleNamespace(
            index=1,
            vertices=(1, 2),
            use_seam=True,
            use_edge_sharp=False,
        ),
        SimpleNamespace(
            index=2,
            vertices=(2, 3),
            use_seam=False,
            use_edge_sharp=True,
        ),
        SimpleNamespace(
            index=3,
            vertices=(3, 0),
            use_seam=False,
            use_edge_sharp=False,
        ),
    ]
    loops = [
        SimpleNamespace(index=0, vertex_index=0, edge_index=0),
        SimpleNamespace(index=1, vertex_index=1, edge_index=1),
        SimpleNamespace(index=2, vertex_index=2, edge_index=2),
        SimpleNamespace(index=3, vertex_index=3, edge_index=3),
    ]
    polygons = [
        SimpleNamespace(
            index=0,
            loop_start=0,
            loop_total=4,
            material_index=2,
            normal=(0, 0, 1),
            use_smooth=True,
        )
    ]
    uv_layer = _uv_layer(
        "UVMap",
        ((0, 0), (1, 0), (1, 1), (0, 1)),
        active_render=True,
    )
    mesh = SimpleNamespace(
        vertices=vertices,
        edges=edges,
        loops=loops,
        polygons=polygons,
        uv_layers=FakeUvLayers([uv_layer], active=uv_layer),
        attributes=FakeAttributes(
            {
                "uv_seam": _boolean_edge_attribute((False, True, False, False)),
                "sharp_edge": _boolean_edge_attribute((False, False, True, False)),
            }
        ),
    )
    return SimpleNamespace(
        type="MESH",
        name="Quad",
        name_full="Collection/Quad",
        data=mesh,
        matrix_world=FakeMatrix(),
    )


def test_read_source_mesh_snapshot_preserves_face_corner_identity():
    snapshot = read_source_mesh_snapshot(
        make_fake_quad(), source_object_id="source-123"
    )

    MeshSnapshotValidator().validate_or_raise(snapshot)
    assert snapshot.snapshot_id == "source-123:source"
    assert snapshot.object_name == "Collection/Quad"
    assert snapshot.active_uv_layer == "UVMap"
    assert snapshot.render_uv_layer == "UVMap"
    assert snapshot.world_matrix[3] == 2.0
    assert snapshot.world_matrix[7] == 3.0
    assert snapshot.world_matrix[11] == 4.0
    assert tuple(loop.source_id for loop in snapshot.loops) == (
        SourceLoopId("source-123", 0, 0),
        SourceLoopId("source-123", 0, 1),
        SourceLoopId("source-123", 0, 2),
        SourceLoopId("source-123", 0, 3),
    )
    assert snapshot.edges[1].seam is True
    assert snapshot.edges[2].sharp is True
    assert snapshot.faces[0].material_index == 2


def test_active_uv_layer_overrides_stale_active_render_flag():
    obj = make_fake_quad()
    uv_map = obj.data.uv_layers[0]
    source_uv = _uv_layer(
        "SourceUV",
        ((0.25, 0.5), (0.25, 0.5), (0.25, 0.5), (0.25, 0.5)),
        active_render=False,
    )
    obj.data.uv_layers.append(source_uv)
    obj.data.uv_layers.active = source_uv
    uv_map.active_render = True

    snapshot = read_source_mesh_snapshot(obj)

    assert snapshot.uv_layer_names == ("UVMap", "SourceUV")
    assert snapshot.active_uv_layer == "SourceUV"
    assert snapshot.render_uv_layer == "SourceUV"


def test_reader_rejects_non_mesh_and_missing_uv_layer():
    with pytest.raises(MeshReadError):
        read_source_mesh_snapshot(SimpleNamespace(type="LIGHT", data=None))

    obj = make_fake_quad()
    with pytest.raises(MeshReadError, match="Requested UV layers are missing"):
        read_source_mesh_snapshot(obj, uv_layer_names=("Missing",))
