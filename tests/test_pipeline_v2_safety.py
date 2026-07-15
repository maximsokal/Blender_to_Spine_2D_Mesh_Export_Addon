from unittest.mock import MagicMock, patch

import pytest

from Blender_to_Spine2D_Mesh_Exporter import blender_context, main, pipeline_v2


def test_pipeline_is_installed_behind_legacy_main_api():
    assert main.save_uv_as_json is pipeline_v2.save_uv_as_json
    assert main.collect_vertices is pipeline_v2.collect_vertices
    assert main.triangulate_mesh is pipeline_v2.triangulate_mesh


def test_managed_bmesh_writes_back_and_frees_exactly_once():
    mesh = MagicMock()
    bm = MagicMock()

    with patch.object(blender_context.bmesh, "new", return_value=bm):
        with blender_context.managed_bmesh(mesh, write_back=True) as yielded:
            assert yielded is bm

    bm.from_mesh.assert_called_once_with(mesh)
    bm.to_mesh.assert_called_once_with(mesh)
    bm.free.assert_called_once_with()
    mesh.update.assert_called_once_with()


def test_managed_bmesh_frees_on_exception_without_partial_writeback():
    mesh = MagicMock()
    bm = MagicMock()

    with patch.object(blender_context.bmesh, "new", return_value=bm):
        with pytest.raises(RuntimeError):
            with blender_context.managed_bmesh(mesh, write_back=True):
                raise RuntimeError("boom")

    bm.to_mesh.assert_not_called()
    bm.free.assert_called_once_with()


def test_triangulate_mesh_transfers_ownership_to_caller():
    obj = MagicMock(type="MESH")
    obj.name = "Mesh"
    bm = MagicMock()
    bm.faces = [MagicMock()]

    with patch.object(pipeline_v2.bmesh, "new", return_value=bm), patch.object(
        pipeline_v2.bmesh.ops, "triangulate"
    ) as triangulate:
        result = pipeline_v2.triangulate_mesh(obj)

    assert result is bm
    bm.from_mesh.assert_called_once_with(obj.data)
    triangulate.assert_called_once()
    bm.free.assert_not_called()


def test_triangulate_mesh_frees_when_triangulation_fails():
    obj = MagicMock(type="MESH")
    obj.name = "Mesh"
    bm = MagicMock()
    bm.faces = [MagicMock()]

    with patch.object(pipeline_v2.bmesh, "new", return_value=bm), patch.object(
        pipeline_v2.bmesh.ops,
        "triangulate",
        side_effect=RuntimeError("invalid topology"),
    ):
        with pytest.raises(RuntimeError):
            pipeline_v2.triangulate_mesh(obj)

    bm.free.assert_called_once_with()


def test_collect_vertices_does_not_free_borrowed_bmesh():
    bm = MagicMock()
    bm.loops.layers.uv.active = None

    with pytest.raises(pipeline_v2.ExportPipelineError):
        pipeline_v2.collect_vertices(bm, "Mesh")

    bm.free.assert_not_called()


def test_get_texture_dimensions_uses_defaults_for_missing_material():
    obj = MagicMock()
    obj.active_material = None
    assert pipeline_v2.get_texture_dimensions(obj, 256, 128) == (256, 128)
