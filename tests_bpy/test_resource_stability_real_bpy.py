"""Fault-injection, user-map, repeated-operation, and blend round-trip regressions."""

from __future__ import annotations

from pathlib import Path

import bpy
import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import evaluated_mesh_reader
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.evaluated_mesh_reader import (
    EvaluatedMeshReadError,
    read_evaluated_mesh_snapshot,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.mesh_edge_attributes import (
    SHARP_EDGE_ATTRIBUTE,
    UV_SEAM_ATTRIBUTE,
    read_boolean_edge_attribute,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.mesh_reader import (
    read_source_mesh_snapshot,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import mesh_writer
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.mesh_writer import (
    MeshWriteError,
    temporary_mesh_object,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.mesh_uv_attributes import (
    read_uv_coordinates,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.uv_unwrap import (
    unwrap_snapshot_uv,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import UvUnwrapSettings


_ID_COLLECTIONS = (
    "objects",
    "meshes",
    "collections",
    "materials",
    "images",
    "node_groups",
    "actions",
    "cameras",
    "lights",
    "curves",
    "armatures",
)


def _id_key(value) -> tuple[str, str]:
    rna = getattr(getattr(value, "bl_rna", None), "identifier", type(value).__name__)
    name = str(getattr(value, "name_full", None) or getattr(value, "name", ""))
    return str(rna), name


def _datablock_signature() -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    result = []
    for collection_name in _ID_COLLECTIONS:
        collection = getattr(bpy.data, collection_name, None)
        if collection is not None:
            result.append(
                (collection_name, tuple(sorted(_id_key(item) for item in collection)))
            )
    return tuple(result)


def _user_map_signature() -> tuple[tuple[tuple[str, str], tuple[tuple[str, str], ...]], ...]:
    user_map = bpy.data.user_map()
    return tuple(
        sorted(
            (_id_key(datablock), tuple(sorted(_id_key(user) for user in users)))
            for datablock, users in user_map.items()
        )
    )


def _temporary_names() -> tuple[str, ...]:
    prefixes = ("__Spine2D_Eval_", "__Spine2D_UV")
    names = []
    for collection_name in ("objects", "meshes", "collections"):
        collection = getattr(bpy.data, collection_name)
        names.extend(
            item.name_full
            for item in collection
            if item.name_full.startswith(prefixes)
        )
    return tuple(sorted(names))


def test_evaluated_mesh_fault_after_to_mesh_clears_every_temporary_resource(
    quad_object,
    monkeypatch: pytest.MonkeyPatch,
):
    ids_before = _datablock_signature()
    users_before = _user_map_signature()

    def fail_after_to_mesh(**_kwargs):
        raise RuntimeError("forced snapshot-build failure")

    monkeypatch.setattr(
        evaluated_mesh_reader,
        "_build_snapshot_from_evaluated_mesh",
        fail_after_to_mesh,
    )

    with pytest.raises(EvaluatedMeshReadError, match="forced snapshot-build failure"):
        read_evaluated_mesh_snapshot(
            quad_object,
            depsgraph=bpy.context.evaluated_depsgraph_get(),
            scene=bpy.context.scene,
        )

    assert _temporary_names() == ()
    assert _datablock_signature() == ids_before
    assert _user_map_signature() == users_before


def test_materialization_fault_removes_object_mesh_collection_and_users(
    quad_object,
    monkeypatch: pytest.MonkeyPatch,
):
    snapshot = read_source_mesh_snapshot(quad_object)
    ids_before = _datablock_signature()
    users_before = _user_map_signature()

    def fail_uv_write(*_args, **_kwargs):
        raise RuntimeError("forced UV materialization failure")

    monkeypatch.setattr(mesh_writer, "_write_uv_layers", fail_uv_write)

    with pytest.raises(MeshWriteError, match="forced UV materialization failure"):
        with temporary_mesh_object(snapshot, scene=bpy.context.scene):
            pytest.fail("temporary object must not be yielded after setup failure")

    assert _temporary_names() == ()
    assert _datablock_signature() == ids_before
    assert _user_map_signature() == users_before


def test_repeated_evaluated_snapshot_and_uv_unwrap_have_no_datablock_growth(
    quad_object,
):
    source_snapshot = read_source_mesh_snapshot(quad_object)
    ids_before = _datablock_signature()
    users_before = _user_map_signature()

    for _iteration in range(25):
        result = read_evaluated_mesh_snapshot(
            quad_object,
            depsgraph=bpy.context.evaluated_depsgraph_get(),
            scene=bpy.context.scene,
        )
        assert len(result.snapshot.vertices) == 4
        assert _temporary_names() == ()
        assert _datablock_signature() == ids_before
        assert _user_map_signature() == users_before

    for _iteration in range(10):
        result = unwrap_snapshot_uv(
            source_snapshot,
            UvUnwrapSettings(),
            context=bpy.context,
            scene=bpy.context.scene,
        )
        assert result.snapshot.active_uv_layer == "SpineBakeUV"
        assert _temporary_names() == ()
        assert _datablock_signature() == ids_before
        assert _user_map_signature() == users_before


def test_blend_save_open_roundtrip_preserves_mesh_uv_attributes_and_properties(
    quad_object,
    tmp_path: Path,
):
    quad_object["spine2d_roundtrip"] = "Юнікод_日本語"
    mesh = quad_object.data
    active_uv = mesh.uv_layers.active
    assert active_uv is not None

    expected = {
        "vertices": tuple(tuple(float(value) for value in vertex.co) for vertex in mesh.vertices),
        "edges": tuple(tuple(int(value) for value in edge.vertices) for edge in mesh.edges),
        "faces": tuple(tuple(int(value) for value in polygon.vertices) for polygon in mesh.polygons),
        "uv": read_uv_coordinates(active_uv, expected_length=len(mesh.loops)),
        "seam": read_boolean_edge_attribute(mesh, UV_SEAM_ATTRIBUTE),
        "sharp": read_boolean_edge_attribute(mesh, SHARP_EDGE_ATTRIBUTE),
        "property": quad_object["spine2d_roundtrip"],
    }

    blend_path = tmp_path / "Spine2D_Юнікод_日本語.blend"
    result = bpy.ops.wm.save_as_mainfile(
        filepath=str(blend_path),
        check_existing=False,
    )
    assert "FINISHED" in result
    assert blend_path.is_file() and blend_path.stat().st_size > 0

    result = bpy.ops.wm.open_mainfile(filepath=str(blend_path), load_ui=False)
    assert "FINISHED" in result

    restored = bpy.data.objects.get("Spine2D_TestQuad")
    assert restored is not None and restored.type == "MESH"
    restored_mesh = restored.data
    restored_uv = restored_mesh.uv_layers.active
    assert restored_uv is not None
    actual = {
        "vertices": tuple(tuple(float(value) for value in vertex.co) for vertex in restored_mesh.vertices),
        "edges": tuple(tuple(int(value) for value in edge.vertices) for edge in restored_mesh.edges),
        "faces": tuple(tuple(int(value) for value in polygon.vertices) for polygon in restored_mesh.polygons),
        "uv": read_uv_coordinates(restored_uv, expected_length=len(restored_mesh.loops)),
        "seam": read_boolean_edge_attribute(restored_mesh, UV_SEAM_ATTRIBUTE),
        "sharp": read_boolean_edge_attribute(restored_mesh, SHARP_EDGE_ATTRIBUTE),
        "property": restored["spine2d_roundtrip"],
    }
    assert actual == expected
