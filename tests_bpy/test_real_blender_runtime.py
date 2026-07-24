"""Smoke tests proving the suite executes inside the official Blender Python module."""

from __future__ import annotations

from types import ModuleType
from unittest.mock import Mock

import bmesh
import bpy


def test_runtime_is_real_blender_520_not_a_mock(real_blender_runtime):
    assert real_blender_runtime is bpy
    assert isinstance(bpy, ModuleType)
    assert isinstance(bmesh, ModuleType)
    assert not isinstance(bpy, Mock)
    assert not isinstance(bmesh, Mock)
    assert tuple(int(value) for value in bpy.app.version[:3]) == (5, 2, 0)
    assert bpy.app.background is True
    assert callable(bpy.context.evaluated_depsgraph_get)
    assert callable(bmesh.new)


def test_bmesh_create_cube_roundtrip_frees_owned_bmesh(clean_blender_data):
    mesh = bpy.data.meshes.new("Spine2D_BMeshRoundtrip")
    bm = bmesh.new()
    try:
        bmesh.ops.create_cube(bm, size=2.0)
        assert len(bm.verts) == 8
        assert len(bm.edges) == 12
        assert len(bm.faces) == 6
        bm.to_mesh(mesh)
    finally:
        # This test owns the BMesh because it was created through bmesh.new().
        bm.free()

    mesh.update(calc_edges=True, calc_edges_loose=True)
    assert len(mesh.vertices) == 8
    assert len(mesh.edges) == 12
    assert len(mesh.polygons) == 6


def test_real_depsgraph_updates_linked_mesh(quad_object):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()

    evaluated = quad_object.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.to_mesh(
        preserve_all_data_layers=True,
        depsgraph=depsgraph,
    )
    try:
        assert evaluated_mesh is not None
        assert len(evaluated_mesh.vertices) == 4
        assert len(evaluated_mesh.edges) == 4
        assert len(evaluated_mesh.polygons) == 1
    finally:
        evaluated.to_mesh_clear()
