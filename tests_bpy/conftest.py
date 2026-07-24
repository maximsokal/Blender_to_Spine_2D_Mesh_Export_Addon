"""Fixtures for tests that run against the official Blender 5.2 bpy wheel.

This directory deliberately lives outside ``tests/`` because the legacy test conftest
installs global MagicMock replacements for bpy, bmesh, and mathutils.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest


bpy = pytest.importorskip(
    "bpy",
    reason="official bpy 5.2.0 wheel is required for tests_bpy",
)
bmesh = pytest.importorskip(
    "bmesh",
    reason="real bmesh module from bpy 5.2.0 is required for tests_bpy",
)

EXPECTED_BLENDER_VERSION = (5, 2, 0)


def _require_real_blender_runtime() -> None:
    """Fail closed when mocks or an unintended Blender runtime were imported."""

    if isinstance(bpy, Mock) or isinstance(bmesh, Mock):
        raise RuntimeError("tests_bpy imported mocked Blender modules")
    if not callable(getattr(bmesh, "new", None)):
        raise RuntimeError("real bmesh.new() is unavailable")
    runtime_version = tuple(int(value) for value in bpy.app.version[:3])
    if runtime_version != EXPECTED_BLENDER_VERSION:
        raise RuntimeError(
            f"tests_bpy requires Blender {EXPECTED_BLENDER_VERSION}, got {runtime_version}"
        )


def _remove_all(collection, *, do_unlink: bool = False) -> None:
    """Remove a stable copy so Blender collection mutation cannot skip entries."""

    for datablock in tuple(collection):
        if do_unlink:
            collection.remove(datablock, do_unlink=True)
        elif int(getattr(datablock, "users", 0) or 0) == 0:
            collection.remove(datablock)


def _reset_blender_data() -> None:
    """Return the bpy module to a deterministic empty-data state."""

    _remove_all(bpy.data.objects, do_unlink=True)
    _remove_all(bpy.data.collections, do_unlink=True)

    # Remove orphaned data after objects and collections have released their users.
    for collection_name in (
        "meshes",
        "materials",
        "images",
        "cameras",
        "lights",
        "curves",
        "armatures",
        "node_groups",
    ):
        collection = getattr(bpy.data, collection_name, None)
        if collection is not None:
            _remove_all(collection)

    scene = bpy.context.scene
    scene.frame_set(0)
    for candidate in tuple(scene.collection.objects):
        scene.collection.objects.unlink(candidate)


@pytest.fixture(scope="session", autouse=True)
def real_blender_runtime():
    """Prove once that this suite owns a real, exact Blender 5.2 runtime."""

    _require_real_blender_runtime()
    yield bpy


@pytest.fixture(autouse=True)
def clean_blender_data(real_blender_runtime):
    """Isolate every test and surface leaked temporary datablocks."""

    _reset_blender_data()
    yield real_blender_runtime
    _reset_blender_data()


@pytest.fixture
def quad_object(clean_blender_data):
    """Create one real quad with UV, seam, and sharp-edge attributes."""

    from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.mesh_edge_attributes import (
        SHARP_EDGE_ATTRIBUTE,
        UV_SEAM_ATTRIBUTE,
        write_boolean_edge_attribute,
    )
    from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.mesh_uv_attributes import (
        write_uv_coordinate,
    )

    mesh = bpy.data.meshes.new("Spine2D_TestQuadMesh")
    mesh.from_pydata(
        (
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
        ),
        (),
        ((0, 1, 2, 3),),
    )
    mesh.validate(clean_customdata=False)
    mesh.update(calc_edges=True, calc_edges_loose=True)

    uv_layer = mesh.uv_layers.new(name="UVMap")
    mesh.uv_layers.active = uv_layer
    uv_layer.active_render = True
    coordinates = (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    )
    for loop_index, coordinate in enumerate(coordinates):
        write_uv_coordinate(
            uv_layer,
            loop_index,
            coordinate,
            expected_length=len(mesh.loops),
        )

    seam_values = tuple(
        set(int(value) for value in edge.vertices) == {0, 1}
        for edge in mesh.edges
    )
    sharp_values = tuple(
        set(int(value) for value in edge.vertices) == {1, 2}
        for edge in mesh.edges
    )
    write_boolean_edge_attribute(mesh, UV_SEAM_ATTRIBUTE, seam_values)
    write_boolean_edge_attribute(mesh, SHARP_EDGE_ATTRIBUTE, sharp_values)

    obj = bpy.data.objects.new("Spine2D_TestQuad", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.context.view_layer.update()
    return obj
