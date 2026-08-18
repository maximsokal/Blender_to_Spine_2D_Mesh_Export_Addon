"""Prove production export operators do not pollute Blender's undo history."""

from __future__ import annotations

import hashlib

import bpy
import pytest

import Blender_to_Spine2D_Mesh_Exporter as addon
from Blender_to_Spine2D_Mesh_Exporter import single_object_operator


def _create_source(output_root):
    mesh = bpy.data.meshes.new("UndoMesh")
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
    mesh.update(calc_edges=True)

    material = bpy.data.materials.new("UndoMaterial")
    material.diffuse_color = (0.2, 0.7, 0.3, 1.0)
    mesh.materials.append(material)

    obj = bpy.data.objects.new("UndoHero", mesh)
    bpy.context.scene.collection.objects.link(obj)
    for candidate in bpy.context.view_layer.objects:
        candidate.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    scene = bpy.context.scene
    scene.spine2d_json_path = str(output_root)
    scene.spine2d_images_path = "images"
    scene.spine2d_texture_size = 64
    scene.spine2d_angle_limit = 30
    scene.spine2d_seam_maker_mode = "AUTO"
    scene.spine2d_control_icons = False
    scene.spine2d_export_preview_animation = False
    scene.spine2d_frames_for_render = 0
    scene.spine2d_bake_frame_start = 0

    camera_data = bpy.data.cameras.new("UndoCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 5.0
    camera = bpy.data.objects.new("UndoCamera", camera_data)
    camera.location = (0.0, 0.0, 5.0)
    scene.collection.objects.link(camera)
    scene.camera = camera

    light_data = bpy.data.lights.new("UndoLight", "AREA")
    light_data.energy = 1000.0
    light_data.size = 5.0
    light = bpy.data.objects.new("UndoLight", light_data)
    light.location = (0.0, 0.0, 4.0)
    scene.collection.objects.link(light)
    return obj


def _temporary_names():
    prefixes = ("__Spine2D_", ".spine2d")
    values = []
    for collection_name in (
        "objects",
        "meshes",
        "materials",
        "images",
        "collections",
        "node_groups",
    ):
        collection = getattr(bpy.data, collection_name)
        values.extend(
            item.name_full
            for item in collection
            if item.name_full.startswith(prefixes)
        )
    return tuple(sorted(values))


def test_successful_export_does_not_push_temporary_scene_state_to_undo(tmp_path):
    addon.register()
    try:
        bpy.context.preferences.edit.use_global_undo = True
        obj = _create_source(tmp_path)

        assert "FINISHED" in bpy.ops.ed.undo_push(message="Spine2D initial")
        obj.location.x = 1.0
        bpy.context.view_layer.update()
        assert "FINISHED" in bpy.ops.ed.undo_push(message="Spine2D moved")

        result = set(bpy.ops.object.save_uv_as_json())
        assert "FINISHED" in result
        json_path = next(tmp_path.glob("*.json"))
        digest = hashlib.sha256(json_path.read_bytes()).hexdigest()
        assert bpy.data.objects["UndoHero"].location.x == pytest.approx(1.0)
        assert _temporary_names() == ()

        assert "FINISHED" in bpy.ops.ed.undo()
        restored = bpy.data.objects["UndoHero"]
        assert restored.location.x == pytest.approx(0.0)
        assert _temporary_names() == ()

        assert "FINISHED" in bpy.ops.ed.redo()
        restored = bpy.data.objects["UndoHero"]
        assert restored.location.x == pytest.approx(1.0)
        assert _temporary_names() == ()

        for candidate in bpy.context.view_layer.objects:
            candidate.select_set(False)
        restored.select_set(True)
        bpy.context.view_layer.objects.active = restored

        assert "FINISHED" in bpy.ops.object.save_uv_as_json()
        assert hashlib.sha256(json_path.read_bytes()).hexdigest() == digest
        assert _temporary_names() == ()
    finally:
        addon.unregister()


def test_cancelled_operator_does_not_consume_undo_step(tmp_path, monkeypatch):
    addon.register()
    try:
        bpy.context.preferences.edit.use_global_undo = True
        obj = _create_source(tmp_path)

        assert "FINISHED" in bpy.ops.ed.undo_push(message="Spine2D initial failure")
        obj.location.x = 2.0
        bpy.context.view_layer.update()
        assert "FINISHED" in bpy.ops.ed.undo_push(message="Spine2D moved failure")

        def fail_export(_context):
            raise RuntimeError("forced production operator failure")

        monkeypatch.setattr(
            single_object_operator,
            "export_active_object_a1",
            fail_export,
        )

        result = set(bpy.ops.object.save_uv_as_json())
        assert "CANCELLED" in result

        assert "FINISHED" in bpy.ops.ed.undo()
        assert bpy.data.objects["UndoHero"].location.x == pytest.approx(0.0)
        assert not tuple(tmp_path.rglob("*.json"))
        assert not tuple(tmp_path.rglob("*.png"))
        assert _temporary_names() == ()
    finally:
        addon.unregister()
