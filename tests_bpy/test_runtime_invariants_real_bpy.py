"""Real Blender source fingerprints, context rollback, and operator-poll matrix."""

from __future__ import annotations

import bpy
import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.context_state import (
    BlenderContextState,
    activate_object_for_operator,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.evaluated_mesh_reader import (
    read_evaluated_mesh_snapshot,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.mesh_reader import (
    read_source_mesh_snapshot,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.mesh_uv_attributes import (
    read_uv_coordinates,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.uv_unwrap import (
    unwrap_snapshot_uv,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import UvUnwrapSettings
from Blender_to_Spine2D_Mesh_Exporter.single_object_operator import (
    OBJECT_OT_SaveUVAsJSON,
)
from Blender_to_Spine2D_Mesh_Exporter.ui import (
    OBJECT_OT_Spine2DMultiExport,
    OBJECT_OT_Spine2DSingleExport,
)


def _round_tuple(values, digits=9):
    return tuple(round(float(value), digits) for value in values)


def _material_fingerprint(material):
    if material is None:
        return None
    tree = material.node_tree
    nodes = ()
    links = ()
    if tree is not None:
        nodes = tuple(
            sorted(
                (
                    node.name,
                    node.bl_idname,
                    bool(node.mute),
                )
                for node in tree.nodes
            )
        )
        links = tuple(
            sorted(
                (
                    link.from_node.name,
                    link.from_socket.name,
                    link.to_node.name,
                    link.to_socket.name,
                )
                for link in tree.links
            )
        )
    return (
        material.name_full,
        bool(material.use_nodes),
        str(material.surface_render_method),
        nodes,
        links,
    )


def _object_fingerprint(obj):
    mesh = obj.data
    uv_layers = tuple(
        (
            layer.name,
            layer is mesh.uv_layers.active,
            bool(layer.active_render),
            read_uv_coordinates(layer, expected_length=len(mesh.loops)),
        )
        for layer in mesh.uv_layers
    )
    custom_properties = tuple(
        sorted(
            (key, repr(obj[key]))
            for key in obj.keys()
            if key != "_RNA_UI"
        )
    )
    return (
        obj.name_full,
        _round_tuple(value for row in obj.matrix_world for value in row),
        None if obj.parent is None else obj.parent.name_full,
        bool(obj.hide_viewport),
        bool(obj.hide_render),
        tuple(sorted(collection.name_full for collection in obj.users_collection)),
        custom_properties,
        tuple(
            (
                modifier.name,
                modifier.type,
                bool(modifier.show_viewport),
                bool(modifier.show_render),
            )
            for modifier in obj.modifiers
        ),
        tuple(
            (constraint.name, constraint.type, float(constraint.influence))
            for constraint in obj.constraints
        ),
        tuple(_round_tuple(vertex.co) for vertex in mesh.vertices),
        tuple(tuple(int(value) for value in edge.vertices) for edge in mesh.edges),
        tuple(
            tuple(int(value) for value in polygon.vertices)
            for polygon in mesh.polygons
        ),
        uv_layers,
        tuple(_material_fingerprint(material) for material in mesh.materials),
    )


def _context_signature():
    state = BlenderContextState.capture(bpy.context)
    return (
        None if state.active_object is None else state.active_object.name_full,
        tuple(obj.name_full for obj in state.selected_objects),
        state.active_mode,
        int(bpy.context.scene.frame_current),
    )


def _create_mesh_object(name: str):
    mesh = bpy.data.meshes.new(f"{name}Mesh")
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
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def test_source_fingerprint_survives_repeated_evaluated_and_uv_operations(quad_object):
    quad_object.location = (2.0, -3.0, 4.0)
    quad_object.rotation_euler = (0.1, 0.2, 0.3)
    quad_object.scale = (1.25, 0.75, 2.0)
    quad_object["spine2d_source_marker"] = "Юнікод_日本語"
    modifier = quad_object.modifiers.new("SourceDisplace", "DISPLACE")
    modifier.strength = 0.125
    modifier.mid_level = 0.0
    constraint = quad_object.constraints.new("LIMIT_LOCATION")
    constraint.name = "SourceLimit"
    constraint.influence = 0.35

    material = bpy.data.materials.new("SourceMaterial")
    material.use_nodes = True
    quad_object.data.materials.append(material)
    bpy.context.view_layer.update()

    source_snapshot = read_source_mesh_snapshot(quad_object)
    before = _object_fingerprint(quad_object)

    for _iteration in range(12):
        evaluated = read_evaluated_mesh_snapshot(
            quad_object,
            depsgraph=bpy.context.evaluated_depsgraph_get(),
            scene=bpy.context.scene,
        )
        assert len(evaluated.snapshot.vertices) >= len(source_snapshot.vertices)
        assert _object_fingerprint(quad_object) == before

    for _iteration in range(6):
        unwrapped = unwrap_snapshot_uv(
            source_snapshot,
            UvUnwrapSettings(),
            context=bpy.context,
            scene=bpy.context.scene,
        )
        assert unwrapped.snapshot.active_uv_layer == "SpineBakeUV"
        assert _object_fingerprint(quad_object) == before


def test_activate_object_failure_restores_edit_mode_selection_frame_and_active_object(
    quad_object,
):
    target = _create_mesh_object("TemporaryTarget")
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    quad_object.select_set(True)
    bpy.context.view_layer.objects.active = quad_object
    bpy.context.scene.frame_set(27)
    result = bpy.ops.object.mode_set(mode="EDIT")
    assert "FINISHED" in result
    before = _context_signature()

    with pytest.raises(RuntimeError, match="forced operator-body failure"):
        with activate_object_for_operator(target, context=bpy.context):
            assert bpy.context.view_layer.objects.active is target
            assert tuple(bpy.context.selected_objects) == (target,)
            assert target.mode == "OBJECT"
            bpy.context.scene.frame_set(99)
            raise RuntimeError("forced operator-body failure")

    # Context ownership restores selection/mode; timeline ownership belongs to bake state.
    after = _context_signature()
    assert after[:3] == before[:3]
    assert after[3] == 99
    assert quad_object.mode == "EDIT"


def test_real_operator_poll_matrix_rejects_invalid_contexts_and_accepts_meshes(
    clean_blender_data,
):
    mesh_obj = _create_mesh_object("PollMesh")
    camera_data = bpy.data.cameras.new("PollCameraData")
    camera_obj = bpy.data.objects.new("PollCamera", camera_data)
    bpy.context.scene.collection.objects.link(camera_obj)

    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    bpy.context.view_layer.objects.active = None
    assert not OBJECT_OT_SaveUVAsJSON.poll(bpy.context)
    assert not OBJECT_OT_Spine2DSingleExport.poll(bpy.context)
    assert not OBJECT_OT_Spine2DMultiExport.poll(bpy.context)

    camera_obj.select_set(True)
    bpy.context.view_layer.objects.active = camera_obj
    assert not OBJECT_OT_SaveUVAsJSON.poll(bpy.context)
    assert not OBJECT_OT_Spine2DSingleExport.poll(bpy.context)
    assert not OBJECT_OT_Spine2DMultiExport.poll(bpy.context)

    mesh_obj.select_set(True)
    assert OBJECT_OT_Spine2DMultiExport.poll(bpy.context)
    bpy.context.view_layer.objects.active = mesh_obj
    assert OBJECT_OT_SaveUVAsJSON.poll(bpy.context)
    assert OBJECT_OT_Spine2DSingleExport.poll(bpy.context)
