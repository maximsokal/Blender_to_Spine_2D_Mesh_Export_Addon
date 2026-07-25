"""Shared Material/Image/NodeGroup ownership must survive temporary bake copies."""

from __future__ import annotations

import bpy
import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import bake_materials
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.bake_materials import (
    BakeMaterialError,
    temporary_bake_materials,
)


def _quad(name: str):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)),
        (),
        ((0, 1, 2, 3),),
    )
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _shared_material_graph():
    image = bpy.data.images.new("SharedImage", width=4, height=4, alpha=True)
    group = bpy.data.node_groups.new("SharedShaderGroup", "ShaderNodeTree")
    material = bpy.data.materials.new("SharedMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    image_node = nodes.new("ShaderNodeTexImage")
    image_node.image = image
    group_node = nodes.new("ShaderNodeGroup")
    group_node.node_tree = group
    return material, image, group


def _temporary_material_names():
    return tuple(
        sorted(
            material.name_full
            for material in bpy.data.materials
            if material.name_full.startswith(("__Spine2D_Bake_", "__Spine2D_EmptySlot_"))
        )
    )


def test_shared_material_image_and_group_users_return_after_body_failure(clean_blender_data):
    source = _quad("SharedSource")
    peer = _quad("SharedPeer")
    target = _quad("SharedTarget")
    material, image, group = _shared_material_graph()
    source.data.materials.append(material)
    peer.data.materials.append(material)

    baseline = (material.users, image.users, group.users)
    assert baseline[0] == 2

    with pytest.raises(RuntimeError, match="forced body failure"):
        with temporary_bake_materials(
            source,
            target,
            used_material_indices=(0,),
            face_material_indices=(0,),
        ) as prepared:
            assert prepared.materials[0] is not material
            assert material.users == baseline[0]
            assert image.users >= baseline[1]
            assert group.users >= baseline[2]
            raise RuntimeError("forced body failure")

    assert tuple(target.data.materials) == ()
    assert source.data.materials[0] is material
    assert peer.data.materials[0] is material
    assert (material.users, image.users, group.users) == baseline
    assert _temporary_material_names() == ()
    assert bpy.data.images.get(image.name_full) is image
    assert bpy.data.node_groups.get(group.name_full) is group


def test_setup_failure_removes_partial_copies_without_touching_shared_users(
    clean_blender_data,
    monkeypatch: pytest.MonkeyPatch,
):
    source = _quad("FailureSource")
    peer = _quad("FailurePeer")
    target = _quad("FailureTarget")
    material, image, group = _shared_material_graph()
    source.data.materials.append(material)
    peer.data.materials.append(material)
    baseline = (material.users, image.users, group.users)

    def _fail_node(*_args, **_kwargs):
        raise BakeMaterialError("forced active-node failure")

    monkeypatch.setattr(bake_materials, "_create_active_bake_node", _fail_node)

    with pytest.raises(BakeMaterialError, match="forced active-node failure"):
        with temporary_bake_materials(
            source,
            target,
            used_material_indices=(0,),
            face_material_indices=(0,),
        ):
            pytest.fail("setup failure must happen before yield")

    assert tuple(target.data.materials) == ()
    assert source.data.materials[0] is material
    assert peer.data.materials[0] is material
    assert (material.users, image.users, group.users) == baseline
    assert _temporary_material_names() == ()


def test_shared_datablocks_survive_repeated_temporary_material_cycles(clean_blender_data):
    source = _quad("RepeatSource")
    peer = _quad("RepeatPeer")
    target = _quad("RepeatTarget")
    material, image, group = _shared_material_graph()
    source.data.materials.append(material)
    peer.data.materials.append(material)
    baseline = (material.users, image.users, group.users)

    for _index in range(20):
        with temporary_bake_materials(
            source,
            target,
            used_material_indices=(0,),
            face_material_indices=(0,),
        ) as prepared:
            assert len(prepared.materials) == 1
            assert len(prepared.image_nodes) == 1
        assert tuple(target.data.materials) == ()
        assert (material.users, image.users, group.users) == baseline
        assert _temporary_material_names() == ()

    bpy.data.orphans_purge(do_recursive=True)
    assert source.data.materials[0] is material
    assert peer.data.materials[0] is material
    assert bpy.data.materials.get(material.name_full) is material
    assert bpy.data.images.get(image.name_full) is image
    assert bpy.data.node_groups.get(group.name_full) is group
    assert (material.users, image.users, group.users) == baseline
