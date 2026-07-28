"""Shared datablock ownership must survive temporary bake copies."""

from __future__ import annotations

import bpy
import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import bake_materials
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.bake_materials import (
    BakeMaterialError,
    temporary_bake_materials,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.mesh_reader import (
    read_source_mesh_snapshot,
)


def _quad(name: str):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")

    resolved_name = name.strip()
    mesh = bpy.data.meshes.new(f"{resolved_name}Mesh")
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

    obj = bpy.data.objects.new(resolved_name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _shared_material_graph():
    image = bpy.data.images.new(
        "SharedImage",
        width=4,
        height=4,
        alpha=True,
    )
    group = bpy.data.node_groups.new(
        "SharedShaderGroup",
        "ShaderNodeTree",
    )
    material = bpy.data.materials.new("SharedMaterial")
    material.use_nodes = True

    image_node = material.node_tree.nodes.new("ShaderNodeTexImage")
    image_node.image = image

    group_node = material.node_tree.nodes.new("ShaderNodeGroup")
    group_node.node_tree = group

    return material, image, group


def _target_snapshot(target):
    if target is None:
        raise TypeError("target cannot be None")
    if getattr(target, "type", None) != "MESH":
        raise TypeError("target must be a Blender MESH object")

    return read_source_mesh_snapshot(
        target,
        snapshot_id=f"{target.name_full}:temporary-material-target",
        source_object_id=target.name_full,
    )


def _temporary_material_names() -> tuple[str, ...]:
    return tuple(
        sorted(
            material.name_full
            for material in bpy.data.materials
            if material.name_full.startswith(
                (
                    "__Spine2D_Bake_",
                    "__Spine2D_EmptySlot_",
                )
            )
        )
    )


def test_shared_material_image_and_group_users_return_after_body_failure(
    clean_blender_data,
):
    source = _quad("SharedSource")
    peer = _quad("SharedPeer")
    target = _quad("SharedTarget")
    target_snapshot = _target_snapshot(target)

    material, image, group = _shared_material_graph()
    source.data.materials.append(material)
    peer.data.materials.append(material)
    baseline = (material.users, image.users, group.users)

    with pytest.raises(RuntimeError, match="forced body failure"):
        with temporary_bake_materials(
            source,
            target,
            target_snapshot=target_snapshot,
            used_material_indices=(0,),
            face_material_indices=(0,),
        ) as prepared:
            assert prepared.materials[0] is not material
            raise RuntimeError("forced body failure")

    assert tuple(target.data.materials) == ()
    assert source.data.materials[0] is material
    assert peer.data.materials[0] is material
    assert (material.users, image.users, group.users) == baseline
    assert _temporary_material_names() == ()


def test_setup_failure_removes_partial_copies_without_touching_shared_users(
    clean_blender_data,
    monkeypatch,
):
    source = _quad("FailureSource")
    peer = _quad("FailurePeer")
    target = _quad("FailureTarget")
    target_snapshot = _target_snapshot(target)

    material, image, group = _shared_material_graph()
    source.data.materials.append(material)
    peer.data.materials.append(material)
    baseline = (material.users, image.users, group.users)

    def _raise_active_node_failure(*_args, **_kwargs):
        raise BakeMaterialError("forced active-node failure")

    monkeypatch.setattr(
        bake_materials,
        "_create_active_bake_node",
        _raise_active_node_failure,
    )

    with pytest.raises(BakeMaterialError, match="forced active-node failure"):
        with temporary_bake_materials(
            source,
            target,
            target_snapshot=target_snapshot,
            used_material_indices=(0,),
            face_material_indices=(0,),
        ):
            pytest.fail("setup failure must happen before yield")

    assert tuple(target.data.materials) == ()
    assert (material.users, image.users, group.users) == baseline
    assert _temporary_material_names() == ()


def test_shared_datablocks_survive_repeated_temporary_material_cycles(
    clean_blender_data,
):
    source = _quad("RepeatSource")
    peer = _quad("RepeatPeer")
    target = _quad("RepeatTarget")
    target_snapshot = _target_snapshot(target)

    material, image, group = _shared_material_graph()
    source.data.materials.append(material)
    peer.data.materials.append(material)
    baseline = (material.users, image.users, group.users)

    for _index in range(20):
        with temporary_bake_materials(
            source,
            target,
            target_snapshot=target_snapshot,
            used_material_indices=(0,),
            face_material_indices=(0,),
        ):
            pass

        assert tuple(target.data.materials) == ()
        assert (material.users, image.users, group.users) == baseline
        assert _temporary_material_names() == ()

    bpy.data.orphans_purge(do_recursive=True)
    assert bpy.data.materials.get(material.name_full) is material
    assert bpy.data.images.get(image.name_full) is image
    assert bpy.data.node_groups.get(group.name_full) is group


def test_shared_mesh_and_fake_user_packed_assets_survive_cleanup(
    clean_blender_data,
    tmp_path,
):
    source = _quad("SharedMeshSource")
    peer = bpy.data.objects.new("SharedMeshPeer", source.data)
    bpy.context.scene.collection.objects.link(peer)
    target = _quad("SharedMeshTarget")
    target_snapshot = _target_snapshot(target)

    material, image, group = _shared_material_graph()
    material.use_fake_user = True
    image.use_fake_user = True
    group.use_fake_user = True
    image.generated_color = (0.2, 0.4, 0.6, 0.8)
    image.filepath_raw = str(tmp_path / "packed-shared-image.png")
    image.file_format = "PNG"
    image.save()
    image.pack()
    source.data.materials.append(material)

    baseline = (
        source.data.users,
        material.users,
        image.users,
        group.users,
        image.packed_file.size,
    )

    for _index in range(12):
        with temporary_bake_materials(
            source,
            target,
            target_snapshot=target_snapshot,
            used_material_indices=(0,),
            face_material_indices=(0,),
        ):
            pass

        assert source.data is peer.data
        assert (
            source.data.users,
            material.users,
            image.users,
            group.users,
            image.packed_file.size,
        ) == baseline
        assert _temporary_material_names() == ()

    bpy.data.orphans_purge(do_recursive=True)
    assert bpy.data.meshes.get(source.data.name_full) is source.data
    assert material.use_fake_user
    assert image.use_fake_user
    assert group.use_fake_user


def test_linked_library_material_is_copied_locally_but_never_deleted(
    clean_blender_data,
    tmp_path,
):
    original = bpy.data.materials.new("LibraryOwnedMaterial")
    original.use_nodes = True
    library_path = tmp_path / "shared-material-library.blend"
    bpy.data.libraries.write(str(library_path), {original})
    bpy.data.materials.remove(original)

    with bpy.data.libraries.load(str(library_path), link=True) as (
        data_from,
        data_to,
    ):
        assert "LibraryOwnedMaterial" in data_from.materials
        data_to.materials = ["LibraryOwnedMaterial"]

    linked = data_to.materials[0]
    source = _quad("LinkedSource")
    target = _quad("LinkedTarget")
    target_snapshot = _target_snapshot(target)
    source.data.materials.append(linked)
    baseline = linked.users

    with temporary_bake_materials(
        source,
        target,
        target_snapshot=target_snapshot,
        used_material_indices=(0,),
        face_material_indices=(0,),
    ) as prepared:
        assert prepared.materials[0] is not linked
        assert prepared.materials[0].library is None

    assert source.data.materials[0] is linked
    assert linked.users == baseline
    assert linked.library is not None
    assert _temporary_material_names() == ()
