#!/usr/bin/env python3
"""Create small deterministic public .blend fixtures for Rewrite regression tests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import bpy


FIXTURE_NAMES = (
    "procedural_noise",
    "nested_node_groups",
    "overlapping_uv",
    "non_manifold",
    "negative_scale_modifier",
)


def _arguments(argv: Sequence[str]) -> list[str]:
    """Return script arguments passed after Blender's ``--`` separator."""

    try:
        return list(argv[argv.index("--") + 1 :])
    except ValueError:
        return list(argv[1:])


def _reset() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _material(name: str, color: tuple[float, float, float, float]):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is None:
        raise RuntimeError(f"material {name!r} has no Principled BSDF node")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = 0.6
    return material


def _object_from_pydata(name, vertices, faces):
    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(vertices, (), faces)
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _quad(name: str = "Hero"):
    obj = _object_from_pydata(
        name,
        (
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
        ),
        ((0, 1, 2, 3),),
    )
    uv_layer = obj.data.uv_layers.new(name="UVMap")
    for loop, value in zip(
        uv_layer.data,
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        strict=True,
    ):
        loop.uv = value
    return obj


def _activate(obj) -> None:
    for candidate in bpy.context.view_layer.objects:
        candidate.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _ensure_camera() -> None:
    camera_data = bpy.data.cameras.new("Spine2D_FixtureCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 5.0
    camera = bpy.data.objects.new("Spine2D_FixtureCamera", camera_data)
    camera.location = (0.0, 0.0, 5.0)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    light_data = bpy.data.lights.new("Spine2D_FixtureLight", "AREA")
    light_data.energy = 1000.0
    light_data.shape = "DISK"
    light_data.size = 5.0
    light = bpy.data.objects.new("Spine2D_FixtureLight", light_data)
    light.location = (0.0, 0.0, 4.0)
    bpy.context.scene.collection.objects.link(light)


def _save(path: Path, obj) -> None:
    _ensure_camera()
    _activate(obj)
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 2
    path.parent.mkdir(parents=True, exist_ok=True)
    result = bpy.ops.wm.save_as_mainfile(
        filepath=str(path),
        check_existing=False,
    )
    if "FINISHED" not in result:
        raise RuntimeError(f"unable to save fixture {path}: {result}")


def procedural_noise(path: Path) -> None:
    _reset()
    obj = _quad()
    material = _material(
        "ProceduralNoiseMaterial",
        (0.2, 0.4, 0.8, 1.0),
    )
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get("Principled BSDF")
    if principled is None:
        raise RuntimeError("ProceduralNoiseMaterial has no Principled BSDF")

    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0
    noise.inputs["Detail"].default_value = 2.0

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.02, 0.1, 0.8, 1.0)
    ramp.color_ramp.elements[1].color = (1.0, 0.2, 0.02, 1.0)

    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], principled.inputs["Base Color"])
    obj.data.materials.append(material)
    _save(path, obj)


def nested_node_groups(path: Path) -> None:
    _reset()
    obj = _quad()
    material = _material(
        "NestedGroupMaterial",
        (0.1, 0.8, 0.2, 1.0),
    )
    tree = material.node_tree
    principled = tree.nodes.get("Principled BSDF")
    if principled is None:
        raise RuntimeError("NestedGroupMaterial has no Principled BSDF")

    inner = bpy.data.node_groups.new(
        "Spine2D_InnerColor",
        "ShaderNodeTree",
    )
    inner.interface.new_socket(
        name="Color",
        in_out="OUTPUT",
        socket_type="NodeSocketColor",
    )
    inner_output = inner.nodes.new("NodeGroupOutput")
    rgb = inner.nodes.new("ShaderNodeRGB")
    rgb.outputs["Color"].default_value = (0.75, 0.08, 0.9, 1.0)
    inner.links.new(rgb.outputs["Color"], inner_output.inputs["Color"])

    outer = bpy.data.node_groups.new(
        "Spine2D_OuterColor",
        "ShaderNodeTree",
    )
    outer.interface.new_socket(
        name="Color",
        in_out="OUTPUT",
        socket_type="NodeSocketColor",
    )
    outer_output = outer.nodes.new("NodeGroupOutput")
    inner_node = outer.nodes.new("ShaderNodeGroup")
    inner_node.node_tree = inner
    outer.links.new(inner_node.outputs["Color"], outer_output.inputs["Color"])

    group_node = tree.nodes.new("ShaderNodeGroup")
    group_node.node_tree = outer
    tree.links.new(group_node.outputs["Color"], principled.inputs["Base Color"])
    obj.data.materials.append(material)
    _save(path, obj)


def overlapping_uv(path: Path) -> None:
    _reset()
    obj = _object_from_pydata(
        "Hero",
        (
            (-2.0, -1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 1.0, 0.0),
            (-2.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (2.0, -1.0, 0.0),
            (2.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        ((0, 1, 2, 3), (4, 5, 6, 7)),
    )
    uv_layer = obj.data.uv_layers.new(name="UVMap")
    values = (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ) * 2
    for loop, value in zip(uv_layer.data, values, strict=True):
        loop.uv = value

    obj.data.materials.append(
        _material("LeftMaterial", (0.9, 0.1, 0.05, 1.0))
    )
    obj.data.materials.append(
        _material("RightMaterial", (0.05, 0.2, 0.9, 1.0))
    )
    obj.data.polygons[0].material_index = 0
    obj.data.polygons[1].material_index = 1
    _save(path, obj)


def non_manifold(path: Path) -> None:
    _reset()
    obj = _object_from_pydata(
        "Hero",
        (
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        ((0, 1, 2), (1, 0, 3), (0, 1, 4)),
    )
    obj.data.materials.append(
        _material("NonManifoldMaterial", (0.1, 0.7, 0.7, 1.0))
    )
    _save(path, obj)


def negative_scale_modifier(path: Path) -> None:
    _reset()
    obj = _quad()
    obj.scale = (-1.0, 1.5, 1.0)
    solidify = obj.modifiers.new("FixtureSolidify", "SOLIDIFY")
    solidify.thickness = 0.05
    obj.data.materials.append(
        _material("NegativeScaleMaterial", (0.9, 0.55, 0.05, 1.0))
    )
    _save(path, obj)


def create_all(output: Path) -> tuple[Path, ...]:
    if not isinstance(output, Path):
        raise TypeError("output must be pathlib.Path")

    creators = {
        "procedural_noise": procedural_noise,
        "nested_node_groups": nested_node_groups,
        "overlapping_uv": overlapping_uv,
        "non_manifold": non_manifold,
        "negative_scale_modifier": negative_scale_modifier,
    }
    paths: list[Path] = []
    for name in FIXTURE_NAMES:
        path = output / f"{name}.blend"
        creators[name](path)
        paths.append(path)
    return tuple(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    namespace = parser.parse_args(_arguments(sys.argv))

    for path in create_all(namespace.output.resolve(strict=False)):
        print(path)


if __name__ == "__main__":
    main()
