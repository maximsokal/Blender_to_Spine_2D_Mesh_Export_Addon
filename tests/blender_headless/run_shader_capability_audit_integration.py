"""Blender 4.4 integration checks for diagnostic shader capability auditing."""

from __future__ import annotations

from pathlib import Path
import sys
import traceback

import bpy

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    analyse_material_graph,
    audit_material_graph_capabilities,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    ShaderBakeCapability,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _reset() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _material_from_value_output(
    name: str,
    *,
    node_type: str,
    output_name: str,
):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    source = nodes.new(type=node_type)
    source.name = f"{name} Source"
    emission = nodes.new(type="ShaderNodeEmission")
    source_output = source.outputs.get(output_name)
    _assert(
        source_output is not None,
        f"{node_type} has no output named {output_name!r}; "
        f"available={tuple(socket.name for socket in source.outputs)}",
    )
    material.node_tree.links.new(source_output, emission.inputs["Color"])
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _shader_to_rgb_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
    shader_to_rgb = nodes.new(type="ShaderNodeShaderToRGB")
    emission = nodes.new(type="ShaderNodeEmission")
    material.node_tree.links.new(diffuse.outputs["BSDF"], shader_to_rgb.inputs["Shader"])
    material.node_tree.links.new(shader_to_rgb.outputs["Color"], emission.inputs["Color"])
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _audit(material, render_target: str):
    graph = analyse_material_graph(material, render_target=render_target)
    return audit_material_graph_capabilities(graph, render_target=render_target)


def test_texture_coordinate_outputs_are_socket_specific() -> None:
    uv = _audit(
        _material_from_value_output(
            "TextureCoordUV",
            node_type="ShaderNodeTexCoord",
            output_name="UV",
        ),
        "CYCLES",
    )
    window = _audit(
        _material_from_value_output(
            "TextureCoordWindow",
            node_type="ShaderNodeTexCoord",
            output_name="Window",
        ),
        "CYCLES",
    )
    instancer = _audit(
        _material_from_value_output(
            "TextureCoordInstancer",
            node_type="ShaderNodeTexCoord",
            output_name="From Instancer",
        ),
        "CYCLES",
    )

    _assert(
        uv.required_capability is ShaderBakeCapability.LOCAL_UV_SAFE,
        f"UV output was not local-safe: {uv}",
    )
    _assert(
        window.required_capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
        f"Window output did not require camera render: {window}",
    )
    _assert(
        instancer.required_capability is ShaderBakeCapability.GROUP_RENDER_REQUIRED,
        f"From Instancer did not require group render: {instancer}",
    )


def test_source_context_nodes_are_not_local_uv_safe() -> None:
    cases = (
        ("CameraData", "ShaderNodeCameraData", "View Vector"),
        ("ObjectInfo", "ShaderNodeObjectInfo", "Random"),
        ("Attribute", "ShaderNodeAttribute", "Color"),
    )
    for name, node_type, output_name in cases:
        audit = _audit(
            _material_from_value_output(
                name,
                node_type=node_type,
                output_name=output_name,
            ),
            "CYCLES",
        )
        _assert(
            audit.required_capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
            f"{node_type}/{output_name} was not camera-bound: {audit}",
        )


def test_particle_info_requires_group_context() -> None:
    audit = _audit(
        _material_from_value_output(
            "ParticleInfo",
            node_type="ShaderNodeParticleInfo",
            output_name="Age",
        ),
        "CYCLES",
    )
    _assert(
        audit.required_capability is ShaderBakeCapability.GROUP_RENDER_REQUIRED,
        f"Particle Info was not group-bound: {audit}",
    )


def test_shader_to_rgb_is_eevee_only() -> None:
    material = _shader_to_rgb_material("ShaderToRGB")
    eevee = _audit(material, "EEVEE")
    cycles = _audit(material, "CYCLES")

    _assert(
        eevee.required_capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
        f"Eevee Shader to RGB was not camera-render-bound: {eevee}",
    )
    _assert(
        cycles.required_capability is ShaderBakeCapability.UNSUPPORTED,
        f"Cycles Shader to RGB was not rejected: {cycles}",
    )


def main() -> None:
    _reset()
    tests = (
        test_texture_coordinate_outputs_are_socket_specific,
        test_source_context_nodes_are_not_local_uv_safe,
        test_particle_info_requires_group_context,
        test_shader_to_rgb_is_eevee_only,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"Shader capability audit integration passed: {len(tests)} tests")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
