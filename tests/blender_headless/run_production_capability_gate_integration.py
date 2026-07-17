"""Blender 4.4 integration checks for production shader capability routing."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import traceback

import bpy

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.application import A1SingleObjectStage  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    A1ObjectPreparationError,
    prepare_a1_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    BakePlanError,
    CameraProjectionPlan,
)
from run_bake_integration import _assert  # noqa: E402
from run_camera_projection_integration import (  # noqa: E402
    _create_quad,
    _prepare_scene_with_sentinel,
    _settings,
)
from run_recursive_node_group_integration import _new_group_socket  # noqa: E402


def _alpha_group_material(name: str):
    group = bpy.data.node_groups.new(f"{name}_Group", "ShaderNodeTree")
    _new_group_socket(group, "Shader", in_out="OUTPUT", socket_type="NodeSocketShader")
    group_output = group.nodes.new(type="NodeGroupOutput")
    group_output.is_active_output = True
    principled = group.nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (0.8, 0.08, 0.02, 1.0)
    principled.inputs["Roughness"].default_value = 1.0
    principled.inputs["Alpha"].default_value = 0.42
    group.links.new(principled.outputs["BSDF"], group_output.inputs["Shader"])

    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    material.node_tree.nodes.clear()
    output = material.node_tree.nodes.new(type="ShaderNodeOutputMaterial")
    instance = material.node_tree.nodes.new(type="ShaderNodeGroup")
    instance.name = "Alpha Group Instance"
    instance.node_tree = group
    material.node_tree.links.new(instance.outputs["Shader"], output.inputs["Surface"])
    return material


def _alpha_reroute_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (0.1, 0.7, 0.2, 1.0)
    principled.inputs["Alpha"].default_value = 0.35
    reroute = nodes.new(type="NodeReroute")
    material.node_tree.links.new(principled.outputs["BSDF"], reroute.inputs[0])
    material.node_tree.links.new(reroute.outputs[0], output.inputs["Surface"])
    return material


def _muted_alpha_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (0.7, 0.1, 0.05, 1.0)
    principled.inputs["Alpha"].default_value = 0.51
    transparent = nodes.new(type="ShaderNodeBsdfTransparent")
    mix = nodes.new(type="ShaderNodeMixShader")
    mix.name = "Muted Alpha Mix"
    mix.mute = True
    material.node_tree.links.new(principled.outputs["BSDF"], mix.inputs[1])
    material.node_tree.links.new(transparent.outputs["BSDF"], mix.inputs[2])
    material.node_tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return material


def _value_emission_material(name: str, node_type: str, output_name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    source = nodes.new(type=node_type)
    emission = nodes.new(type="ShaderNodeEmission")
    source_output = source.outputs.get(output_name)
    _assert(source_output is not None, f"{node_type} output {output_name!r} is missing")
    material.node_tree.links.new(source_output, emission.inputs["Color"])
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _prepare_with_material(material, stem: str):
    with tempfile.TemporaryDirectory(prefix=f"spine2d-capability-{stem}-") as directory:
        source = _create_quad(f"{stem}Source")
        source.data.materials.append(material)
        return prepare_a1_object(source, _settings(Path(directory), stem))


def test_alpha_proxy_boundaries_route_to_b4() -> None:
    _prepare_scene_with_sentinel()
    for stem, material in (
        ("GroupAlpha", _alpha_group_material("GroupAlphaMaterial")),
        ("RerouteAlpha", _alpha_reroute_material("RerouteAlphaMaterial")),
        ("MutedAlpha", _muted_alpha_material("MutedAlphaMaterial")),
    ):
        prepared = _prepare_with_material(material, stem)
        _assert(
            isinstance(prepared.bake_plan, CameraProjectionPlan),
            f"{stem} did not route to B4: {type(prepared.bake_plan).__name__}",
        )
        _assert(
            prepared.statistics["shader_capability"] == "CAMERA_RENDER_REQUIRED",
            f"{stem} has wrong capability: {prepared.statistics['shader_capability']}",
        )


def test_object_and_camera_inputs_route_to_b4() -> None:
    _prepare_scene_with_sentinel()
    for stem, node_type, output_name in (
        ("ObjectInfo", "ShaderNodeObjectInfo", "Random"),
        ("CameraData", "ShaderNodeCameraData", "View Distance"),
    ):
        prepared = _prepare_with_material(
            _value_emission_material(
                f"{stem}Material",
                node_type,
                output_name,
            ),
            stem,
        )
        _assert(
            isinstance(prepared.bake_plan, CameraProjectionPlan),
            f"{stem} did not route to B4",
        )


def test_particle_context_fails_before_bake() -> None:
    _prepare_scene_with_sentinel()
    material = _value_emission_material(
        "ParticleCapabilityMaterial",
        "ShaderNodeParticleInfo",
        "Age",
    )
    try:
        _prepare_with_material(material, "ParticleCapability")
    except A1ObjectPreparationError as exc:
        _assert(
            exc.stage is A1SingleObjectStage.PLAN_BAKE,
            f"Particle capability failed at wrong stage: {exc.stage}",
        )
        _assert(isinstance(exc.cause, BakePlanError), "Particle failure is not BakePlanError")
        _assert(
            "GROUP_RENDER_REQUIRED" in str(exc.cause),
            f"Particle failure lacks capability code: {exc.cause}",
        )
    else:
        raise AssertionError("Particle Info was silently assigned to a per-object bake")


def main() -> None:
    tests = (
        test_alpha_proxy_boundaries_route_to_b4,
        test_object_and_camera_inputs_route_to_b4,
        test_particle_context_fails_before_bake,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"Production capability gate integration passed: {len(tests)} tests")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
