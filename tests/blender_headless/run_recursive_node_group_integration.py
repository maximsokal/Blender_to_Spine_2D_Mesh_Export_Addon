"""Blender 4.4 integration tests for recursive Shader Node Group analysis."""

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

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    analyse_object_materials,
    export_a1_single_object,
    prepare_a1_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    CameraProjectionPlan,
    MaterialDependencyKind,
    MaterialKind,
    MaterialSemanticChannel,
)
from run_bake_integration import _assert  # noqa: E402
from run_camera_projection_integration import (  # noqa: E402
    _create_cube,
    _create_quad,
    _prepare_scene_with_sentinel,
    _read_pixels,
    _settings,
    _visible_and_transparent_counts,
)


def _new_group_socket(tree, name: str, *, in_out: str, socket_type: str):
    interface = getattr(tree, "interface", None)
    if interface is None or not hasattr(interface, "new_socket"):
        raise AssertionError("Blender node-tree interface API is unavailable")
    return interface.new_socket(name=name, in_out=in_out, socket_type=socket_type)


def _create_nested_layer_weight_material(name: str):
    inner = bpy.data.node_groups.new(f"{name}_Inner", "ShaderNodeTree")
    _new_group_socket(inner, "Shader", in_out="OUTPUT", socket_type="NodeSocketShader")
    inner_output = inner.nodes.new(type="NodeGroupOutput")
    inner_output.is_active_output = True
    layer_weight = inner.nodes.new(type="ShaderNodeLayerWeight")
    layer_weight.name = "Nested Layer Weight"
    ramp = inner.nodes.new(type="ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.01, 0.04, 0.3, 1.0)
    ramp.color_ramp.elements[1].color = (1.0, 0.05, 0.01, 1.0)
    emission = inner.nodes.new(type="ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 2.0
    inner.links.new(layer_weight.outputs["Facing"], ramp.inputs["Fac"])
    inner.links.new(ramp.outputs["Color"], emission.inputs["Color"])
    inner.links.new(emission.outputs["Emission"], inner_output.inputs["Shader"])

    outer = bpy.data.node_groups.new(f"{name}_Outer", "ShaderNodeTree")
    _new_group_socket(outer, "Shader", in_out="OUTPUT", socket_type="NodeSocketShader")
    outer_output = outer.nodes.new(type="NodeGroupOutput")
    outer_output.is_active_output = True
    inner_instance = outer.nodes.new(type="ShaderNodeGroup")
    inner_instance.name = "Inner Instance"
    inner_instance.node_tree = inner
    outer.links.new(inner_instance.outputs["Shader"], outer_output.inputs["Shader"])

    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    material_output = nodes.new(type="ShaderNodeOutputMaterial")
    outer_instance = nodes.new(type="ShaderNodeGroup")
    outer_instance.name = "Outer Instance"
    outer_instance.node_tree = outer
    material.node_tree.links.new(
        outer_instance.outputs["Shader"], material_output.inputs["Surface"]
    )
    return material


def _create_nested_volume_material(name: str):
    group = bpy.data.node_groups.new(f"{name}_Group", "ShaderNodeTree")
    _new_group_socket(group, "Volume", in_out="OUTPUT", socket_type="NodeSocketShader")
    group_output = group.nodes.new(type="NodeGroupOutput")
    group_output.is_active_output = True
    volume = group.nodes.new(type="ShaderNodeVolumePrincipled")
    volume.name = "Nested Principled Volume"
    volume.inputs["Density"].default_value = 1.3
    volume.inputs["Color"].default_value = (0.04, 0.1, 0.8, 1.0)
    emission_color = volume.inputs.get("Emission Color") or volume.inputs.get("Emission")
    if emission_color is not None:
        emission_color.default_value = (0.02, 0.08, 0.5, 1.0)
    emission_strength = volume.inputs.get("Emission Strength")
    if emission_strength is not None:
        emission_strength.default_value = 0.6
    group.links.new(volume.outputs["Volume"], group_output.inputs["Volume"])

    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    instance = nodes.new(type="ShaderNodeGroup")
    instance.name = "Volume Instance"
    instance.node_tree = group
    material.node_tree.links.new(instance.outputs["Volume"], output.inputs["Volume"])
    return material


def _create_group_with_unused_camera_input(name: str):
    group = bpy.data.node_groups.new(f"{name}_Group", "ShaderNodeTree")
    _new_group_socket(group, "Unused View", in_out="INPUT", socket_type="NodeSocketFloat")
    _new_group_socket(group, "Shader", in_out="OUTPUT", socket_type="NodeSocketShader")
    group.nodes.new(type="NodeGroupInput")
    group_output = group.nodes.new(type="NodeGroupOutput")
    group_output.is_active_output = True
    diffuse = group.nodes.new(type="ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (0.7, 0.15, 0.03, 1.0)
    group.links.new(diffuse.outputs["BSDF"], group_output.inputs["Shader"])

    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    layer_weight = nodes.new(type="ShaderNodeLayerWeight")
    layer_weight.name = "Unused Parent Layer Weight"
    instance = nodes.new(type="ShaderNodeGroup")
    instance.name = "Precise Instance"
    instance.node_tree = group
    material.node_tree.links.new(
        layer_weight.outputs["Facing"], instance.inputs["Unused View"]
    )
    material.node_tree.links.new(instance.outputs["Shader"], output.inputs["Surface"])
    return material


def _create_nested_image_material(name: str):
    group = bpy.data.node_groups.new(f"{name}_Group", "ShaderNodeTree")
    _new_group_socket(group, "Shader", in_out="OUTPUT", socket_type="NodeSocketShader")
    group_output = group.nodes.new(type="NodeGroupOutput")
    group_output.is_active_output = True
    generated = bpy.data.images.new(
        f"{name}_Generated",
        width=8,
        height=8,
        alpha=True,
    )
    generated.generated_color = (0.1, 0.7, 0.2, 1.0)
    image = group.nodes.new(type="ShaderNodeTexImage")
    image.name = "Nested Reachable Image"
    image.image = generated
    emission = group.nodes.new(type="ShaderNodeEmission")
    group.links.new(image.outputs["Color"], emission.inputs["Color"])
    group.links.new(emission.outputs["Emission"], group_output.inputs["Shader"])

    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    instance = nodes.new(type="ShaderNodeGroup")
    instance.name = "Nested Image Instance"
    instance.node_tree = group
    unused = nodes.new(type="ShaderNodeTexImage")
    unused.name = "Unreachable Missing Image"
    material.node_tree.links.new(instance.outputs["Shader"], output.inputs["Surface"])
    return material, generated


def _create_muted_camera_group_material(name: str):
    group = bpy.data.node_groups.new(f"{name}_Group", "ShaderNodeTree")
    _new_group_socket(group, "Shader", in_out="INPUT", socket_type="NodeSocketShader")
    _new_group_socket(group, "Shader", in_out="OUTPUT", socket_type="NodeSocketShader")
    group.nodes.new(type="NodeGroupInput")
    output = group.nodes.new(type="NodeGroupOutput")
    output.is_active_output = True
    layer_weight = group.nodes.new(type="ShaderNodeLayerWeight")
    layer_weight.name = "Muted Nested Layer Weight"
    emission = group.nodes.new(type="ShaderNodeEmission")
    group.links.new(layer_weight.outputs["Facing"], emission.inputs["Color"])
    group.links.new(emission.outputs["Emission"], output.inputs["Shader"])

    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    material_output = nodes.new(type="ShaderNodeOutputMaterial")
    diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (0.2, 0.6, 0.9, 1.0)
    instance = nodes.new(type="ShaderNodeGroup")
    instance.name = "Muted Camera Group"
    instance.node_tree = group
    instance.mute = True
    material.node_tree.links.new(diffuse.outputs["BSDF"], instance.inputs["Shader"])
    material.node_tree.links.new(
        instance.outputs["Shader"], material_output.inputs["Surface"]
    )
    return material, instance


def _create_renderer_specific_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()

    cycles_output = nodes.new(type="ShaderNodeOutputMaterial")
    cycles_output.name = "Cycles Material Output"
    cycles_output.target = "CYCLES"
    eevee_output = nodes.new(type="ShaderNodeOutputMaterial")
    eevee_output.name = "Eevee Material Output"
    eevee_output.target = "EEVEE"

    layer_weight = nodes.new(type="ShaderNodeLayerWeight")
    layer_weight.name = "Cycles Layer Weight"
    emission = nodes.new(type="ShaderNodeEmission")
    material.node_tree.links.new(layer_weight.outputs["Facing"], emission.inputs["Color"])
    material.node_tree.links.new(
        emission.outputs["Emission"], cycles_output.inputs["Surface"]
    )

    diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (0.8, 0.25, 0.05, 1.0)
    material.node_tree.links.new(diffuse.outputs["BSDF"], eevee_output.inputs["Surface"])
    return material


def test_nested_camera_group_routes_to_b4_and_renders() -> None:
    _prepare_scene_with_sentinel()
    with tempfile.TemporaryDirectory(prefix="spine2d-recursive-camera-") as directory:
        output_directory = Path(directory)
        source = _create_quad("NestedCameraSource")
        material = _create_nested_layer_weight_material("NestedCameraMaterial")
        source.data.materials.append(material)

        analysis = analyse_object_materials(source, source_object_id=source.name)
        graph = analysis.slots[0].graph
        _assert(graph is not None, "nested camera material has no graph snapshot")
        dependencies = set(graph.dependencies)
        _assert(MaterialDependencyKind.NODE_GROUP in dependencies, "group marker is missing")
        _assert(MaterialDependencyKind.CAMERA in dependencies, "nested camera dependency missing")
        _assert(MaterialDependencyKind.VIEW in dependencies, "nested view dependency missing")
        nested_node = next(
            node for node in graph.reachable_nodes if node.node_name == "Nested Layer Weight"
        )
        _assert(
            nested_node.group_path == ("Outer Instance", "Inner Instance"),
            f"unexpected nested group path: {nested_node.group_path}",
        )

        prepared = prepare_a1_object(
            source,
            _settings(output_directory, "NestedCameraProjection"),
        )
        _assert(
            isinstance(prepared.bake_plan, CameraProjectionPlan),
            f"nested camera graph did not select B4: {type(prepared.bake_plan).__name__}",
        )
        result = export_a1_single_object(
            source,
            _settings(output_directory, "NestedCameraProjection"),
        )
        _assert(result.success, f"nested camera export failed: {result.issues}")
        pixels = _read_pixels(
            output_directory / "images" / "NestedCameraProjection_Baked.png"
        )
        visible, transparent = _visible_and_transparent_counts(pixels)
        _assert(visible > 100, "nested camera render has too few visible pixels")
        _assert(transparent > 100, "nested camera render lost transparent background")


def test_nested_volume_group_routes_to_b4_and_renders() -> None:
    _prepare_scene_with_sentinel()
    with tempfile.TemporaryDirectory(prefix="spine2d-recursive-volume-") as directory:
        output_directory = Path(directory)
        source = _create_cube("NestedVolumeSource")
        material = _create_nested_volume_material("NestedVolumeMaterial")
        source.data.materials.append(material)

        analysis = analyse_object_materials(source, source_object_id=source.name)
        graph = analysis.slots[0].graph
        _assert(graph is not None, "nested volume material has no graph snapshot")
        _assert(
            MaterialSemanticChannel.VOLUME in graph.semantic_channels,
            f"nested Volume channel is missing: {graph.semantic_channels}",
        )
        prepared = prepare_a1_object(
            source,
            _settings(output_directory, "NestedVolumeProjection"),
        )
        _assert(isinstance(prepared.bake_plan, CameraProjectionPlan), "Volume group missed B4")
        result = export_a1_single_object(
            source,
            _settings(output_directory, "NestedVolumeProjection"),
        )
        _assert(result.success, f"nested Volume export failed: {result.issues}")
        pixels = _read_pixels(
            output_directory / "images" / "NestedVolumeProjection_Baked.png"
        )
        visible, transparent = _visible_and_transparent_counts(pixels)
        _assert(visible > 20, "nested Volume render has no visible contribution")
        _assert(transparent > 20, "nested Volume render lost transparent background")


def test_unused_group_input_does_not_select_camera_projection() -> None:
    _prepare_scene_with_sentinel()
    with tempfile.TemporaryDirectory(prefix="spine2d-recursive-precise-") as directory:
        output_directory = Path(directory)
        source = _create_quad("PreciseGroupSource")
        material = _create_group_with_unused_camera_input("PreciseGroupMaterial")
        source.data.materials.append(material)

        analysis = analyse_object_materials(source, source_object_id=source.name)
        graph = analysis.slots[0].graph
        _assert(graph is not None, "precise group material has no graph snapshot")
        _assert(
            MaterialDependencyKind.CAMERA not in graph.dependencies,
            f"unused group input leaked CAMERA: {graph.dependencies}",
        )
        _assert(
            all(node.node_name != "Unused Parent Layer Weight" for node in graph.reachable_nodes),
            "unused parent branch was marked reachable",
        )
        prepared = prepare_a1_object(
            source,
            _settings(output_directory, "PreciseGroupLocal"),
        )
        _assert(
            not isinstance(prepared.bake_plan, CameraProjectionPlan),
            "unused group input incorrectly selected B4",
        )


def test_nested_image_controls_material_kind_and_ignores_orphan_image() -> None:
    _prepare_scene_with_sentinel()
    source = _create_quad("NestedImageSource")
    material, generated = _create_nested_image_material("NestedImageMaterial")
    source.data.materials.append(material)

    analysis = analyse_object_materials(source, source_object_id=source.name)
    slot = analysis.slots[0]
    _assert(slot.kind is MaterialKind.IMAGE, f"unexpected nested image kind: {slot.kind}")
    _assert(
        tuple(item.image_name for item in slot.image_dependencies) == (generated.name,),
        f"nested image dependency mismatch: {slot.image_dependencies}",
    )
    _assert(
        all("Unreachable Missing Image" not in issue for issue in slot.issues),
        f"orphan image leaked into issues: {slot.issues}",
    )


def test_muted_group_uses_internal_bypass_and_stays_local() -> None:
    _prepare_scene_with_sentinel()
    with tempfile.TemporaryDirectory(prefix="spine2d-recursive-muted-") as directory:
        output_directory = Path(directory)
        source = _create_quad("MutedGroupSource")
        material, instance = _create_muted_camera_group_material("MutedGroupMaterial")
        source.data.materials.append(material)
        _assert(tuple(instance.internal_links), "Blender did not create muted group bypass links")

        analysis = analyse_object_materials(source, source_object_id=source.name)
        graph = analysis.slots[0].graph
        _assert(graph is not None, "muted group material has no graph snapshot")
        _assert(
            MaterialDependencyKind.CAMERA not in graph.dependencies,
            f"muted nested camera dependency leaked: {graph.dependencies}",
        )
        _assert(
            MaterialDependencyKind.NODE_GROUP not in graph.dependencies,
            f"muted group was treated as evaluated: {graph.dependencies}",
        )
        prepared = prepare_a1_object(
            source,
            _settings(output_directory, "MutedGroupLocal"),
        )
        _assert(
            not isinstance(prepared.bake_plan, CameraProjectionPlan),
            "muted group incorrectly selected B4",
        )


def test_renderer_specific_material_outputs_follow_active_engine() -> None:
    _prepare_scene_with_sentinel()
    source = _create_quad("RendererSpecificSource")
    source.data.materials.append(_create_renderer_specific_material("RendererSpecificMaterial"))
    scene = bpy.context.scene
    original_engine = scene.render.engine
    try:
        scene.render.engine = "CYCLES"
        cycles = analyse_object_materials(source, source_object_id=source.name)
        cycles_graph = cycles.slots[0].graph
        _assert(cycles_graph is not None, "Cycles graph snapshot is missing")
        _assert(
            cycles_graph.active_output_node_id == "Cycles Material Output",
            f"wrong Cycles output: {cycles_graph.active_output_node_id}",
        )
        _assert(
            MaterialDependencyKind.CAMERA in cycles_graph.dependencies,
            f"Cycles camera dependency missing: {cycles_graph.dependencies}",
        )

        scene.render.engine = "BLENDER_EEVEE_NEXT"
        eevee = analyse_object_materials(source, source_object_id=source.name)
        eevee_graph = eevee.slots[0].graph
        _assert(eevee_graph is not None, "Eevee graph snapshot is missing")
        _assert(
            eevee_graph.active_output_node_id == "Eevee Material Output",
            f"wrong Eevee output: {eevee_graph.active_output_node_id}",
        )
        _assert(
            MaterialDependencyKind.CAMERA not in eevee_graph.dependencies,
            f"Cycles branch leaked into Eevee: {eevee_graph.dependencies}",
        )
    finally:
        scene.render.engine = original_engine


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    tests = (
        test_nested_camera_group_routes_to_b4_and_renders,
        test_nested_volume_group_routes_to_b4_and_renders,
        test_unused_group_input_does_not_select_camera_projection,
        test_nested_image_controls_material_kind_and_ignores_orphan_image,
        test_muted_group_uses_internal_bypass_and_stays_local,
        test_renderer_specific_material_outputs_follow_active_engine,
    )
    for test in tests:
        print(f"[RECURSIVE-GROUP] RUN {test.__name__}")
        test()
        print(f"[RECURSIVE-GROUP] PASS {test.__name__}")
    print(f"[RECURSIVE-GROUP] PASS {len(tests)} integration tests")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
