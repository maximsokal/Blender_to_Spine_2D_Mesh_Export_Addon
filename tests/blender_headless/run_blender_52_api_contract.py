"""Real Blender 5.2+ smoke test for APIs used by the Rewrite adapter layer.

Run with:
    blender --background --factory-startup --python \
        tests/blender_headless/run_blender_52_api_contract.py
"""

from __future__ import annotations

import traceback

import bpy


MINIMUM_VERSION = (5, 2, 0)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _remove_object(obj) -> None:
    if obj is not None:
        bpy.data.objects.remove(obj, do_unlink=True)


def _remove_mesh(mesh) -> None:
    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def _remove_material(material) -> None:
    if material is not None and material.users == 0:
        bpy.data.materials.remove(material)


def _remove_image(image) -> None:
    if image is not None:
        bpy.data.images.remove(image, do_unlink=True)


def test_blender_version_and_eevee_identifier() -> None:
    version = tuple(int(value) for value in bpy.app.version[:3])
    _assert(version >= MINIMUM_VERSION, f"Blender 5.2+ required, detected {version}")

    scene = bpy.context.scene
    original_engine = scene.render.engine
    try:
        scene.render.engine = "BLENDER_EEVEE"
        _assert(
            scene.render.engine == "BLENDER_EEVEE",
            f"Unexpected Blender 5.2 EEVEE identifier: {scene.render.engine!r}",
        )
        rejected_old_identifier = False
        try:
            scene.render.engine = "BLENDER_EEVEE_NEXT"
        except (TypeError, ValueError):
            rejected_old_identifier = True
        _assert(
            rejected_old_identifier,
            "Removed BLENDER_EEVEE_NEXT identifier was unexpectedly accepted",
        )
    finally:
        scene.render.engine = original_engine


def test_mesh_edge_and_uv_attribute_contract() -> None:
    mesh = None
    obj = None
    try:
        mesh = bpy.data.meshes.new("__Spine2D_ApiContractMesh")
        mesh.from_pydata(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((0, 1), (1, 2), (2, 0)),
            ((0, 1, 2),),
        )
        mesh.update(calc_edges=True)
        obj = bpy.data.objects.new("__Spine2D_ApiContractObject", mesh)
        bpy.context.scene.collection.objects.link(obj)

        seam = mesh.attributes.new(name="uv_seam", type="BOOLEAN", domain="EDGE")
        sharp = mesh.attributes.new(name="sharp_edge", type="BOOLEAN", domain="EDGE")
        _assert(len(seam.data) == len(mesh.edges), "uv_seam EDGE length mismatch")
        _assert(len(sharp.data) == len(mesh.edges), "sharp_edge EDGE length mismatch")
        seam.data[0].value = True
        sharp.data[1].value = True
        _assert(bool(seam.data[0].value), "uv_seam BOOLEAN write/read failed")
        _assert(bool(sharp.data[1].value), "sharp_edge BOOLEAN write/read failed")

        uv_layer = mesh.uv_layers.new(name="BakeUV")
        _assert(len(uv_layer.uv) == len(mesh.loops), "UV attribute length mismatch")
        uv_layer.uv[0].vector = (0.125, 0.875)
        coordinate = tuple(float(value) for value in uv_layer.uv[0].vector)
        _assert(coordinate == (0.125, 0.875), f"UV vector round trip failed: {coordinate}")

        mesh.uv_layers.active = uv_layer
        uv_layer.active_render = True
        _assert(mesh.uv_layers.active is uv_layer, "Active UV layer was not retained")
        _assert(bool(uv_layer.active_render), "Render UV layer was not retained")
    finally:
        _remove_object(obj)
        _remove_mesh(mesh)


def test_material_node_tree_and_output_target_contract() -> None:
    material = None
    try:
        material = bpy.data.materials.new("__Spine2D_ApiContractMaterial")
        node_tree = material.node_tree
        _assert(node_tree is not None, "Blender 5.2 Material has no node_tree")
        _assert(node_tree.nodes is not None, "Material node tree has no nodes")
        _assert(node_tree.links is not None, "Material node tree has no links")

        cycles_output = node_tree.get_output_node("CYCLES")
        eevee_output = node_tree.get_output_node("EEVEE")
        _assert(cycles_output is not None, "get_output_node('CYCLES') returned None")
        _assert(eevee_output is not None, "get_output_node('EEVEE') returned None")
        _assert(cycles_output.type == "OUTPUT_MATERIAL", "Cycles output has wrong type")
        _assert(eevee_output.type == "OUTPUT_MATERIAL", "EEVEE output has wrong type")

        emission = node_tree.nodes.new(type="ShaderNodeEmission")
        mix_rgb = node_tree.nodes.new(type="ShaderNodeMixRGB")
        vertex_color = node_tree.nodes.new(type="ShaderNodeVertexColor")
        _assert(emission.type == "EMISSION", "ShaderNodeEmission is unavailable")
        _assert(mix_rgb.type == "MIX_RGB", "ShaderNodeMixRGB is unavailable")
        _assert(vertex_color.type == "VERTEX_COLOR", "ShaderNodeVertexColor is unavailable")
    finally:
        _remove_material(material)


def test_image_alpha_and_color_attribute_contract() -> None:
    image = None
    mesh = None
    try:
        image = bpy.data.images.new(
            "__Spine2D_ApiContractImage",
            width=4,
            height=4,
            alpha=True,
            float_buffer=True,
        )
        image.alpha_mode = "STRAIGHT"
        _assert(image.alpha_mode == "STRAIGHT", "Image alpha_mode was not retained")

        mesh = bpy.data.meshes.new("__Spine2D_ApiContractColorMesh")
        mesh.from_pydata(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            (),
            ((0, 1, 2),),
        )
        mesh.update(calc_edges=True)
        attribute = mesh.color_attributes.new(
            name="Spine2DGeneratedColor",
            type="FLOAT_COLOR",
            domain="CORNER",
        )
        attribute.data[0].color_srgb = (0.25, 0.5, 0.75, 1.0)
        stored = tuple(float(value) for value in attribute.data[0].color_srgb)
        _assert(
            all(abs(actual - expected) < 1e-6 for actual, expected in zip(stored, (0.25, 0.5, 0.75, 1.0))),
            f"FloatColorAttributeValue.color_srgb round trip failed: {stored}",
        )
    finally:
        _remove_image(image)
        _remove_mesh(mesh)


def main() -> None:
    tests = (
        test_blender_version_and_eevee_identifier,
        test_mesh_edge_and_uv_attribute_contract,
        test_material_node_tree_and_output_target_contract,
        test_image_alpha_and_color_attribute_contract,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"Blender 5.2 API contract passed: {len(tests)} tests")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
