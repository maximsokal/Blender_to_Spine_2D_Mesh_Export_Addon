"""Real Blender 5.2+ smoke test for APIs used by the Rewrite adapter layer.

Run with:
    blender --background --factory-startup --python \
        tests/blender_headless/run_blender_52_api_contract.py
"""

from __future__ import annotations

from math import isfinite, radians
import traceback

import bpy


MINIMUM_VERSION = (5, 2, 0)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _require_finished(result, label: str) -> None:
    _assert(isinstance(result, set), f"{label} returned non-set result: {result!r}")
    _assert("FINISHED" in result, f"{label} did not finish: {result!r}")


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


def _restore_object_mode() -> None:
    active = bpy.context.view_layer.objects.active
    if active is None:
        return
    if str(getattr(active, "mode", "OBJECT")) == "OBJECT":
        return
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        traceback.print_exc()


def _activate_only(obj) -> None:
    for candidate in tuple(bpy.context.selected_objects):
        candidate.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _operator_properties(operator):
    rna_type = operator.get_rna_type()
    return {
        prop.identifier: prop
        for prop in rna_type.properties
        if prop.identifier != "rna_type"
    }


def _require_operator_contract(
    operator,
    *,
    required_properties,
    required_enum_items=None,
) -> None:
    properties = _operator_properties(operator)
    missing = tuple(sorted(set(required_properties) - set(properties)))
    _assert(not missing, f"{operator.idname()} is missing properties: {missing}")

    for property_name, expected_items in (required_enum_items or {}).items():
        prop = properties[property_name]
        actual_items = {
            item.identifier for item in getattr(prop, "enum_items", ())
        }
        missing_items = tuple(sorted(set(expected_items) - actual_items))
        _assert(
            not missing_items,
            f"{operator.idname()}.{property_name} is missing enum identifiers: "
            f"{missing_items}; actual={tuple(sorted(actual_items))}",
        )


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


def _activate_uv_layer_for_display_and_render(mesh, uv_layer, expected_name: str) -> None:
    """Use Blender 5.2 UVLoopLayers collection ownership for active UV roles."""

    if mesh is None:
        raise ValueError("mesh cannot be None")
    if uv_layer is None:
        raise ValueError("uv_layer cannot be None")
    if not isinstance(expected_name, str) or not expected_name:
        raise ValueError("expected_name must be a non-empty string")

    layers = mesh.uv_layers
    layers.active = uv_layer
    layers.active_render = uv_layer

    active_uv = layers.active
    render_uv = layers.active_render
    _assert(active_uv is not None, "Active UV layer became None")
    _assert(render_uv is not None, "Render UV layer became None")
    _assert(
        active_uv.name == expected_name,
        f"Active UV layer was not retained: {getattr(active_uv, 'name', None)!r}",
    )
    _assert(
        render_uv.name == expected_name,
        f"Render UV layer was not retained: {getattr(render_uv, 'name', None)!r}",
    )
    _assert(
        int(layers.active_index) == int(layers.find(expected_name)),
        f"Active UV index does not identify {expected_name!r}",
    )
    _assert(
        int(layers.active_render_index) == int(layers.find(expected_name)),
        f"Render UV index does not identify {expected_name!r}",
    )


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

        _activate_uv_layer_for_display_and_render(mesh, uv_layer, "BakeUV")
    finally:
        _remove_object(obj)
        _remove_mesh(mesh)


def test_uv_operator_rna_and_headless_execution_contract() -> None:
    smart_project_properties = {
        "angle_limit",
        "margin_method",
        "rotate_method",
        "island_margin",
        "area_weight",
        "correct_aspect",
        "scale_to_bounds",
    }
    unwrap_properties = {
        "method",
        "fill_holes",
        "correct_aspect",
        "use_subsurf_data",
        "margin_method",
        "margin",
        "no_flip",
        "iterations",
        "use_weights",
        "weight_group",
        "weight_factor",
    }
    pack_properties = {
        "udim_source",
        "rotate",
        "rotate_method",
        "scale",
        "merge_overlap",
        "margin_method",
        "margin",
        "pin",
        "pin_method",
        "shape_method",
    }
    _require_operator_contract(
        bpy.ops.uv.smart_project,
        required_properties=smart_project_properties,
        required_enum_items={
            "margin_method": {"SCALED", "ADD", "FRACTION"},
            "rotate_method": {
                "AXIS_ALIGNED",
                "AXIS_ALIGNED_X",
                "AXIS_ALIGNED_Y",
            },
        },
    )
    _require_operator_contract(
        bpy.ops.uv.unwrap,
        required_properties=unwrap_properties,
        required_enum_items={
            "method": {"ANGLE_BASED", "CONFORMAL", "MINIMUM_STRETCH"},
            "margin_method": {"SCALED", "ADD", "FRACTION"},
        },
    )
    _require_operator_contract(
        bpy.ops.uv.pack_islands,
        required_properties=pack_properties,
        required_enum_items={
            "udim_source": {"CLOSEST_UDIM", "ACTIVE_UDIM", "ORIGINAL_AABB"},
            "rotate_method": {
                "ANY",
                "CARDINAL",
                "AXIS_ALIGNED",
                "AXIS_ALIGNED_X",
                "AXIS_ALIGNED_Y",
            },
            "pin_method": {"SCALE", "ROTATION", "ROTATION_SCALE", "LOCKED"},
            "shape_method": {"CONCAVE", "CONVEX", "AABB"},
        },
    )

    mesh = None
    obj = None
    try:
        mesh = bpy.data.meshes.new("__Spine2D_UvOperatorContractMesh")
        mesh.from_pydata(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
            (),
            ((0, 1, 2, 3),),
        )
        mesh.update(calc_edges=True)
        obj = bpy.data.objects.new("__Spine2D_UvOperatorContractObject", mesh)
        bpy.context.scene.collection.objects.link(obj)
        uv_layer = mesh.uv_layers.new(name="SpineBakeUV")
        _activate_uv_layer_for_display_and_render(mesh, uv_layer, "SpineBakeUV")

        _activate_only(obj)
        _require_finished(
            bpy.ops.object.mode_set(mode="EDIT"),
            "bpy.ops.object.mode_set(EDIT)",
        )
        _assert(bpy.ops.mesh.select_all.poll(), "mesh.select_all poll failed")
        _require_finished(
            bpy.ops.mesh.select_all(action="SELECT"),
            "bpy.ops.mesh.select_all",
        )
        _assert(bpy.ops.uv.select_all.poll(), "uv.select_all poll failed")
        _require_finished(
            bpy.ops.uv.select_all(action="SELECT"),
            "bpy.ops.uv.select_all",
        )
        _assert(bpy.ops.uv.smart_project.poll(), "uv.smart_project poll failed")
        _require_finished(
            bpy.ops.uv.smart_project(
                angle_limit=radians(66.0),
                margin_method="SCALED",
                rotate_method="AXIS_ALIGNED_Y",
                island_margin=0.001,
                area_weight=0.0,
                correct_aspect=True,
                scale_to_bounds=True,
            ),
            "bpy.ops.uv.smart_project",
        )
        _assert(bpy.ops.uv.pack_islands.poll(), "uv.pack_islands poll failed")
        _require_finished(
            bpy.ops.uv.pack_islands(
                udim_source="CLOSEST_UDIM",
                rotate=True,
                rotate_method="ANY",
                scale=True,
                merge_overlap=False,
                margin_method="SCALED",
                margin=0.001,
                pin=False,
                pin_method="LOCKED",
                shape_method="CONCAVE",
            ),
            "bpy.ops.uv.pack_islands after smart_project",
        )
        _require_finished(
            bpy.ops.object.mode_set(mode="OBJECT"),
            "bpy.ops.object.mode_set(OBJECT)",
        )

        first_coordinates = tuple(
            tuple(float(component) for component in item.vector)
            for item in uv_layer.uv
        )
        _assert(
            len(first_coordinates) == len(mesh.loops),
            "smart_project returned incomplete UV data",
        )
        _assert(
            all(isfinite(component) for uv in first_coordinates for component in uv),
            "smart_project returned non-finite UV data",
        )

        _require_finished(
            bpy.ops.object.mode_set(mode="EDIT"),
            "bpy.ops.object.mode_set(EDIT) for unwrap",
        )
        _require_finished(
            bpy.ops.mesh.select_all(action="SELECT"),
            "bpy.ops.mesh.select_all for unwrap",
        )
        _require_finished(
            bpy.ops.uv.select_all(action="SELECT"),
            "bpy.ops.uv.select_all for unwrap",
        )
        _assert(bpy.ops.uv.unwrap.poll(), "uv.unwrap poll failed")
        _require_finished(
            bpy.ops.uv.unwrap(
                method="CONFORMAL",
                fill_holes=True,
                correct_aspect=True,
                use_subsurf_data=False,
                margin_method="SCALED",
                margin=0.001,
                no_flip=False,
                iterations=10,
                use_weights=False,
                weight_group="uv_importance",
                weight_factor=1.0,
            ),
            "bpy.ops.uv.unwrap",
        )
        _require_finished(
            bpy.ops.uv.pack_islands(
                udim_source="CLOSEST_UDIM",
                rotate=True,
                rotate_method="ANY",
                scale=True,
                merge_overlap=False,
                margin_method="SCALED",
                margin=0.001,
                pin=False,
                pin_method="LOCKED",
                shape_method="CONCAVE",
            ),
            "bpy.ops.uv.pack_islands after unwrap",
        )
        _require_finished(
            bpy.ops.object.mode_set(mode="OBJECT"),
            "bpy.ops.object.mode_set(OBJECT) after unwrap",
        )

        second_coordinates = tuple(
            tuple(float(component) for component in item.vector)
            for item in uv_layer.uv
        )
        _assert(
            len(second_coordinates) == len(mesh.loops),
            "unwrap returned incomplete UV data",
        )
        _assert(
            all(isfinite(component) for uv in second_coordinates for component in uv),
            "unwrap returned non-finite UV data",
        )
    finally:
        _restore_object_mode()
        _remove_object(obj)
        _remove_mesh(mesh)


def test_evaluated_object_to_mesh_lifetime_contract() -> None:
    mesh = None
    obj = None
    evaluated_object = None
    evaluated_mesh = None
    try:
        mesh = bpy.data.meshes.new("__Spine2D_EvaluatedContractMesh")
        mesh.from_pydata(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
            (),
            ((0, 1, 2, 3),),
        )
        mesh.update(calc_edges=True)
        obj = bpy.data.objects.new("__Spine2D_EvaluatedContractObject", mesh)
        bpy.context.scene.collection.objects.link(obj)
        triangulate = obj.modifiers.new(name="Triangulate", type="TRIANGULATE")
        _assert(triangulate.type == "TRIANGULATE", "Triangulate modifier unavailable")

        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
        evaluated_object = obj.evaluated_get(depsgraph)
        evaluated_mesh = evaluated_object.to_mesh(
            preserve_all_data_layers=True,
            depsgraph=depsgraph,
        )
        _assert(evaluated_mesh is not None, "Object.to_mesh returned None")
        _assert(len(evaluated_mesh.polygons) == 2, "Evaluated triangulation was not applied")
        _assert(len(evaluated_mesh.loops) == 6, "Evaluated triangle loop count mismatch")
    finally:
        if evaluated_object is not None and evaluated_mesh is not None:
            evaluated_object.to_mesh_clear()
        _remove_object(obj)
        _remove_mesh(mesh)


def test_material_node_tree_and_output_target_contract() -> None:
    material = None
    try:
        material = bpy.data.materials.new("__Spine2D_ApiContractMaterial")
        material.use_nodes = True
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

        expected_srgb = (0.25, 0.5, 0.75, 1.0)
        attribute.data[0].color_srgb = expected_srgb
        stored_srgb = tuple(
            float(value)
            for value in attribute.data[0].color_srgb
        )
        stored_linear = tuple(
            float(value)
            for value in attribute.data[0].color
        )

        _assert(
            all(isfinite(value) for value in (*stored_srgb, *stored_linear)),
            "FloatColorAttributeValue returned non-finite color values",
        )

        rgb_errors = tuple(
            abs(actual - expected)
            for actual, expected in zip(stored_srgb[:3], expected_srgb[:3])
        )
        _assert(
            max(rgb_errors) <= 5e-5,
            "FloatColorAttributeValue.color_srgb round trip exceeded the "
            f"32-bit sRGB/linear conversion tolerance: stored={stored_srgb}, "
            f"errors={rgb_errors}",
        )
        _assert(
            abs(stored_srgb[3] - expected_srgb[3]) <= 1e-7,
            f"FloatColorAttributeValue alpha round trip failed: {stored_srgb[3]}",
        )
        _assert(
            all(
                0.0 <= linear < srgb
                for linear, srgb in zip(stored_linear[:3], stored_srgb[:3])
            ),
            "FloatColorAttributeValue.color_srgb did not convert to scene-linear "
            f"FLOAT_COLOR storage: srgb={stored_srgb}, linear={stored_linear}",
        )
        _assert(
            abs(stored_linear[3] - expected_srgb[3]) <= 1e-7,
            f"Scene-linear alpha storage changed unexpectedly: {stored_linear[3]}",
        )
    finally:
        _remove_image(image)
        _remove_mesh(mesh)


def main() -> None:
    tests = (
        test_blender_version_and_eevee_identifier,
        test_mesh_edge_and_uv_attribute_contract,
        test_uv_operator_rna_and_headless_execution_contract,
        test_evaluated_object_to_mesh_lifetime_contract,
        test_material_node_tree_and_output_target_contract,
        test_image_alpha_and_color_attribute_contract,
    )
    failures: list[tuple[str, BaseException]] = []
    for test in tests:
        print(f"[API-CONTRACT] RUN {test.__name__}")
        try:
            test()
        except BaseException as exc:
            failures.append((test.__name__, exc))
            print(f"[API-CONTRACT] FAIL {test.__name__}: {exc}")
            traceback.print_exc()
        else:
            print(f"[API-CONTRACT] PASS {test.__name__}")

    if failures:
        summary = "; ".join(
            f"{name}: {type(exc).__name__}: {exc}"
            for name, exc in failures
        )
        raise AssertionError(
            f"Blender 5.2 API contract failed: {len(failures)}/{len(tests)}; {summary}"
        ) from failures[0][1]

    print(f"Blender 5.2 API contract passed: {len(tests)} tests")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
