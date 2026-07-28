"""Blender 5.2 regression for independent source-sampling and bake-target UV roles."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import traceback
import warnings

import bpy

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    export_a1_single_object,
    prepare_a1_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    BakeExecutionSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import (  # noqa: E402
    UvUnwrapSettings,
)


TEMPORARY_PREFIX = "__Spine2D"
SOURCE_RENDER_UV = "SourceUV"
BAKE_DESTINATION_UV = "SpineBakeUV"
SOURCE_TEXTURE_SIZE = 4


def _assert(condition: bool, message: str) -> None:
    if not isinstance(condition, bool):
        raise TypeError("condition must be bool")
    if not isinstance(message, str) or not message:
        raise ValueError("message must be a non-empty string")
    if not condition:
        raise AssertionError(message)


def _enable_material_nodes(material: bpy.types.Material) -> None:
    if material is None:
        raise TypeError("material cannot be None")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Material.use_nodes.*",
            category=DeprecationWarning,
        )
        material.use_nodes = True


def _active_render_uv_names(mesh: bpy.types.Mesh) -> tuple[str, ...]:
    if mesh is None:
        raise TypeError("mesh cannot be None")
    return tuple(
        str(layer.name)
        for layer in mesh.uv_layers
        if bool(getattr(layer, "active_render", False))
    )


def _set_render_uv_layer(mesh: bpy.types.Mesh, layer_name: str) -> None:
    if mesh is None:
        raise TypeError("mesh cannot be None")
    if not isinstance(layer_name, str) or not layer_name.strip():
        raise ValueError("layer_name must be a non-empty string")
    resolved_name = layer_name.strip()
    if mesh.uv_layers.get(resolved_name) is None:
        raise AssertionError(f"UV layer does not exist: {resolved_name}")

    for layer in mesh.uv_layers:
        layer.active_render = str(layer.name) == resolved_name

    actual = _active_render_uv_names(mesh)
    if actual != (resolved_name,):
        raise AssertionError(
            f"Unable to set unique active_render UV '{resolved_name}': {actual}"
        )


def _clear_scene() -> None:
    try:
        active = bpy.context.object
        if active is not None and active.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception as exc:
        raise RuntimeError("Unable to restore Object Mode before scene cleanup") from exc

    try:
        for obj in tuple(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        for collection in tuple(bpy.data.collections):
            if collection is not bpy.context.scene.collection:
                bpy.data.collections.remove(collection, do_unlink=True)
        for datablocks in (
            bpy.data.meshes,
            bpy.data.materials,
            bpy.data.images,
        ):
            for datablock in tuple(datablocks):
                if int(getattr(datablock, "users", 0) or 0) == 0:
                    datablocks.remove(datablock)
    except Exception as exc:
        raise RuntimeError("Unable to clear Blender UV-role test fixtures") from exc


def _create_quad(name: str) -> bpy.types.Object:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    resolved_name = name.strip()

    mesh = None
    obj = None
    try:
        mesh = bpy.data.meshes.new(f"{resolved_name}_Mesh")
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

        layer = mesh.uv_layers.new(name="UVMap")
        coordinates = (
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        )
        polygon = mesh.polygons[0]
        for corner_index, coordinate in enumerate(coordinates):
            loop_index = int(polygon.loop_start) + corner_index
            layer.uv[loop_index].vector = coordinate
        mesh.uv_layers.active = layer
        _set_render_uv_layer(mesh, "UVMap")

        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        return obj
    except Exception:
        if obj is not None:
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except Exception:
                pass
        if mesh is not None and int(getattr(mesh, "users", 0) or 0) == 0:
            try:
                bpy.data.meshes.remove(mesh)
            except Exception:
                pass
        raise


def _temporary_datablock_names() -> tuple[str, ...]:
    names: list[str] = []
    for collection in (
        bpy.data.objects,
        bpy.data.meshes,
        bpy.data.collections,
        bpy.data.materials,
        bpy.data.images,
    ):
        names.extend(
            str(item.name)
            for item in collection
            if str(getattr(item, "name", "")).startswith(TEMPORARY_PREFIX)
        )
    return tuple(sorted(names))


def _settings(output_directory: Path, stem: str) -> A1SingleObjectExportSettings:
    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be pathlib.Path")
    if not isinstance(stem, str) or not stem.strip():
        raise ValueError("stem must be a non-empty string")

    resolved_stem = stem.strip()
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=64,
            texture_height=64,
            output_directory=output_directory,
            images_relative_path="images",
            bake_margin=1,
        ),
        prefix=resolved_stem,
        output_stem=resolved_stem,
        json_output_stem=resolved_stem,
        source_geometry_mode=A1SourceGeometryMode.EVALUATED,
        uv=UvUnwrapSettings(layer_name=BAKE_DESTINATION_UV),
        bake_execution=BakeExecutionSettings(samples=2),
    )


def _image_pixels(image: bpy.types.Image) -> tuple[float, ...]:
    if image is None:
        raise TypeError("image cannot be None")
    expected = int(image.size[0]) * int(image.size[1]) * 4
    values = [0.0] * expected
    image.pixels.foreach_get(values)
    if len(values) != expected:
        raise AssertionError(
            f"Image returned {len(values)} pixel values, expected {expected}"
        )
    return tuple(float(value) for value in values)


def _read_pixels(path: Path) -> tuple[float, ...]:
    if not isinstance(path, Path):
        path = Path(path)
    if not path.is_file():
        raise AssertionError(f"Baked image does not exist: {path}")

    image = None
    try:
        image = bpy.data.images.load(str(path), check_existing=False)
        return _image_pixels(image)
    except Exception as exc:
        raise RuntimeError(f"Unable to read baked image pixels from '{path}'") from exc
    finally:
        if image is not None:
            try:
                bpy.data.images.remove(image, do_unlink=True)
            except TypeError:
                bpy.data.images.remove(image)


def _write_source_texture(path: Path, name: str) -> bpy.types.Image:
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    path.parent.mkdir(parents=True, exist_ok=True)

    generated = None
    try:
        generated = bpy.data.images.new(
            name=f"{name.strip()}_Generated",
            width=SOURCE_TEXTURE_SIZE,
            height=SOURCE_TEXTURE_SIZE,
            alpha=True,
            float_buffer=False,
        )
        generated.alpha_mode = "STRAIGHT"
        pixels: list[float] = []
        for _y in range(SOURCE_TEXTURE_SIZE):
            for x in range(SOURCE_TEXTURE_SIZE):
                pixels.extend(
                    (1.0, 0.0, 0.0, 1.0)
                    if x < SOURCE_TEXTURE_SIZE // 2
                    else (0.0, 0.0, 1.0, 1.0)
                )
        generated.pixels.foreach_set(pixels)
        generated.update()
        generated.file_format = "PNG"
        generated.filepath_raw = str(path)
        generated.save()
        if not path.is_file() or path.stat().st_size <= 8:
            raise AssertionError(f"Source texture was not written: {path}")
    except Exception as exc:
        raise RuntimeError(f"Unable to write source UV fixture '{path}'") from exc
    finally:
        if generated is not None:
            try:
                bpy.data.images.remove(generated, do_unlink=True)
            except TypeError:
                bpy.data.images.remove(generated)

    try:
        loaded = bpy.data.images.load(str(path), check_existing=False)
        loaded.alpha_mode = "STRAIGHT"
        try:
            loaded.colorspace_settings.name = "Non-Color"
        except Exception:
            pass
        values = _image_pixels(loaded)
        width = int(loaded.size[0])
        height = int(loaded.size[1])
        if (width, height) != (SOURCE_TEXTURE_SIZE, SOURCE_TEXTURE_SIZE):
            raise AssertionError(
                f"Source texture size changed: {(width, height)}"
            )
        first = values[0:4]
        last = values[-4:]
        if first[0] < 0.9 or first[2] > 0.1:
            raise AssertionError(f"Left source texel is not red: {first}")
        if last[2] < 0.9 or last[0] > 0.1:
            raise AssertionError(f"Right source texel is not blue: {last}")
        return loaded
    except Exception:
        loaded = locals().get("loaded")
        if loaded is not None:
            try:
                bpy.data.images.remove(loaded, do_unlink=True)
            except Exception:
                pass
        raise


def _source_uv_image_material(name: str, image_path: Path):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    resolved_name = name.strip()

    material = bpy.data.materials.new(name=resolved_name)
    _enable_material_nodes(material)
    node_tree = material.node_tree
    if node_tree is None:
        raise RuntimeError("Created material has no node tree")
    nodes = node_tree.nodes
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs["Roughness"].default_value = 1.0
    texture_coordinate = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    image_node = nodes.new(type="ShaderNodeTexImage")
    image_node.interpolation = "Closest"
    image_node.extension = "CLIP"

    image = _write_source_texture(image_path, resolved_name)
    image_node.image = image

    node_tree.links.new(
        texture_coordinate.outputs["UV"],
        mapping.inputs["Vector"],
    )
    node_tree.links.new(
        mapping.outputs["Vector"],
        image_node.inputs["Vector"],
    )
    node_tree.links.new(
        image_node.outputs["Color"],
        principled.inputs["Base Color"],
    )
    node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return material, image


def _assign_constant_source_uv(obj: bpy.types.Object) -> None:
    if obj is None or getattr(obj, "type", None) != "MESH":
        raise TypeError("obj must be a Blender Mesh object")

    mesh = obj.data
    layers = mesh.uv_layers
    source = layers.get(SOURCE_RENDER_UV) or layers.new(name=SOURCE_RENDER_UV)
    for item in source.uv:
        item.vector = (0.25, 0.5)
    layers.active = source
    _set_render_uv_layer(mesh, SOURCE_RENDER_UV)


def test_source_render_uv_is_not_replaced_by_spine_bake_uv() -> None:
    _clear_scene()
    bpy.context.scene.render.engine = "CYCLES"

    with tempfile.TemporaryDirectory(prefix="spine2d-uv-sampling-role-") as directory:
        output_directory = Path(directory)
        source = _create_quad("UvSamplingRole")
        _assign_constant_source_uv(source)
        source_active = source.data.uv_layers.active
        _assert(
            source_active is not None and source_active.name == SOURCE_RENDER_UV,
            f"source active UV is wrong: {getattr(source_active, 'name', None)}",
        )
        _assert(
            _active_render_uv_names(source.data) == (SOURCE_RENDER_UV,),
            f"source active_render UV is wrong: {_active_render_uv_names(source.data)}",
        )

        source_texture_path = output_directory / "source_fixture.png"
        material, source_image = _source_uv_image_material(
            "UvSamplingRoleMaterial",
            source_texture_path,
        )
        source.data.materials.append(material)
        settings = _settings(output_directory, "UvSamplingRole")

        prepared = prepare_a1_object(source, settings)
        target = prepared.bake_target_snapshot
        _assert(
            target.active_uv_layer == BAKE_DESTINATION_UV,
            f"wrong bake target UV: {target.active_uv_layer}",
        )
        _assert(
            target.render_uv_layer == SOURCE_RENDER_UV,
            f"source render UV was lost: {target.render_uv_layer}",
        )
        _assert(
            {SOURCE_RENDER_UV, BAKE_DESTINATION_UV}.issubset(
                set(target.uv_layer_names)
            ),
            f"UV layers were not preserved: {target.uv_layer_names}",
        )
        source_coordinates = tuple(loop.uv(SOURCE_RENDER_UV) for loop in target.loops)
        _assert(
            all(
                coordinate is not None
                and abs(coordinate[0] - 0.25) <= 1.0e-6
                and abs(coordinate[1] - 0.5) <= 1.0e-6
                for coordinate in source_coordinates
            ),
            f"SourceUV coordinates changed before bake: {source_coordinates}",
        )

        pass_signature = tuple(
            (
                pass_plan.strategy_id.value,
                pass_plan.bake_mode.value,
                pass_plan.material_slot_indices,
            )
            for pass_plan in prepared.bake_plan.passes
        )
        print(
            "[UV_ROLE_PLAN] "
            f"passes={pass_signature} composite={prepared.bake_plan.composite.mode.value}"
        )

        result = export_a1_single_object(source, settings)
        _assert(result.success, f"UV-role export failed: {result.issues}")
        image_paths = tuple(
            Path(path)
            for path in result.output_files
            if Path(path).suffix.lower() == ".png"
            and Path(path).name != source_texture_path.name
        )
        _assert(
            len(image_paths) == 1,
            f"expected one exported PNG in output_files, got: {result.output_files}",
        )

        pixels = _read_pixels(image_paths[0])
        covered = [
            (
                float(pixels[offset]),
                float(pixels[offset + 1]),
                float(pixels[offset + 2]),
            )
            for offset in range(0, len(pixels), 4)
            if float(pixels[offset + 3]) > 0.5
        ]
        _assert(len(covered) > 20, "UV-role bake produced too few covered pixels")
        mean_red = sum(value[0] for value in covered) / len(covered)
        mean_blue = sum(value[2] for value in covered) / len(covered)
        print(
            "[UV_ROLE] "
            f"covered={len(covered)} mean_red={mean_red:.6f} "
            f"mean_blue={mean_blue:.6f}"
        )
        _assert(
            mean_red > 0.75,
            f"source render UV did not sample the red texel: red={mean_red}",
        )
        _assert(
            mean_blue < 0.2,
            f"SpineBakeUV leaked into source texture sampling: blue={mean_blue}",
        )
        _assert(not _temporary_datablock_names(), "UV-role bake leaked temporary data")
        _assert(source_image.name in bpy.data.images, "source image was removed")
        _assert(source_texture_path.is_file(), "source texture file was removed")


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    _assert(bpy.app.version >= (5, 2, 0), "Blender 5.2 or newer is required")
    test_source_render_uv_is_not_replaced_by_spine_bake_uv()
    print("[PASS] test_source_render_uv_is_not_replaced_by_spine_bake_uv")
    print("UV sampling role integration passed: 1 test")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
