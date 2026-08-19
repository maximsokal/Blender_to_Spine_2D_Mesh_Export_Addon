"""Legacy-derived Blender 5.2 bake compatibility matrix.

A PNG signature is not enough: Blender may report FINISHED and write an opaque black
image. These tests decode every output and exercise the production A1 material modes,
multiple material slots, per-object sequences, a common connected rig, and rollback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
import tempfile
import traceback
from unittest import mock

import bpy


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1MultiObjectStage,
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    A1MultiObjectSource,
    BakeExecutionError,
    export_a1_multi_object,
    export_a1_single_object,
)
import Blender_to_Spine2D_Mesh_Exporter.blender_adapter.semantic_bake_execution as bake_module  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    BakeExecutionSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import UvUnwrapSettings  # noqa: E402
from run_bake_integration import (  # noqa: E402
    PNG_SIGNATURE,
    _activate_only,
    _assert,
    _capture_context,
    _capture_scene_bake_state,
    _clear_scene,
    _configure_cycles_scene,
    _create_mesh_object,
    _create_sentinel,
    _material_fingerprint,
    _temporary_datablock_names,
)


ALPHA_THRESHOLD = 0.01
COLOR_THRESHOLD = 0.01
DOMINANCE_MARGIN = 0.04


@dataclass(frozen=True)
class DecodedImage:
    width: int
    height: int
    pixels: tuple[float, ...] = field(repr=False)
    opaque_pixels: int = 0
    colored_pixels: int = 0
    red_dominant_pixels: int = 0
    green_dominant_pixels: int = 0
    blue_dominant_pixels: int = 0
    maximum_rgb: float = 0.0

    @property
    def pixel_count(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class MatrixFixture:
    sources: tuple[A1MultiObjectSource, ...]
    materials: tuple[object, ...]


def _create_two_material_panel(name: str, *, location=(0.0, 0.0, 0.0)):
    """Create two connected quads with UVs occupying the left/right texture halves."""

    obj = _create_mesh_object(
        name,
        (
            (-1.0, -1.0, 0.0),
            (0.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        ),
        ((0, 1, 4, 3), (1, 2, 5, 4)),
    )
    obj.location = location
    return obj


def _principled_material(name: str, color: tuple[float, float, float, float]):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    material.diffuse_color = color
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is None:
        raise RuntimeError(f"Material '{name}' has no Principled BSDF")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = 0.5
    return material


def _animated_principled_material(
    name: str,
    keyframes: tuple[tuple[int, tuple[float, float, float, float]], ...],
):
    material = _principled_material(name, keyframes[0][1])
    socket = material.node_tree.nodes["Principled BSDF"].inputs["Base Color"]
    for frame_number, color in keyframes:
        socket.default_value = color
        socket.keyframe_insert(data_path="default_value", frame=frame_number)
    return material


def _checker_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    checker = nodes.new(type="ShaderNodeTexChecker")
    checker.inputs["Color1"].default_value = (0.02, 0.25, 0.9, 1.0)
    checker.inputs["Color2"].default_value = (0.9, 0.25, 0.02, 1.0)
    checker.inputs["Scale"].default_value = 5.0
    links.new(checker.outputs["Color"], principled.inputs["Base Color"])
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return material


def _generated_image_material(name: str):
    image = bpy.data.images.new(
        name=f"{name}_Generated",
        width=2,
        height=2,
        alpha=True,
    )
    image.pixels[:] = (
        0.95,
        0.04,
        0.02,
        1.0,
        0.75,
        0.02,
        0.01,
        1.0,
        1.0,
        0.2,
        0.02,
        1.0,
        0.6,
        0.01,
        0.01,
        1.0,
    )
    image.update()

    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    texture = nodes.new(type="ShaderNodeTexImage")
    texture.image = image
    texture.interpolation = "Closest"
    links.new(texture.outputs["Color"], principled.inputs["Base Color"])
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return material


def _assign_two_materials(obj, first, second) -> None:
    """Add slots before writing indices; Blender clamps indices without existing slots."""

    obj.data.materials.clear()
    obj.data.materials.append(first)
    obj.data.materials.append(second)
    _assert(len(obj.data.polygons) == 2, f"{obj.name} must have two polygons")
    obj.data.polygons[0].material_index = 0
    obj.data.polygons[1].material_index = 1
    obj.data.update()
    _assert(
        tuple(polygon.material_index for polygon in obj.data.polygons) == (0, 1),
        f"{obj.name} material indices were not retained",
    )


def _single_settings(
    output_directory: Path,
    *,
    prefix: str,
    sequence_start: int = 0,
    sequence_count: int = 0,
) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=64,
            texture_height=64,
            output_directory=output_directory,
            images_relative_path="images",
            bake_margin=2,
            sequence_start_frame=sequence_start,
            sequence_frame_count=sequence_count,
        ),
        prefix=prefix,
        output_stem=prefix,
        source_geometry_mode=A1SourceGeometryMode.ORIGINAL,
        uv=UvUnwrapSettings(layer_name="SpineBakeUV"),
        bake_execution=BakeExecutionSettings(samples=1),
    )


def _source(
    obj,
    output_directory: Path,
    *,
    component_id: str,
    prefix: str,
    sequence_start: int = 0,
    sequence_count: int = 0,
) -> A1MultiObjectSource:
    return A1MultiObjectSource(
        source_object=obj,
        component_id=component_id,
        animation_namespace=component_id,
        settings=_single_settings(
            output_directory,
            prefix=prefix,
            sequence_start=sequence_start,
            sequence_count=sequence_count,
        ),
    )


def _read_png(path: Path) -> DecodedImage:
    _assert(path.is_file(), f"PNG is missing: {path}")
    _assert(path.read_bytes()[:8] == PNG_SIGNATURE, f"File is not PNG: {path}")
    image = None
    try:
        image = bpy.data.images.load(str(path), check_existing=False)
        width, height = int(image.size[0]), int(image.size[1])
        pixels = tuple(float(value) for value in image.pixels[:])
        _assert(len(pixels) == width * height * 4, f"bad pixel array: {path}")

        opaque = colored = red = green = blue = 0
        maximum = 0.0
        for index in range(0, len(pixels), 4):
            r, g, b, a = pixels[index : index + 4]
            maximum = max(maximum, r, g, b)
            if a <= ALPHA_THRESHOLD:
                continue
            opaque += 1
            if max(r, g, b) > COLOR_THRESHOLD:
                colored += 1
            if r > g + DOMINANCE_MARGIN and r > b + DOMINANCE_MARGIN:
                red += 1
            if g > r + DOMINANCE_MARGIN and g > b + DOMINANCE_MARGIN:
                green += 1
            if b > r + DOMINANCE_MARGIN and b > g + DOMINANCE_MARGIN:
                blue += 1
        return DecodedImage(
            width=width,
            height=height,
            pixels=pixels,
            opaque_pixels=opaque,
            colored_pixels=colored,
            red_dominant_pixels=red,
            green_dominant_pixels=green,
            blue_dominant_pixels=blue,
            maximum_rgb=maximum,
        )
    finally:
        if image is not None:
            bpy.data.images.remove(image)


def _assert_usable(decoded: DecodedImage, label: str) -> None:
    _assert(
        decoded.opaque_pixels >= max(4, decoded.pixel_count // 100),
        f"{label} is transparent/empty: {decoded}",
    )
    _assert(decoded.colored_pixels > 0, f"{label} is black: {decoded}")
    _assert(decoded.maximum_rgb > COLOR_THRESHOLD, f"{label} has no RGB: {decoded}")


def _pixel_delta(first: DecodedImage, second: DecodedImage) -> float:
    _assert((first.width, first.height) == (second.width, second.height), "size changed")
    return sum(abs(a - b) for a, b in zip(first.pixels, second.pixels)) / len(
        first.pixels
    )


def _material_state(materials: tuple[object, ...]):
    return tuple(_material_fingerprint(material) for material in materials)


def _assert_restored(context_before, scene_before, materials, material_before) -> None:
    _assert(_capture_context() == context_before, "context was not restored")
    _assert(_capture_scene_bake_state() == scene_before, "scene state was not restored")
    _assert(_material_state(materials) == material_before, "source materials changed")
    _assert(not _temporary_datablock_names(), "temporary datablocks leaked")


def _mesh_attachments(document: dict):
    result = []
    for skin in document.get("skins", ()):
        for slot_name, attachments in skin.get("attachments", {}).items():
            for attachment_name, attachment in attachments.items():
                if attachment.get("type") == "mesh":
                    result.append((slot_name, attachment_name, attachment))
    return tuple(result)


def _build_matrix(output_directory: Path) -> MatrixFixture:
    static_a = _create_two_material_panel("StaticA", location=(0.0, 0.0, 0.0))
    animated_b = _create_two_material_panel("AnimatedB", location=(2.5, 0.5, 0.5))
    static_c = _create_two_material_panel("StaticC", location=(-2.5, -0.5, -0.5))

    materials = (
        _generated_image_material("StaticA_Image"),
        _principled_material("StaticA_Green", (0.02, 0.85, 0.08, 1.0)),
        _animated_principled_material(
            "AnimatedB_Keyframed",
            (
                (1, (0.9, 0.03, 0.02, 1.0)),
                (2, (0.02, 0.08, 0.95, 1.0)),
                (3, (0.8, 0.02, 0.65, 1.0)),
            ),
        ),
        _principled_material("AnimatedB_Yellow", (0.9, 0.7, 0.02, 1.0)),
        _checker_material("StaticC_Checker"),
        _principled_material("StaticC_Cyan", (0.02, 0.8, 0.8, 1.0)),
    )
    _assign_two_materials(static_a, materials[0], materials[1])
    _assign_two_materials(animated_b, materials[2], materials[3])
    _assign_two_materials(static_c, materials[4], materials[5])

    return MatrixFixture(
        sources=(
            _source(
                static_a,
                output_directory,
                component_id="static_a",
                prefix="StaticA",
            ),
            _source(
                animated_b,
                output_directory,
                component_id="animated_b",
                prefix="AnimatedB",
                sequence_start=1,
                sequence_count=3,
            ),
            _source(
                static_c,
                output_directory,
                component_id="static_c",
                prefix="StaticC",
            ),
        ),
        materials=materials,
    )


def _connected_settings(output_directory: Path, stem: str):
    return A1MultiObjectExportSettings(
        output_directory=output_directory,
        output_stem=stem,
        mode=A1MultiObjectMode.CONNECTED,
        anchor_component_id="static_a",
    )


def _prepare_state(sources, materials, frame_number: int):
    _configure_cycles_scene()
    sentinel = _create_sentinel()
    _activate_only(sentinel)
    for source in sources:
        source.source_object.select_set(False)
    bpy.context.scene.frame_set(frame_number)
    return (
        _capture_context(),
        _capture_scene_bake_state(),
        _material_state(materials),
    )


def test_standard_principled_multi_material_bake_is_not_black() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-standard-materials-") as directory:
        output_directory = Path(directory)
        source = _create_two_material_panel("StandardMaterials")
        red = _principled_material("StandardRed", (0.9, 0.02, 0.01, 1.0))
        green = _principled_material("StandardGreen", (0.01, 0.85, 0.03, 1.0))
        _assign_two_materials(source, red, green)
        settings = _single_settings(output_directory, prefix="StandardMaterials")
        context_before, scene_before, material_before = _prepare_state(
            (_source(source, output_directory, component_id="unused", prefix="unused"),),
            (red, green),
            9,
        )

        result = export_a1_single_object(source, settings)

        _assert(result.success, f"standard export failed: {result.issues}")
        decoded = _read_png(output_directory / "images" / "StandardMaterials_Baked.png")
        _assert_usable(decoded, "standard multi-material PNG")
        _assert(decoded.red_dominant_pixels > 4, f"red slot missing: {decoded}")
        _assert(decoded.green_dominant_pixels > 4, f"green slot missing: {decoded}")
        _assert_restored(
            context_before,
            scene_before,
            (red, green),
            material_before,
        )


def test_common_rig_supports_one_sequence_and_two_static_multi_material_objects() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-connected-sequence-") as directory:
        output_directory = Path(directory)
        fixture = _build_matrix(output_directory)
        context_before, scene_before, material_before = _prepare_state(
            fixture.sources,
            fixture.materials,
            11,
        )

        result = export_a1_multi_object(
            fixture.sources,
            _connected_settings(output_directory, "ConnectedSequenceRig"),
        )

        _assert(result.success, f"connected sequence failed: {result.issues}")
        expected = (
            (output_directory / "ConnectedSequenceRig.json").resolve(),
            (output_directory / "images" / "StaticA_Baked.png").resolve(),
            (output_directory / "images" / "AnimatedB_Baked_0001.png").resolve(),
            (output_directory / "images" / "AnimatedB_Baked_0002.png").resolve(),
            (output_directory / "images" / "AnimatedB_Baked_0003.png").resolve(),
            (output_directory / "images" / "StaticC_Baked.png").resolve(),
        )
        _assert(result.output_files == expected, f"wrong output set: {result.output_files}")

        decoded = {path.name: _read_png(path) for path in expected[1:]}
        for filename, image in decoded.items():
            _assert_usable(image, filename)
        _assert(
            _pixel_delta(
                decoded["AnimatedB_Baked_0001.png"],
                decoded["AnimatedB_Baked_0002.png"],
            )
            > 0.001,
            "sequence frames 1/2 are identical",
        )
        _assert(
            _pixel_delta(
                decoded["AnimatedB_Baked_0002.png"],
                decoded["AnimatedB_Baked_0003.png"],
            )
            > 0.001,
            "sequence frames 2/3 are identical",
        )

        document = json.loads(expected[0].read_text(encoding="utf-8"))
        bones = {bone["name"]: bone for bone in document["bones"]}
        _assert(tuple(bones).count("root") == 1, "root duplicated")
        _assert("all_objects_main" in bones, "connected all_objects rig missing")
        for prefix in ("StaticA", "AnimatedB", "StaticC"):
            _assert(
                str(bones[f"{prefix}_main"].get("parent", "")).startswith(
                    "all_objects_layer_"
                ),
                f"{prefix}_main is outside common rig",
            )

        attachments = _mesh_attachments(document)
        animated = tuple(item for item in attachments if item[0].startswith("AnimatedB_"))
        static = tuple(
            item
            for item in attachments
            if item[0].startswith("StaticA_") or item[0].startswith("StaticC_")
        )
        _assert(animated and static, "mesh attachments missing")
        for _, _, attachment in animated:
            _assert(attachment.get("path") == "images/AnimatedB_Baked_", attachment)
            _assert(
                attachment.get("sequence")
                == {"count": 3, "start": 1, "digits": 4, "setup": 1},
                attachment,
            )
        for _, _, attachment in static:
            _assert("sequence" not in attachment, f"static attachment became sequence")

        names = tuple(sorted(path.name for path in expected[1:]))
        actual_names = tuple(
            sorted(path.name for path in (output_directory / "images").glob("*.png"))
        )
        _assert(actual_names == names, f"extra/missing PNGs: {actual_names}")
        _assert_restored(
            context_before,
            scene_before,
            fixture.materials,
            material_before,
        )


def test_forced_second_object_failure_rolls_back_everything() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-matrix-rollback-") as directory:
        output_directory = Path(directory)
        fixture = _build_matrix(output_directory)
        context_before, scene_before, material_before = _prepare_state(
            fixture.sources,
            fixture.materials,
            7,
        )
        existing = output_directory / "images" / "AnimatedB_Baked_0002.png"
        existing.parent.mkdir(parents=True, exist_ok=True)
        original_bytes = b"existing-user-texture"
        existing.write_bytes(original_bytes)

        original_call = bake_module._call_bake_operator
        calls = 0

        def fail_after_first_bake(bpy_module, bake_type, *, uv_layer_name):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise BakeExecutionError("forced legacy-derived second bake failure")
            return original_call(
                bpy_module,
                bake_type,
                uv_layer_name=uv_layer_name,
            )

        with mock.patch.object(
            bake_module,
            "_call_bake_operator",
            side_effect=fail_after_first_bake,
        ):
            result = export_a1_multi_object(
                fixture.sources,
                _connected_settings(output_directory, "RollbackRig"),
            )

        _assert(not result.success, "forced failure unexpectedly succeeded")
        _assert(
            result.statistics.get("stage") == A1MultiObjectStage.STAGE_OUTPUTS.value,
            f"wrong failed stage: {result.statistics}",
        )
        _assert(existing.read_bytes() == original_bytes, "existing output was changed")
        _assert(
            not (output_directory / "RollbackRig.json").exists(),
            "failed export committed JSON",
        )
        files = tuple(
            sorted(
                path.relative_to(output_directory).as_posix()
                for path in output_directory.rglob("*")
                if path.is_file()
            )
        )
        _assert(files == ("images/AnimatedB_Baked_0002.png",), f"rollback leaks: {files}")
        _assert_restored(
            context_before,
            scene_before,
            fixture.materials,
            material_before,
        )


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    tests = (
        test_standard_principled_multi_material_bake_is_not_black,
        test_common_rig_supports_one_sequence_and_two_static_multi_material_objects,
        test_forced_second_object_failure_rolls_back_everything,
    )
    for test in tests:
        print(f"[LEGACY-MATRIX] RUN {test.__name__}")
        test()
        print(f"[LEGACY-MATRIX] PASS {test.__name__}")
    print(f"[LEGACY-MATRIX] PASS {len(tests)} integration tests")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
