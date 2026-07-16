"""Legacy-derived Blender 4.4 bake compatibility matrix.

The old exporter treated a written PNG as a successful bake even when Blender produced
fully transparent or black pixels. These integration tests preserve the useful legacy
input/output contract while explicitly rejecting unusable image content.

Covered production scenarios:

* one connected mesh using several standard Blender materials;
* one connected multi-object rig where exactly one object is an image sequence;
* static and sequence objects each using several material slots;
* atomic rollback when a later sequence frame fails after earlier bakes succeeded.

All fixtures are created at runtime and use the rewritten A1 services with their normal
DIFFUSE/COMBINED material policy. Only Cycles sample count is reduced to keep CI fast.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    bake_executor as bake_module,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    BakeExecutionSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import (  # noqa: E402
    UvUnwrapSettings,
)
from run_bake_integration import (  # noqa: E402
    PNG_SIGNATURE,
    _activate_only,
    _assert,
    _capture_context,
    _capture_scene_bake_state,
    _clear_scene,
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
    pixels: tuple[float, ...]
    opaque_pixels: int
    colored_pixels: int
    red_dominant_pixels: int
    green_dominant_pixels: int
    blue_dominant_pixels: int
    maximum_rgb: float
    mean_opaque_rgb: tuple[float, float, float]

    @property
    def pixel_count(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class MatrixFixture:
    sources: tuple[A1MultiObjectSource, ...]
    materials: tuple[object, ...]
    static_a: object
    animated_b: object
    static_c: object



def _create_two_material_panel(name: str, *, location=(0.0, 0.0, 0.0)):
    """Create two connected quads with non-overlapping UV halves and two slots."""

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
        (
            (0, 1, 4, 3),
            (1, 2, 5, 4),
        ),
    )
    obj.location = location
    obj.data.polygons[0].material_index = 0
    obj.data.polygons[1].material_index = 1
    obj.data.update()
    return obj



def _principled_material(name: str, color: tuple[float, float, float, float]):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    material.diffuse_color = color
    nodes = material.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    if principled is None:
        raise RuntimeError(f"Material '{name}' has no Principled BSDF node")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = 0.5
    return material



def _animated_principled_material(
    name: str,
    keyframes: tuple[tuple[int, tuple[float, float, float, float]], ...],
):
    if not keyframes:
        raise ValueError("keyframes cannot be empty")
    material = _principled_material(name, keyframes[0][1])
    principled = material.node_tree.nodes.get("Principled BSDF")
    socket = principled.inputs["Base Color"]
    for frame, color in keyframes:
        socket.default_value = color
        socket.keyframe_insert(data_path="default_value", frame=frame)
    return material



def _checker_material(
    name: str,
    color_a: tuple[float, float, float, float],
    color_b: tuple[float, float, float, float],
):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    checker = nodes.new(type="ShaderNodeTexChecker")
    checker.inputs["Color1"].default_value = color_a
    checker.inputs["Color2"].default_value = color_b
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
    image.generated_color = (0.8, 0.05, 0.02, 1.0)
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



def _append_materials(obj, materials: tuple[object, object]) -> None:
    if len(materials) != 2:
        raise ValueError("two materials are required")
    for material in materials:
        obj.data.materials.append(material)
    _assert(len(obj.material_slots) == 2, f"{obj.name} did not receive two slots")



def _single_settings(
    output_directory: Path,
    *,
    prefix: str,
    output_stem: str,
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
        output_stem=output_stem,
        source_geometry_mode=A1SourceGeometryMode.ORIGINAL,
        uv=UvUnwrapSettings(layer_name="SpineBakeUV"),
        # Keep production DIFFUSE/COMBINED mode selection. Only samples are lowered.
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
            output_stem=prefix,
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
        width, height = (int(image.size[0]), int(image.size[1]))
        pixels = tuple(float(value) for value in image.pixels[:])
        expected_length = width * height * 4
        _assert(
            len(pixels) == expected_length,
            f"Decoded pixel length is wrong for {path.name}: {len(pixels)}",
        )

        opaque = 0
        colored = 0
        red = 0
        green = 0
        blue = 0
        maximum_rgb = 0.0
        sum_r = 0.0
        sum_g = 0.0
        sum_b = 0.0
        for index in range(0, len(pixels), 4):
            r, g, b, a = pixels[index : index + 4]
            maximum_rgb = max(maximum_rgb, r, g, b)
            if a <= ALPHA_THRESHOLD:
                continue
            opaque += 1
            sum_r += r
            sum_g += g
            sum_b += b
            if max(r, g, b) > COLOR_THRESHOLD:
                colored += 1
            if r > g + DOMINANCE_MARGIN and r > b + DOMINANCE_MARGIN:
                red += 1
            if g > r + DOMINANCE_MARGIN and g > b + DOMINANCE_MARGIN:
                green += 1
            if b > r + DOMINANCE_MARGIN and b > g + DOMINANCE_MARGIN:
                blue += 1

        mean = (
            (sum_r / opaque) if opaque else 0.0,
            (sum_g / opaque) if opaque else 0.0,
            (sum_b / opaque) if opaque else 0.0,
        )
        return DecodedImage(
            width=width,
            height=height,
            pixels=pixels,
            opaque_pixels=opaque,
            colored_pixels=colored,
            red_dominant_pixels=red,
            green_dominant_pixels=green,
            blue_dominant_pixels=blue,
            maximum_rgb=maximum_rgb,
            mean_opaque_rgb=mean,
        )
    finally:
        if image is not None:
            bpy.data.images.remove(image)



def _assert_usable_bake(decoded: DecodedImage, label: str) -> None:
    minimum_opaque = max(4, int(decoded.pixel_count * 0.01))
    _assert(
        decoded.opaque_pixels >= minimum_opaque,
        f"{label} is fully transparent or nearly empty: {decoded}",
    )
    _assert(
        decoded.colored_pixels > 0,
        f"{label} is black: {decoded}",
    )
    _assert(
        decoded.maximum_rgb > COLOR_THRESHOLD,
        f"{label} contains no usable RGB values: {decoded}",
    )



def _mean_absolute_pixel_delta(first: DecodedImage, second: DecodedImage) -> float:
    _assert(
        (first.width, first.height) == (second.width, second.height),
        "sequence frame dimensions changed",
    )
    _assert(len(first.pixels) == len(second.pixels), "sequence pixel lengths changed")
    return sum(abs(a - b) for a, b in zip(first.pixels, second.pixels)) / len(
        first.pixels
    )



def _mesh_attachments(document: dict) -> tuple[tuple[str, str, dict], ...]:
    result: list[tuple[str, str, dict]] = []
    for skin in document.get("skins", ()):  # Spine 4.2 array skin format.
        for slot_name, attachments in skin.get("attachments", {}).items():
            for attachment_name, attachment in attachments.items():
                if attachment.get("type") == "mesh":
                    result.append((slot_name, attachment_name, attachment))
    return tuple(result)



def _capture_materials(materials: tuple[object, ...]) -> tuple[tuple[object, ...], ...]:
    return tuple(_material_fingerprint(material) for material in materials)



def _assert_restored(
    *,
    context_before,
    scene_before,
    materials: tuple[object, ...],
    material_before: tuple[tuple[object, ...], ...],
) -> None:
    _assert(_capture_context() == context_before, "bake matrix changed Blender context")
    _assert(
        _capture_scene_bake_state() == scene_before,
        "bake matrix changed scene bake settings or timeline frame",
    )
    _assert(
        _capture_materials(materials) == material_before,
        "bake matrix mutated source material node trees",
    )
    _assert(
        not _temporary_datablock_names(),
        "bake matrix leaked temporary Blender datablocks",
    )



def _build_connected_matrix(output_directory: Path) -> MatrixFixture:
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
        _checker_material(
            "StaticC_Checker",
            (0.02, 0.25, 0.9, 1.0),
            (0.9, 0.25, 0.02, 1.0),
        ),
        _principled_material("StaticC_Cyan", (0.02, 0.8, 0.8, 1.0)),
    )
    _append_materials(static_a, materials[0:2])
    _append_materials(animated_b, materials[2:4])
    _append_materials(static_c, materials[4:6])

    sources = (
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
    )
    return MatrixFixture(
        sources=sources,
        materials=materials,
        static_a=static_a,
        animated_b=animated_b,
        static_c=static_c,
    )



def _connected_settings(output_directory: Path, output_stem: str):
    return A1MultiObjectExportSettings(
        output_directory=output_directory,
        output_stem=output_stem,
        mode=A1MultiObjectMode.CONNECTED,
        anchor_component_id="static_a",
    )



def test_standard_principled_multi_material_bake_is_not_black() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-legacy-standard-materials-") as directory:
        output_directory = Path(directory)
        source = _create_two_material_panel("StandardMaterials")
        red = _principled_material("StandardRed", (0.9, 0.02, 0.01, 1.0))
        green = _principled_material("StandardGreen", (0.01, 0.85, 0.03, 1.0))
        _append_materials(source, (red, green))
        sentinel = _create_sentinel()
        _activate_only(sentinel)
        source.select_set(False)
        bpy.context.scene.frame_set(9)
        context_before = _capture_context()
        scene_before = _capture_scene_bake_state()
        materials = (red, green)
        material_before = _capture_materials(materials)

        result = export_a1_single_object(
            source,
            _single_settings(
                output_directory,
                prefix="StandardMaterials",
                output_stem="StandardMaterials",
            ),
        )

        _assert(result.success, f"standard multi-material export failed: {result.issues}")
        png = output_directory / "images" / "StandardMaterials_Baked.png"
        decoded = _read_png(png)
        _assert_usable_bake(decoded, "standard Principled multi-material PNG")
        _assert(
            decoded.red_dominant_pixels > 4,
            f"red material disappeared from bake: {decoded}",
        )
        _assert(
            decoded.green_dominant_pixels > 4,
            f"green material disappeared from bake: {decoded}",
        )
        _assert_restored(
            context_before=context_before,
            scene_before=scene_before,
            materials=materials,
            material_before=material_before,
        )



def test_connected_rig_supports_one_sequence_and_two_static_multi_material_objects() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-legacy-connected-sequence-") as directory:
        output_directory = Path(directory)
        fixture = _build_connected_matrix(output_directory)
        sentinel = _create_sentinel()
        _activate_only(sentinel)
        for source in fixture.sources:
            source.source_object.select_set(False)
        bpy.context.scene.frame_set(11)
        context_before = _capture_context()
        scene_before = _capture_scene_bake_state()
        material_before = _capture_materials(fixture.materials)

        result = export_a1_multi_object(
            fixture.sources,
            _connected_settings(output_directory, "ConnectedSequenceRig"),
        )

        _assert(result.success, f"connected sequence export failed: {result.issues}")
        expected_json = (output_directory / "ConnectedSequenceRig.json").resolve()
        expected_paths = (
            expected_json,
            (output_directory / "images" / "StaticA_Baked.png").resolve(),
            (output_directory / "images" / "AnimatedB_Baked_0001.png").resolve(),
            (output_directory / "images" / "AnimatedB_Baked_0002.png").resolve(),
            (output_directory / "images" / "AnimatedB_Baked_0003.png").resolve(),
            (output_directory / "images" / "StaticC_Baked.png").resolve(),
        )
        _assert(
            result.output_files == expected_paths,
            f"static/sequence output set or order changed: {result.output_files}",
        )

        decoded = {path.name: _read_png(path) for path in expected_paths[1:]}
        for name, image in decoded.items():
            _assert_usable_bake(image, name)
        _assert(
            _mean_absolute_pixel_delta(
                decoded["AnimatedB_Baked_0001.png"],
                decoded["AnimatedB_Baked_0002.png"],
            )
            > 0.001,
            "AnimatedB frames 1 and 2 are pixel-identical",
        )
        _assert(
            _mean_absolute_pixel_delta(
                decoded["AnimatedB_Baked_0002.png"],
                decoded["AnimatedB_Baked_0003.png"],
            )
            > 0.001,
            "AnimatedB frames 2 and 3 are pixel-identical",
        )

        document = json.loads(expected_json.read_text(encoding="utf-8"))
        bone_names = tuple(item["name"] for item in document["bones"])
        _assert(bone_names.count("root") == 1, "common rig duplicated root")
        _assert("all_objects_main" in bone_names, "common connected rig is missing")
        for prefix in ("StaticA", "AnimatedB", "StaticC"):
            main = next(item for item in document["bones"] if item["name"] == f"{prefix}_main")
            _assert(
                str(main.get("parent", "")).startswith("all_objects_layer_"),
                f"{prefix}_main is not parented into the common rig: {main}",
            )

        attachments = _mesh_attachments(document)
        animated = tuple(item for item in attachments if item[0].startswith("AnimatedB_"))
        static = tuple(
            item
            for item in attachments
            if item[0].startswith("StaticA_") or item[0].startswith("StaticC_")
        )
        _assert(animated, "AnimatedB mesh attachment is missing")
        _assert(static, "static mesh attachments are missing")
        for _, _, attachment in animated:
            _assert(
                attachment.get("path") == "images/AnimatedB_Baked_",
                f"sequence attachment path is wrong: {attachment}",
            )
            _assert(
                attachment.get("sequence")
                == {"count": 3, "start": 1, "digits": 4, "setup": 1},
                f"sequence metadata is wrong: {attachment.get('sequence')}",
            )
        for _, _, attachment in static:
            _assert("sequence" not in attachment, f"static attachment became a sequence: {attachment}")

        actual_png_names = tuple(
            sorted(path.name for path in (output_directory / "images").glob("*.png"))
        )
        _assert(
            actual_png_names
            == (
                "AnimatedB_Baked_0001.png",
                "AnimatedB_Baked_0002.png",
                "AnimatedB_Baked_0003.png",
                "StaticA_Baked.png",
                "StaticC_Baked.png",
            ),
            f"unexpected extra or missing sequence files: {actual_png_names}",
        )
        _assert_restored(
            context_before=context_before,
            scene_before=scene_before,
            materials=fixture.materials,
            material_before=material_before,
        )



def test_sequence_frame_failure_rolls_back_common_json_static_and_all_sequence_files() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-legacy-sequence-rollback-") as directory:
        output_directory = Path(directory)
        fixture = _build_connected_matrix(output_directory)
        sentinel = _create_sentinel()
        _activate_only(sentinel)
        for source in fixture.sources:
            source.source_object.select_set(False)
        bpy.context.scene.frame_set(13)
        context_before = _capture_context()
        scene_before = _capture_scene_bake_state()
        material_before = _capture_materials(fixture.materials)

        final_paths = (
            output_directory / "ConnectedSequenceRollback.json",
            output_directory / "images" / "StaticA_Baked.png",
            output_directory / "images" / "AnimatedB_Baked_0001.png",
            output_directory / "images" / "AnimatedB_Baked_0002.png",
            output_directory / "images" / "AnimatedB_Baked_0003.png",
            output_directory / "images" / "StaticC_Baked.png",
        )
        previous = {}
        for index, path in enumerate(final_paths):
            path.parent.mkdir(parents=True, exist_ok=True)
            content = f"previous-output-{index}".encode("utf-8")
            path.write_bytes(content)
            previous[path] = content

        original_call = bake_module._call_bake_operator
        call_count = 0

        def fail_second_sequence_frame(bpy_module, bake_type):
            nonlocal call_count
            call_count += 1
            # StaticA is call 1, AnimatedB frame 1 is call 2, frame 2 is call 3.
            if call_count == 3:
                raise BakeExecutionError("forced failure in AnimatedB sequence frame 2")
            return original_call(bpy_module, bake_type)

        with mock.patch.object(
            bake_module,
            "_call_bake_operator",
            side_effect=fail_second_sequence_frame,
        ):
            result = export_a1_multi_object(
                fixture.sources,
                _connected_settings(output_directory, "ConnectedSequenceRollback"),
            )

        _assert(not result.success, "forced sequence-frame failure returned success")
        _assert(call_count == 3, f"expected three bake calls, got {call_count}")
        issue = result.issues[-1]
        _assert(
            issue.stage == A1MultiObjectStage.STAGE_OUTPUTS.value,
            f"sequence rollback reported wrong stage: {issue.stage}",
        )
        _assert(
            issue.code == A1MultiObjectStage.STAGE_OUTPUTS.error_code,
            f"sequence rollback reported wrong code: {issue.code}",
        )
        for path, content in previous.items():
            _assert(path.read_bytes() == content, f"rollback corrupted {path.name}")

        leftovers = tuple(
            sorted(
                str(path.relative_to(output_directory))
                for path in output_directory.rglob("*")
                if path.is_file()
            )
        )
        expected_leftovers = tuple(
            sorted(str(path.relative_to(output_directory)) for path in final_paths)
        )
        _assert(
            leftovers == expected_leftovers,
            f"sequence rollback left staged/backup files: {leftovers}",
        )
        _assert_restored(
            context_before=context_before,
            scene_before=scene_before,
            materials=fixture.materials,
            material_before=material_before,
        )



def main() -> None:
    tests = (
        test_standard_principled_multi_material_bake_is_not_black,
        test_connected_rig_supports_one_sequence_and_two_static_multi_material_objects,
        test_sequence_frame_failure_rolls_back_common_json_static_and_all_sequence_files,
    )
    print(f"Blender version: {bpy.app.version_string}")
    for test in tests:
        print(f"[LEGACY_BAKE_MATRIX] RUN {test.__name__}")
        test()
        print(f"[LEGACY_BAKE_MATRIX] PASS {test.__name__}")
    print(f"[LEGACY_BAKE_MATRIX] PASS {len(tests)} integration tests")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
