"""Real Blender 5.2 Normal-UV sequence acceptance with animated transforms.

This runner deliberately exercises the complete user-facing export path rather than a
planner fixture. It bakes 128x128 PNG sequences from a material that consumes Texture
Coordinate Camera once and Reflection twice while the source object changes location,
rotation, and scale between frames.
"""

from __future__ import annotations

import json
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

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    export_a1_single_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    A1TextureExportMode,
    BakeExecutionSettings,
    TextureSequenceTiming,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (  # noqa: E402
    A1RigProfile,
    SpineJsonTarget,
    SpineTextureAnimationEncoding,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import UvUnwrapSettings  # noqa: E402
from run_bake_integration import (  # noqa: E402
    _activate_only,
    _assert,
    _clear_scene,
    _create_mesh_object,
)
from run_camera_projection_integration import (  # noqa: E402
    _configure_scene,
    _create_camera,
    _read_image,
)


_TEXTURE_SIZE = 128
_ANALYSIS_FRAME = 19
_FLOAT32_EPSILON = 2.0 ** -23
_MATRIX_COMPARE_ULPS = 4.0


def _create_pentagon(name: str):
    return _create_mesh_object(
        name,
        (
            (-1.45, -0.75, 0.0),
            (-0.55, -1.25, 0.0),
            (1.25, -0.55, 0.0),
            (1.05, 1.0, 0.0),
            (-0.85, 1.35, 0.0),
        ),
        ((0, 1, 2, 3, 4),),
    )


def _scaled_math(nodes, name: str, factor: float):
    node = nodes.new(type="ShaderNodeMath")
    node.name = name
    node.operation = "MULTIPLY"
    node.inputs[1].default_value = factor
    return node


def _add_math(nodes, name: str, offset: float | None = None):
    node = nodes.new(type="ShaderNodeMath")
    node.name = name
    node.operation = "ADD"
    if offset is not None:
        node.inputs[1].default_value = offset
    return node


def _create_animated_camera_reflection_material(name: str):
    """Create the capability shape reported by the user's crystal material."""

    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    coordinates = nodes.new(type="ShaderNodeTexCoord")

    camera_xyz = nodes.new(type="ShaderNodeSeparateXYZ")
    camera_xyz.name = "Camera Coordinates"
    reflection_xyz_a = nodes.new(type="ShaderNodeSeparateXYZ")
    reflection_xyz_a.name = "Reflection Coordinates A"
    reflection_xyz_b = nodes.new(type="ShaderNodeSeparateXYZ")
    reflection_xyz_b.name = "Reflection Coordinates B"

    camera_scale = _scaled_math(nodes, "Camera Scale", 0.30)
    reflection_scale_a = _scaled_math(nodes, "Reflection X Scale", 0.20)
    reflection_scale_b = _scaled_math(nodes, "Reflection Z Scale", 0.20)
    camera_plus_reflection = _add_math(nodes, "Camera Plus Reflection")
    reflection_sum = _add_math(nodes, "Reflection Sum")
    centered_factor = _add_math(nodes, "Centered Factor", 0.50)
    centered_factor.use_clamp = True
    ramp = nodes.new(type="ShaderNodeValToRGB")

    links.new(coordinates.outputs["Camera"], camera_xyz.inputs["Vector"])
    links.new(coordinates.outputs["Reflection"], reflection_xyz_a.inputs["Vector"])
    links.new(coordinates.outputs["Reflection"], reflection_xyz_b.inputs["Vector"])
    links.new(camera_xyz.outputs["Z"], camera_scale.inputs[0])
    links.new(reflection_xyz_a.outputs["X"], reflection_scale_a.inputs[0])
    links.new(reflection_xyz_b.outputs["Z"], reflection_scale_b.inputs[0])
    links.new(camera_scale.outputs[0], camera_plus_reflection.inputs[0])
    links.new(reflection_scale_a.outputs[0], camera_plus_reflection.inputs[1])
    links.new(camera_plus_reflection.outputs[0], reflection_sum.inputs[0])
    links.new(reflection_scale_b.outputs[0], reflection_sum.inputs[1])
    links.new(reflection_sum.outputs[0], centered_factor.inputs[0])
    links.new(centered_factor.outputs[0], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    left = ramp.color_ramp.elements[0]
    right = ramp.color_ramp.elements[1]
    for frame, left_color, right_color, strength in (
        (1, (0.02, 0.15, 1.0, 1.0), (0.05, 1.0, 0.75, 1.0), 0.8),
        (2, (0.9, 0.02, 0.35, 1.0), (1.0, 0.45, 0.02, 1.0), 1.4),
        (3, (0.3, 0.02, 1.0, 1.0), (0.95, 0.05, 1.0, 1.0), 2.0),
    ):
        left.color = left_color
        right.color = right_color
        emission.inputs["Strength"].default_value = strength
        left.keyframe_insert(data_path="color", frame=frame)
        right.keyframe_insert(data_path="color", frame=frame)
        emission.inputs["Strength"].keyframe_insert(
            data_path="default_value",
            frame=frame,
        )
    return material


def _animate_source_transform(source) -> None:
    """Key bake frames and a distinct analysis frame for pipeline regression coverage."""

    transforms = (
        (1, (-0.85, 0.10, 0.00), (0.15, -0.35, -0.30), (0.75, 1.10, 1.00)),
        (2, (0.65, 0.30, 0.45), (0.45, 0.20, 0.50), (1.20, 0.70, 1.00)),
        (3, (0.20, -0.55, -0.20), (-0.35, 0.55, 0.90), (0.90, 1.35, 1.00)),
        (
            _ANALYSIS_FRAME,
            (1.35, -0.80, 0.65),
            (0.80, -0.90, 1.20),
            (1.40, 0.85, 1.10),
        ),
    )
    for frame, location, rotation, scale in transforms:
        source.location = location
        source.rotation_euler = rotation
        source.scale = scale
        source.keyframe_insert(data_path="location", frame=frame)
        source.keyframe_insert(data_path="rotation_euler", frame=frame)
        source.keyframe_insert(data_path="scale", frame=frame)


def _matrix_tuple(obj) -> tuple[float, ...]:
    matrix = obj.matrix_world
    return tuple(
        float(matrix[row][column])
        for row in range(4)
        for column in range(4)
    )


def _maximum_matrix_delta(
    first: tuple[float, ...],
    second: tuple[float, ...],
) -> float:
    _assert(len(first) == 16 and len(second) == 16, "matrix must contain 16 values")
    return max(abs(left - right) for left, right in zip(first, second, strict=True))


def _matrices_equal_at_float32_precision(
    first: tuple[float, ...],
    second: tuple[float, ...],
) -> bool:
    if len(first) != 16 or len(second) != 16:
        return False
    tolerance_scale = _FLOAT32_EPSILON * _MATRIX_COMPARE_ULPS
    return all(
        abs(left - right)
        <= tolerance_scale * max(1.0, abs(left), abs(right)) + tolerance_scale
        for left, right in zip(first, second, strict=True)
    )


def _frame_matrices(scene, source, frame_count: int) -> tuple[tuple[float, ...], ...]:
    original_frame = int(scene.frame_current)
    try:
        values = []
        for frame in range(1, frame_count + 1):
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            values.append(_matrix_tuple(source))
        return tuple(values)
    finally:
        scene.frame_set(original_frame)
        bpy.context.view_layer.update()


def _settings(
    output_directory: Path,
    *,
    target: SpineJsonTarget,
    frame_count: int,
    stem: str,
) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=_TEXTURE_SIZE,
            texture_height=_TEXTURE_SIZE,
            output_directory=output_directory,
            images_relative_path="images",
            spine_version=target.exact_version,
            rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE.value,
            bake_margin=1,
            sequence_start_frame=1,
            sequence_frame_count=frame_count,
            sequence_timing=TextureSequenceTiming(
                scene_fps=24,
                scene_fps_base=1.0,
            ),
        ),
        prefix=stem,
        output_stem=stem,
        json_output_stem=stem,
        source_geometry_mode=A1SourceGeometryMode.EVALUATED,
        uv=UvUnwrapSettings(layer_name="SpineBakeUV"),
        bake_execution=BakeExecutionSettings(
            samples=1,
            texture_export_mode=A1TextureExportMode.NORMAL_UV_SEGMENTS,
        ),
    )


def _rgb_signature(pixels: tuple[float, ...]) -> tuple[float, float, float]:
    red = green = blue = weight = 0.0
    for offset in range(0, len(pixels), 4):
        alpha = float(pixels[offset + 3])
        if alpha <= 0.01:
            continue
        red += float(pixels[offset]) * alpha
        green += float(pixels[offset + 1]) * alpha
        blue += float(pixels[offset + 2]) * alpha
        weight += alpha
    _assert(weight > 0.0, "Normal UV frame contains no visible pixels")
    return red / weight, green / weight, blue / weight


def _read_and_validate_frames(paths: tuple[Path, ...]) -> None:
    images = tuple(_read_image(path) for path in paths)
    for path, (size, pixels) in zip(paths, images, strict=True):
        _assert(
            size == (_TEXTURE_SIZE, _TEXTURE_SIZE),
            f"wrong PNG size for {path}: {size}",
        )
        _assert(
            len(pixels) == _TEXTURE_SIZE * _TEXTURE_SIZE * 4,
            "wrong RGBA buffer size",
        )

    signatures = tuple(_rgb_signature(pixels) for _, pixels in images)
    for first_index in range(len(signatures)):
        for second_index in range(first_index + 1, len(signatures)):
            delta = sum(
                abs(first - second)
                for first, second in zip(
                    signatures[first_index],
                    signatures[second_index],
                    strict=True,
                )
            )
            _assert(
                delta > 0.10,
                "Normal UV sequence frames are not visually distinct; "
                f"frames={(first_index, second_index)}, delta={delta}",
            )


def _visual_slot(payload: dict[str, object]) -> tuple[str, str]:
    slots = payload.get("slots")
    _assert(isinstance(slots, list), "serialized slots must be a list")
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        slot_name = slot.get("name")
        attachment_name = slot.get("attachment")
        if isinstance(slot_name, str) and isinstance(attachment_name, str):
            return slot_name, attachment_name
    raise AssertionError("serialized sequence document has no visual setup slot")


def _skin_attachments(
    payload: dict[str, object],
    slot_name: str,
) -> dict[str, object]:
    skins = payload.get("skins")
    _assert(isinstance(skins, list) and skins, "serialized skins must be non-empty")
    for skin in skins:
        if not isinstance(skin, dict):
            continue
        attachments = skin.get("attachments")
        if not isinstance(attachments, dict):
            continue
        slot_attachments = attachments.get(slot_name)
        if isinstance(slot_attachments, dict):
            return slot_attachments
    raise AssertionError(f"no skin attachments found for slot {slot_name!r}")


def _contains_sequence(value: object) -> bool:
    if isinstance(value, dict):
        return "sequence" in value or any(
            _contains_sequence(child) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_sequence(child) for child in value)
    return False


def _assert_mesh_topology(attachments: dict[str, object]) -> None:
    _assert(attachments, "sequence attachment set must not be empty")
    for name, attachment in attachments.items():
        _assert(
            isinstance(attachment, dict),
            f"attachment {name!r} must be a mapping",
        )
        _assert(
            attachment.get("type") == "mesh",
            f"attachment {name!r} is not mesh",
        )
        uvs = attachment.get("uvs")
        _assert(isinstance(uvs, list), f"attachment {name!r} UVs must be a list")
        _assert(
            len(uvs) == 10,
            "Normal UV export must preserve the five-vertex source topology; "
            f"attachment={name!r}, UV scalar count={len(uvs)}",
        )


def _assert_sequence_payload(
    payload: dict[str, object],
    *,
    target: SpineJsonTarget,
    frame_count: int,
) -> None:
    slot_name, setup_attachment = _visual_slot(payload)
    attachments = _skin_attachments(payload, slot_name)
    _assert_mesh_topology(attachments)

    if target.texture_animation_encoding is SpineTextureAnimationEncoding.NATIVE_SEQUENCE:
        _assert(len(attachments) == 1, "native target must keep one mesh attachment")
        attachment = attachments[setup_attachment]
        sequence = attachment.get("sequence")
        _assert(isinstance(sequence, dict), "native sequence metadata is missing")
        _assert(sequence.get("count") == frame_count, "native sequence count is wrong")
        timeline = payload["animations"]["animation"]["attachments"]["default"][
            slot_name
        ][setup_attachment]["sequence"]
        _assert(len(timeline) == 2, "native Loop sequence must use two boundary keys")
        _assert(timeline[0].get("mode") == "loop", "native sequence is not Loop")
        return

    _assert(
        not _contains_sequence(payload),
        "legacy target must not contain sequence metadata",
    )
    _assert(
        len(attachments) == frame_count,
        "legacy target must contain one mesh attachment per frame",
    )
    timeline = payload["animations"]["animation"]["slots"][slot_name]["attachment"]
    _assert(
        len(timeline) == frame_count + 1,
        "legacy Loop must contain every frame plus one wrap key",
    )
    frame_names = tuple(key.get("name") for key in timeline[:-1])
    _assert(len(set(frame_names)) == frame_count, "legacy frame keys are not unique")
    _assert(
        all(name in attachments for name in frame_names),
        "legacy key references missing mesh",
    )
    _assert(timeline[-1].get("name") == frame_names[0], "legacy Loop does not wrap")


def _run_export_case(
    *,
    target: SpineJsonTarget,
    frame_count: int,
    stem: str,
) -> None:
    _clear_scene()
    _configure_scene()
    _create_camera()
    source = _create_pentagon(f"{stem}_Source")
    source.data.materials.append(
        _create_animated_camera_reflection_material(f"{stem}_Material")
    )
    _animate_source_transform(source)
    _activate_only(source)

    scene = bpy.context.scene
    scene.frame_set(_ANALYSIS_FRAME)
    bpy.context.view_layer.update()
    original_matrix = _matrix_tuple(source)
    matrices = _frame_matrices(scene, source, frame_count)

    for first_index in range(frame_count):
        for second_index in range(first_index + 1, frame_count):
            delta = _maximum_matrix_delta(
                matrices[first_index],
                matrices[second_index],
            )
            _assert(
                delta > 1.0e-3,
                "test fixture must provide a meaningfully different matrix_world "
                f"for bake frames {(first_index + 1, second_index + 1)}; delta={delta}",
            )
    for frame_index, matrix in enumerate(matrices, start=1):
        delta = _maximum_matrix_delta(matrix, original_matrix)
        _assert(
            delta > 1.0e-3,
            "analysis-frame matrix must differ from every requested bake frame; "
            f"frame={frame_index}, delta={delta}",
        )

    with tempfile.TemporaryDirectory(prefix=f"spine2d-{stem.casefold()}-") as directory:
        output_directory = Path(directory)
        result = export_a1_single_object(
            source,
            _settings(
                output_directory,
                target=target,
                frame_count=frame_count,
                stem=stem,
            ),
            context=bpy.context,
            scene=scene,
        )
        _assert(result.success, f"Normal UV sequence export failed: {result.issues}")
        _assert(
            scene.frame_current == _ANALYSIS_FRAME,
            "sequence export did not restore scene.frame_current",
        )
        bpy.context.view_layer.update()
        restored_matrix = _matrix_tuple(source)
        _assert(
            _matrices_equal_at_float32_precision(restored_matrix, original_matrix),
            "source matrix_world was not restored with the analysis frame; "
            f"maximum_delta={_maximum_matrix_delta(restored_matrix, original_matrix)}",
        )
        _assert(
            len(result.output_files) == frame_count + 1,
            "expected one JSON plus one PNG per sequence frame",
        )

        json_path = result.output_files[0]
        texture_paths = tuple(result.output_files[1:])
        _assert(json_path.is_file(), f"missing JSON output: {json_path}")
        _assert(all(path.is_file() for path in texture_paths), "missing sequence PNG")
        _read_and_validate_frames(texture_paths)

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        _assert_sequence_payload(
            payload,
            target=target,
            frame_count=frame_count,
        )


def test_two_frame_spine38_normal_uv_sequence_with_animated_transform() -> None:
    _run_export_case(
        target=SpineJsonTarget.SPINE_3_8,
        frame_count=2,
        stem="CrystalNormalUv38",
    )


def test_three_frame_spine42_normal_uv_sequence_with_animated_transform() -> None:
    _run_export_case(
        target=SpineJsonTarget.SPINE_4_2,
        frame_count=3,
        stem="CrystalNormalUv42",
    )


def main() -> None:
    tests = (
        test_two_frame_spine38_normal_uv_sequence_with_animated_transform,
        test_three_frame_spine42_normal_uv_sequence_with_animated_transform,
    )
    failures: list[tuple[str, str]] = []
    print(f"Blender version: {bpy.app.version_string}")
    for test in tests:
        print(f"[NORMAL-UV-SEQUENCE-128] RUN {test.__name__}")
        try:
            test()
        except Exception:
            failures.append((test.__name__, traceback.format_exc()))
            print(f"[NORMAL-UV-SEQUENCE-128] FAIL {test.__name__}")
        else:
            print(f"[NORMAL-UV-SEQUENCE-128] PASS {test.__name__}")
        finally:
            _clear_scene()

    if failures:
        for name, details in failures:
            print(f"\n--- {name} ---\n{details}")
        raise SystemExit(1)
    print(f"[NORMAL-UV-SEQUENCE-128] PASS {len(tests)} integration tests")


if __name__ == "__main__":
    main()
