"""Real Blender 5.2 acceptance for camera-context materials in Normal UV mode."""

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
    _read_pixels,
)


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


def _create_animated_camera_context_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    coordinates = nodes.new(type="ShaderNodeTexCoord")
    separate = nodes.new(type="ShaderNodeSeparateXYZ")
    normal_scale = nodes.new(type="ShaderNodeMath")
    normal_scale.operation = "MULTIPLY"
    normal_scale.inputs[1].default_value = 0.35
    normal_offset = nodes.new(type="ShaderNodeMath")
    normal_offset.operation = "ADD"
    normal_offset.inputs[1].default_value = 0.5
    fresnel = nodes.new(type="ShaderNodeFresnel")
    fresnel.inputs["IOR"].default_value = 1.45
    fresnel_scale = nodes.new(type="ShaderNodeMath")
    fresnel_scale.operation = "MULTIPLY"
    fresnel_scale.inputs[1].default_value = 0.45
    factor = nodes.new(type="ShaderNodeMath")
    factor.operation = "ADD"
    factor.use_clamp = True
    ramp = nodes.new(type="ShaderNodeValToRGB")

    links.new(coordinates.outputs["Normal"], separate.inputs["Vector"])
    links.new(separate.outputs["Z"], normal_scale.inputs[0])
    links.new(normal_scale.outputs[0], normal_offset.inputs[0])
    links.new(fresnel.outputs["Fac"], fresnel_scale.inputs[0])
    links.new(normal_offset.outputs[0], factor.inputs[0])
    links.new(fresnel_scale.outputs[0], factor.inputs[1])
    links.new(factor.outputs[0], ramp.inputs["Fac"])
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


def _settings(output_directory: Path) -> A1SingleObjectExportSettings:
    target = SpineJsonTarget.SPINE_4_2
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=64,
            texture_height=64,
            output_directory=output_directory,
            images_relative_path="images",
            spine_version=target.exact_version,
            rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE.value,
            bake_margin=1,
            sequence_start_frame=1,
            sequence_frame_count=3,
            sequence_timing=TextureSequenceTiming(
                scene_fps=24,
                scene_fps_base=1.0,
            ),
        ),
        prefix="CrystalNormalUv",
        output_stem="CrystalNormalUv",
        json_output_stem="CrystalNormalUv",
        source_geometry_mode=A1SourceGeometryMode.EVALUATED,
        uv=UvUnwrapSettings(layer_name="SpineBakeUV"),
        bake_execution=BakeExecutionSettings(
            samples=2,
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


def _assert_distinct_frames(paths: tuple[Path, ...]) -> None:
    signatures = tuple(_rgb_signature(_read_pixels(path)) for path in paths)
    _assert(len(signatures) == 3, "expected exactly three Normal UV sequence frames")
    for first_index in range(3):
        for second_index in range(first_index + 1, 3):
            delta = sum(
                abs(first - second)
                for first, second in zip(
                    signatures[first_index],
                    signatures[second_index],
                    strict=True,
                )
            )
            _assert(
                delta > 0.12,
                "camera-context Normal UV frames are not visually distinct; "
                f"frames={(first_index, second_index)}, delta={delta}",
            )


def _first_mesh_attachment(payload: dict[str, object]) -> dict[str, object]:
    skins = payload.get("skins")
    _assert(isinstance(skins, list) and skins, "serialized skins must be non-empty")
    for skin in skins:
        if not isinstance(skin, dict):
            continue
        attachments = skin.get("attachments")
        if not isinstance(attachments, dict):
            continue
        for slot_attachments in attachments.values():
            if not isinstance(slot_attachments, dict):
                continue
            for attachment in slot_attachments.values():
                if isinstance(attachment, dict) and attachment.get("type") == "mesh":
                    return attachment
    raise AssertionError("serialized document contains no mesh attachment")


def test_normal_uv_camera_context_material_exports_sequence() -> None:
    _clear_scene()
    _configure_scene()
    _create_camera()
    source = _create_pentagon("CrystalSource")
    source.data.materials.append(
        _create_animated_camera_context_material("CrystalCameraContextMaterial")
    )
    _activate_only(source)
    scene = bpy.context.scene
    scene.frame_set(19)

    with tempfile.TemporaryDirectory(prefix="spine2d-normal-uv-camera-context-") as directory:
        output_directory = Path(directory)
        result = export_a1_single_object(
            source,
            _settings(output_directory),
            context=bpy.context,
            scene=scene,
        )
        _assert(result.success, f"Normal UV camera-context export failed: {result.issues}")
        _assert(scene.frame_current == 19, "Normal UV export did not restore frame_current")
        _assert(len(result.output_files) == 4, "expected JSON plus three PNG files")

        json_path = result.output_files[0]
        texture_paths = tuple(result.output_files[1:])
        _assert(json_path.is_file(), f"missing JSON output: {json_path}")
        _assert(all(path.is_file() for path in texture_paths), "missing Normal UV PNG")
        _assert_distinct_frames(texture_paths)

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        attachment = _first_mesh_attachment(payload)
        uvs = attachment.get("uvs")
        _assert(isinstance(uvs, list), "mesh attachment UVs must be a list")
        _assert(
            len(uvs) == 10,
            "Normal UV export must preserve the five-vertex source topology; "
            f"actual UV scalar count={len(uvs)}",
        )


def main() -> None:
    tests = (test_normal_uv_camera_context_material_exports_sequence,)
    failures: list[tuple[str, str]] = []
    print(f"Blender version: {bpy.app.version_string}")
    for test in tests:
        print(f"[NORMAL-UV-CAMERA-CONTEXT] RUN {test.__name__}")
        try:
            test()
        except Exception:
            failures.append((test.__name__, traceback.format_exc()))
            print(f"[NORMAL-UV-CAMERA-CONTEXT] FAIL {test.__name__}")
        else:
            print(f"[NORMAL-UV-CAMERA-CONTEXT] PASS {test.__name__}")
        finally:
            _clear_scene()

    if failures:
        for name, details in failures:
            print(f"\n--- {name} ---\n{details}")
        raise SystemExit(1)
    print(f"[NORMAL-UV-CAMERA-CONTEXT] PASS {len(tests)} integration test")


if __name__ == "__main__":
    main()
