"""Real Blender 5.2 acceptance for animated texture sequence export."""

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
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.camera_projection_state import (  # noqa: E402
    configure_camera_visibility,
    preserve_camera_projection_state,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    A1TextureExportMode,
    BakeExecutionSettings,
    CameraProjectionInfluencePolicy,
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
    _read_pixels,
)


_TARGETS = tuple(SpineJsonTarget)


def _create_quad(name: str):
    return _create_mesh_object(
        name,
        (
            (-1.5, -1.0, 0.0),
            (1.5, -1.0, 0.0),
            (1.5, 1.0, 0.0),
            (-1.5, 1.0, 0.0),
        ),
        ((0, 1, 2, 3),),
    )


def _create_animated_emission_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    material.node_tree.links.new(
        emission.outputs["Emission"],
        output.inputs["Surface"],
    )

    color = emission.inputs["Color"]
    strength = emission.inputs["Strength"]
    for frame, color_value, strength_value in (
        (1, (1.0, 0.02, 0.01, 1.0), 0.6),
        (2, (0.01, 1.0, 0.03, 1.0), 1.4),
        (3, (0.02, 0.05, 1.0, 1.0), 2.2),
    ):
        color.default_value = color_value
        strength.default_value = strength_value
        color.keyframe_insert(data_path="default_value", frame=frame)
        strength.keyframe_insert(data_path="default_value", frame=frame)
    return material


def _settings(
    output_directory: Path,
    stem: str,
    target: SpineJsonTarget,
) -> A1SingleObjectExportSettings:
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
                scene_fps=24000,
                scene_fps_base=1001.0,
            ),
        ),
        prefix=stem,
        output_stem=stem,
        json_output_stem=stem,
        source_geometry_mode=A1SourceGeometryMode.EVALUATED,
        uv=UvUnwrapSettings(layer_name="SpineBakeUV"),
        bake_execution=BakeExecutionSettings(
            samples=2,
            texture_export_mode=A1TextureExportMode.CAMERA_PROJECTION,
        ),
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
    _assert(weight > 0.0, "sequence frame contains no visible pixels")
    return red / weight, green / weight, blue / weight


def _assert_distinct_frames(paths: tuple[Path, ...]) -> None:
    _assert(len(paths) == 3, f"expected 3 texture frames, got {len(paths)}")
    signatures = tuple(_rgb_signature(_read_pixels(path)) for path in paths)
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
                delta > 0.15,
                "animated material frames are not visually distinct; "
                f"frames={(first_index, second_index)}, delta={delta}",
            )


def _assert_native_payload(
    payload: dict[str, object],
    slot_name: str,
    attachment_name: str,
) -> None:
    attachments = _skin_attachments(payload, slot_name)
    _assert(tuple(attachments) == (attachment_name,), "native target must keep one attachment")
    attachment = attachments[attachment_name]
    _assert(isinstance(attachment, dict), "native attachment must be a mapping")
    sequence = attachment.get("sequence")
    _assert(isinstance(sequence, dict), "native attachment sequence is missing")
    _assert(sequence.get("count") == 3, "native sequence count must be 3")
    _assert(sequence.get("start") == 1, "native sequence start must be 1")
    timeline = payload["animations"]["animation"]["attachments"]["default"][
        slot_name
    ][attachment_name]["sequence"]
    _assert(len(timeline) == 2, "native Loop sequence must use two boundary keys")
    _assert(timeline[0]["mode"] == "loop", "native sequence mode must be Loop")
    _assert(timeline[0]["index"] == 0, "native sequence must start at index zero")


def _assert_attachment_swap_payload(
    payload: dict[str, object],
    slot_name: str,
    setup_attachment_name: str,
) -> None:
    _assert(not _contains_sequence(payload), "legacy target must contain no sequence member")
    attachments = _skin_attachments(payload, slot_name)
    _assert(len(attachments) == 3, "legacy target must contain one attachment per frame")
    _assert(set(attachments) == {
        setup_attachment_name,
        setup_attachment_name[:-1] + "2",
        setup_attachment_name[:-1] + "3",
    }, "legacy frame attachment names do not match 0001..0003")
    timeline = payload["animations"]["animation"]["slots"][slot_name]["attachment"]
    _assert(len(timeline) == 4, "legacy Loop requires three frame keys plus wrap key")
    _assert(
        tuple(key["name"] for key in timeline) == (
            setup_attachment_name,
            setup_attachment_name[:-1] + "2",
            setup_attachment_name[:-1] + "3",
            setup_attachment_name,
        ),
        "legacy attachment key order is incorrect",
    )


def test_all_targets_export_animated_material_sequence() -> None:
    _clear_scene()
    _configure_scene()
    _create_camera()
    source = _create_quad("AnimatedSequenceSource")
    source.data.materials.append(
        _create_animated_emission_material("AnimatedSequenceMaterial")
    )
    _activate_only(source)
    scene = bpy.context.scene
    scene.frame_set(17)

    with tempfile.TemporaryDirectory(prefix="spine2d-sequence-") as directory:
        root = Path(directory)
        for target in _TARGETS:
            output_directory = root / target.value
            stem = f"Animated_{target.family.replace('.', '_')}"
            result = export_a1_single_object(
                source,
                _settings(output_directory, stem, target),
                context=bpy.context,
                scene=scene,
            )
            _assert(
                result.success,
                f"{target.value} sequence export failed: {result.issues}",
            )
            _assert(scene.frame_current == 17, "sequence export did not restore frame_current")
            _assert(len(result.output_files) == 4, "expected JSON plus three PNG files")
            json_path = result.output_files[0]
            texture_paths = tuple(result.output_files[1:])
            _assert(json_path.is_file(), f"missing JSON output: {json_path}")
            _assert(all(path.is_file() for path in texture_paths), "missing sequence PNG")
            _assert_distinct_frames(texture_paths)

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            slot_name, setup_attachment_name = _visual_slot(payload)
            expected_fps = round(24000.0 / 1001.0, 6)
            _assert(
                abs(float(payload["skeleton"]["fps"]) - expected_fps) < 1.0e-6,
                "serialized sequence FPS differs from Scene FPS",
            )
            if (
                target.texture_animation_encoding
                is SpineTextureAnimationEncoding.NATIVE_SEQUENCE
            ):
                _assert_native_payload(
                    payload,
                    slot_name,
                    setup_attachment_name,
                )
            else:
                _assert_attachment_swap_payload(
                    payload,
                    slot_name,
                    setup_attachment_name,
                )


def test_real_blender_camera_influence_flags_restore() -> None:
    _clear_scene()
    _configure_scene()
    _create_camera()
    source = _create_quad("InfluenceSource")
    dependency = _create_quad("InfluenceDependency")
    dependency.location.x = 4.0
    _activate_only(source)
    scene = bpy.context.scene
    original_world = scene.world
    original = (
        dependency.visible_camera,
        dependency.visible_shadow,
        dependency.visible_glossy,
        dependency.visible_transmission,
    )

    with preserve_camera_projection_state(scene):
        configure_camera_visibility(
            source,
            scene,
            isolate=True,
            influence_policy=CameraProjectionInfluencePolicy(
                include_scene_shadows=False,
                include_scene_reflection_transmission=False,
                world_affects_lighting_reflections=False,
            ),
        )
        scene.world = None
        _assert(dependency.visible_camera is False, "dependency must be camera-hidden")
        _assert(dependency.visible_shadow is False, "dependency shadow flag was not disabled")
        _assert(dependency.visible_glossy is False, "dependency glossy flag was not disabled")
        _assert(
            dependency.visible_transmission is False,
            "dependency transmission flag was not disabled",
        )

    _assert(scene.world is original_world, "Scene World was not restored")
    _assert(
        (
            dependency.visible_camera,
            dependency.visible_shadow,
            dependency.visible_glossy,
            dependency.visible_transmission,
        ) == original,
        "dependency ray visibility was not restored",
    )


def main() -> None:
    tests = (
        test_all_targets_export_animated_material_sequence,
        test_real_blender_camera_influence_flags_restore,
    )
    failures: list[tuple[str, str]] = []
    print(f"Blender version: {bpy.app.version_string}")
    for test in tests:
        print(f"[TEXTURE-SEQUENCE] RUN {test.__name__}")
        try:
            test()
        except Exception:
            failures.append((test.__name__, traceback.format_exc()))
            print(f"[TEXTURE-SEQUENCE] FAIL {test.__name__}")
        else:
            print(f"[TEXTURE-SEQUENCE] PASS {test.__name__}")
        finally:
            _clear_scene()

    if failures:
        for name, details in failures:
            print(f"\n--- {name} ---\n{details}")
        raise SystemExit(1)
    print(f"[TEXTURE-SEQUENCE] PASS {len(tests)} integration tests")


if __name__ == "__main__":
    main()
