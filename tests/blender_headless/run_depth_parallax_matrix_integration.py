"""Blender 5.2 matrix acceptance for Depth parallax reserve camera variants.

The established positive Perspective smoke remains the minimal production path. This
runner adds two independent boundaries without duplicating exporter implementation:

* Orthographic static export with a fitted virtual reserve camera.
* Perspective two-frame material sequence with FRONT and reserve view-owned crops.

Every case uses the public prepare/export routes and verifies that Scene, camera,
materials, frame, selection, and temporary datablocks are restored.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
import tempfile

import bpy


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    export_a1_single_object,
    prepare_a1_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    TextureSequenceTiming,
)
from run_bake_integration import (  # noqa: E402
    PNG_SIGNATURE,
    _assert,
    _capture_context,
    _capture_scene_bake_state,
    _material_fingerprint,
    _temporary_datablock_names,
)
from run_camera_projection_integration import (  # noqa: E402
    _read_image,
    _scene_render_fingerprint,
    _visible_and_transparent_counts,
)
import run_depth_parallax_integration as smoke  # noqa: E402


_SEQUENCE_START = 1
_SEQUENCE_COUNT = 2


@dataclass(frozen=True, slots=True)
class _Case:
    key: str
    camera_type: str
    sequence_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("case key must be a non-empty string")
        if self.camera_type not in {"PERSP", "ORTHO"}:
            raise ValueError(f"unsupported camera_type: {self.camera_type}")
        if (
            isinstance(self.sequence_count, bool)
            or not isinstance(self.sequence_count, int)
            or self.sequence_count < 0
        ):
            raise ValueError("sequence_count must be a non-negative integer")

    @property
    def frame_count(self) -> int:
        return max(1, self.sequence_count)


_CASES = (
    _Case("orthographic-static", "ORTHO"),
    _Case("perspective-sequence", "PERSP", _SEQUENCE_COUNT),
)


def _emission_color_socket(material: object):
    node_tree = getattr(material, "node_tree", None)
    nodes = getattr(node_tree, "nodes", None)
    if nodes is None:
        raise AssertionError(f"material {getattr(material, 'name', '<unknown>')} has no nodes")
    emission_nodes = tuple(
        node for node in nodes if str(getattr(node, "bl_idname", "")) == "ShaderNodeEmission"
    )
    _assert(
        len(emission_nodes) == 1,
        f"expected one emission node in {getattr(material, 'name', '<unknown>')}",
    )
    socket = emission_nodes[0].inputs.get("Color")
    _assert(socket is not None, "emission node has no Color input")
    return socket


def _animate_emission(
    material: object,
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> None:
    socket = _emission_color_socket(material)
    socket.default_value = first
    socket.keyframe_insert(data_path="default_value", frame=_SEQUENCE_START)
    socket.default_value = second
    socket.keyframe_insert(data_path="default_value", frame=_SEQUENCE_START + 1)


def _case_settings(output_directory: Path, case: _Case):
    base = smoke._settings(output_directory)
    export = replace(
        base.export,
        output_directory=output_directory,
        sequence_start_frame=_SEQUENCE_START,
        sequence_frame_count=case.sequence_count,
        sequence_timing=TextureSequenceTiming(
            scene_fps=24,
            scene_fps_base=1.0,
        ),
    )
    return replace(base, export=export)


def _prepare_case_scene(case: _Case):
    source, front_material, reserve_material, camera = smoke._prepare_scene()
    camera.data.type = case.camera_type
    if case.camera_type == "ORTHO":
        camera.data.ortho_scale = 3.8
    if case.sequence_count:
        _animate_emission(
            front_material,
            (1.0, 0.02, 0.01, 1.0),
            (1.0, 0.02, 0.35, 1.0),
        )
        _animate_emission(
            reserve_material,
            (0.01, 1.0, 0.04, 1.0),
            (0.18, 1.0, 0.02, 1.0),
        )
    bpy.context.scene.frame_set(_SEQUENCE_START)
    bpy.context.view_layer.update()
    return source, front_material, reserve_material, camera


def _assert_view_images(
    label: str,
    paths: tuple[Path, ...],
    *,
    expected_frames: int,
    expect_red: bool,
) -> tuple[tuple[int, int], tuple[tuple[float, ...], ...]]:
    _assert(len(paths) == expected_frames, f"{label} frame count differs")
    image_data = []
    for path in paths:
        _assert(path.read_bytes().startswith(PNG_SIGNATURE), f"invalid PNG: {path}")
        image_data.append(_read_image(path))

    sizes = tuple(size for size, _pixels in image_data)
    _assert(
        len(set(sizes)) == 1,
        f"{label} sequence crop changed between frames: {sizes}",
    )
    size = sizes[0]
    _assert(
        1 <= size[0] <= smoke._TEXTURE_SIZE
        and 1 <= size[1] <= smoke._TEXTURE_SIZE,
        f"invalid {label} crop size: {size}",
    )

    pixels_by_frame = tuple(pixels for _size, pixels in image_data)
    for frame_index, pixels in enumerate(pixels_by_frame):
        visible, transparent = _visible_and_transparent_counts(pixels)
        _assert(
            visible > 100,
            f"{label} frame {frame_index} has too few visible pixels: {visible}",
        )
        _assert(
            transparent > 0,
            f"{label} frame {frame_index} crop has no transparent padding",
        )
        red, green = smoke._dominant_color_counts(pixels)
        if expect_red:
            _assert(red > 100, f"{label} frame {frame_index} lost front material: {red}")
            _assert(
                green < max(8, red // 50),
                f"reserve material leaked into {label} frame {frame_index}: green={green}",
            )
        else:
            _assert(
                green > 20,
                f"{label} frame {frame_index} lost reserve material: {green}",
            )
            _assert(
                red < max(8, green // 50),
                f"front material leaked into {label} frame {frame_index}: red={red}",
            )

    if expected_frames > 1:
        _assert(
            len(set(pixels_by_frame)) == expected_frames,
            f"{label} sequence frames are pixel-identical",
        )
    return size, pixels_by_frame


def _assert_sequence_metadata(
    document: dict[str, object],
    slot_name: str,
    sequence_count: int,
) -> None:
    group = smoke._attachment_group(document, slot_name)
    attachment = group.get(slot_name)
    _assert(isinstance(attachment, dict), f"attachment missing for {slot_name}")
    if sequence_count == 0:
        _assert("sequence" not in attachment, f"static attachment has sequence: {slot_name}")
        return
    sequence = attachment.get("sequence")
    _assert(isinstance(sequence, dict), f"sequence metadata missing for {slot_name}")
    _assert(
        int(sequence.get("count", -1)) == sequence_count,
        f"sequence count differs for {slot_name}: {sequence}",
    )


def _run_case(output_root: Path, case: _Case) -> None:
    output_directory = output_root / case.key
    output_directory.mkdir(parents=True, exist_ok=False)
    source, front_material, reserve_material, camera = _prepare_case_scene(case)
    settings = _case_settings(output_directory, case)

    context_before = _capture_context()
    bake_before = _capture_scene_bake_state()
    render_before = _scene_render_fingerprint()
    camera_before = smoke._camera_fingerprint(camera)
    front_material_before = _material_fingerprint(front_material)
    reserve_material_before = _material_fingerprint(reserve_material)
    temporary_before = _temporary_datablock_names()

    prepared = prepare_a1_object(
        source,
        settings,
        context=bpy.context,
        scene=bpy.context.scene,
    )
    reserve_slot, front_slot, prepared_triangles = smoke._assert_prepared(prepared)
    _assert(
        len(prepared.bake_plan.frame_tasks) == case.frame_count,
        "FRONT frame-task count differs",
    )
    _assert(
        len(prepared.reserve_bake_plans[0].frame_tasks) == case.frame_count,
        "reserve frame-task count differs",
    )
    _assert(
        str(camera.data.type) == case.camera_type,
        "prepare changed source camera type",
    )
    _assert(_capture_context() == context_before, "prepare changed Blender context")
    _assert(_capture_scene_bake_state() == bake_before, "prepare changed bake state")
    _assert(_scene_render_fingerprint() == render_before, "prepare changed render state")
    _assert(smoke._camera_fingerprint(camera) == camera_before, "prepare changed camera")
    _assert(_temporary_datablock_names() == temporary_before, "prepare leaked datablocks")

    result = export_a1_single_object(
        source,
        settings,
        context=bpy.context,
        scene=bpy.context.scene,
    )
    _assert(result.success, f"Depth parallax export failed for {case.key}: {result.issues}")

    front_paths = tuple(task.output_path for task in prepared.bake_plan.frame_tasks)
    reserve_paths = tuple(
        task.output_path for task in prepared.reserve_bake_plans[0].frame_tasks
    )
    expected_paths = (prepared.output_paths.json_path, *front_paths, *reserve_paths)
    _assert(
        tuple(path.resolve(strict=False) for path in result.output_files)
        == tuple(path.resolve(strict=False) for path in expected_paths),
        f"unexpected output order for {case.key}: {result.output_files}",
    )

    json_path = result.output_files[0]
    front_size, front_pixels = _assert_view_images(
        f"{case.key}:FRONT",
        tuple(result.output_files[1 : 1 + case.frame_count]),
        expected_frames=case.frame_count,
        expect_red=True,
    )
    reserve_size, reserve_pixels = _assert_view_images(
        f"{case.key}:RESERVE",
        tuple(result.output_files[1 + case.frame_count :]),
        expected_frames=case.frame_count,
        expect_red=False,
    )
    _assert(
        front_size != reserve_size or front_pixels != reserve_pixels,
        f"FRONT and reserve renders are identical for {case.key}",
    )

    document = json.loads(json_path.read_text(encoding="utf-8"))
    _assert(isinstance(document, dict), "serialized Spine document must be object")
    slots = document.get("slots")
    _assert(isinstance(slots, list), "serialized slots must be an array")
    visual_slots = tuple(
        str(slot.get("name"))
        for slot in slots
        if isinstance(slot, dict)
        and str(slot.get("name", "")).startswith(smoke._PREFIX)
    )
    _assert(
        visual_slots == (reserve_slot, front_slot),
        f"reserve/front draw order changed for {case.key}: {visual_slots}",
    )
    smoke._assert_serialized_attachment(
        document,
        reserve_slot,
        reserve_size,
        prepared_triangles[reserve_slot],
    )
    smoke._assert_serialized_attachment(
        document,
        front_slot,
        front_size,
        prepared_triangles[front_slot],
    )
    _assert_sequence_metadata(document, reserve_slot, case.sequence_count)
    _assert_sequence_metadata(document, front_slot, case.sequence_count)
    if case.sequence_count:
        animations = document.get("animations")
        _assert(isinstance(animations, dict), "sequence animations missing")
        _assert("sequence" in json.dumps(animations), "sequence timeline missing")

    expected_output_count = 1 + case.frame_count * 2
    expected_statistics = {
        "depth_parallax_cropped_view_count": 2,
        "parallax_texture_view_count": 2,
        "parallax_texture_output_count": case.frame_count * 2,
        "output_file_count": expected_output_count,
    }
    for key, expected in expected_statistics.items():
        _assert(
            int(result.statistics.get(key, -1)) == expected,
            f"result statistic {key!r} differs for {case.key}",
        )

    _assert(_capture_context() == context_before, "export changed Blender context")
    _assert(_capture_scene_bake_state() == bake_before, "export changed bake state")
    _assert(_scene_render_fingerprint() == render_before, "export changed render state")
    _assert(smoke._camera_fingerprint(camera) == camera_before, "export changed camera")
    _assert(
        _material_fingerprint(front_material) == front_material_before,
        "export changed front material",
    )
    _assert(
        _material_fingerprint(reserve_material) == reserve_material_before,
        "export changed reserve material",
    )
    _assert(_temporary_datablock_names() == temporary_before, "export leaked datablocks")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="spine2d-depth-parallax-matrix-") as directory:
        output_root = Path(directory)
        for case in _CASES:
            _run_case(output_root, case)
    print(
        "[DEPTH-PARALLAX-MATRIX] PASS cases=2 cameras=ORTHO,PERSP "
        "sequence_frames=2 attachments_per_case=2 view_crops=independent"
    )


if __name__ == "__main__":
    main()
