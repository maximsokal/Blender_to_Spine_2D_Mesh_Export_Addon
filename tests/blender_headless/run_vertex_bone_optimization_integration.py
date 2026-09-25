"""Blender 5.2 integration gate for shared segment vertex-bone optimization."""

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

import Blender_to_Spine2D_Mesh_Exporter as addon  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_bridge import (  # noqa: E402
    export_active_object_a1,
)
from run_normal_uv_pyramid_mode_integration import (  # noqa: E402
    _assert,
    _clear_scene,
    _configure_scene,
    _create_pyramid,
    _load_exported_outputs,
)


def _decode_single_influence_indices(stream: list[float | int]) -> tuple[int, ...]:
    indices: list[int] = []
    cursor = 0
    while cursor < len(stream):
        influence_count = int(stream[cursor])
        _assert(influence_count == 1, f"Expected one influence, got {influence_count}")
        cursor += 1
        bone_index = int(stream[cursor])
        x = float(stream[cursor + 1])
        y = float(stream[cursor + 2])
        weight = float(stream[cursor + 3])
        _assert((x, y, weight) == (0.0, 0.0, 1.0), "Local weight data changed")
        indices.append(bone_index)
        cursor += 4
    return tuple(indices)


def test_pyramid_exports_four_shared_vertex_bones_for_twelve_mesh_vertices() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-shared-vertex-bones-") as directory:
        output_directory = Path(directory)
        _create_pyramid()
        _configure_scene(output_directory, seam_mode="AUTO")

        result = export_active_object_a1(bpy.context)
        _assert(result.success, f"Pyramid export failed: {result.issues}")

        document, _texture_path = _load_exported_outputs(result)
        bones = document["bones"]
        vertex_bone_indices = tuple(
            index
            for index, bone in enumerate(bones)
            if "_Segment_" in bone["name"] and "_vertex_" in bone["name"]
        )
        _assert(
            len(vertex_bone_indices) == 4,
            f"Expected 4 shared pyramid vertex bones, got {len(vertex_bone_indices)}",
        )

        vertex_bone_keys = tuple(
            (
                bones[index].get("parent"),
                float(bones[index].get("x", 0.0)),
                float(bones[index].get("y", 0.0)),
            )
            for index in vertex_bone_indices
        )
        _assert(
            len(set(vertex_bone_keys)) == 4,
            f"Optimized vertex bones still contain duplicate setup keys: {vertex_bone_keys}",
        )

        attachments = document["skins"][0]["attachments"]
        referenced_indices: list[int] = []
        weighted_vertex_count = 0
        for segment_index in range(4):
            name = f"Pyramid_Segment_{segment_index}"
            attachment = attachments[name][name]
            indices = _decode_single_influence_indices(attachment["vertices"])
            _assert(len(indices) == 3, f"{name} no longer contains three mesh vertices")
            _assert(
                len(attachment["uvs"]) == 6,
                f"{name} UV stream changed during bone optimization",
            )
            _assert(
                len(attachment["triangles"]) == 3,
                f"{name} triangle stream changed during bone optimization",
            )
            referenced_indices.extend(indices)
            weighted_vertex_count += len(indices)

        _assert(weighted_vertex_count == 12, "Segment mesh vertices were removed")
        _assert(
            set(referenced_indices) == set(vertex_bone_indices),
            "Weighted meshes do not reference exactly the four canonical vertex bones",
        )


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    _assert(bpy.app.version >= (5, 2, 0), "Blender 5.2+ is required")
    addon.register()
    try:
        print("[VERTEX_BONE_OPTIMIZATION] RUN pyramid shared-bone regression")
        test_pyramid_exports_four_shared_vertex_bones_for_twelve_mesh_vertices()
        print("[VERTEX_BONE_OPTIMIZATION] PASS pyramid shared-bone regression")
    finally:
        addon.unregister()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
