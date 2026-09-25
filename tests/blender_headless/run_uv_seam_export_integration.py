"""Real Blender 5.2 export fixture for loop-level UV seam duplication.

Two triangles form one manifold disk with a 90-degree fold. A1 segmentation keeps
them in one export region, while Smart Project unwrap deliberately splits the fold.
The final Spine attachment must therefore contain six UV-specific vertices for four
geometric vertices, while its physical XY convex hull remains the three projected
outer points. Final publication may canonicalize coincident generated vertex bones,
but every weighted attachment vertex must remain present and correctly remapped.
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
    BakeExecutionSettings,
    BakeMode,
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
    _configure_cycles_scene,
    _create_emission_material,
    _create_mesh_object,
    _create_sentinel,
    _material_fingerprint,
    _temporary_datablock_names,
)


def _create_folded_disk():
    """Create two triangles sharing an edge with a 90-degree normal change."""

    return _create_mesh_object(
        "UvSeamSource",
        (
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        (
            (0, 1, 2),
            (0, 2, 3),
        ),
    )


def _build_settings(output_directory: Path) -> A1SingleObjectExportSettings:
    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be pathlib.Path")
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=32,
            texture_height=32,
            output_directory=output_directory,
            images_relative_path="images",
            angle_limit_degrees=179.0,
            bake_margin=1,
        ),
        prefix="UVSeam",
        output_stem="UVSeam",
        source_geometry_mode=A1SourceGeometryMode.ORIGINAL,
        uv=UvUnwrapSettings(
            layer_name="SpineBakeUV",
            smart_angle_limit_degrees=30.0,
        ),
        diffuse_mode=BakeMode.EMIT,
        procedural_mode=BakeMode.EMIT,
        bake_execution=BakeExecutionSettings(samples=1),
    )


def _decode_single_influence_indices(stream: list[float | int]) -> tuple[int, ...]:
    if not isinstance(stream, list):
        raise TypeError("weighted vertex stream must be a list")

    indices: list[int] = []
    cursor = 0
    while cursor < len(stream):
        influence_count = int(stream[cursor])
        _assert(
            influence_count == 1,
            f"expected one influence per UV-specific vertex, got {influence_count}",
        )
        cursor += 1
        _assert(cursor + 3 < len(stream), "weighted vertex stream is truncated")
        bone_index = int(stream[cursor])
        x = float(stream[cursor + 1])
        y = float(stream[cursor + 2])
        weight = float(stream[cursor + 3])
        _assert(
            (x, y, weight) == (0.0, 0.0, 1.0),
            f"UV seam local weight data changed: {(x, y, weight)}",
        )
        indices.append(bone_index)
        cursor += 4
    return tuple(indices)


def test_complete_service_exports_uv_split_region() -> None:
    _clear_scene()
    _configure_cycles_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-uv-seam-") as directory:
        output_directory = Path(directory)
        source = _create_folded_disk()
        material = _create_emission_material(source)
        sentinel = _create_sentinel()
        _activate_only(sentinel)
        source.select_set(False)
        context_before = _capture_context()
        scene_before = _capture_scene_bake_state()
        material_before = _material_fingerprint(material)

        result = export_a1_single_object(
            source,
            _build_settings(output_directory),
        )

        _assert(result.success, f"UV seam service failed: {result.issues}")
        _assert(result.statistics["segment_count"] == 1, "fold was segmented")
        _assert(result.statistics["region_count"] == 1, "fold became multiple regions")
        expected_json = output_directory / "UVSeam.json"
        expected_png = output_directory / "images" / "UVSeam_Baked.png"
        _assert(
            result.output_files == (expected_json.resolve(), expected_png.resolve()),
            f"unexpected outputs: {result.output_files}",
        )
        _assert(expected_png.read_bytes()[:8] == PNG_SIGNATURE, "PNG output invalid")

        document = json.loads(expected_json.read_text(encoding="utf-8"))
        _assert(len(document["slots"]) == 1, "expected one attachment slot")
        slot_name = document["slots"][0]["name"]
        attachment = document["skins"][0]["attachments"][slot_name][slot_name]
        vertex_count = len(attachment["uvs"]) // 2
        _assert(vertex_count == 6, f"expected six UV vertices, got {vertex_count}")
        _assert(
            int(attachment["hull"]) == 3,
            f"physical XY hull should contain three outer points: {attachment['hull']}",
        )
        _assert(
            int(attachment["hull"]) < vertex_count,
            "UV seam duplication was incorrectly promoted into the physical hull",
        )
        _assert(
            len(attachment["triangles"]) == 6,
            "two source triangles were not preserved",
        )
        _assert(
            max(attachment["triangles"]) < vertex_count,
            "triangle references an invalid duplicated vertex",
        )
        _assert(
            len(attachment.get("edges", ())) == 12,
            "UV-split attachment edge topology is incomplete",
        )

        # Publication runs the generated vertex-bone optimizer after the attachment
        # builder. The builder still creates one bone per attachment vertex, but the
        # optimizer deterministically merges bones with identical final setup semantics
        # and remaps the weighted stream. This folded fixture has six UV-specific mesh
        # vertices but only four unique physical setup positions.
        vertex_bone_prefix = f"{slot_name}_vertex_"
        vertex_bone_indices = tuple(
            index
            for index, bone in enumerate(document["bones"])
            if str(bone.get("name", "")).startswith(vertex_bone_prefix)
        )
        _assert(
            len(vertex_bone_indices) == 4,
            "UV seam canonicalization should retain exactly four physical vertex bones; "
            f"indices={vertex_bone_indices}",
        )
        vertex_bone_keys = tuple(
            (
                document["bones"][index].get("parent"),
                float(document["bones"][index].get("x", 0.0)),
                float(document["bones"][index].get("y", 0.0)),
            )
            for index in vertex_bone_indices
        )
        _assert(
            len(set(vertex_bone_keys)) == 4,
            f"canonical UV seam bones still contain duplicate setup keys: {vertex_bone_keys}",
        )

        weighted_indices = _decode_single_influence_indices(attachment["vertices"])
        _assert(
            len(weighted_indices) == vertex_count == 6,
            f"UV-specific weighted vertices were removed: {weighted_indices}",
        )
        _assert(
            set(weighted_indices) == set(vertex_bone_indices),
            "weighted UV seam vertices do not reference exactly the canonical bones; "
            f"weighted={weighted_indices}, canonical={vertex_bone_indices}",
        )
        _assert(
            len(set(weighted_indices)) < len(weighted_indices),
            "UV seam duplicate positions were not shared through canonical vertex bones",
        )

        _assert(_capture_context() == context_before, "UV seam export changed context")
        _assert(
            _capture_scene_bake_state() == scene_before,
            "UV seam export changed scene state",
        )
        _assert(
            _material_fingerprint(material) == material_before,
            "UV seam export mutated source material",
        )
        _assert(
            not _temporary_datablock_names(),
            "UV seam export leaked temporary Blender datablocks",
        )


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    print(f"[UV-SEAM] RUN {test_complete_service_exports_uv_split_region.__name__}")
    test_complete_service_exports_uv_split_region()
    print(f"[UV-SEAM] PASS {test_complete_service_exports_uv_split_region.__name__}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise