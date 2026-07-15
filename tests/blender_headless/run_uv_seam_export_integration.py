"""Real Blender 4.4 export fixture for loop-level UV seam duplication.

Two triangles form one manifold disk with a 90-degree fold. A1 segmentation keeps
them in one export region, while Smart Project unwrap deliberately splits the fold.
The final Spine attachment must therefore contain six UV-specific vertices for four
geometric vertices and remain fully exportable through Cycles and atomic JSON/PNG
commit.
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


def test_complete_service_exports_uv_split_region() -> None:
    _clear_scene()
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
        _assert(attachment["hull"] == 6, f"unexpected hull: {attachment['hull']}")
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

        vertex_bones = tuple(
            bone
            for bone in document["bones"]
            if bone["name"].startswith("UVSeam_Segment_0_vertex_")
        )
        _assert(len(vertex_bones) == 6, "one bone per UV-specific vertex is required")
        positions = tuple(
            (float(bone.get("x", 0.0)), float(bone.get("y", 0.0)))
            for bone in vertex_bones
        )
        _assert(
            len(set(positions)) == 4,
            f"UV duplication changed geometric positions: {positions}",
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
