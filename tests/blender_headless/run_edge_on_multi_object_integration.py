"""Real Blender 5.2 regression for edge-on faces in two-axis multi export."""

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
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    A1MultiObjectSource,
    export_a1_multi_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    BakeExecutionSettings,
    BakeMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (  # noqa: E402
    A1RigProfile,
    A1RigSetupPoseMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import UvUnwrapSettings  # noqa: E402
from run_bake_integration import (  # noqa: E402
    PNG_SIGNATURE,
    _assert,
    _clear_scene,
    _configure_cycles_scene,
    _create_emission_material,
    _create_mesh_object,
    _create_quad,
    _temporary_datablock_names,
)


def _object_settings(
    output_directory: Path,
    *,
    prefix: str,
) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=32,
            texture_height=32,
            output_directory=output_directory,
            images_relative_path="images",
            rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE.value,
            bake_margin=1,
        ),
        prefix=prefix,
        output_stem=prefix,
        source_geometry_mode=A1SourceGeometryMode.ORIGINAL,
        uv=UvUnwrapSettings(layer_name="SpineBakeUV"),
        diffuse_mode=BakeMode.EMIT,
        procedural_mode=BakeMode.EMIT,
        bake_execution=BakeExecutionSettings(samples=1),
        rig_setup_pose_mode=A1RigSetupPoseMode.PRESERVE_COMPOSITION,
    )


def _create_visible_face_with_edge_on_side(name: str):
    # The second triangle has valid three-dimensional area, but vertices 1 and 3
    # project to the same XY position. It represents a side wall that is invisible
    # in Spine's two-dimensional projection.
    return _create_mesh_object(
        name,
        (
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, -1.0, 1.0),
        ),
        (
            (0, 1, 2),
            (0, 3, 1),
        ),
    )


def test_two_axis_standalone_multi_export_skips_edge_on_side_face() -> None:
    _clear_scene()
    _configure_cycles_scene()

    with tempfile.TemporaryDirectory(prefix="spine2d-edge-on-multi-") as directory:
        output_directory = Path(directory)
        first = _create_quad("VisibleComponent")
        second = _create_visible_face_with_edge_on_side("EdgeOnComponent")
        first.location = (-2.0, 0.0, 0.0)
        second.location = (2.0, 0.0, 0.0)
        _create_emission_material(first)
        _create_emission_material(second)

        sources = (
            A1MultiObjectSource(
                source_object=first,
                component_id="visible_component",
                animation_namespace="visible_component",
                settings=_object_settings(
                    output_directory,
                    prefix="VisibleComponent",
                ),
            ),
            A1MultiObjectSource(
                source_object=second,
                component_id="edge_on_component",
                animation_namespace="edge_on_component",
                settings=_object_settings(
                    output_directory,
                    prefix="EdgeOnComponent",
                ),
            ),
        )
        settings = A1MultiObjectExportSettings(
            output_directory=output_directory,
            output_stem="EdgeOnSword",
            mode=A1MultiObjectMode.STANDALONE,
        )

        result = export_a1_multi_object(sources, settings)

        _assert(result.success, f"edge-on multi export failed: {result.issues}")
        json_path = (output_directory / "EdgeOnSword.json").resolve()
        first_png = (
            output_directory / "images" / "VisibleComponent_Baked.png"
        ).resolve()
        second_png = (
            output_directory / "images" / "EdgeOnComponent_Baked.png"
        ).resolve()
        _assert(
            result.output_files == (json_path, first_png, second_png),
            f"unexpected outputs: {result.output_files}",
        )
        _assert(first_png.read_bytes()[:8] == PNG_SIGNATURE, "first PNG invalid")
        _assert(second_png.read_bytes()[:8] == PNG_SIGNATURE, "second PNG invalid")

        document = json.loads(json_path.read_text(encoding="utf-8"))
        slot_names = tuple(slot["name"] for slot in document["slots"])
        _assert(
            "EdgeOnComponent_Segment_0" in slot_names,
            f"visible EdgeOnComponent attachment missing: {slot_names}",
        )
        _assert(
            "EdgeOnComponent_Segment_1" not in slot_names,
            f"edge-on side face produced an attachment: {slot_names}",
        )
        edge_on_slot = next(
            slot for slot in document["slots"]
            if slot["name"] == "EdgeOnComponent_Segment_0"
        )
        edge_on_attachment = document["skins"][0]["attachments"][
            edge_on_slot["name"]
        ][edge_on_slot["name"]]
        _assert(
            len(edge_on_attachment["triangles"]) == 3,
            f"expected one visible triangle: {edge_on_attachment['triangles']}",
        )
        _assert(
            not _temporary_datablock_names(),
            "edge-on multi export leaked temporary Blender datablocks",
        )


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    print("[EDGE_ON_MULTI] RUN two-axis standalone edge-on regression")
    test_two_axis_standalone_multi_export_skips_edge_on_side_face()
    print("[EDGE_ON_MULTI] PASS two-axis standalone edge-on regression")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
