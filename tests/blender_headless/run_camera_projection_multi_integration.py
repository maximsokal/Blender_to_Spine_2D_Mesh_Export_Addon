"""Blender 4.4 production tests for cropped B4 multi-object composition."""

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
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    A1MultiObjectSource,
    export_a1_mixed_object,
    export_a1_multi_object,
)
from run_bake_integration import _assert, _temporary_datablock_names  # noqa: E402
from run_camera_projection_integration import (  # noqa: E402
    _create_layer_weight_material,
    _create_quad,
    _prepare_scene_with_sentinel,
    _read_image,
    _settings,
)


def _source(
    output_directory: Path,
    *,
    object_name: str,
    component_id: str,
    prefix: str,
    scale: tuple[float, float, float],
) -> A1MultiObjectSource:
    obj = _create_quad(object_name)
    obj.scale = scale
    obj.data.materials.append(_create_layer_weight_material(f"{object_name}_Material"))
    return A1MultiObjectSource(
        source_object=obj,
        component_id=component_id,
        animation_namespace=component_id,
        settings=_settings(output_directory, prefix),
    )


def _mesh_attachments(document: dict) -> tuple[dict, ...]:
    result = []
    for skin in document.get("skins", []):
        for slot_attachments in skin.get("attachments", {}).values():
            for attachment in slot_attachments.values():
                if isinstance(attachment, dict) and attachment.get("type") == "mesh":
                    result.append(attachment)
    return tuple(result)


def _attachment_for_stem(document: dict, stem: str) -> dict:
    suffix = f"{stem}_Baked"
    matches = tuple(
        attachment
        for attachment in _mesh_attachments(document)
        if str(attachment.get("path", "")).replace("\\", "/").endswith(suffix)
    )
    _assert(len(matches) == 1, f"expected one attachment for {stem}, found {len(matches)}")
    return matches[0]


def _assert_attachment_matches_image(
    document: dict,
    output_directory: Path,
    stem: str,
) -> tuple[int, int]:
    image_path = output_directory / "images" / f"{stem}_Baked.png"
    image_size, pixels = _read_image(image_path)
    _assert(image_size != (64, 64), f"{stem} was not cropped")
    _assert(0 < image_size[0] <= 64 and 0 < image_size[1] <= 64, f"bad size {image_size}")
    _assert(any(pixels[index] > 0.08 for index in range(3, len(pixels), 4)), f"{stem} is empty")

    attachment = _attachment_for_stem(document, stem)
    hull = int(attachment["hull"])
    _assert(hull >= 3, f"{stem} hull is degenerate")
    _assert(len(attachment["uvs"]) == hull * 2, f"{stem} UV/hull mismatch")
    _assert(
        len(attachment["triangles"]) == (hull - 2) * 3,
        f"{stem} triangle fan mismatch",
    )
    _assert(float(attachment["width"]) == float(image_size[0]), f"{stem} width mismatch")
    _assert(float(attachment["height"]) == float(image_size[1]), f"{stem} height mismatch")
    return image_size


def test_standalone_multi_recomposes_cropped_projection_documents() -> None:
    _prepare_scene_with_sentinel()
    with tempfile.TemporaryDirectory(prefix="spine2d-b4-multi-crop-") as directory:
        output_directory = Path(directory)
        sources = (
            _source(
                output_directory,
                object_name="ProjectionMultiA",
                component_id="projection_a",
                prefix="ProjectionA",
                scale=(1.0, 1.0, 1.0),
            ),
            _source(
                output_directory,
                object_name="ProjectionMultiB",
                component_id="projection_b",
                prefix="ProjectionB",
                scale=(0.55, 0.7, 1.0),
            ),
        )
        result = export_a1_multi_object(
            sources,
            A1MultiObjectExportSettings(
                output_directory=output_directory,
                output_stem="ProjectionStandaloneGroup",
                mode=A1MultiObjectMode.STANDALONE,
            ),
        )
        _assert(result.success, f"standalone B4 multi failed: {result.issues}")
        document = json.loads(
            (output_directory / "ProjectionStandaloneGroup.json").read_text("utf-8")
        )
        size_a = _assert_attachment_matches_image(document, output_directory, "ProjectionA")
        size_b = _assert_attachment_matches_image(document, output_directory, "ProjectionB")
        _assert(size_a != size_b, f"independent crops unexpectedly match: {size_a}")
        _assert(
            result.statistics["projection_cropped_component_count"] == 2,
            f"bad cropped component stats: {result.statistics}",
        )
        _assert(not _temporary_datablock_names(), "standalone B4 multi leaked data")


def test_connected_multi_recomposes_cropped_projection_documents() -> None:
    _prepare_scene_with_sentinel()
    with tempfile.TemporaryDirectory(prefix="spine2d-b4-connected-crop-") as directory:
        output_directory = Path(directory)
        sources = (
            _source(
                output_directory,
                object_name="ProjectionConnectedA",
                component_id="connected_a",
                prefix="ConnectedProjectionA",
                scale=(0.8, 1.0, 1.0),
            ),
            _source(
                output_directory,
                object_name="ProjectionConnectedB",
                component_id="connected_b",
                prefix="ConnectedProjectionB",
                scale=(0.45, 0.55, 1.0),
            ),
        )
        result = export_a1_multi_object(
            sources,
            A1MultiObjectExportSettings(
                output_directory=output_directory,
                output_stem="ProjectionConnectedGroup",
                mode=A1MultiObjectMode.CONNECTED,
                anchor_component_id="connected_a",
            ),
        )
        _assert(result.success, f"connected B4 multi failed: {result.issues}")
        document = json.loads(
            (output_directory / "ProjectionConnectedGroup.json").read_text("utf-8")
        )
        _assert_attachment_matches_image(document, output_directory, "ConnectedProjectionA")
        _assert_attachment_matches_image(document, output_directory, "ConnectedProjectionB")
        bone_names = {bone["name"] for bone in document["bones"]}
        _assert("all_objects_main" in bone_names, "connected global rig was lost")
        _assert(not _temporary_datablock_names(), "connected B4 multi leaked data")


def test_mixed_export_recomposes_all_cropped_projection_documents() -> None:
    _prepare_scene_with_sentinel()
    with tempfile.TemporaryDirectory(prefix="spine2d-b4-mixed-crop-") as directory:
        output_directory = Path(directory)
        connected_sources = (
            _source(
                output_directory,
                object_name="ProjectionMixedConnectedA",
                component_id="mixed_connected_a",
                prefix="MixedConnectedA",
                scale=(0.9, 0.9, 1.0),
            ),
            _source(
                output_directory,
                object_name="ProjectionMixedConnectedB",
                component_id="mixed_connected_b",
                prefix="MixedConnectedB",
                scale=(0.55, 0.65, 1.0),
            ),
        )
        standalone_sources = (
            _source(
                output_directory,
                object_name="ProjectionMixedStandalone",
                component_id="mixed_standalone",
                prefix="MixedStandalone",
                scale=(0.35, 0.45, 1.0),
            ),
        )
        result = export_a1_mixed_object(
            connected_sources,
            standalone_sources,
            A1MultiObjectExportSettings(
                output_directory=output_directory,
                output_stem="ProjectionMixedGroup",
                mode=A1MultiObjectMode.MIXED,
                anchor_component_id="mixed_connected_a",
            ),
        )
        _assert(result.success, f"mixed B4 export failed: {result.issues}")
        document = json.loads(
            (output_directory / "ProjectionMixedGroup.json").read_text("utf-8")
        )
        for stem in ("MixedConnectedA", "MixedConnectedB", "MixedStandalone"):
            _assert_attachment_matches_image(document, output_directory, stem)
        _assert(
            result.statistics["projection_cropped_component_count"] == 3,
            f"bad mixed crop statistics: {result.statistics}",
        )
        _assert(not _temporary_datablock_names(), "mixed B4 export leaked data")


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    tests = (
        test_standalone_multi_recomposes_cropped_projection_documents,
        test_connected_multi_recomposes_cropped_projection_documents,
        test_mixed_export_recomposes_all_cropped_projection_documents,
    )
    for test in tests:
        print(f"[CAMERA-PROJECTION-MULTI] RUN {test.__name__}")
        test()
        print(f"[CAMERA-PROJECTION-MULTI] PASS {test.__name__}")
    print(f"[CAMERA-PROJECTION-MULTI] PASS {len(tests)} integration tests")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
