"""Real Blender 4.4 fixture for mixed connected and standalone selection."""

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
)
from run_bake_integration import (  # noqa: E402
    PNG_SIGNATURE,
    _activate_only,
    _assert,
    _capture_context,
    _capture_scene_bake_state,
    _clear_scene,
    _create_emission_material,
    _create_quad,
    _create_sentinel,
    _material_fingerprint,
    _temporary_datablock_names,
)
from run_multi_object_export_integration import (  # noqa: E402
    _build_object_settings,
)


def test_mixed_selection_preserves_connected_and_standalone_semantics() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-mixed-") as directory:
        output_directory = Path(directory)
        object_a = _create_quad("MixedSourceA")
        object_b = _create_quad("MixedSourceB")
        object_c = _create_quad("MixedSourceC")
        object_a.location = (0.0, 0.0, 0.0)
        object_b.location = (2.0, 1.0, 0.5)
        object_c.location = (-3.0, 2.0, -1.0)
        materials = (
            _create_emission_material(object_a),
            _create_emission_material(object_b),
            _create_emission_material(object_c),
        )
        connected_sources = (
            A1MultiObjectSource(
                source_object=object_a,
                component_id="component_a",
                animation_namespace="object_1",
                settings=_build_object_settings(
                    output_directory,
                    prefix="ObjectA",
                    output_stem="ObjectA",
                ),
            ),
            A1MultiObjectSource(
                source_object=object_b,
                component_id="component_b",
                animation_namespace="object_2",
                settings=_build_object_settings(
                    output_directory,
                    prefix="ObjectB",
                    output_stem="ObjectB",
                ),
            ),
        )
        standalone_sources = (
            A1MultiObjectSource(
                source_object=object_c,
                component_id="component_c",
                animation_namespace="object_3",
                settings=_build_object_settings(
                    output_directory,
                    prefix="ObjectC",
                    output_stem="ObjectC",
                ),
            ),
        )
        settings = A1MultiObjectExportSettings(
            output_directory=output_directory,
            output_stem="MixedGroup",
            mode=A1MultiObjectMode.MIXED,
            anchor_component_id="component_a",
        )

        sentinel = _create_sentinel()
        _activate_only(sentinel)
        for source in connected_sources + standalone_sources:
            source.source_object.select_set(False)
        context_before = _capture_context()
        scene_before = _capture_scene_bake_state()
        material_before = tuple(_material_fingerprint(item) for item in materials)

        result = export_a1_mixed_object(
            connected_sources,
            standalone_sources,
            settings,
        )

        _assert(result.success, f"mixed export failed: {result.issues}")
        expected_json = (output_directory / "MixedGroup.json").resolve()
        expected_textures = tuple(
            (output_directory / "images" / f"Object{name}_Baked.png").resolve()
            for name in ("A", "B", "C")
        )
        _assert(
            result.output_files == (expected_json, *expected_textures),
            f"unexpected mixed outputs: {result.output_files}",
        )
        for path in expected_textures:
            _assert(path.read_bytes()[:8] == PNG_SIGNATURE, f"invalid PNG: {path}")

        document = json.loads(expected_json.read_text(encoding="utf-8"))
        bones = {bone["name"]: bone for bone in document["bones"]}
        _assert(
            tuple(bone["name"] for bone in document["bones"]).count("root") == 1,
            "mixed composition duplicated root",
        )
        _assert("all_objects_main" in bones, "connected global rig missing")
        _assert(
            bones["ObjectA_main"]["parent"].startswith("all_objects_layer_"),
            "ObjectA was not connected",
        )
        _assert(
            bones["ObjectB_main"]["parent"].startswith("all_objects_layer_"),
            "ObjectB was not connected",
        )
        _assert(
            bones["ObjectC_main"]["parent"] == "root",
            f"standalone ObjectC was incorrectly connected: {bones['ObjectC_main']}",
        )
        slot_names = tuple(slot["name"] for slot in document["slots"])
        _assert(
            slot_names
            == (
                "ObjectA_Segment_0",
                "ObjectB_Segment_0",
                "ObjectC_Segment_0",
            ),
            f"mixed slot order changed: {slot_names}",
        )
        constraints = tuple(document.get("ik", ())) + tuple(document.get("transform", ()))
        orders = tuple(int(item["order"]) for item in constraints)
        _assert(len(orders) == len(set(orders)), "mixed constraint orders collide")
        _assert(result.statistics["connected_object_count"] == 2, "connected count wrong")
        _assert(result.statistics["standalone_object_count"] == 1, "standalone count wrong")

        _assert(_capture_context() == context_before, "mixed export changed context")
        _assert(_capture_scene_bake_state() == scene_before, "mixed export changed scene")
        _assert(
            tuple(_material_fingerprint(item) for item in materials) == material_before,
            "mixed export mutated source materials",
        )
        _assert(not _temporary_datablock_names(), "mixed export leaked temporary data")


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    print(
        "[MIXED] RUN "
        "test_mixed_selection_preserves_connected_and_standalone_semantics"
    )
    test_mixed_selection_preserves_connected_and_standalone_semantics()
    print(
        "[MIXED] PASS "
        "test_mixed_selection_preserves_connected_and_standalone_semantics"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
