"""Real Blender 5.2 integration tests for the rewritten multi-object service.

The fixtures are generated at runtime and exercise both typed composition modes plus
one shared filesystem transaction. The failure case deliberately lets the first Cycles
bake finish and fails the second bake, proving that the final JSON and every texture are
rolled back together.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import traceback
from unittest import mock

import bpy


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1MultiObjectStage,
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    A1MultiObjectSource,
    BakeExecutionError,
    export_a1_multi_object,
)
import Blender_to_Spine2D_Mesh_Exporter.blender_adapter.semantic_bake_execution as bake_module  # noqa: E402
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
    _create_quad,
    _create_sentinel,
    _material_fingerprint,
    _temporary_datablock_names,
)


def _build_object_settings(
    output_directory: Path,
    *,
    prefix: str,
    output_stem: str,
) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=32,
            texture_height=32,
            output_directory=output_directory,
            images_relative_path="images",
            bake_margin=1,
        ),
        prefix=prefix,
        output_stem=output_stem,
        source_geometry_mode=A1SourceGeometryMode.ORIGINAL,
        uv=UvUnwrapSettings(layer_name="SpineBakeUV"),
        diffuse_mode=BakeMode.EMIT,
        procedural_mode=BakeMode.EMIT,
        bake_execution=BakeExecutionSettings(samples=1),
    )


def _build_sources(output_directory: Path):
    first = _create_quad("MultiSourceA")
    second = _create_quad("MultiSourceB")
    first.location = (0.0, 0.0, 0.0)
    second.location = (2.0, 1.0, 0.5)
    first_material = _create_emission_material(first)
    second_material = _create_emission_material(second)
    sources = (
        A1MultiObjectSource(
            source_object=first,
            component_id="component_a",
            animation_namespace="object_1",
            settings=_build_object_settings(
                output_directory,
                prefix="ObjectA",
                output_stem="ObjectA",
            ),
        ),
        A1MultiObjectSource(
            source_object=second,
            component_id="component_b",
            animation_namespace="object_2",
            settings=_build_object_settings(
                output_directory,
                prefix="ObjectB",
                output_stem="ObjectB",
            ),
        ),
    )
    return sources, (first_material, second_material)


def _multi_settings(
    output_directory: Path,
    *,
    mode: A1MultiObjectMode,
    output_stem: str,
) -> A1MultiObjectExportSettings:
    return A1MultiObjectExportSettings(
        output_directory=output_directory,
        output_stem=output_stem,
        mode=mode,
        anchor_component_id=(
            "component_a" if mode is A1MultiObjectMode.CONNECTED else None
        ),
    )


def _prepare_state(output_directory: Path):
    _configure_cycles_scene()
    sources, materials = _build_sources(output_directory)
    sentinel = _create_sentinel()
    _activate_only(sentinel)
    for source in sources:
        source.source_object.select_set(False)
    return (
        sources,
        materials,
        _capture_context(),
        _capture_scene_bake_state(),
        tuple(_material_fingerprint(material) for material in materials),
    )


def _assert_state_restored(
    *,
    context_before,
    scene_before,
    materials,
    material_fingerprints,
) -> None:
    _assert(_capture_context() == context_before, "multi export changed context")
    _assert(
        _capture_scene_bake_state() == scene_before,
        "multi export changed scene bake state",
    )
    _assert(
        tuple(_material_fingerprint(material) for material in materials)
        == material_fingerprints,
        "multi export mutated source materials",
    )
    _assert(
        not _temporary_datablock_names(),
        "multi export leaked temporary Blender datablocks",
    )


def test_standalone_multi_export_commits_one_json_and_two_textures() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-multi-standalone-") as directory:
        output_directory = Path(directory)
        (
            sources,
            materials,
            context_before,
            scene_before,
            material_fingerprints,
        ) = _prepare_state(output_directory)
        settings = _multi_settings(
            output_directory,
            mode=A1MultiObjectMode.STANDALONE,
            output_stem="StandaloneGroup",
        )

        result = export_a1_multi_object(sources, settings)

        _assert(result.success, f"standalone multi export failed: {result.issues}")
        expected_json = (output_directory / "StandaloneGroup.json").resolve()
        expected_a = (output_directory / "images" / "ObjectA_Baked.png").resolve()
        expected_b = (output_directory / "images" / "ObjectB_Baked.png").resolve()
        _assert(
            result.output_files == (expected_json, expected_a, expected_b),
            f"unexpected standalone outputs: {result.output_files}",
        )
        _assert(expected_a.read_bytes()[:8] == PNG_SIGNATURE, "ObjectA PNG invalid")
        _assert(expected_b.read_bytes()[:8] == PNG_SIGNATURE, "ObjectB PNG invalid")

        document = json.loads(expected_json.read_text(encoding="utf-8"))
        bone_names = tuple(bone["name"] for bone in document["bones"])
        _assert(bone_names.count("root") == 1, "standalone composition duplicated root")
        _assert("ObjectA_main" in bone_names, "ObjectA main bone missing")
        _assert("ObjectB_main" in bone_names, "ObjectB main bone missing")
        slot_names = tuple(slot["name"] for slot in document["slots"])
        _assert(
            slot_names == ("ObjectA_Segment_0", "ObjectB_Segment_0"),
            f"standalone slot order changed: {slot_names}",
        )
        _assert(
            result.statistics["object_count"] == 2,
            "standalone statistics lost object count",
        )
        _assert_state_restored(
            context_before=context_before,
            scene_before=scene_before,
            materials=materials,
            material_fingerprints=material_fingerprints,
        )


def test_connected_multi_export_builds_global_rig_and_offsets() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-multi-connected-") as directory:
        output_directory = Path(directory)
        (
            sources,
            materials,
            context_before,
            scene_before,
            material_fingerprints,
        ) = _prepare_state(output_directory)
        settings = _multi_settings(
            output_directory,
            mode=A1MultiObjectMode.CONNECTED,
            output_stem="ConnectedGroup",
        )

        result = export_a1_multi_object(sources, settings)

        _assert(result.success, f"connected multi export failed: {result.issues}")
        expected_json = (output_directory / "ConnectedGroup.json").resolve()
        expected_a = (output_directory / "images" / "ObjectA_Baked.png").resolve()
        expected_b = (output_directory / "images" / "ObjectB_Baked.png").resolve()
        _assert(
            result.output_files == (expected_json, expected_a, expected_b),
            f"unexpected connected outputs: {result.output_files}",
        )

        document = json.loads(expected_json.read_text(encoding="utf-8"))
        bones = {bone["name"]: bone for bone in document["bones"]}
        _assert("all_objects_main" in bones, "global main bone missing")
        _assert("all_objects_rotation_X" in bones, "global X controller missing")
        _assert("all_objects_rotation_Y" in bones, "global Y controller missing")
        _assert("all_objects_rotation_Z" in bones, "global Z controller missing")
        _assert(
            bones["ObjectA_main"]["parent"] == "all_objects_layer_1",
            f"anchor object layer is wrong: {bones['ObjectA_main']}",
        )
        _assert(
            bones["ObjectB_main"]["parent"] == "all_objects_layer_0",
            f"elevated object layer is wrong: {bones['ObjectB_main']}",
        )
        _assert(float(bones["ObjectA_main"].get("x", 0.0)) == 0.0, "anchor X moved")
        _assert(float(bones["ObjectA_main"].get("y", 0.0)) == 0.0, "anchor Y moved")
        _assert(float(bones["ObjectB_main"].get("x", 0.0)) == 64.0, "ObjectB X offset wrong")
        _assert(float(bones["ObjectB_main"].get("y", 0.0)) == 32.0, "ObjectB Y offset wrong")

        constraints = tuple(document.get("ik", ())) + tuple(document.get("transform", ()))
        orders = tuple(int(item["order"]) for item in constraints)
        _assert(len(orders) == len(set(orders)), "connected constraint orders collide")
        _assert(
            result.statistics["connected_layer_count"] == 2,
            "connected Z layer count is wrong",
        )
        _assert_state_restored(
            context_before=context_before,
            scene_before=scene_before,
            materials=materials,
            material_fingerprints=material_fingerprints,
        )


def test_second_bake_failure_rolls_back_json_and_both_textures() -> None:
    _clear_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-multi-rollback-") as directory:
        output_directory = Path(directory)
        (
            sources,
            materials,
            context_before,
            scene_before,
            material_fingerprints,
        ) = _prepare_state(output_directory)
        settings = _multi_settings(
            output_directory,
            mode=A1MultiObjectMode.STANDALONE,
            output_stem="RollbackGroup",
        )
        final_json = output_directory / "RollbackGroup.json"
        final_a = output_directory / "images" / "ObjectA_Baked.png"
        final_b = output_directory / "images" / "ObjectB_Baked.png"
        final_a.parent.mkdir(parents=True, exist_ok=True)
        previous = {
            final_json: b"previous-multi-json",
            final_a: b"previous-object-a-png",
            final_b: b"previous-object-b-png",
        }
        for path, content in previous.items():
            path.write_bytes(content)

        original_call = bake_module._call_bake_operator
        call_count = 0

        def fail_second_bake(
            bpy_module,
            bake_type,
            *,
            uv_layer_name,
        ):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise BakeExecutionError("forced second multi-object bake failure")
            return original_call(
                bpy_module,
                bake_type,
                uv_layer_name=uv_layer_name,
            )

        with mock.patch.object(
            bake_module,
            "_call_bake_operator",
            side_effect=fail_second_bake,
        ):
            result = export_a1_multi_object(sources, settings)

        _assert(not result.success, "forced second bake failure returned success")
        _assert(call_count == 2, f"expected two bake calls, got {call_count}")
        primary_issue = result.issues[-1]
        _assert(
            primary_issue.stage == A1MultiObjectStage.STAGE_OUTPUTS.value,
            f"unexpected rollback stage: {primary_issue.stage}",
        )
        _assert(
            primary_issue.code == A1MultiObjectStage.STAGE_OUTPUTS.error_code,
            f"unexpected rollback code: {primary_issue.code}",
        )
        for path, content in previous.items():
            _assert(path.read_bytes() == content, f"rollback corrupted {path.name}")

        leftovers = tuple(
            sorted(
                str(path.relative_to(output_directory))
                for path in output_directory.rglob("*")
                if path.is_file()
            )
        )
        _assert(
            leftovers
            == (
                "RollbackGroup.json",
                "images/ObjectA_Baked.png",
                "images/ObjectB_Baked.png",
            ),
            f"multi rollback left staged or backup files: {leftovers}",
        )
        _assert_state_restored(
            context_before=context_before,
            scene_before=scene_before,
            materials=materials,
            material_fingerprints=material_fingerprints,
        )


def main() -> None:
    tests = (
        test_standalone_multi_export_commits_one_json_and_two_textures,
        test_connected_multi_export_builds_global_rig_and_offsets,
        test_second_bake_failure_rolls_back_json_and_both_textures,
    )
    print(f"Blender version: {bpy.app.version_string}")
    for test in tests:
        print(f"[MULTI] RUN {test.__name__}")
        test()
        print(f"[MULTI] PASS {test.__name__}")
    print(f"[MULTI] PASS {len(tests)} integration tests")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
