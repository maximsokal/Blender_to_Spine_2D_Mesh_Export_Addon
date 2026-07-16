"""Execute supported B3 scene baking and validate the camera projection boundary."""

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

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    analyse_bake_contexts,
    analyse_object_materials,
    read_source_mesh_snapshot,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    BakeMode,
    BakePlanError,
    BakeSettings,
    build_bake_plan,
)
from run_bake_integration import (  # noqa: E402
    _activate_only,
    _assert,
    _capture_context,
    _capture_scene_bake_state,
    _clear_scene,
    _create_mesh_object,
    _create_sentinel,
    _material_fingerprint,
    _temporary_datablock_names,
)
from run_scene_bake_integration import (  # noqa: E402
    _configure_cycles_scene,
    _create_camera,
    _create_layer_weight_emission_material,
    test_mixed_local_and_scene_slots_are_composed_without_double_counting,
    test_scene_combined_responds_to_light_energy,
)


def test_camera_graph_is_rejected_at_projection_boundary() -> None:
    _clear_scene()
    _configure_cycles_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-camera-boundary-") as directory:
        output_directory = Path(directory)
        source = _create_mesh_object(
            "CameraBoundary",
            (
                (-1.0, -1.0, 0.0),
                (1.0, -1.0, 0.0),
                (1.0, 1.0, 0.0),
                (-1.0, 1.0, 0.0),
            ),
            ((0, 1, 2, 3),),
        )
        material = _create_layer_weight_emission_material("CameraBoundaryMaterial")
        source.data.materials.append(material)
        _create_camera()
        sentinel = _create_sentinel()
        sentinel.location.x = 20.0
        _activate_only(sentinel)
        source.select_set(False)

        context_before = _capture_context()
        scene_before = _capture_scene_bake_state()
        material_before = _material_fingerprint(material)
        hide_before = bool(source.hide_render)
        snapshot = read_source_mesh_snapshot(source)
        analysis = analyse_object_materials(
            source,
            source_object_id=snapshot.source_object_id,
        )
        object_context, scene_context = analyse_bake_contexts(
            source,
            scene=bpy.context.scene,
            context=bpy.context,
        )
        dependencies = {item.value for item in analysis.slots[0].dependencies}
        _assert("VIEW" in dependencies, "Layer Weight did not report VIEW")
        _assert("CAMERA" in dependencies, "Layer Weight did not report CAMERA")

        try:
            build_bake_plan(
                analysis,
                BakeSettings(
                    width=64,
                    height=64,
                    output_directory=output_directory,
                    output_stem="CameraBoundary",
                    uv_layer_name="UVMap",
                    margin_pixels=1,
                    diffuse_mode=BakeMode.DIFFUSE,
                    procedural_mode=BakeMode.DIFFUSE,
                ),
                object_context=object_context,
                scene_context=scene_context,
            )
        except BakePlanError as exc:
            _assert(
                "camera-render projection" in str(exc),
                f"camera boundary error is not actionable: {exc}",
            )
        else:
            raise AssertionError(
                "camera-dependent graph produced an object-bake plan even though "
                "Blender 4.4 has no camera-ray bake type"
            )

        _assert(bool(source.hide_render) == hide_before, "planning changed hide_render")
        _assert(_capture_context() == context_before, "camera planning changed context")
        _assert(_capture_scene_bake_state() == scene_before, "camera planning changed scene")
        _assert(_material_fingerprint(material) == material_before, "planning mutated material")
        _assert(not _temporary_datablock_names(), "camera planning leaked temporary data")


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    tests = (
        test_scene_combined_responds_to_light_energy,
        test_camera_graph_is_rejected_at_projection_boundary,
        test_mixed_local_and_scene_slots_are_composed_without_double_counting,
    )
    for test in tests:
        print(f"[SCENE-BAKE] RUN {test.__name__}")
        test()
        print(f"[SCENE-BAKE] PASS {test.__name__}")
    print(f"[SCENE-BAKE] PASS {len(tests)} integration tests")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
