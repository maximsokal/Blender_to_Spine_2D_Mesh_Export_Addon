"""Real Blender 4.4 fail-closed planning/execution scene parity fixtures."""

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
    execute_bake_plan,
    execute_camera_projection_plan,
    prepare_a1_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    BakeExecutionSettings,
    CameraProjectionPlan,
)
from run_bake_integration import (  # noqa: E402
    _assert,
    _clear_scene,
    _create_mesh_object,
    _temporary_datablock_names,
)
from run_camera_projection_integration import (  # noqa: E402
    _create_layer_weight_material,
    _create_quad,
    _prepare_scene_with_sentinel,
    _settings,
)
from run_scene_bake_extended_integration import (  # noqa: E402
    _build_plan,
    _create_ao_emission_material,
    _create_cube_mesh,
)
from run_scene_bake_integration import _configure_cycles_scene  # noqa: E402


def test_b3_rejects_scene_object_set_change_before_output_reservation() -> None:
    _clear_scene()
    _configure_cycles_scene()
    with tempfile.TemporaryDirectory(prefix="spine2d-scene-parity-") as directory:
        output_directory = Path(directory)
        source = _create_mesh_object(
            "ParityAoSurface",
            ((-2.0, -2.0, 0.0), (2.0, -2.0, 0.0), (2.0, 2.0, 0.0), (-2.0, 2.0, 0.0)),
            ((0, 1, 2, 3),),
        )
        source.data.materials.append(_create_ao_emission_material("ParitySceneAO"))
        occluder = _create_cube_mesh(
            "ParityAoOccluder",
            center=(0.0, 0.0, 0.55),
            size=1.4,
        )
        snapshot, _, plan = _build_plan(
            source,
            output_directory,
            "ParityAo",
        )
        output_path = plan.representative_task.output_path
        _assert(not output_path.exists(), "B3 output existed before execution")

        occluder.hide_render = True
        bpy.context.view_layer.update()
        try:
            execute_bake_plan(
                source,
                snapshot,
                plan,
                BakeExecutionSettings(samples=1),
                context=bpy.context,
                scene=bpy.context.scene,
            )
        except Exception as exc:
            message = str(exc)
            _assert(
                "render-visible object set changed" in message,
                f"B3 parity error was not actionable: {message}",
            )
            _assert(
                "shadow-caster set changed" in message,
                f"B3 parity missed shadow set: {message}",
            )
        else:
            raise AssertionError("B3 executed after scene object set changed")

        _assert(not output_path.exists(), "B3 parity failure reserved or wrote output")
        _assert(not _temporary_datablock_names(), "B3 parity failure leaked temporary data")


def test_b4_rejects_color_management_change_before_output_reservation() -> None:
    _prepare_scene_with_sentinel()
    with tempfile.TemporaryDirectory(prefix="spine2d-b4-parity-") as directory:
        output_directory = Path(directory)
        source = _create_quad("ParityProjectionSource")
        source.data.materials.append(
            _create_layer_weight_material("ParityProjectionMaterial")
        )
        settings = _settings(output_directory, "ParityProjection")
        prepared = prepare_a1_object(
            source,
            settings,
            context=bpy.context,
            scene=bpy.context.scene,
        )
        plan = prepared.bake_plan
        _assert(isinstance(plan, CameraProjectionPlan), "fixture did not select B4")
        output_path = plan.representative_task.output_path
        _assert(not output_path.exists(), "B4 output existed before execution")

        bpy.context.scene.view_settings.view_transform = "AgX"
        bpy.context.view_layer.update()
        try:
            execute_camera_projection_plan(
                source,
                plan,
                settings.bake_execution,
                context=bpy.context,
                scene=bpy.context.scene,
            )
        except Exception as exc:
            message = str(exc)
            _assert(
                "color management changed" in message,
                f"B4 parity error was not actionable: {message}",
            )
        else:
            raise AssertionError("B4 executed after color management changed")

        _assert(not output_path.exists(), "B4 parity failure reserved or wrote output")
        _assert(not _temporary_datablock_names(), "B4 parity failure leaked temporary data")


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    tests = (
        test_b3_rejects_scene_object_set_change_before_output_reservation,
        test_b4_rejects_color_management_change_before_output_reservation,
    )
    for test in tests:
        print(f"[SCENE-PARITY] RUN {test.__name__}")
        test()
        print(f"[SCENE-PARITY] PASS {test.__name__}")
    print(f"[SCENE-PARITY] PASS {len(tests)} integration tests")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
