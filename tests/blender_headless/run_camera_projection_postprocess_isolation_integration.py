"""Blender 4.4 regression for B4 compositor and sequencer isolation."""

from __future__ import annotations

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

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    export_a1_single_object,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import bake_executor as bake_module  # noqa: E402
from run_bake_integration import _assert  # noqa: E402
from run_camera_projection_integration import (  # noqa: E402
    _create_layer_weight_material,
    _create_quad,
    _prepare_scene_with_sentinel,
    _read_pixels,
    _settings,
    _visible_and_transparent_counts,
)


def test_b4_disables_postprocess_only_during_render() -> None:
    _prepare_scene_with_sentinel()
    scene = bpy.context.scene
    scene.render.use_compositing = True
    scene.render.use_sequencer = True

    with tempfile.TemporaryDirectory(prefix="spine2d-postprocess-isolation-") as directory:
        source = _create_quad("PostprocessIsolationSource")
        source.data.materials.append(
            _create_layer_weight_material("PostprocessIsolationMaterial")
        )
        observed = []
        original_render = bake_module._call_render_operator

        def guarded_render(bpy_module):
            observed.append(
                (
                    bool(scene.render.use_compositing),
                    bool(scene.render.use_sequencer),
                )
            )
            _assert(
                observed[-1] == (False, False),
                f"B4 render still has postprocess enabled: {observed[-1]}",
            )
            return original_render(bpy_module)

        with mock.patch.object(
            bake_module,
            "_call_render_operator",
            side_effect=guarded_render,
        ):
            result = export_a1_single_object(
                source,
                _settings(Path(directory), "PostprocessIsolation"),
            )

        _assert(observed, "B4 render operator was not called")
        _assert(scene.render.use_compositing, "Compositor setting was not restored")
        _assert(scene.render.use_sequencer, "Sequencer setting was not restored")
        pixels = _read_pixels(result.image_paths[0])
        visible, transparent = _visible_and_transparent_counts(pixels)
        _assert(visible > 20, "isolated B4 image has no visible source pixels")
        _assert(transparent > 20, "isolated B4 image lost transparent background")


def main() -> None:
    test_b4_disables_postprocess_only_during_render()
    print("[PASS] test_b4_disables_postprocess_only_during_render")
    print("B4 postprocess isolation integration passed: 1 test")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
