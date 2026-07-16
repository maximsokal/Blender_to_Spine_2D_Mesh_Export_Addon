"""Execute supported B3 scene baking and validate the camera projection boundary."""

from __future__ import annotations

from pathlib import Path
import sys
import traceback

import bpy

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_scene_bake_integration import (  # noqa: E402
    test_camera_graph_stops_at_projection_boundary,
    test_mixed_local_and_scene_slots_are_composed_without_double_counting,
    test_scene_combined_responds_to_light_energy,
)


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    tests = (
        test_scene_combined_responds_to_light_energy,
        test_camera_graph_stops_at_projection_boundary,
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
