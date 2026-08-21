"""Static bootstrap contract for real-coin thin Blender entrypoints.

Blender executes these files by path. The headless interpreter does not guarantee that
``tests/blender_headless`` or the repository root is importable before the script runs, so
each thin wrapper must establish both paths before importing sibling core modules or the
add-on package.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADLESS = ROOT / "tests" / "blender_headless"

_RUNNERS_AND_FIRST_RUNTIME_IMPORTS = {
    "run_coin_star_normal_camera_root_modes_integration.py": (
        "from coin_star_normal_camera_root_modes_core import",
    ),
    "run_coin_star_normal_object_root_setup_compensation_integration.py": (
        "from coin_star_normal_object_root_setup_core import",
    ),
    "run_coin_star_normal_projection_parity_integration.py": (
        "from Blender_to_Spine2D_Mesh_Exporter.application import",
        "from coin_star_normal_projection_parity_core import",
    ),
    "run_coin_star_normal_side_segment_retention_integration.py": (
        "from coin_star_normal_side_segment_core import",
    ),
}


def test_thin_real_coin_runners_bootstrap_import_paths_before_runtime_imports() -> None:
    for filename, runtime_imports in _RUNNERS_AND_FIRST_RUNTIME_IMPORTS.items():
        source = (HEADLESS / filename).read_text(encoding="utf-8")

        bootstrap_index = source.index(
            "for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):"
        )
        insert_index = source.index("sys.path.insert(0, str(path))")

        assert "SCRIPT_DIRECTORY = Path(__file__).resolve().parent" in source
        assert "REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]" in source
        assert bootstrap_index < insert_index

        for runtime_import in runtime_imports:
            runtime_import_index = source.index(runtime_import)
            assert insert_index < runtime_import_index, (
                f"{filename} imports runtime dependency before sys.path bootstrap: "
                f"{runtime_import}"
            )
