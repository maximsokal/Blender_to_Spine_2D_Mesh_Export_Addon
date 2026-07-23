"""Blender 5.2+ regressions replacing the retired non-node material fallback."""

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
    export_a1_single_object,
)
from run_bake_integration import (  # noqa: E402
    _assert,
    _clear_scene,
    _create_emission_material,
    _create_quad,
    _material_fingerprint,
    _temporary_datablock_names,
)
from run_camera_projection_integration import _settings  # noqa: E402


def test_blender_52_node_material_is_baked_from_an_owned_copy() -> None:
    _clear_scene()
    bpy.context.scene.render.engine = "CYCLES"
    with tempfile.TemporaryDirectory(prefix="spine2d-node-material-") as directory:
        source = _create_quad("NodeMaterialSource")
        material = _create_emission_material(source)
        before = _material_fingerprint(material)

        _assert(material.node_tree is not None, "Blender 5.2 material has no node tree")
        result = export_a1_single_object(
            source,
            _settings(Path(directory), "NodeMaterial"),
        )

        _assert(result.success, f"Blender 5.2 node material export failed: {result.issues}")
        _assert(
            _material_fingerprint(material) == before,
            "source Blender 5.2 material graph was mutated",
        )
        _assert(not _temporary_datablock_names(), "node material export leaked temporary data")


def test_material_graph_without_output_fails_without_source_mutation() -> None:
    _clear_scene()
    bpy.context.scene.render.engine = "CYCLES"
    with tempfile.TemporaryDirectory(prefix="spine2d-invalid-node-material-") as directory:
        source = _create_quad("InvalidNodeMaterialSource")
        material = bpy.data.materials.new(name="InvalidNodeMaterial")
        _assert(material.node_tree is not None, "Blender 5.2 material has no node tree")
        material.node_tree.nodes.clear()
        source.data.materials.append(material)
        before = _material_fingerprint(material)

        result = export_a1_single_object(
            source,
            _settings(Path(directory), "InvalidNodeMaterial"),
        )

        _assert(not result.success, "material without an output node exported silently")
        errors = tuple(issue for issue in result.issues if issue.severity.value == "ERROR")
        _assert(errors, f"invalid material export has no error issue: {result.issues}")
        _assert(
            _material_fingerprint(material) == before,
            "failed material analysis mutated the source graph",
        )
        _assert(not _temporary_datablock_names(), "invalid material export leaked temporary data")


def main() -> None:
    _assert(tuple(bpy.app.version[:3]) >= (5, 2, 0), "Blender 5.2+ is required")
    tests = (
        test_blender_52_node_material_is_baked_from_an_owned_copy,
        test_material_graph_without_output_fails_without_source_mutation,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"Blender 5.2 material contract integration passed: {len(tests)} tests")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
