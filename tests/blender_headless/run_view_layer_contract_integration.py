"""Blender 5.2 regression for B4 active View Layer source visibility."""

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
import Blender_to_Spine2D_Mesh_Exporter.blender_adapter.camera_projection_execution as render_module  # noqa: E402
from run_bake_integration import _assert, _temporary_datablock_names  # noqa: E402
from run_camera_projection_integration import (  # noqa: E402
    _create_layer_weight_material,
    _create_quad,
    _prepare_scene_with_sentinel,
    _settings,
)


def _find_layer_collection(root, collection):
    if getattr(root, "collection", None) is collection:
        return root
    for child in tuple(getattr(root, "children", ())):
        found = _find_layer_collection(child, collection)
        if found is not None:
            return found
    return None


def _move_to_collection_only(obj, collection) -> None:
    old_collections = tuple(obj.users_collection)
    if obj.name not in collection.objects:
        collection.objects.link(obj)
    for old in old_collections:
        if old is collection:
            continue
        old.objects.unlink(obj)


def test_holdout_only_source_is_rejected_before_render() -> None:
    _prepare_scene_with_sentinel()
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    with tempfile.TemporaryDirectory(prefix="spine2d-view-layer-contract-") as directory:
        source = _create_quad("HoldoutOnlySource")
        source.data.materials.append(
            _create_layer_weight_material("HoldoutOnlyMaterial")
        )
        collection = bpy.data.collections.new("HoldoutOnlyCollection")
        scene.collection.children.link(collection)
        _move_to_collection_only(source, collection)
        view_layer.update()
        layer_collection = _find_layer_collection(view_layer.layer_collection, collection)
        _assert(layer_collection is not None, "Holdout Layer Collection was not found")
        layer_collection.holdout = True
        view_layer.update()

        with mock.patch.object(
            render_module,
            "_call_render_operator",
            side_effect=AssertionError("render operator must not run for Holdout source"),
        ):
            result = export_a1_single_object(
                source,
                _settings(Path(directory), "HoldoutOnly"),
            )

        _assert(not result.success, "Holdout-only source was rendered as a normal attachment")
        errors = tuple(issue for issue in result.issues if issue.severity.value == "ERROR")
        _assert(errors, f"Holdout failure has no error issue: {result.issues}")
        _assert(
            any("Holdout" in issue.message for issue in errors),
            f"Holdout failure is not actionable: {errors}",
        )
        _assert(not result.committed_paths, "Holdout failure committed output files")
        _assert(not _temporary_datablock_names(), "Holdout failure leaked temporary data")


def main() -> None:
    test_holdout_only_source_is_rejected_before_render()
    print("[PASS] test_holdout_only_source_is_rejected_before_render")
    print("View Layer camera projection contract passed: 1 test")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
