"""Verify that production selected-object UI export ignores persisted Connect flags.

The per-object RNA is retained so older .blend files load without data loss. Production
Analyze/Export must nevertheless build a standalone plan. Connected and mixed routing is
available only through the explicitly named development-only plan builder.
"""

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

import Blender_to_Spine2D_Mesh_Exporter as addon  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1MultiObjectMode,
    resolve_a1_multi_object_preparation_settings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_export_plan import (  # noqa: E402
    build_development_connected_ui_export_plan,
    build_selected_ui_export_plan,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import (  # noqa: E402
    A1RigProfile,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import (  # noqa: E402
    SpineJsonTarget,
)


PASS_MARKER = (
    "[PRODUCTION_UI_STANDALONE] PASS persisted Connect flags ignored by production"
)


def _assert(condition: bool, message: str) -> None:
    """Raise one deterministic assertion with a useful Blender-console message."""

    if not condition:
        raise AssertionError(message)


def _mesh_identity(mesh: bpy.types.Mesh) -> tuple[str, int]:
    """Return a stable process-local identity for one Blender Mesh datablock."""

    pointer = getattr(mesh, "as_pointer", None)
    if callable(pointer):
        resolved = int(pointer())
        if resolved:
            return "RNA_POINTER", resolved
    return "PYTHON_ID", id(mesh)


def _clear_scene_objects() -> None:
    """Remove every scene object and each now-unused owned mesh without operators."""

    meshes_by_identity: dict[tuple[str, int], bpy.types.Mesh] = {}
    for obj in tuple(bpy.data.objects):
        mesh = obj.data if getattr(obj, "type", None) == "MESH" else None
        if mesh is not None:
            meshes_by_identity.setdefault(_mesh_identity(mesh), mesh)
        bpy.data.objects.remove(obj, do_unlink=True)

    for mesh in tuple(meshes_by_identity.values()):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def _create_mesh_object(name: str) -> bpy.types.Object:
    """Create one minimal real Mesh object suitable for UI request capture."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        (),
        ((0, 1, 2),),
    )
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _select_objects(objects: tuple[bpy.types.Object, ...]) -> None:
    """Make the supplied Mesh objects the exact deterministic Blender selection."""

    if len(objects) < 2:
        raise ValueError("objects must contain at least two Mesh objects")
    for obj in tuple(bpy.context.selected_objects):
        obj.select_set(False)
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]


def _configure_scene(output_directory: Path) -> None:
    """Configure exact Spine 4.1 production UI settings for the request capture."""

    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be pathlib.Path")
    scene = bpy.context.scene
    scene.spine2d_target_spine_version = SpineJsonTarget.SPINE_4_1.value
    scene.spine2d_rig_profile = A1RigProfile.TWO_AXIS_ROTATION_SCALE.value
    scene.spine2d_texture_size = 64
    scene.spine2d_json_path = str(output_directory)
    scene.spine2d_images_path = "images"
    scene.spine2d_frames_for_render = 0
    scene.spine2d_bake_frame_start = 0


def _assert_production_plan(objects: tuple[bpy.types.Object, ...]) -> None:
    """Require standalone production routing and accepted Spine 4.1 preflight."""

    production = build_selected_ui_export_plan(bpy.context)
    _assert(
        production.settings.mode is A1MultiObjectMode.STANDALONE,
        f"production mode changed: {production.settings.mode!r}",
    )
    _assert(production.connected_sources == (), "production retained connected sources")
    _assert(
        len(production.standalone_sources) == len(objects),
        "production standalone source count changed",
    )
    _assert(
        production.settings.anchor_component_id is None,
        "standalone production plan unexpectedly has an anchor",
    )
    _assert(production.issues == (), "production plan emitted a Connect fallback warning")

    for source in production.all_sources:
        resolved = resolve_a1_multi_object_preparation_settings(
            source.settings,
            production.settings.mode,
        )
        _assert(resolved is source.settings, "Spine 4.1 standalone preflight rewrote settings")
        _assert(
            source.settings.export.spine_target is SpineJsonTarget.SPINE_4_1,
            "production source target changed",
        )


def _assert_development_plan() -> None:
    """Prove that persisted flags remain available only to the explicit dev builder."""

    development = build_development_connected_ui_export_plan(bpy.context)
    _assert(
        development.settings.mode is A1MultiObjectMode.MIXED,
        f"development mode changed: {development.settings.mode!r}",
    )
    _assert(len(development.connected_sources) == 2, "development Connect flags were lost")
    _assert(len(development.standalone_sources) == 1, "development partition changed")


def _run_integration() -> None:
    """Exercise production and development routing against the same persisted state."""

    with tempfile.TemporaryDirectory(prefix="spine2d-ui-standalone-") as output_text:
        _clear_scene_objects()
        objects = tuple(
            _create_mesh_object(name)
            for name in ("StaleConnectA", "StaleConnectB", "StandaloneC")
        )
        _select_objects(objects)
        _configure_scene(Path(output_text))

        objects[0].spine2d_connect_settings.enabled = True
        objects[1].spine2d_connect_settings.enabled = True
        objects[2].spine2d_connect_settings.enabled = False

        _assert_production_plan(objects)
        _assert_development_plan()


def main() -> None:
    """Register the Rewrite extension, run the gate, and always restore Blender state."""

    registered = False
    try:
        addon.register()
        registered = True
        _run_integration()
        print(PASS_MARKER)
    finally:
        try:
            _clear_scene_objects()
        finally:
            if registered:
                addon.unregister()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # pragma: no cover - Blender process reports the traceback.
        traceback.print_exc()
        raise
