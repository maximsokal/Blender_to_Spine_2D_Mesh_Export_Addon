"""Validate public projection-direction RNA, planning, reset, and persistence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

import bpy


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import Blender_to_Spine2D_Mesh_Exporter as addon  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1MultiObjectMode,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_export_plan import (  # noqa: E402
    build_active_ui_export_plan,
    build_selected_ui_export_plan,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    A1TextureExportMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import (  # noqa: E402
    A1ProjectionDirection,
)


_OBJECT_NAMES = ("ProjectionUiAlpha", "ProjectionUiBeta")
_PERSISTED_DIRECTION = A1ProjectionDirection.NEGATIVE_Y


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("routes", "write", "read"),
        default="routes",
    )
    parser.add_argument("--blend", type=Path)
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else ()
    return parser.parse_args(values)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _prepare_output_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for mesh in tuple(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def _create_mesh_object(name: str, x: float):
    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(
        (
            (-0.5, -0.5, 0.0),
            (0.5, -0.5, 0.0),
            (0.5, 0.5, 0.0),
            (-0.5, 0.5, 0.0),
        ),
        (),
        ((0, 1, 2, 3),),
    )
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location.x = float(x)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _activate(objects, active) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = active
    bpy.context.view_layer.update()


def _configure_scene(output_root: Path) -> tuple[object, object]:
    scene = bpy.context.scene
    scene.spine2d_json_path = str(output_root)
    scene.spine2d_images_path = "images"
    scene.spine2d_texture_size = 256
    scene.spine2d_texture_export_mode = (
        A1TextureExportMode.NORMAL_UV_SEGMENTS.value
    )
    alpha = _create_mesh_object(_OBJECT_NAMES[0], -1.0)
    beta = _create_mesh_object(_OBJECT_NAMES[1], 1.0)
    _activate((alpha, beta), alpha)
    return alpha, beta


def _enum_contract(scene) -> dict[str, object]:
    property_definition = bpy.types.Scene.bl_rna.properties[
        "spine2d_projection_direction"
    ]
    identifiers = tuple(item.identifier for item in property_definition.enum_items)
    expected = tuple(direction.value for direction in A1ProjectionDirection)
    _assert(
        identifiers == expected,
        f"Projection enum identifiers mismatch: actual={identifiers}, expected={expected}",
    )
    scene_default = str(scene.spine2d_projection_direction)
    _assert(
        scene_default == A1ProjectionDirection.POSITIVE_Z.value,
        f"Projection Scene default is not POSITIVE_Z: {scene_default!r}",
    )
    return {
        "identifiers": list(identifiers),
        "default": scene_default,
    }


def _route_contract(scene) -> tuple[dict[str, object], ...]:
    results: list[dict[str, object]] = []
    for direction in A1ProjectionDirection:
        scene.spine2d_texture_export_mode = (
            A1TextureExportMode.NORMAL_UV_SEGMENTS.value
        )
        scene.spine2d_projection_direction = direction.value

        active_plan = build_active_ui_export_plan(bpy.context)
        selected_plan = build_selected_ui_export_plan(bpy.context)
        _assert(
            active_plan.settings.projection_direction is direction,
            f"Active plan lost {direction.value}",
        )
        _assert(
            selected_plan.settings.mode is A1MultiObjectMode.STANDALONE,
            "Public selected-object plan is not standalone",
        )
        source_directions = tuple(
            source.settings.projection_direction
            for source in selected_plan.standalone_sources
        )
        _assert(
            source_directions == (direction, direction),
            f"Selected plan lost {direction.value}: {source_directions}",
        )
        _assert(
            not selected_plan.connected_sources,
            "Public selected plan unexpectedly contains connected sources",
        )
        results.append(
            {
                "direction": direction.value,
                "active": active_plan.settings.projection_direction.value,
                "selected": [item.value for item in source_directions],
                "mode": selected_plan.settings.mode.value,
            }
        )
    return tuple(results)


def _rendered_camera_isolation(scene) -> dict[str, object]:
    scene.spine2d_texture_export_mode = A1TextureExportMode.CAMERA_PROJECTION.value
    scene.spine2d_projection_direction = A1ProjectionDirection.ACTIVE_CAMERA.value

    active_plan = build_active_ui_export_plan(bpy.context)
    selected_plan = build_selected_ui_export_plan(bpy.context)
    active_direction = active_plan.settings.projection_direction
    selected_directions = tuple(
        source.settings.projection_direction
        for source in selected_plan.standalone_sources
    )
    expected = A1ProjectionDirection.POSITIVE_Z
    _assert(active_direction is expected, "Rendered Camera active plan entered camera object-bake")
    _assert(
        selected_directions == (expected, expected),
        "Rendered Camera selected plan entered camera object-bake",
    )
    return {
        "storedDirection": scene.spine2d_projection_direction,
        "activeEffectiveDirection": active_direction.value,
        "selectedEffectiveDirections": [item.value for item in selected_directions],
    }


def _reset_contract(scene) -> dict[str, object]:
    scene.spine2d_projection_direction = A1ProjectionDirection.NEGATIVE_X.value
    result = bpy.ops.spine2d.reset_settings()
    _assert(result == {"FINISHED"}, f"Reset operator failed: {result}")
    _assert(
        scene.spine2d_projection_direction
        == A1ProjectionDirection.POSITIVE_Z.value,
        "Reset did not restore POSITIVE_Z",
    )
    return {
        "operatorResult": sorted(result),
        "direction": scene.spine2d_projection_direction,
    }


def _write_report(output_root: Path, name: str, payload: dict[str, object]) -> Path:
    path = output_root / name
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _run_routes(output_root: Path) -> Path:
    _clear_scene()
    _configure_scene(output_root)
    scene = bpy.context.scene
    report = {
        "status": "passed",
        "phase": "routes",
        "blenderVersion": bpy.app.version_string,
        "enum": _enum_contract(scene),
        "routes": list(_route_contract(scene)),
        "renderedCameraIsolation": _rendered_camera_isolation(scene),
        "reset": _reset_contract(scene),
        "publicMultiMode": A1MultiObjectMode.STANDALONE.value,
    }
    return _write_report(
        output_root,
        "projection_ui_routes_acceptance.json",
        report,
    )


def _run_write(output_root: Path, blend_path: Path) -> Path:
    _clear_scene()
    objects = _configure_scene(output_root)
    scene = bpy.context.scene
    scene.spine2d_texture_export_mode = (
        A1TextureExportMode.NORMAL_UV_SEGMENTS.value
    )
    scene.spine2d_projection_direction = _PERSISTED_DIRECTION.value
    _activate(objects, objects[0])
    resolved_blend = blend_path.expanduser().resolve(strict=False)
    resolved_blend.parent.mkdir(parents=True, exist_ok=True)
    result = bpy.ops.wm.save_as_mainfile(filepath=str(resolved_blend))
    _assert(result == {"FINISHED"}, f"Unable to save persistence fixture: {result}")
    return _write_report(
        output_root,
        "projection_ui_persistence_write.json",
        {
            "status": "passed",
            "phase": "write",
            "blend": str(resolved_blend),
            "direction": scene.spine2d_projection_direction,
        },
    )


def _run_read(output_root: Path, blend_path: Path) -> Path:
    resolved_blend = blend_path.expanduser().resolve(strict=True)
    _assert(
        Path(bpy.data.filepath).resolve(strict=False) == resolved_blend,
        f"Blender opened unexpected file: {bpy.data.filepath}",
    )
    scene = bpy.context.scene
    _assert(
        scene.spine2d_projection_direction == _PERSISTED_DIRECTION.value,
        "Persisted projection direction was not restored",
    )
    objects = tuple(bpy.data.objects[name] for name in _OBJECT_NAMES)
    _activate(objects, objects[0])
    active_plan = build_active_ui_export_plan(bpy.context)
    selected_plan = build_selected_ui_export_plan(bpy.context)
    _assert(
        active_plan.settings.projection_direction is _PERSISTED_DIRECTION,
        "Persisted active plan lost projection direction",
    )
    _assert(
        tuple(
            source.settings.projection_direction
            for source in selected_plan.standalone_sources
        )
        == (_PERSISTED_DIRECTION, _PERSISTED_DIRECTION),
        "Persisted selected plan lost projection direction",
    )
    return _write_report(
        output_root,
        "projection_ui_persistence_read.json",
        {
            "status": "passed",
            "phase": "read",
            "blend": str(resolved_blend),
            "direction": scene.spine2d_projection_direction,
            "activeDirection": active_plan.settings.projection_direction.value,
            "selectedDirections": [
                source.settings.projection_direction.value
                for source in selected_plan.standalone_sources
            ],
        },
    )


def run(arguments: argparse.Namespace) -> Path:
    output_root = _prepare_output_directory(arguments.output)
    blend_path = arguments.blend or (output_root / "projection_ui_persistence.blend")
    if arguments.phase == "routes":
        return _run_routes(output_root)
    if arguments.phase == "write":
        return _run_write(output_root, blend_path)
    if arguments.phase == "read":
        return _run_read(output_root, blend_path)
    raise AssertionError(f"Unhandled phase: {arguments.phase}")


def main() -> None:
    arguments = _parse_arguments()
    print(f"Blender version: {bpy.app.version_string}")
    addon.register()
    try:
        report_path = run(arguments)
        print(f"[PROJECTION_UI] REPORT {report_path}")
        print(f"[PROJECTION_UI] PASS phase={arguments.phase}")
    finally:
        addon.unregister()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
