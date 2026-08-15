#!/usr/bin/env python3
"""Save or verify installed Spine2D exact-version AddonPreferences in Blender."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import traceback

import bpy


def _arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True)
    parser.add_argument("--mode", required=True, choices=("save", "verify"))
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(values)


def _write_report(path: Path, payload: dict[str, object]) -> None:
    resolved = path.resolve(strict=False)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(resolved)


def _custom_exact_version(default_version: str) -> str:
    major_text, minor_text, patch_text = default_version.split(".")
    patch = int(patch_text)
    custom_patch = patch - 1 if patch > 0 else patch + 1
    return f"{int(major_text)}.{int(minor_text)}.{custom_patch}"


def _create_source_object():
    mesh = bpy.data.meshes.new("Spine2D_VersionPreferenceGateMesh")
    mesh.from_pydata(
        (
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
        ),
        (),
        ((0, 1, 2, 3),),
    )
    mesh.update(calc_edges=True)
    material = bpy.data.materials.new("Spine2D_VersionPreferenceGateMaterial")
    material.diffuse_color = (0.2, 0.6, 0.9, 1.0)
    mesh.materials.append(material)
    obj = bpy.data.objects.new("Spine2D_VersionPreferenceGate", mesh)
    bpy.context.scene.collection.objects.link(obj)
    for candidate in bpy.context.view_layer.objects:
        candidate.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    return obj


def _prepare_scene(output_root: Path) -> None:
    scene = bpy.context.scene
    scene.spine2d_json_path = str(output_root)
    scene.spine2d_images_path = "images"
    scene.spine2d_texture_size = 64
    scene.spine2d_angle_limit = 30
    scene.spine2d_seam_maker_mode = "AUTO"
    scene.spine2d_control_icons = False
    scene.spine2d_export_preview_animation = False
    scene.spine2d_frames_for_render = 0
    scene.spine2d_bake_frame_start = 0
    scene.spine2d_texture_export_mode = "NORMAL_UV_SEGMENTS"
    scene.spine2d_projection_direction = "POSITIVE_Z"


def _verify_real_exports(
    namespace: argparse.Namespace,
    helper,
    expected: dict[str, str],
) -> list[dict[str, object]]:
    output_root = namespace.output_root.resolve(strict=False)
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"Export output root must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    _create_source_object()

    plan_module = importlib.import_module(
        f"{namespace.module}.blender_adapter.a1_ui_export_plan"
    )
    router_module = importlib.import_module(
        f"{namespace.module}.blender_adapter.a1_ui_router"
    )
    scene = bpy.context.scene
    results: list[dict[str, object]] = []

    for spec in helper.SPINE_EXACT_VERSION_PREFERENCE_SPECS:
        exact_version = expected[spec.target.value]
        target_output = output_root / spec.target.value.lower()
        target_output.mkdir(parents=False, exist_ok=False)
        _prepare_scene(target_output)
        scene.spine2d_target_spine_version = spec.target.value

        plan = plan_module.build_active_ui_export_plan(bpy.context)
        if plan.settings.export.spine_target is not spec.target:
            raise RuntimeError(
                "Public UI plan resolved wrong Spine codec family: "
                f"expected={spec.target.value}, "
                f"actual={plan.settings.export.spine_target.value}"
            )
        if plan.settings.export.spine_version != exact_version:
            raise RuntimeError(
                "Public UI plan ignored persisted exact project version: "
                f"target={spec.target.value}, expected={exact_version!r}, "
                f"actual={plan.settings.export.spine_version!r}"
            )

        export_result = router_module.export_active_object_a1(bpy.context)
        if not export_result.success:
            raise RuntimeError(
                f"Production export failed for {spec.target.value}: "
                f"issues={export_result.issues!r}"
            )
        json_files = tuple(sorted(target_output.glob("*.json")))
        if len(json_files) != 1:
            raise RuntimeError(
                f"Expected one JSON for {spec.target.value}; actual={json_files!r}"
            )
        json_path = json_files[0]
        if exact_version not in json_path.name:
            raise RuntimeError(
                "JSON filename does not use persisted exact project version: "
                f"expected={exact_version!r}, actual={json_path.name!r}"
            )
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        actual_spine = payload.get("skeleton", {}).get("spine")
        if actual_spine != exact_version:
            raise RuntimeError(
                "skeleton.spine does not use persisted exact project version: "
                f"target={spec.target.value}, expected={exact_version!r}, "
                f"actual={actual_spine!r}"
            )
        png_files = tuple(sorted(target_output.rglob("*.png")))
        if not png_files:
            raise RuntimeError(
                f"Production export emitted no PNG for {spec.target.value}"
            )
        results.append(
            {
                "target": spec.target.value,
                "exact_version": exact_version,
                "json": str(json_path),
                "png_count": len(png_files),
            }
        )
    return results


def _run(namespace: argparse.Namespace) -> dict[str, object]:
    module = importlib.import_module(namespace.module)
    helper = importlib.import_module(
        f"{namespace.module}.blender_adapter.spine_version_preferences"
    )
    preferences = helper.get_spine_addon_preferences(required=True)
    expected = {
        spec.target.value: _custom_exact_version(spec.default_version)
        for spec in helper.SPINE_EXACT_VERSION_PREFERENCE_SPECS
    }

    if namespace.mode == "save":
        for spec in helper.SPINE_EXACT_VERSION_PREFERENCE_SPECS:
            helper.assign_spine_project_exact_version(
                preferences,
                spec.target,
                expected[spec.target.value],
            )
        save_result = set(bpy.ops.wm.save_userpref())
        if "FINISHED" not in save_result:
            raise RuntimeError(
                f"bpy.ops.wm.save_userpref returned {sorted(save_result)!r}"
            )
    else:
        save_result = set()

    actual = {
        spec.target.value: str(getattr(preferences, spec.property_name))
        for spec in helper.SPINE_EXACT_VERSION_PREFERENCE_SPECS
    }
    if actual != expected:
        raise RuntimeError(
            "Exact Spine project preferences differ after "
            f"{namespace.mode}: expected={expected!r}, actual={actual!r}"
        )

    exports = (
        _verify_real_exports(namespace, helper, expected)
        if namespace.mode == "verify"
        else []
    )
    return {
        "status": "passed",
        "mode": namespace.mode,
        "module": module.__name__,
        "expected": expected,
        "actual": actual,
        "save_operator_result": sorted(save_result),
        "exports": exports,
    }


def main() -> None:
    namespace = _arguments()
    try:
        payload = _run(namespace)
    except Exception as exc:
        payload = {
            "status": "failed",
            "mode": namespace.mode,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_report(namespace.report_json, payload)
        raise
    _write_report(namespace.report_json, payload)
    print(
        "[SPINE-VERSION-PREFERENCES] PASS "
        f"mode={namespace.mode} values={payload['actual']!r} "
        f"exports={len(payload['exports'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
