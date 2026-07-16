#!/usr/bin/env python3
"""Run one Legacy or Rewrite fixture export inside a fresh Blender process."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence

import bpy


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import Blender_to_Spine2D_Mesh_Exporter as addon  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    sanitize_filename_stem,
)


class FixtureWorkerError(RuntimeError):
    """Raised when a fixture cannot be reproduced exactly inside Blender."""


def _arguments_after_separator(argv: Sequence[str]) -> list[str]:
    try:
        separator = argv.index("--")
    except ValueError:
        return []
    return list(argv[separator + 1 :])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-json", type=Path, required=True)
    parser.add_argument("--backend", choices=("LEGACY", "REWRITE"), required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    return parser


def _load_payload(path: Path) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise FixtureWorkerError(f"Worker payload does not exist: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise FixtureWorkerError(f"Unable to read worker payload: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise FixtureWorkerError(
            f"Invalid worker payload at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(value, Mapping):
        raise FixtureWorkerError("Worker payload root must be an object")
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve(strict=False)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=False)
    return {
        "relative_path": resolved.relative_to(root).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _output_file_records(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    return [_file_record(path, root) for path in files]


def _datablock_snapshot() -> dict[str, list[str]]:
    return {
        "objects": sorted(item.name_full for item in bpy.data.objects),
        "meshes": sorted(item.name_full for item in bpy.data.meshes),
        "collections": sorted(item.name_full for item in bpy.data.collections),
        "materials": sorted(item.name_full for item in bpy.data.materials),
        "images": sorted(item.name_full for item in bpy.data.images),
    }


def _datablock_additions(
    before: Mapping[str, list[str]],
    after: Mapping[str, list[str]],
) -> dict[str, list[str]]:
    return {
        key: sorted(set(after.get(key, ())) - set(before.get(key, ())))
        for key in before
    }


def _context_snapshot() -> dict[str, Any]:
    active = bpy.context.view_layer.objects.active
    return {
        "active_object": None if active is None else active.name_full,
        "selected_objects": sorted(item.name_full for item in bpy.context.selected_objects),
        "mode": bpy.context.mode,
        "frame_current": bpy.context.scene.frame_current,
        "render_engine": bpy.context.scene.render.engine,
    }


def _mesh_snapshot(object_names: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in object_names:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH" or obj.data is None:
            continue
        mesh = obj.data
        result[name] = {
            "mesh_name": mesh.name_full,
            "vertex_count": len(mesh.vertices),
            "edge_count": len(mesh.edges),
            "polygon_count": len(mesh.polygons),
            "uv_layers": [layer.name for layer in mesh.uv_layers],
            "seam_edge_indices": [edge.index for edge in mesh.edges if edge.use_seam],
            "material_names": [
                None if material is None else material.name_full
                for material in mesh.materials
            ],
        }
    return result


def _require_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise FixtureWorkerError(f"payload.{key} must be a non-empty string")
    return value


def _require_string_array(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise FixtureWorkerError(f"payload.{key} must be an array of object names")
    if len(value) != len(set(value)):
        raise FixtureWorkerError(f"payload.{key} contains duplicates")
    return tuple(value)


def _resolve_objects(names: Sequence[str]) -> tuple[Any, ...]:
    result: list[Any] = []
    missing: list[str] = []
    invalid: list[str] = []
    for name in names:
        obj = bpy.data.objects.get(name)
        if obj is None:
            missing.append(name)
            continue
        if obj.type != "MESH" or obj.data is None:
            invalid.append(name)
            continue
        result.append(obj)
    if missing:
        raise FixtureWorkerError("Objects not found: " + ", ".join(missing))
    if invalid:
        raise FixtureWorkerError("Objects are not Mesh values: " + ", ".join(invalid))
    return tuple(result)


def _configure_selection(active_name: str, selected_names: Sequence[str]) -> None:
    objects = _resolve_objects(selected_names)
    active = bpy.data.objects.get(active_name)
    if active is None or active not in objects:
        raise FixtureWorkerError("active_object is not in the selected Mesh set")
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in bpy.context.view_layer.objects:
        obj.select_set(False)
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = active


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FixtureWorkerError(f"{label} must be an object")
    return value


def _configure_scene(payload: Mapping[str, Any], backend: str) -> None:
    scene = bpy.context.scene
    settings = _mapping(payload.get("settings"), "settings")
    output_directory = Path(_require_string(payload, "output_directory")).resolve(
        strict=False
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    scene.spine2d_texture_size = int(settings.get("texture_size", 1024))
    scene.spine2d_json_path = str(output_directory)
    scene.spine2d_images_path = str(settings.get("images_path", "images"))
    scene.spine2d_seam_maker_mode = str(settings.get("seam_mode", "AUTO"))
    scene.spine2d_angle_limit = float(settings.get("angle_limit", 30.0))
    scene.spine2d_control_icons = bool(settings.get("control_icons", True))
    scene.spine2d_export_preview_animation = bool(
        settings.get("preview_animation", True)
    )

    mode = _require_string(payload, "mode").lower()
    selected_names = _require_string_array(payload, "selected_objects")
    connected_names = set(_require_string_array(payload, "connected_objects"))
    sequence = _mapping(settings.get("sequence", {}), "settings.sequence")
    per_object_sequence = _mapping(
        settings.get("per_object_sequence", {}),
        "settings.per_object_sequence",
    )

    if mode == "single":
        scene.spine2d_single_export_backend = backend
        scene.spine2d_bake_frame_start = int(sequence.get("start_frame", 0))
        scene.spine2d_frames_for_render = int(sequence.get("frame_count", 0))
    elif mode == "multi":
        scene.spine2d_multi_export_backend = backend
        for name in selected_names:
            obj = bpy.data.objects.get(name)
            if obj is None:
                raise FixtureWorkerError(f"Selected object disappeared: {name}")
            obj.spine2d_connect_settings.enabled = name in connected_names
            raw_sequence = _mapping(
                per_object_sequence.get(name, {}),
                f"per_object_sequence.{name}",
            )
            obj.spine2d_bake_settings.bake_frame_start = int(
                raw_sequence.get("start_frame", 0)
            )
            obj.spine2d_bake_settings.frames_for_render = int(
                raw_sequence.get("frame_count", 0)
            )
    else:
        raise FixtureWorkerError(f"Unsupported fixture mode: {mode}")


def _expected_json_name(payload: Mapping[str, Any]) -> str:
    override = payload.get("expected_json_name")
    if override is not None:
        if not isinstance(override, str) or not override.endswith(".json"):
            raise FixtureWorkerError("expected_json_name must be a .json filename")
        return override
    active = sanitize_filename_stem(_require_string(payload, "active_object"))
    mode = _require_string(payload, "mode").lower()
    selected_count = len(_require_string_array(payload, "selected_objects"))
    if mode == "single":
        return f"{active}_merged.json"
    return f"{active}_plus_{selected_count - 1}_objects.json"


def _invoke_operator(payload: Mapping[str, Any]) -> set[str]:
    mode = _require_string(payload, "mode").lower()
    if mode == "single":
        return set(bpy.ops.object.save_uv_as_json())
    if mode == "multi":
        return set(bpy.ops.object.spine2d_multi_export())
    raise FixtureWorkerError(f"Unsupported fixture mode: {mode}")


def _run(payload: Mapping[str, Any], backend: str) -> dict[str, Any]:
    source_path = Path(bpy.data.filepath).expanduser().resolve(strict=False)
    if not source_path.is_file():
        raise FixtureWorkerError(
            "Blender process must be started with one existing .blend fixture"
        )
    selected_names = _require_string_array(payload, "selected_objects")
    active_name = _require_string(payload, "active_object")
    output_directory = Path(_require_string(payload, "output_directory")).resolve(
        strict=False
    )

    source_hash_before = _sha256(source_path)
    addon.register()
    try:
        _configure_selection(active_name, selected_names)
        _configure_scene(payload, backend)
        context_before = _context_snapshot()
        mesh_before = _mesh_snapshot(selected_names)
        datablocks_before = _datablock_snapshot()

        operator_result = _invoke_operator(payload)
        if "FINISHED" not in operator_result:
            raise FixtureWorkerError(
                f"Export operator returned {sorted(operator_result)}"
            )

        context_after = _context_snapshot()
        mesh_after = _mesh_snapshot(selected_names)
        datablocks_after = _datablock_snapshot()
        expected_json = output_directory / _expected_json_name(payload)
        if not expected_json.is_file():
            raise FixtureWorkerError(
                f"Expected final JSON was not created: {expected_json}"
            )
        outputs = _output_file_records(output_directory)
        source_hash_after = _sha256(source_path)
        additions = _datablock_additions(datablocks_before, datablocks_after)
        return {
            "success": True,
            "case_id": _require_string(payload, "case_id"),
            "backend": backend,
            "mode": _require_string(payload, "mode").lower(),
            "blender_version": bpy.app.version_string,
            "source_blend": str(source_path),
            "source_sha256_before": source_hash_before,
            "source_sha256_after": source_hash_after,
            "source_unchanged": source_hash_before == source_hash_after,
            "operator_result": sorted(operator_result),
            "expected_json": str(expected_json),
            "output_files": outputs,
            "context_before": context_before,
            "context_after": context_after,
            "context_restored": context_before == context_after,
            "mesh_before": mesh_before,
            "mesh_after": mesh_after,
            "mesh_restored": mesh_before == mesh_after,
            "datablock_additions": additions,
            "temporary_datablocks_clean": not any(additions.values()),
        }
    finally:
        addon.unregister()


def main() -> None:
    namespace = _build_parser().parse_args(_arguments_after_separator(sys.argv))
    report_path = namespace.report_json.expanduser().resolve(strict=False)
    payload: Mapping[str, Any] | None = None
    try:
        payload = _load_payload(namespace.payload_json)
        report = _run(payload, namespace.backend)
        _write_json_atomic(report_path, report)
    except Exception as exc:
        traceback.print_exc()
        failure = {
            "success": False,
            "case_id": None if payload is None else payload.get("case_id"),
            "backend": namespace.backend,
            "blender_version": bpy.app.version_string,
            "error_type": type(exc).__name__,
            "error": str(exc) or type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
        try:
            _write_json_atomic(report_path, failure)
        except Exception:
            traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
