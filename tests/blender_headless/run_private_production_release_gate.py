"""Run manifest-driven private production .blend parity and fail closed for release."""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Mapping

import bpy

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import Blender_to_Spine2D_Mesh_Exporter as addon  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.domain.release_gate import (  # noqa: E402
    PrivateFixtureSpec,
    PrivateReleaseGateError,
    parse_private_release_manifest,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (  # noqa: E402
    A1ParitySettings,
    compare_a1_exports,
)


class PrivateProductionGateFailure(RuntimeError):
    pass


def _arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rewrite-sha", type=str, default="")
    return parser.parse_args(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrivateProductionGateFailure(f"Unable to read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise PrivateProductionGateFailure(f"JSON root must be an object: {path}")
    return value


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PrivateProductionGateFailure(f"Unable to resolve git HEAD: {exc}") from exc


def _replace_placeholders(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        result = value
        for key, replacement in replacements.items():
            result = result.replace(key, replacement)
        return result
    if isinstance(value, list):
        return [_replace_placeholders(item, replacements) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_placeholders(item, replacements) for item in value)
    if isinstance(value, Mapping):
        return {
            str(key): _replace_placeholders(item, replacements)
            for key, item in value.items()
        }
    return value


def _set_attribute_path(root: Any, path: str, value: Any) -> None:
    if not isinstance(path, str) or not path.strip():
        raise PrivateProductionGateFailure("attribute path must be non-empty")
    parts = path.split(".")
    current = root
    for part in parts[:-1]:
        if not hasattr(current, part):
            raise PrivateProductionGateFailure(
                f"Missing RNA path segment {part!r} in {path!r}"
            )
        current = getattr(current, part)
    if not hasattr(current, parts[-1]):
        raise PrivateProductionGateFailure(f"Missing writable RNA attribute {path!r}")
    setattr(current, parts[-1], value)


def _ensure_addon_registered() -> None:
    try:
        addon.register()
    except Exception as exc:
        text = str(exc).lower()
        if "already registered" not in text:
            raise PrivateProductionGateFailure(
                f"Unable to register rewrite add-on: {exc}"
            ) from exc


def _temporary_datablock_names() -> tuple[str, ...]:
    collections = (
        bpy.data.objects,
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.node_groups,
        bpy.data.collections,
    )
    return tuple(
        sorted(
            f"{type(item).__name__}:{item.name}"
            for collection in collections
            for item in collection
            if str(item.name).startswith("__Spine2D_")
        )
    )


def _rounded(values) -> tuple[float, ...]:
    return tuple(round(float(value), 9) for value in values)


def _mesh_fingerprint(obj: Any) -> Mapping[str, Any] | None:
    mesh = getattr(obj, "data", None)
    if getattr(obj, "type", None) != "MESH" or mesh is None:
        return None
    return {
        "vertices": tuple(_rounded(vertex.co) for vertex in mesh.vertices),
        "edges": tuple(tuple(int(value) for value in edge.vertices) for edge in mesh.edges),
        "polygons": tuple(
            {
                "vertices": tuple(int(value) for value in polygon.vertices),
                "material_index": int(polygon.material_index),
                "use_smooth": bool(polygon.use_smooth),
            }
            for polygon in mesh.polygons
        ),
        "uv_layers": tuple(
            {
                "name": layer.name,
                "active": layer is mesh.uv_layers.active,
                "uv": tuple(_rounded(loop.uv) for loop in layer.data),
            }
            for layer in mesh.uv_layers
        ),
        "materials": tuple(
            None if material is None else material.name_full
            for material in mesh.materials
        ),
    }


def _runtime_fingerprint(selected_names: tuple[str, ...]) -> Mapping[str, Any]:
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    active = getattr(view_layer.objects, "active", None)
    selected = tuple(sorted(obj.name_full for obj in bpy.context.selected_objects))
    objects = []
    for name in sorted(selected_names):
        obj = bpy.data.objects.get(name)
        if obj is None:
            objects.append({"name": name, "missing": True})
            continue
        objects.append(
            {
                "name": obj.name_full,
                "type": obj.type,
                "matrix_world": _rounded(
                    component for row in obj.matrix_world for component in row
                ),
                "hide_render": bool(obj.hide_render),
                "visible_camera": (
                    bool(obj.visible_camera) if hasattr(obj, "visible_camera") else None
                ),
                "modifiers": tuple(
                    (modifier.name, modifier.type, bool(modifier.show_render))
                    for modifier in obj.modifiers
                ),
                "mesh": _mesh_fingerprint(obj),
            }
        )
    return {
        "active_object": None if active is None else active.name_full,
        "selected_objects": selected,
        "mode": str(getattr(bpy.context, "mode", "")),
        "frame_current": int(scene.frame_current),
        "render": {
            "engine": scene.render.engine,
            "resolution_x": int(scene.render.resolution_x),
            "resolution_y": int(scene.render.resolution_y),
            "resolution_percentage": int(scene.render.resolution_percentage),
            "filepath": scene.render.filepath,
            "film_transparent": bool(scene.render.film_transparent),
            "use_compositing": bool(scene.render.use_compositing),
            "use_sequencer": bool(scene.render.use_sequencer),
            "file_format": scene.render.image_settings.file_format,
            "color_mode": scene.render.image_settings.color_mode,
            "color_depth": scene.render.image_settings.color_depth,
        },
        "objects": tuple(objects),
    }


def _prepare_fixture_scene(
    fixture: PrivateFixtureSpec,
    *,
    manifest_directory: Path,
    output_directory: Path,
) -> None:
    source_path = (manifest_directory / fixture.source_blend).resolve(strict=True)
    bpy.ops.wm.open_mainfile(filepath=str(source_path), load_ui=False)
    _ensure_addon_registered()
    replacements = {
        "${OUTPUT_DIR}": str(output_directory),
        "${FIXTURE_DIR}": str(source_path.parent),
        "${MANIFEST_DIR}": str(manifest_directory),
    }
    scene = bpy.context.scene
    for path, value in fixture.scene_attributes.items():
        _set_attribute_path(
            scene,
            str(path),
            _replace_placeholders(value, replacements),
        )
    for key, value in fixture.scene_custom_properties.items():
        scene[str(key)] = _replace_placeholders(value, replacements)
    for object_name, attributes in fixture.object_attributes.items():
        obj = bpy.data.objects.get(object_name)
        if obj is None:
            raise PrivateProductionGateFailure(
                f"Fixture {fixture.fixture_id!r} has no object {object_name!r}"
            )
        for path, value in attributes.items():
            _set_attribute_path(
                obj,
                str(path),
                _replace_placeholders(value, replacements),
            )
    for object_name, properties in fixture.object_custom_properties.items():
        obj = bpy.data.objects.get(object_name)
        if obj is None:
            raise PrivateProductionGateFailure(
                f"Fixture {fixture.fixture_id!r} has no object {object_name!r}"
            )
        for key, value in properties.items():
            obj[str(key)] = _replace_placeholders(value, replacements)

    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    selected = []
    for object_name in fixture.selected_objects:
        obj = bpy.data.objects.get(object_name)
        if obj is None:
            raise PrivateProductionGateFailure(
                f"Fixture {fixture.fixture_id!r} has no selected object {object_name!r}"
            )
        obj.select_set(True)
        selected.append(obj)
    active = bpy.data.objects.get(fixture.active_object)
    if active is None:
        raise PrivateProductionGateFailure(
            f"Fixture {fixture.fixture_id!r} has no active object {fixture.active_object!r}"
        )
    bpy.context.view_layer.objects.active = active
    update = getattr(bpy.context.view_layer, "update", None)
    if callable(update):
        update()


def _invoke_operator(fixture: PrivateFixtureSpec, replacements: Mapping[str, str]) -> None:
    module_name, operator_name = fixture.operator.split(".")
    module = getattr(bpy.ops, module_name, None)
    operator = None if module is None else getattr(module, operator_name, None)
    if operator is None:
        raise PrivateProductionGateFailure(
            f"Blender operator is unavailable: {fixture.operator}"
        )
    kwargs = _replace_placeholders(dict(fixture.operator_kwargs), replacements)
    try:
        result = operator(**kwargs)
    except Exception as exc:
        raise PrivateProductionGateFailure(
            f"Operator {fixture.operator} raised: {exc}"
        ) from exc
    if "FINISHED" not in set(result):
        raise PrivateProductionGateFailure(
            f"Operator {fixture.operator} did not finish: {set(result)}"
        )


def _image_pixels(path: Path) -> tuple[tuple[int, int], array, str]:
    image = None
    try:
        image = bpy.data.images.load(str(path), check_existing=False)
        size = tuple(int(value) for value in image.size[:2])
        pixels = array("f", [0.0]) * (size[0] * size[1] * 4)
        image.pixels.foreach_get(pixels)
        return size, pixels, str(getattr(image, "alpha_mode", ""))
    except Exception as exc:
        raise PrivateProductionGateFailure(f"Unable to decode image {path}: {exc}") from exc
    finally:
        if image is not None:
            bpy.data.images.remove(image)


def _compare_image_pair(expected: Path, actual: Path, spec) -> Mapping[str, Any]:
    if not expected.is_file():
        raise PrivateProductionGateFailure(f"Expected image is missing: {expected}")
    if not actual.is_file():
        raise PrivateProductionGateFailure(f"Actual image is missing: {actual}")
    expected_size, expected_pixels, expected_alpha_mode = _image_pixels(expected)
    actual_size, actual_pixels, actual_alpha_mode = _image_pixels(actual)
    errors: list[str] = []
    if expected_size != actual_size:
        errors.append(f"size mismatch: expected={expected_size}, actual={actual_size}")
        return {
            "expected": str(expected),
            "actual": str(actual),
            "expected_size": expected_size,
            "actual_size": actual_size,
            "errors": errors,
        }
    differences = [
        abs(float(expected_value) - float(actual_value))
        for expected_value, actual_value in zip(expected_pixels, actual_pixels)
    ]
    rgb_differences = [
        difference
        for index, difference in enumerate(differences)
        if index % 4 != 3
    ]
    alpha_differences = differences[3::4]
    maximum = max(rgb_differences, default=0.0)
    mean = sum(rgb_differences) / max(1, len(rgb_differences))
    alpha_maximum = max(alpha_differences, default=0.0)
    if maximum > spec.maximum_absolute_error:
        errors.append(
            f"RGB max error {maximum} exceeds {spec.maximum_absolute_error}"
        )
    if mean > spec.mean_absolute_error:
        errors.append(f"RGB mean error {mean} exceeds {spec.mean_absolute_error}")
    if alpha_maximum > spec.alpha_maximum_absolute_error:
        errors.append(
            "alpha max error "
            f"{alpha_maximum} exceeds {spec.alpha_maximum_absolute_error}"
        )
    return {
        "expected": str(expected),
        "actual": str(actual),
        "expected_sha256": _sha256(expected),
        "actual_sha256": _sha256(actual),
        "size": expected_size,
        "expected_alpha_mode": expected_alpha_mode,
        "actual_alpha_mode": actual_alpha_mode,
        "rgb_maximum_absolute_error": maximum,
        "rgb_mean_absolute_error": mean,
        "alpha_maximum_absolute_error": alpha_maximum,
        "errors": errors,
    }


def _parity_report(fixture: PrivateFixtureSpec, expected_path: Path, actual_path: Path):
    expected = _load_json(expected_path)
    actual = _load_json(actual_path)
    defaults = A1ParitySettings()
    settings = A1ParitySettings(
        absolute_tolerance=fixture.absolute_tolerance,
        relative_tolerance=fixture.relative_tolerance,
        ignored_paths=defaults.ignored_paths + fixture.ignored_paths,
        compare_animations=fixture.compare_animations,
        nonessential_mesh_edges_are_errors=fixture.strict_edges,
    )
    return compare_a1_exports(expected, actual, settings)


def _issue_mapping(issue) -> Mapping[str, Any]:
    return {
        "severity": issue.severity.value,
        "code": issue.code,
        "path": issue.path,
        "message": issue.message,
        "expected": issue.expected,
        "actual": issue.actual,
    }


def _run_fixture(
    fixture: PrivateFixtureSpec,
    *,
    manifest_directory: Path,
    output_root: Path,
) -> Mapping[str, Any]:
    source_path = (manifest_directory / fixture.source_blend).resolve(strict=True)
    legacy_json_path = (manifest_directory / fixture.legacy_json).resolve(strict=True)
    output_directory = output_root / fixture.fixture_id
    output_directory.mkdir(parents=True, exist_ok=True)
    actual_json_path = output_directory / fixture.actual_json
    source_digest_before = _sha256(source_path)

    _prepare_fixture_scene(
        fixture,
        manifest_directory=manifest_directory,
        output_directory=output_directory,
    )
    runtime_before = _runtime_fingerprint(fixture.selected_objects)
    temporary_before = set(_temporary_datablock_names())
    replacements = {
        "${OUTPUT_DIR}": str(output_directory),
        "${FIXTURE_DIR}": str(source_path.parent),
        "${MANIFEST_DIR}": str(manifest_directory),
    }
    _invoke_operator(fixture, replacements)
    runtime_after = _runtime_fingerprint(fixture.selected_objects)
    temporary_after = set(_temporary_datablock_names())
    source_digest_after = _sha256(source_path)

    errors: list[str] = []
    if source_digest_before != source_digest_after:
        errors.append("source .blend file digest changed during rewrite export")
    if runtime_before != runtime_after:
        errors.append("selected source/runtime state changed during rewrite export")
    leaked = tuple(sorted(temporary_after - temporary_before))
    if leaked:
        errors.append(f"temporary Blender datablocks leaked: {leaked}")
    if not actual_json_path.is_file():
        errors.append(f"rewrite JSON is missing: {actual_json_path}")
        parity = None
        parity_issues = []
        parity_errors = 1
        parity_warnings = 0
    else:
        parity = _parity_report(fixture, legacy_json_path, actual_json_path)
        parity_issues = [_issue_mapping(issue) for issue in parity.issues]
        parity_errors = parity.error_count
        parity_warnings = parity.warning_count
        if parity_errors:
            errors.append(f"JSON parity found {parity_errors} errors")

    warning_codes = {
        issue["code"]
        for issue in parity_issues
        if issue["severity"] == "WARNING"
    }
    accepted = set(fixture.accepted_warning_codes)
    unaccepted = tuple(sorted(warning_codes - accepted))
    unused_acceptances = tuple(sorted(accepted - warning_codes))
    if unaccepted:
        errors.append(f"unaccepted parity warning codes: {unaccepted}")
    if unused_acceptances:
        errors.append(f"stale accepted warning codes: {unused_acceptances}")

    image_reports = []
    for pair in fixture.image_pairs:
        report = _compare_image_pair(
            (manifest_directory / pair.expected).resolve(strict=False),
            (output_directory / pair.actual).resolve(strict=False),
            pair,
        )
        image_reports.append(report)
        errors.extend(str(message) for message in report.get("errors", ()))

    return {
        "fixture_id": fixture.fixture_id,
        "passed": not errors,
        "capabilities": fixture.capabilities,
        "source_blend": str(source_path),
        "source_sha256_before": source_digest_before,
        "source_sha256_after": source_digest_after,
        "runtime_state_unchanged": runtime_before == runtime_after,
        "temporary_datablock_leaks": leaked,
        "legacy_json": str(legacy_json_path),
        "actual_json": str(actual_json_path),
        "json_compatible": False if parity is None else parity.compatible,
        "json_error_count": parity_errors,
        "json_warning_count": parity_warnings,
        "json_issues": parity_issues,
        "accepted_warning_codes": fixture.accepted_warning_codes,
        "image_reports": image_reports,
        "errors": errors,
    }


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    namespace = _arguments()
    manifest_path = namespace.manifest.expanduser().resolve(strict=True)
    manifest_mapping = _load_json(manifest_path)
    manifest = parse_private_release_manifest(manifest_mapping)
    actual_blender = ".".join(str(value) for value in bpy.app.version[:3])
    if actual_blender != manifest.blender_version:
        raise PrivateProductionGateFailure(
            f"Blender version mismatch: manifest={manifest.blender_version}, actual={actual_blender}"
        )
    actual_sha = _git_head()
    if namespace.rewrite_sha and actual_sha != namespace.rewrite_sha:
        raise PrivateProductionGateFailure(
            f"rewrite SHA mismatch: requested={namespace.rewrite_sha}, actual={actual_sha}"
        )

    _ensure_addon_registered()
    fixture_reports = []
    with tempfile.TemporaryDirectory(prefix="spine2d-private-release-") as directory:
        output_root = Path(directory)
        for fixture in manifest.fixtures:
            print(f"[PRIVATE-GATE] RUN {fixture.fixture_id}")
            try:
                report = _run_fixture(
                    fixture,
                    manifest_directory=manifest_path.parent,
                    output_root=output_root,
                )
            except Exception as exc:
                traceback.print_exc()
                report = {
                    "fixture_id": fixture.fixture_id,
                    "passed": False,
                    "capabilities": fixture.capabilities,
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }
            fixture_reports.append(report)
            print(
                f"[PRIVATE-GATE] {'PASS' if report['passed'] else 'FAIL'} "
                f"{fixture.fixture_id}"
            )

    passed = all(bool(report.get("passed")) for report in fixture_reports)
    if not manifest.release_gate.allow_unaccepted_warnings:
        passed = passed and all(
            not any("unaccepted parity warning" in error for error in report.get("errors", ()))
            for report in fixture_reports
        )
    payload = {
        "schema_version": 1,
        "suite_id": manifest.suite_id,
        "passed": passed,
        "rewrite_sha": actual_sha,
        "blender_version": actual_blender,
        "manifest_path": str(manifest_path),
        "required_capabilities": manifest.release_gate.required_capabilities,
        "fixture_count": len(fixture_reports),
        "fixtures": fixture_reports,
    }
    _write_report(namespace.report, payload)
    print(
        f"[PRIVATE-GATE] {'PASS' if passed else 'FAIL'} "
        f"{len(fixture_reports)} private production fixtures"
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except (PrivateReleaseGateError, PrivateProductionGateFailure) as exc:
        print(f"[PRIVATE-GATE] INVALID: {exc}", file=sys.stderr)
        raise SystemExit(2)
