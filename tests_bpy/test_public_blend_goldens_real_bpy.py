"""Generate, reopen, export, and compare public .blend fixtures to reviewed goldens."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import bpy
import pytest

import Blender_to_Spine2D_Mesh_Exporter as addon
from tools.create_public_blend_fixtures import create_all


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "tests" / "fixtures" / "public_blend_golden.json"
CANDIDATE_PATH = ROOT / ".pytest_cache" / "public_blend_golden_candidate.json"
GOLDEN_DOCUMENT = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
GOLDEN = GOLDEN_DOCUMENT["cases"]


@pytest.fixture(scope="session")
def generated_public_blend_fixtures(tmp_path_factory):
    fixture_root = tmp_path_factory.mktemp("spine2d-public-blend-fixtures")
    created = create_all(fixture_root)
    assert tuple(sorted(path.stem for path in created)) == tuple(sorted(GOLDEN))
    for path in created:
        assert path.is_file()
        assert path.stat().st_size > 1024
    return fixture_root


def _register_steps():
    completed = []
    try:
        for step in addon.REGISTRATION_STEPS:
            step[1]()
            completed.append(step)
        return completed
    except Exception:
        for step in reversed(completed):
            step[2]()
        raise


def _unregister_steps(completed):
    failures = []
    for label, _register, unregister in reversed(completed):
        try:
            unregister()
        except Exception as exc:
            failures.append(f"{label}: {exc}")
    assert not failures, failures


def _source_fingerprint(obj):
    mesh = obj.data
    return (
        obj.name_full,
        tuple(round(float(value), 7) for row in obj.matrix_world for value in row),
        tuple(round(float(value), 7) for value in obj.scale),
        tuple((item.name, item.type) for item in obj.modifiers),
        tuple(tuple(int(value) for value in edge.vertices) for edge in mesh.edges),
        tuple(
            tuple(int(value) for value in polygon.vertices)
            for polygon in mesh.polygons
        ),
        tuple(
            (
                layer.name,
                tuple(
                    tuple(round(float(value), 7) for value in item.uv)
                    for item in layer.data
                ),
            )
            for layer in mesh.uv_layers
        ),
        tuple(
            material.name_full if material else None
            for material in mesh.materials
        ),
    )


def _image_metrics(path: Path) -> dict[str, object]:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width = int(image.size[0])
        height = int(image.size[1])
        channels = int(image.channels)
        values = tuple(float(value) for value in image.pixels[:])
        count = width * height
        means: list[float] = []
        for channel in range(4):
            if channel >= channels:
                means.append(1.0)
                continue
            means.append(
                sum(
                    values[index * channels + channel]
                    for index in range(count)
                )
                / count
            )
        alpha_index = 3 if channels >= 4 else None
        coverage = (
            sum(
                1
                for index in range(count)
                if values[index * channels + alpha_index] > (1.0 / 255.0)
            )
            / count
            if alpha_index is not None
            else 1.0
        )
        return {
            "name": path.name,
            "width": width,
            "height": height,
            "mean_rgba": [round(value, 6) for value in means],
            "alpha_coverage": round(coverage, 6),
        }
    finally:
        bpy.data.images.remove(image)


def _configure(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
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
    obj = bpy.data.objects["Hero"]
    for candidate in bpy.context.view_layer.objects:
        candidate.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    return obj


def _collect_passed_case(output_root: Path) -> dict[str, object]:
    json_files = tuple(sorted(output_root.glob("*.json")))
    assert len(json_files) == 1, f"expected one JSON output, got: {json_files}"
    json_path = json_files[0]
    payload = json_path.read_bytes()
    parsed = json.loads(payload.decode("utf-8"))
    assert isinstance(parsed, dict), f"{json_path.name} root must be a JSON object"

    images = tuple(
        path
        for path in sorted((output_root / "images").glob("*"))
        if path.is_file()
    )
    return {
        "status": "passed",
        "json_sha256": hashlib.sha256(payload).hexdigest(),
        "json_size": len(payload),
        "images": [_image_metrics(path) for path in images],
    }


def _compare_number(
    *,
    case_id: str,
    field: str,
    expected: object,
    actual: object,
    tolerance: float,
) -> list[str]:
    expected_value = float(expected)
    actual_value = float(actual)
    if abs(expected_value - actual_value) <= tolerance:
        return []
    return [
        f"{case_id}.{field}: expected {expected_value}, got {actual_value}"
    ]


def _compare_case(
    case_id: str,
    expected: dict[str, object],
    actual: dict[str, object],
) -> list[str]:
    mismatches: list[str] = []
    expected_status = str(expected.get("status"))
    actual_status = str(actual.get("status"))
    if actual_status != expected_status:
        return [
            f"{case_id}.status: expected {expected_status!r}, got {actual_status!r}"
        ]
    if expected_status != "passed":
        return mismatches

    for field in ("json_size", "json_sha256"):
        if actual.get(field) != expected.get(field):
            mismatches.append(
                f"{case_id}.{field}: expected {expected.get(field)!r}, "
                f"got {actual.get(field)!r}"
            )

    expected_images = tuple(expected.get("images", ()))
    actual_images = tuple(actual.get("images", ()))
    if len(actual_images) != len(expected_images):
        mismatches.append(
            f"{case_id}.images: expected {len(expected_images)} outputs, "
            f"got {len(actual_images)}"
        )
        return mismatches

    for index, (expected_image, actual_image) in enumerate(
        zip(expected_images, actual_images, strict=True)
    ):
        prefix = f"images[{index}]"
        for field in ("name", "width", "height"):
            if actual_image.get(field) != expected_image.get(field):
                mismatches.append(
                    f"{case_id}.{prefix}.{field}: "
                    f"expected {expected_image.get(field)!r}, "
                    f"got {actual_image.get(field)!r}"
                )

        mismatches.extend(
            _compare_number(
                case_id=case_id,
                field=f"{prefix}.alpha_coverage",
                expected=expected_image["alpha_coverage"],
                actual=actual_image["alpha_coverage"],
                tolerance=0.01,
            )
        )

        expected_rgba = tuple(expected_image["mean_rgba"])
        actual_rgba = tuple(actual_image["mean_rgba"])
        if len(expected_rgba) != len(actual_rgba):
            mismatches.append(
                f"{case_id}.{prefix}.mean_rgba: expected {len(expected_rgba)} "
                f"channels, got {len(actual_rgba)}"
            )
            continue
        for channel, (expected_value, actual_value) in enumerate(
            zip(expected_rgba, actual_rgba, strict=True)
        ):
            mismatches.extend(
                _compare_number(
                    case_id=case_id,
                    field=f"{prefix}.mean_rgba[{channel}]",
                    expected=expected_value,
                    actual=actual_value,
                    tolerance=0.03,
                )
            )
    return mismatches


def _write_candidate(actual_cases: dict[str, dict[str, object]]) -> str:
    document = {
        "schema_version": int(GOLDEN_DOCUMENT.get("schema_version", 1)),
        "cases": actual_cases,
    }
    rendered = json.dumps(
        document,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    CANDIDATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE_PATH.write_text(rendered, encoding="utf-8")
    return rendered


def test_public_blend_fixtures_match_reviewed_goldens(
    tmp_path,
    generated_public_blend_fixtures,
):
    actual_cases: dict[str, dict[str, object]] = {}
    mismatches: list[str] = []

    for case_id in sorted(GOLDEN):
        fixture = generated_public_blend_fixtures / f"{case_id}.blend"
        assert fixture.is_file()
        assert fixture.stat().st_size > 1024
        assert "FINISHED" in bpy.ops.wm.open_mainfile(
            filepath=str(fixture),
            load_ui=False,
        )

        completed = _register_steps()
        try:
            output_root = tmp_path / case_id
            obj = _configure(output_root)
            source_before = _source_fingerprint(obj)
            expected = GOLDEN[case_id]

            if expected["status"] == "failed":
                with pytest.raises(
                    RuntimeError,
                    match="A1_PREPARE_GEOMETRY_FAILED",
                ):
                    bpy.ops.object.save_uv_as_json()
                assert not tuple(output_root.rglob("*.json"))
                assert not tuple(output_root.rglob("*.png"))
                actual = dict(expected)
            else:
                result = set(bpy.ops.object.save_uv_as_json())
                assert "FINISHED" in result
                actual = _collect_passed_case(output_root)

            assert _source_fingerprint(obj) == source_before
            assert not tuple(output_root.glob("*.spine2d.lock"))
            assert not tuple(output_root.glob(".spine2d-journal-*.json"))
        finally:
            _unregister_steps(completed)

        actual_cases[case_id] = actual
        mismatches.extend(_compare_case(case_id, expected, actual))

    candidate = _write_candidate(actual_cases)
    if mismatches:
        details = "\n".join(f"- {message}" for message in mismatches)
        pytest.fail(
            "Public blend golden catalog is stale.\n"
            f"Candidate written to: {CANDIDATE_PATH}\n"
            f"Mismatches:\n{details}\n"
            f"Candidate JSON:\n{candidate}",
            pytrace=False,
        )
