import json
from pathlib import Path

import pytest

from tools.a1_fixture_manifest import (
    A1FixtureManifest,
    FixtureManifestError,
    FixtureMode,
    case_to_worker_payload,
    load_fixture_manifest,
)


def _blend(tmp_path: Path, name: str = "source.blend") -> Path:
    path = tmp_path / name
    path.write_bytes(b"BLENDER-fixture-placeholder")
    return path


def _write_manifest(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _valid_single(tmp_path: Path) -> dict:
    _blend(tmp_path)
    return {
        "schema_version": 1,
        "cases": [
            {
                "case_id": "hero-single",
                "blend_file": "source.blend",
                "mode": "single",
                "active_object": "Hero",
                "selected_objects": ["Hero"],
                "settings": {
                    "texture_size": 512,
                    "images_path": "textures/spine",
                    "seam_mode": "CUSTOM",
                    "angle_limit": 45,
                    "sequence": {"start_frame": 7, "frame_count": 3},
                    "control_icons": False,
                    "preview_animation": True,
                },
                "parity": {
                    "compare_animations": True,
                    "strict_edges": True,
                    "ignore_paths": ["skeleton.hash"],
                    "image_absolute_tolerance": 0.001,
                    "image_max_differing_pixel_ratio": 0.02,
                    "image_max_mean_absolute_delta": 0.0001,
                },
            }
        ],
    }


def test_loads_relative_single_fixture_and_normalizes_payload(tmp_path):
    manifest = load_fixture_manifest(
        _write_manifest(tmp_path, _valid_single(tmp_path))
    )

    assert isinstance(manifest, A1FixtureManifest)
    assert manifest.schema_version == 1
    case = manifest.cases[0]
    assert case.mode is FixtureMode.SINGLE
    assert case.blend_file == (tmp_path / "source.blend").resolve()
    assert case.settings.sequence.start_frame == 7
    assert case.settings.sequence.frame_count == 3
    assert case.parity.ignore_paths == ("skeleton.hash",)

    payload = case_to_worker_payload(case, tmp_path / "output")
    assert payload["mode"] == "single"
    assert payload["settings"]["images_path"] == "textures/spine"
    assert payload["settings"]["sequence"] == {
        "start_frame": 7,
        "frame_count": 3,
    }
    assert payload["settings"]["control_icons"] is False


def test_loads_multi_fixture_with_connected_subset_and_per_object_frames(tmp_path):
    _blend(tmp_path, "multi.blend")
    manifest = load_fixture_manifest(
        _write_manifest(
            tmp_path,
            {
                "schema_version": 1,
                "blender_executable": "blender-custom",
                "cases": [
                    {
                        "case_id": "mixed.case",
                        "blend_file": "multi.blend",
                        "mode": "multi",
                        "active_object": "Body",
                        "selected_objects": ["Body", "Arm", "Prop"],
                        "connected_objects": ["Body", "Arm"],
                        "expected_json_name": "custom.json",
                        "settings": {
                            "per_object_sequence": {
                                "Body": {"start_frame": 1, "frame_count": 2},
                                "Arm": {"start_frame": 10, "frame_count": 4},
                            }
                        },
                    }
                ],
            },
        )
    )

    case = manifest.cases[0]
    assert manifest.blender_executable == "blender-custom"
    assert case.connected_objects == ("Body", "Arm")
    assert case.expected_json_name == "custom.json"
    assert case.settings.per_object_sequence["Arm"].frame_count == 4


@pytest.mark.parametrize(
    "mutator, message",
    (
        (lambda data: data.update(schema_version=2), "schema_version"),
        (
            lambda data: data["cases"][0].update(case_id="../escape"),
            "case_id",
        ),
        (
            lambda data: data["cases"][0]["settings"].update(images_path="../images"),
            "images_path",
        ),
        (
            lambda data: data["cases"][0].update(selected_objects=["Other"]),
            "active_object",
        ),
        (
            lambda data: data["cases"][0].update(
                connected_objects=["Hero"]
            ),
            "single fixture",
        ),
        (
            lambda data: data["cases"][0].update(expected_json_name="dir/file.json"),
            "expected_json_name",
        ),
        (
            lambda data: data["cases"][0].update(unknown=True),
            "unknown fields",
        ),
    ),
)
def test_rejects_unsafe_or_inconsistent_manifests(tmp_path, mutator, message):
    data = _valid_single(tmp_path)
    mutator(data)
    with pytest.raises(FixtureManifestError, match=message):
        load_fixture_manifest(_write_manifest(tmp_path, data))


def test_rejects_missing_blend_file(tmp_path):
    data = _valid_single(tmp_path)
    data["cases"][0]["blend_file"] = "missing.blend"
    with pytest.raises(FixtureManifestError, match="does not exist"):
        load_fixture_manifest(_write_manifest(tmp_path, data))


def test_rejects_unselected_per_object_sequence(tmp_path):
    data = _valid_single(tmp_path)
    data["cases"][0].update(
        mode="multi",
        selected_objects=["Hero", "Other"],
    )
    data["cases"][0]["settings"].pop("sequence")
    data["cases"][0]["settings"]["per_object_sequence"] = {
        "Missing": {"start_frame": 0, "frame_count": 1}
    }
    with pytest.raises(FixtureManifestError, match="unselected objects"):
        load_fixture_manifest(_write_manifest(tmp_path, data))
