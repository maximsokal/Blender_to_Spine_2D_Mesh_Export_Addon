"""Pure-Python contracts for the Spine 3.8 Blender/runtime acceptance tool."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from tools import run_spine38_standalone_acceptance as acceptance


def _oracle_report(expected: dict[str, object]) -> dict[str, object]:
    constraints = int(expected["ik"]) + int(expected["transform"])
    return {
        "ok": True,
        "version": acceptance.EXPECTED_VERSION,
        "counts": {
            "bones": expected["bones"],
            "slots": 3,
            "skins": 1,
            "ik": expected["ik"],
            "transform": expected["transform"],
            "path": 0,
            "atlasPages": 1,
            "atlasRegions": 3,
            "setupRenderableAttachments": 3,
        },
        "updateCache": {
            "expectedConstraints": constraints,
            "scheduledConstraints": constraints,
            "everyConstraintScheduledExactlyOnce": True,
        },
        "matrices": {
            "finiteBones": expected["bones"],
            "allFinite": True,
        },
        "bounds": {"x": -10.0, "y": -20.0, "width": 30.0, "height": 40.0},
    }


def _blender_report(output_root: Path) -> dict[str, object]:
    profiles = []
    for profile_name, expected_value in acceptance.EXPECTED_CASES.items():
        expected = dict(expected_value)
        case_directory = output_root / str(expected["directory"])
        case_directory.mkdir(parents=True, exist_ok=True)
        json_path = case_directory / f"{expected['stem']}.json"
        json_path.write_text("{}", encoding="utf-8")
        profiles.append(
            {
                "status": "passed",
                "profile": profile_name,
                "version": acceptance.EXPECTED_VERSION,
                "mode": "STANDALONE",
                "bones": expected["bones"],
                "slots": 3,
                "skins": 1,
                "ik": expected["ik"],
                "transform": expected["transform"],
                "constraints": int(expected["ik"]) + int(expected["transform"]),
                "legacyMixFieldsPresent": True,
                "newMixFieldsPresent": False,
                "connectedWrapperPresent": False,
                "crossObjectReferencesPresent": False,
                "sequencePresent": False,
                "jsonPath": str(json_path.resolve()),
                "texturePaths": [],
                "outputFiles": [str(json_path.resolve())],
            }
        )
    return {
        "status": "passed",
        "version": acceptance.EXPECTED_VERSION,
        "mode": "STANDALONE",
        "profiles": profiles,
    }


def test_blender_command_is_factory_isolated_and_fail_closed(tmp_path: Path) -> None:
    blender = tmp_path / "blender.exe"
    output = tmp_path / "output"

    assert acceptance.build_blender_command(blender, output) == (
        str(blender),
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        str(acceptance.BLENDER_WORKER),
        "--",
        "--output",
        str(output),
    )


def test_oracle_command_passes_runtime_only_as_input(tmp_path: Path) -> None:
    json_path = tmp_path / "project.json"
    runtime_entry = tmp_path / "runtime" / "index.js"

    assert acceptance.build_oracle_command("node", json_path, runtime_entry) == (
        "node",
        str(acceptance.RUNTIME_ORACLE),
        str(json_path),
        str(runtime_entry),
    )


def test_prepare_output_root_rejects_non_empty_without_replace(tmp_path: Path) -> None:
    output = tmp_path / "acceptance" / "result"
    output.mkdir(parents=True)
    (output / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(
        acceptance.Spine38StandaloneAcceptanceError,
        match="not empty",
    ):
        acceptance.prepare_output_root(output, replace=False)

    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_prepare_output_root_replaces_only_explicit_deep_directory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "acceptance" / "result"
    output.mkdir(parents=True)
    (output / "old.txt").write_text("old", encoding="utf-8")

    resolved = acceptance.prepare_output_root(output, replace=True)

    assert resolved == output.resolve()
    assert resolved.is_dir()
    assert not tuple(resolved.iterdir())


def test_validate_blender_report_accepts_both_profiles(tmp_path: Path) -> None:
    report = _blender_report(tmp_path)

    assert acceptance.validate_blender_report(report, output_root=tmp_path) is report


@pytest.mark.parametrize(
    "mutation, message",
    (
        ({"status": "failed"}, "did not pass"),
        ({"version": "4.0.64"}, "version mismatch"),
        ({"mode": "CONNECTED"}, "mode mismatch"),
        ({"profiles": []}, "Profile inventory differs"),
    ),
)
def test_validate_blender_report_rejects_invalid_top_level_evidence(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    report = _blender_report(tmp_path)
    report.update(mutation)

    with pytest.raises(acceptance.Spine38StandaloneAcceptanceError, match=message):
        acceptance.validate_blender_report(report, output_root=tmp_path)


def test_validate_blender_report_rejects_new_mix_claim(tmp_path: Path) -> None:
    report = deepcopy(_blender_report(tmp_path))
    profiles = report["profiles"]
    assert isinstance(profiles, list) and isinstance(profiles[0], dict)
    profiles[0]["newMixFieldsPresent"] = True

    with pytest.raises(
        acceptance.Spine38StandaloneAcceptanceError,
        match="newMixFieldsPresent",
    ):
        acceptance.validate_blender_report(report, output_root=tmp_path)


@pytest.mark.parametrize("profile_name", tuple(acceptance.EXPECTED_CASES))
def test_parse_oracle_report_accepts_complete_evidence(profile_name: str) -> None:
    expected = dict(acceptance.EXPECTED_CASES[profile_name])
    report = _oracle_report(expected)

    assert acceptance.parse_oracle_report(
        json.dumps(report),
        expected=expected,
    ) == report


@pytest.mark.parametrize(
    "mutation, message",
    (
        ({"ok": False}, "reported failure"),
        ({"version": "4.0.64"}, "version mismatch"),
        ({"counts": {}}, "counts.bones mismatch"),
        ({"updateCache": {}}, "update cache evidence"),
        ({"matrices": {"allFinite": False}}, "matrix evidence"),
        ({"bounds": None}, "bounds are missing"),
        ({"bounds": {"x": 0.0, "y": 0.0, "width": 0.0, "height": 1.0}}, "not positive"),
        ({"bounds": {"x": 0.0, "y": 0.0, "width": math.nan, "height": 1.0}}, "not finite"),
    ),
)
def test_parse_oracle_report_rejects_incomplete_evidence(
    mutation: dict[str, object],
    message: str,
) -> None:
    expected = dict(acceptance.EXPECTED_CASES["TWO_AXIS_ROTATION_SCALE"])
    report = _oracle_report(expected)
    report.update(mutation)

    with pytest.raises(acceptance.Spine38StandaloneAcceptanceError, match=message):
        acceptance.parse_oracle_report(json.dumps(report), expected=expected)


def test_runner_never_installs_or_modifies_external_runtime() -> None:
    source = Path(acceptance.__file__).read_text(encoding="utf-8")

    for forbidden in (
        "npm install",
        "npm run",
        "npx",
        "pnpm",
        "yarn",
        "write_text(runtime_entry",
        "shutil.rmtree(runtime_entry",
    ):
        assert forbidden not in source
    assert '"externalRuntimeReadOnly": True' in source
