"""Pure-Python contracts for the Blender Spine 4.3 structural acceptance gate."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools import run_spine43_standalone_acceptance as acceptance


def _write_case_artifacts(
    output_root: Path,
    profile_name: str,
    expected: dict[str, object],
) -> dict[str, object]:
    case_directory = output_root / str(expected["directory"])
    case_directory.mkdir(parents=True, exist_ok=True)
    stem = str(expected["stem"])
    json_path = case_directory / f"{stem}.json"
    constraint_count = int(expected["ik"]) + int(expected["transform"])
    constraints = [
        {
            "name": f"constraint_{index}",
            "type": "ik" if index < int(expected["ik"]) else "transform",
            "bones": ["root"],
            **(
                {"target": "root"}
                if index < int(expected["ik"])
                else {"source": "root", "properties": {}}
            ),
        }
        for index in range(constraint_count)
    ]
    json_path.write_text(
        json.dumps(
            {
                "skeleton": {"spine": acceptance.EXPECTED_VERSION},
                "bones": [{"name": "root"}],
                "slots": [],
                "skins": [],
                "constraints": constraints,
            }
        ),
        encoding="utf-8",
    )
    texture_paths = []
    for index in range(3):
        texture = case_directory / "images" / f"texture_{index}.png"
        texture.parent.mkdir(parents=True, exist_ok=True)
        texture.write_bytes(b"png")
        texture_paths.append(str(texture.resolve()))

    return {
        "status": "passed",
        "profile": profile_name,
        "version": acceptance.EXPECTED_VERSION,
        "mode": "STANDALONE",
        "bones": 1,
        "slots": 3,
        "skins": 1,
        "constraints": constraint_count,
        "ik": expected["ik"],
        "transform": expected["transform"],
        "profileBoneInventoryExact": True,
        "connectedWrapperPresent": False,
        "crossObjectReferencesPresent": False,
        "legacyRootConstraintCollectionsPresent": False,
        "legacyConstraintOrderPresent": False,
        "jsonPath": str(json_path.resolve()),
        "texturePaths": texture_paths,
    }


def _report(output_root: Path) -> dict[str, object]:
    return {
        "status": "passed",
        "version": acceptance.EXPECTED_VERSION,
        "mode": "STANDALONE",
        "profiles": [
            _write_case_artifacts(output_root, profile_name, dict(expected))
            for profile_name, expected in acceptance.EXPECTED_CASES.items()
        ],
        "runtimeValidated": False,
        "manualEditorImportRequired": True,
    }


def test_blender_command_uses_factory_startup_and_fails_closed(tmp_path: Path) -> None:
    blender = tmp_path / "blender.exe"
    output = tmp_path / "acceptance-output"

    command = acceptance.build_blender_command(blender, output)

    assert command == (
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


def test_prepare_output_root_rejects_non_empty_directory_without_replace(
    tmp_path: Path,
) -> None:
    output = tmp_path / "acceptance" / "result"
    output.mkdir(parents=True)
    (output / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(
        acceptance.Spine43StandaloneAcceptanceError,
        match="not empty",
    ):
        acceptance.prepare_output_root(output, replace=False)

    assert (output / "existing.txt").read_text(encoding="utf-8") == "keep"


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


def test_prepare_output_root_refuses_filesystem_root_replacement() -> None:
    root = Path(Path.cwd().anchor or "/").resolve()

    assert acceptance._dangerous_replace_target(root) is True


def test_validate_blender_report_accepts_both_profile_exports(tmp_path: Path) -> None:
    report = _report(tmp_path)

    assert acceptance.validate_blender_report(report, output_root=tmp_path) is report


@pytest.mark.parametrize(
    "mutation, message",
    (
        ({"status": "failed"}, "did not pass"),
        ({"version": "4.2.43"}, "version mismatch"),
        ({"mode": "CONNECTED"}, "mode mismatch"),
        ({"runtimeValidated": True}, "must not claim"),
        ({"manualEditorImportRequired": False}, "must require manual"),
        ({"profiles": []}, "inventory differs"),
    ),
)
def test_validate_blender_report_rejects_invalid_top_level_evidence(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    report = _report(tmp_path)
    report.update(mutation)

    with pytest.raises(
        acceptance.Spine43StandaloneAcceptanceError,
        match=message,
    ):
        acceptance.validate_blender_report(report, output_root=tmp_path)


def test_validate_blender_report_rejects_unverified_profile_bone_inventory(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path)
    broken = deepcopy(report)
    broken_profiles = broken["profiles"]
    assert isinstance(broken_profiles, list)
    assert isinstance(broken_profiles[0], dict)
    broken_profiles[0]["profileBoneInventoryExact"] = False

    with pytest.raises(
        acceptance.Spine43StandaloneAcceptanceError,
        match="exact profile bone inventory",
    ):
        acceptance.validate_blender_report(broken, output_root=tmp_path)


def test_validate_blender_report_rejects_report_json_bone_count_drift(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path)
    broken = deepcopy(report)
    broken_profiles = broken["profiles"]
    assert isinstance(broken_profiles, list)
    assert isinstance(broken_profiles[0], dict)
    broken_profiles[0]["bones"] = 2

    with pytest.raises(
        acceptance.Spine43StandaloneAcceptanceError,
        match="differs between worker report and JSON",
    ):
        acceptance.validate_blender_report(broken, output_root=tmp_path)


def test_validate_blender_report_rejects_connected_wrapper_claim(tmp_path: Path) -> None:
    report = _report(tmp_path)
    broken = deepcopy(report)
    broken_profiles = broken["profiles"]
    assert isinstance(broken_profiles, list)
    assert isinstance(broken_profiles[0], dict)
    broken_profiles[0]["connectedWrapperPresent"] = True

    with pytest.raises(
        acceptance.Spine43StandaloneAcceptanceError,
        match="connected wrapper",
    ):
        acceptance.validate_blender_report(broken, output_root=tmp_path)


def test_validate_blender_report_rejects_persisted_legacy_root_collection(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path)
    profiles = report["profiles"]
    assert isinstance(profiles, list) and isinstance(profiles[0], dict)
    json_path = Path(str(profiles[0]["jsonPath"]))
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["transform"] = []
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        acceptance.Spine43StandaloneAcceptanceError,
        match="legacy root field",
    ):
        acceptance.validate_blender_report(report, output_root=tmp_path)


def test_validate_blender_report_rejects_constraint_order_in_persisted_json(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path)
    profiles = report["profiles"]
    assert isinstance(profiles, list) and isinstance(profiles[0], dict)
    json_path = Path(str(profiles[0]["jsonPath"]))
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["constraints"][0]["order"] = 0
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        acceptance.Spine43StandaloneAcceptanceError,
        match="retained legacy order",
    ):
        acceptance.validate_blender_report(report, output_root=tmp_path)
