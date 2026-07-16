from pathlib import Path

import pytest

from tools.a1_fixture_manifest import (
    A1FixtureCase,
    A1FixtureManifest,
    FixtureMode,
    FixtureParitySettings,
)
from tools.run_a1_fixture_parity import (
    FixtureRunnerError,
    _copy_source_sibling,
    _selected_cases,
    build_export_command,
    build_image_compare_command,
    resolve_blender_executable,
)


def _case(tmp_path: Path, case_id: str = "case-a") -> A1FixtureCase:
    source = tmp_path / f"{case_id}.blend"
    source.write_bytes(b"blend-data")
    return A1FixtureCase(
        case_id=case_id,
        blend_file=source,
        mode=FixtureMode.SINGLE,
        active_object="Hero",
        selected_objects=("Hero",),
        parity=FixtureParitySettings(
            image_absolute_tolerance=0.01,
            image_max_differing_pixel_ratio=0.25,
            image_max_mean_absolute_delta=0.001,
        ),
    )


def test_export_commands_use_distinct_source_backend_and_report_paths(tmp_path):
    payload = tmp_path / "payload.json"
    legacy_source = tmp_path / "legacy.blend"
    rewrite_source = tmp_path / "rewrite.blend"
    legacy_report = tmp_path / "legacy-report.json"
    rewrite_report = tmp_path / "rewrite-report.json"

    legacy = build_export_command(
        "/opt/blender/blender",
        legacy_source,
        payload,
        "LEGACY",
        legacy_report,
    )
    rewrite = build_export_command(
        "/opt/blender/blender",
        rewrite_source,
        payload,
        "REWRITE",
        rewrite_report,
    )

    assert legacy_source.as_posix() in legacy
    assert rewrite_source.as_posix() in rewrite
    assert "LEGACY" in legacy
    assert "REWRITE" in rewrite
    assert str(legacy_report) in legacy
    assert str(rewrite_report) in rewrite
    assert "--python-exit-code" in legacy
    assert legacy != rewrite


def test_image_command_contains_all_manifest_thresholds(tmp_path):
    case = _case(tmp_path)
    command = build_image_compare_command(
        "blender",
        tmp_path / "legacy-images",
        tmp_path / "rewrite-images",
        tmp_path / "report.json",
        case,
    )

    assert command[0] == "blender"
    assert command[command.index("--absolute-tolerance") + 1] == "0.01"
    assert command[command.index("--max-differing-pixel-ratio") + 1] == "0.25"
    assert command[command.index("--max-mean-absolute-delta") + 1] == "0.001"


def test_sibling_source_copies_preserve_data_and_directory(tmp_path):
    case = _case(tmp_path)
    copied = _copy_source_sibling(case.blend_file, "LEGACY")
    try:
        assert copied.parent == case.blend_file.parent
        assert copied != case.blend_file
        assert copied.name.startswith(f".{case.blend_file.stem}.a1-parity-legacy-")
        assert copied.read_bytes() == case.blend_file.read_bytes()
    finally:
        copied.unlink(missing_ok=True)


def test_selecting_cases_preserves_manifest_order(tmp_path):
    first = _case(tmp_path, "first")
    second = _case(tmp_path, "second")
    manifest = A1FixtureManifest(schema_version=1, cases=(first, second))

    assert _selected_cases(manifest, ()) == (first, second)
    assert _selected_cases(manifest, ("second",)) == (second,)
    with pytest.raises(FixtureRunnerError, match="Unknown case IDs"):
        _selected_cases(manifest, ("missing",))


def test_resolve_blender_prefers_cli_and_accepts_executable_path(tmp_path):
    executable = tmp_path / "blender"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    manifest = A1FixtureManifest(
        schema_version=1,
        cases=(_case(tmp_path),),
        blender_executable="manifest-blender",
    )

    assert resolve_blender_executable(str(executable), manifest) == str(executable)


def test_resolve_blender_rejects_missing_command(tmp_path, monkeypatch):
    manifest = A1FixtureManifest(schema_version=1, cases=(_case(tmp_path),))
    monkeypatch.setattr("tools.run_a1_fixture_parity.shutil.which", lambda value: None)
    with pytest.raises(FixtureRunnerError, match="was not found"):
        resolve_blender_executable("definitely-missing-blender", manifest)


def test_build_export_command_rejects_unknown_backend(tmp_path):
    with pytest.raises(ValueError, match="backend"):
        build_export_command(
            "blender",
            tmp_path / "source.blend",
            tmp_path / "payload.json",
            "BROKEN",
            tmp_path / "report.json",
        )
