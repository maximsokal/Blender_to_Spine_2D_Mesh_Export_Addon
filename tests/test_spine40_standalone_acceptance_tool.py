"""Pure-Python contracts for the Blender/runtime Spine 4.0 acceptance gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import run_spine40_standalone_acceptance as acceptance


def _oracle_report() -> dict[str, object]:
    return {
        "ok": True,
        "version": "4.0.64",
        "counts": {"setupRenderableAttachments": 3},
        "updateCache": {"everyConstraintScheduledExactlyOnce": True},
        "matrices": {"allFinite": True},
        "bounds": {"x": -1.0, "y": -2.0, "width": 3.0, "height": 4.0},
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


def test_oracle_command_passes_external_runtime_only_as_input(tmp_path: Path) -> None:
    json_path = tmp_path / "export.json"
    runtime = tmp_path / "vendor" / "spine-webgl-40" / "index.js"

    command = acceptance.build_oracle_command("node", json_path, runtime)

    assert command == (
        "node",
        str(acceptance.RUNTIME_ORACLE),
        str(json_path),
        str(runtime),
    )
    assert "--output" not in command
    assert "--write" not in command


def test_prepare_output_root_rejects_non_empty_directory_without_replace(
    tmp_path: Path,
) -> None:
    output = tmp_path / "acceptance" / "result"
    output.mkdir(parents=True)
    (output / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(
        acceptance.Spine40StandaloneAcceptanceError,
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


def test_parse_oracle_report_accepts_complete_runtime_evidence() -> None:
    report = _oracle_report()

    assert acceptance.parse_oracle_report(json.dumps(report)) == report


@pytest.mark.parametrize(
    "mutation, message",
    (
        ({"ok": False}, "reported failure"),
        ({"version": "4.1.24"}, "version mismatch"),
        ({"counts": {"setupRenderableAttachments": 0}}, "no setup-renderable"),
        (
            {"updateCache": {"everyConstraintScheduledExactlyOnce": False}},
            "did not schedule",
        ),
        ({"matrices": {"allFinite": False}}, "non-finite matrices"),
        ({"bounds": None}, "bounds are missing"),
        (
            {"bounds": {"x": 0.0, "y": 0.0, "width": 0.0, "height": 1.0}},
            "not positive",
        ),
        (
            {
                "bounds": {
                    "x": 0.0,
                    "y": 0.0,
                    "width": float("nan"),
                    "height": 1.0,
                }
            },
            "not finite",
        ),
    ),
)
def test_parse_oracle_report_rejects_incomplete_evidence(
    mutation: dict[str, object],
    message: str,
) -> None:
    report = _oracle_report()
    report.update(mutation)

    with pytest.raises(
        acceptance.Spine40StandaloneAcceptanceError,
        match=message,
    ):
        acceptance.parse_oracle_report(json.dumps(report))
