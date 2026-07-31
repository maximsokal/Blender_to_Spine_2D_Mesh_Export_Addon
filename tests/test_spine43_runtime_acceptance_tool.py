"""Pure-Python contracts for official Spine 4.3 runtime acceptance."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from tools import run_spine43_runtime_acceptance as acceptance


def _runtime_root(tmp_path: Path, *, version: str = "4.3.23") -> Path:
    root = tmp_path / "vendor" / "spine-runtimes-4.3"
    core = root / "spine-ts" / "spine-core"
    core.mkdir(parents=True)
    (core / "package.json").write_text(
        json.dumps({"name": "@esotericsoftware/spine-core", "version": version}),
        encoding="utf-8",
    )
    return root


def _runtime_report(
    *,
    expected_ik: int = 3,
    expected_transform: int = 12,
) -> dict[str, object]:
    expected_constraints = expected_ik + expected_transform
    return {
        "ok": True,
        "version": acceptance.EXPECTED_VERSION,
        "counts": {
            "bones": 52,
            "slots": 3,
            "skins": 1,
            "constraints": expected_constraints,
            "ik": expected_ik,
            "transform": expected_transform,
            "setupRenderableAttachments": 3,
        },
        "updateCache": {
            "expectedConstraints": expected_constraints,
            "scheduledConstraints": expected_constraints,
            "everyConstraintScheduledExactlyOnce": True,
        },
        "matrices": {"finiteBones": 52, "allFinite": True},
        "bounds": {"x": -10.0, "y": -20.0, "width": 30.0, "height": 40.0},
    }


def test_runtime_discovery_prefers_existing_built_esm_over_source(
    tmp_path: Path,
) -> None:
    root = _runtime_root(tmp_path)
    core = root / "spine-ts" / "spine-core"
    built = core / "dist" / "esm" / "spine-core.mjs"
    source = core / "src" / "index.ts"
    built.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    built.write_text("export const built = true;", encoding="utf-8")
    source.write_text("export const source = true;", encoding="utf-8")

    runtime = acceptance.resolve_runtime_entry(root)

    assert runtime.mode == "BUILT_ESM"
    assert runtime.entry_path == built.resolve()
    assert runtime.source_root is None
    assert runtime.package_version == "4.3.23"


def test_runtime_discovery_uses_clean_typescript_source_without_building(
    tmp_path: Path,
) -> None:
    root = _runtime_root(tmp_path)
    source = root / "spine-ts" / "spine-core" / "src" / "index.ts"
    source.parent.mkdir(parents=True)
    source.write_text("export const source = true;", encoding="utf-8")

    runtime = acceptance.resolve_runtime_entry(root)

    assert runtime.mode == "SOURCE_TYPESCRIPT"
    assert runtime.entry_path == source.resolve()
    assert runtime.source_root == source.parent.resolve()
    assert runtime.package_version == "4.3.23"


def test_runtime_discovery_rejects_wrong_package_family(tmp_path: Path) -> None:
    root = _runtime_root(tmp_path, version="4.2.43")
    source = root / "spine-ts" / "spine-core" / "src" / "index.ts"
    source.parent.mkdir(parents=True)
    source.write_text("export {};", encoding="utf-8")

    with pytest.raises(
        acceptance.Spine43RuntimeAcceptanceError,
        match="not 4.3.x",
    ):
        acceptance.resolve_runtime_entry(root)


def test_runtime_discovery_fails_closed_without_built_or_source_entry(
    tmp_path: Path,
) -> None:
    root = _runtime_root(tmp_path)

    with pytest.raises(
        acceptance.Spine43RuntimeAcceptanceError,
        match="No usable spine-core 4.3 entry",
    ):
        acceptance.resolve_runtime_entry(root)


def test_built_runtime_command_imports_only_existing_entry(tmp_path: Path) -> None:
    root = _runtime_root(tmp_path)
    built = root / "spine-ts" / "spine-core" / "dist" / "esm" / "spine-core.mjs"
    built.parent.mkdir(parents=True)
    built.write_text("export {};", encoding="utf-8")
    runtime = acceptance.resolve_runtime_entry(root)
    json_path = tmp_path / "project.json"

    command = acceptance.build_runtime_command("node", json_path, runtime)

    assert command == (
        "node",
        str(acceptance.RUNTIME_ORACLE),
        str(json_path),
        str(built.resolve()),
    )
    assert "--loader" not in command
    assert "--experimental-transform-types" not in command


def test_source_runtime_command_uses_repo_owned_read_only_loader(
    tmp_path: Path,
) -> None:
    root = _runtime_root(tmp_path)
    source = root / "spine-ts" / "spine-core" / "src" / "index.ts"
    source.parent.mkdir(parents=True)
    source.write_text("export {};", encoding="utf-8")
    runtime = acceptance.resolve_runtime_entry(root)
    json_path = tmp_path / "project.json"

    command = acceptance.build_runtime_command("node", json_path, runtime)

    assert command == (
        "node",
        "--no-warnings",
        "--experimental-transform-types",
        "--loader",
        str(acceptance.SOURCE_LOADER),
        str(acceptance.RUNTIME_ORACLE),
        str(json_path),
        str(source.resolve()),
    )
    assert "npm" not in command
    assert "install" not in command
    assert "build" not in command


def test_source_environment_exposes_only_the_allowed_source_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _runtime_root(tmp_path)
    source = root / "spine-ts" / "spine-core" / "src" / "index.ts"
    source.parent.mkdir(parents=True)
    source.write_text("export {};", encoding="utf-8")
    runtime = acceptance.resolve_runtime_entry(root)
    monkeypatch.setenv("SPINE43_RUNTIME_SOURCE_ROOT", "stale")

    environment = acceptance.build_runtime_environment(runtime)

    assert environment["SPINE43_RUNTIME_SOURCE_ROOT"] == str(source.parent.resolve())
    assert os.environ["SPINE43_RUNTIME_SOURCE_ROOT"] == "stale"


def test_built_environment_removes_stale_source_loader_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _runtime_root(tmp_path)
    built = root / "spine-ts" / "spine-core" / "dist" / "esm" / "spine-core.mjs"
    built.parent.mkdir(parents=True)
    built.write_text("export {};", encoding="utf-8")
    runtime = acceptance.resolve_runtime_entry(root)
    monkeypatch.setenv("SPINE43_RUNTIME_SOURCE_ROOT", "stale")

    environment = acceptance.build_runtime_environment(runtime)

    assert "SPINE43_RUNTIME_SOURCE_ROOT" not in environment
    assert os.environ["SPINE43_RUNTIME_SOURCE_ROOT"] == "stale"


def test_parse_runtime_report_accepts_complete_two_axis_evidence() -> None:
    report = _runtime_report(expected_ik=3, expected_transform=12)

    actual = acceptance.parse_runtime_report(
        json.dumps(report),
        expected_ik=3,
        expected_transform=12,
    )

    assert actual == report


def test_parse_runtime_report_accepts_complete_three_axis_evidence() -> None:
    report = _runtime_report(expected_ik=3, expected_transform=15)

    actual = acceptance.parse_runtime_report(
        json.dumps(report),
        expected_ik=3,
        expected_transform=15,
    )

    assert actual == report


@pytest.mark.parametrize(
    "mutation, message",
    (
        ({"ok": False}, "reported failure"),
        ({"version": "4.2.43"}, "version mismatch"),
        ({"counts": {"constraints": 14, "ik": 3, "transform": 12, "setupRenderableAttachments": 3}}, "constraints mismatch"),
        ({"counts": {"constraints": 15, "ik": 2, "transform": 12, "setupRenderableAttachments": 3}}, "ik mismatch"),
        ({"counts": {"constraints": 15, "ik": 3, "transform": 11, "setupRenderableAttachments": 3}}, "transform mismatch"),
        ({"counts": {"constraints": 15, "ik": 3, "transform": 12, "setupRenderableAttachments": 0}}, "no setup-renderable"),
        ({"updateCache": {"expectedConstraints": 15, "scheduledConstraints": 14, "everyConstraintScheduledExactlyOnce": False}}, "update-cache evidence"),
        ({"matrices": {"allFinite": False}}, "non-finite"),
        ({"bounds": None}, "bounds are missing"),
        ({"bounds": {"x": 0.0, "y": 0.0, "width": 0.0, "height": 1.0}}, "not positive"),
        ({"bounds": {"x": 0.0, "y": 0.0, "width": math.nan, "height": 1.0}}, "not finite"),
    ),
)
def test_parse_runtime_report_rejects_incomplete_evidence(
    mutation: dict[str, object],
    message: str,
) -> None:
    report = _runtime_report()
    report.update(mutation)

    with pytest.raises(
        acceptance.Spine43RuntimeAcceptanceError,
        match=message,
    ):
        acceptance.parse_runtime_report(
            json.dumps(report),
            expected_ik=3,
            expected_transform=12,
        )


def test_runner_never_invokes_package_install_or_build_commands() -> None:
    source = (
        Path(acceptance.__file__).resolve().read_text(encoding="utf-8")
    )

    for forbidden in (
        "npm install",
        "npm run",
        "pnpm",
        "yarn",
        "npx",
        "tsc",
        "shutil.rmtree(runtime",
    ):
        assert forbidden not in source
    assert '"externalRuntimeReadOnly": True' in source
