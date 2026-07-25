#!/usr/bin/env python3
"""Build, install, enable, smoke, disable, and uninstall the extension in isolation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib
from typing import Mapping, Sequence

from tools import prepare_package


ROOT = Path(__file__).resolve().parents[1]
SMOKE_WORKER = ROOT / "tools" / "blender_extension_install_smoke.py"


class ExtensionInstallGateError(RuntimeError):
    """Raised when an isolated extension lifecycle step fails."""


@dataclass(frozen=True, slots=True)
class GateStepResult:
    name: str
    command: tuple[str, ...]
    return_code: int
    log_path: Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "Blender_to_Spine2D_Mesh_Exporter",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--keep-work", action="store_true")
    return parser


def _manifest(source: Path) -> Mapping[str, object]:
    with (source / "blender_manifest.toml").open("rb") as stream:
        payload = tomllib.load(stream)
    if not isinstance(payload.get("id"), str) or not payload["id"]:
        raise ExtensionInstallGateError("manifest id is missing")
    return payload


def _build_install_archive(
    blender: str,
    source: Path,
    repository_directory: Path,
    manifest: Mapping[str, object],
) -> tuple[str, Path]:
    """Build one validated archive with the current prepare_package API."""

    executable = prepare_package._resolve_blender_executable(Path(blender))
    extension_id = str(manifest.get("id", "") or "").strip()
    version = str(manifest.get("version", "") or "").strip()
    if not extension_id or not version:
        raise ExtensionInstallGateError("manifest id/version is missing")

    archive = repository_directory / f"{extension_id}-{version}.zip"
    built = prepare_package.build_extension(
        blender=executable,
        source=source,
        output=archive,
    )
    return str(executable), built


def isolated_environment(root: Path) -> dict[str, str]:
    resolved = root.resolve(strict=False)
    environment = dict(os.environ)
    environment.update(
        {
            "BLENDER_USER_CONFIG": str(resolved / "config"),
            "BLENDER_USER_SCRIPTS": str(resolved / "scripts"),
            "BLENDER_USER_DATAFILES": str(resolved / "datafiles"),
            "BLENDER_SYSTEM_EXTENSIONS": str(resolved / "system-extensions"),
        }
    )
    for key in (
        "BLENDER_USER_CONFIG",
        "BLENDER_USER_SCRIPTS",
        "BLENDER_USER_DATAFILES",
        "BLENDER_SYSTEM_EXTENSIONS",
    ):
        Path(environment[key]).mkdir(parents=True, exist_ok=True)
    return environment


def extension_commands(
    blender: str,
    *,
    repository_id: str,
    repository_directory: Path,
    archive: Path,
    extension_id: str,
    smoke_output: Path,
    smoke_report: Path,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    module_name = f"bl_ext.{repository_id}.{extension_id}"
    return (
        (
            "repo-add",
            (
                blender,
                "--command",
                "extension",
                "repo-add",
                repository_id,
                "--name",
                "Spine2D Install Gate",
                "--directory",
                str(repository_directory),
                "--clear-all",
            ),
        ),
        (
            "install-enable",
            (
                blender,
                "--command",
                "extension",
                "install-file",
                "-r",
                repository_id,
                "-e",
                str(archive),
            ),
        ),
        (
            "smoke-export",
            (
                blender,
                "--background",
                "--python-exit-code",
                "1",
                "--python",
                str(SMOKE_WORKER),
                "--",
                "--module",
                module_name,
                "--output-root",
                str(smoke_output),
                "--report-json",
                str(smoke_report),
            ),
        ),
        (
            "remove",
            (
                blender,
                "--command",
                "extension",
                "remove",
                extension_id,
            ),
        ),
        (
            "repo-remove",
            (
                blender,
                "--command",
                "extension",
                "repo-remove",
                repository_id,
            ),
        ),
    )


def _run_step(
    name: str,
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    logs: Path,
) -> GateStepResult:
    log_path = logs / f"{name}.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as stream:
        stream.write("Command:\n" + " ".join(json.dumps(item) for item in command) + "\n\n")
        stream.flush()
        completed = subprocess.run(
            tuple(command),
            cwd=ROOT,
            env=dict(environment),
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return GateStepResult(name, tuple(command), int(completed.returncode), log_path)


def run_gate(
    blender: str,
    source: Path,
    output_root: Path,
    *,
    keep_work: bool = False,
) -> dict[str, object]:
    source = source.resolve(strict=False)
    output_root = output_root.resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(source)
    extension_id = str(manifest["id"])
    repository_id = "spine2d_install_gate"
    work = Path(tempfile.mkdtemp(prefix="spine2d-install-gate-", dir=output_root))
    logs = work / "logs"
    logs.mkdir()
    environment = isolated_environment(work / "blender-user")
    repository_directory = work / "repository"
    repository_directory.mkdir()
    executable, archive = _build_install_archive(
        blender,
        source,
        repository_directory,
        manifest,
    )
    smoke_output = work / "smoke-output"
    smoke_report = work / "smoke-report.json"
    commands = extension_commands(
        executable,
        repository_id=repository_id,
        repository_directory=repository_directory,
        archive=archive,
        extension_id=extension_id,
        smoke_output=smoke_output,
        smoke_report=smoke_report,
    )
    results: list[GateStepResult] = []
    primary_error: str | None = None
    try:
        for name, command in commands[:3]:
            result = _run_step(name, command, environment=environment, logs=logs)
            results.append(result)
            if result.return_code != 0:
                raise ExtensionInstallGateError(
                    f"step '{name}' failed with {result.return_code}; see {result.log_path}"
                )
        smoke = json.loads(smoke_report.read_text(encoding="utf-8"))
        if smoke.get("status") != "passed":
            raise ExtensionInstallGateError(f"smoke report failed: {smoke}")
    except Exception as exc:
        primary_error = str(exc)
    finally:
        for name, command in commands[3:]:
            result = _run_step(name, command, environment=environment, logs=logs)
            results.append(result)
            if result.return_code != 0 and primary_error is None:
                primary_error = (
                    f"cleanup step '{name}' failed with {result.return_code}; "
                    f"see {result.log_path}"
                )

    payload: dict[str, object] = {
        "status": "passed" if primary_error is None else "failed",
        "extension_id": extension_id,
        "archive": str(archive),
        "work_directory": str(work),
        "steps": [
            {
                "name": item.name,
                "return_code": item.return_code,
                "log_path": str(item.log_path),
                "command": list(item.command),
            }
            for item in results
        ],
    }
    if primary_error is not None:
        payload["error"] = primary_error
    report_path = output_root / "extension-install-gate.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not keep_work and primary_error is None:
        shutil.rmtree(work)
        payload["work_directory_removed"] = True
    if primary_error is not None:
        raise ExtensionInstallGateError(primary_error)
    return payload


def main() -> None:
    namespace = _parser().parse_args()
    run_gate(
        namespace.blender,
        namespace.source,
        namespace.output_root,
        keep_work=namespace.keep_work,
    )


if __name__ == "__main__":
    main()
