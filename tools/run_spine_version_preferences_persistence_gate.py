#!/usr/bin/env python3
"""Install the extension and prove exact-version preferences survive Blender restart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

if __package__:
    from tools.run_extension_install_gate import (
        ExtensionInstallGateError,
        _build_install_archive,
        _manifest,
        _run_step,
        extension_commands,
        isolated_environment,
    )
else:
    from run_extension_install_gate import (
        ExtensionInstallGateError,
        _build_install_archive,
        _manifest,
        _run_step,
        extension_commands,
        isolated_environment,
    )


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tools" / "blender_spine_version_preferences_persistence.py"


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
    repository_id = "spine2d_version_preferences_gate"
    work = Path(
        tempfile.mkdtemp(prefix="spine2d-version-prefs-", dir=output_root)
    )
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
    module_name = f"bl_ext.{repository_id}.{extension_id}"

    placeholder_output = work / "unused-smoke-output"
    placeholder_report = work / "unused-smoke-report.json"
    lifecycle_commands = extension_commands(
        executable,
        repository_id=repository_id,
        repository_directory=repository_directory,
        archive=archive,
        extension_id=extension_id,
        smoke_output=placeholder_output,
        smoke_report=placeholder_report,
    )
    save_report = work / "save-report.json"
    verify_report = work / "verify-report.json"
    save_output = work / "save-output"
    verify_output = work / "verify-exports"
    save_command = (
        executable,
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(WORKER),
        "--",
        "--module",
        module_name,
        "--mode",
        "save",
        "--report-json",
        str(save_report),
        "--output-root",
        str(save_output),
    )
    verify_command = (
        executable,
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(WORKER),
        "--",
        "--module",
        module_name,
        "--mode",
        "verify",
        "--report-json",
        str(verify_report),
        "--output-root",
        str(verify_output),
    )

    commands = (
        lifecycle_commands[0],
        lifecycle_commands[1],
        ("save-preferences", save_command),
        ("verify-after-restart", verify_command),
        lifecycle_commands[3],
        lifecycle_commands[4],
    )

    results = []
    primary_error: str | None = None
    verified_exports: list[dict[str, object]] = []
    try:
        for name, command in commands[:4]:
            result = _run_step(
                name,
                command,
                environment=environment,
                logs=logs,
            )
            results.append(result)
            if result.return_code != 0:
                raise ExtensionInstallGateError(
                    f"step {name!r} failed with {result.return_code}; "
                    f"see {result.log_path}"
                )
        saved = json.loads(save_report.read_text(encoding="utf-8"))
        verified = json.loads(verify_report.read_text(encoding="utf-8"))
        if saved.get("status") != "passed" or verified.get("status") != "passed":
            raise ExtensionInstallGateError(
                f"preference worker failed: save={saved!r}, verify={verified!r}"
            )
        if saved.get("actual") != verified.get("actual"):
            raise ExtensionInstallGateError(
                "saved exact-version preferences differ after Blender restart"
            )
        raw_exports = verified.get("exports")
        if not isinstance(raw_exports, list) or len(raw_exports) != 5:
            raise ExtensionInstallGateError(
                "restart verification must complete five real target exports; "
                f"actual={raw_exports!r}"
            )
        verified_exports = raw_exports
        exact_versions = {
            str(item.get("exact_version", ""))
            for item in verified_exports
            if isinstance(item, dict)
        }
        if len(exact_versions) != 5 or "" in exact_versions:
            raise ExtensionInstallGateError(
                f"verified exports contain invalid exact versions: {verified_exports!r}"
            )
    except Exception as exc:
        primary_error = str(exc)
    finally:
        for name, command in commands[4:]:
            result = _run_step(
                name,
                command,
                environment=environment,
                logs=logs,
            )
            results.append(result)
            if result.return_code != 0 and primary_error is None:
                primary_error = (
                    f"cleanup step {name!r} failed with {result.return_code}; "
                    f"see {result.log_path}"
                )

    payload: dict[str, object] = {
        "status": "passed" if primary_error is None else "failed",
        "extension_id": extension_id,
        "archive": str(archive),
        "work_directory": str(work),
        "save_report": str(save_report),
        "verify_report": str(verify_report),
        "verified_exports": verified_exports,
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
    report_path = output_root / "spine-version-preferences-persistence-gate.json"
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
    payload = run_gate(
        namespace.blender,
        namespace.source,
        namespace.output_root,
        keep_work=namespace.keep_work,
    )
    versions = tuple(
        str(item.get("exact_version", ""))
        for item in payload["verified_exports"]
        if isinstance(item, dict)
    )
    print(
        "[SPINE-VERSION-PREFERENCES-PERSISTENCE] PASS "
        f"extension={payload['extension_id']} exports={len(versions)} "
        f"versions={versions!r}",
        flush=True,
    )


if __name__ == "__main__":
    main()
