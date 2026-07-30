#!/usr/bin/env python3
"""Run Blender production export and the exact Spine 4.1 runtime oracle in one gate."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Mapping, Sequence


LOGGER = logging.getLogger("spine41_standalone_acceptance")
ROOT = Path(__file__).resolve().parents[1]
BLENDER_WORKER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_spine41_standalone_multi_object_integration.py"
)
RUNTIME_ORACLE = ROOT / "tools" / "spine41_runtime_oracle.mjs"
EXPECTED_JSON_NAME = "Spine41StandaloneMulti.json"
EXPECTED_VERSION = "4.1.24"


class Spine41StandaloneAcceptanceError(RuntimeError):
    """Raised when one acceptance step is invalid or fails."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blender",
        type=Path,
        required=True,
        help="Path to the Blender 5.2+ executable.",
    )
    parser.add_argument(
        "--runtime-entry",
        type=Path,
        required=True,
        help="Read-only path to vendor/spine-webgl-41/index.js.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory that will receive the Blender export and acceptance reports.",
    )
    parser.add_argument(
        "--node",
        default="node",
        help="Node.js executable or command name. Defaults to node.",
    )
    parser.add_argument(
        "--replace-output",
        action="store_true",
        help="Replace a non-empty output directory after safety checks.",
    )
    return parser


def _resolve_required_file(value: Path, *, label: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{label} must be pathlib.Path")
    resolved = value.expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise Spine41StandaloneAcceptanceError(f"{label} does not exist: {resolved}")
    return resolved


def _resolve_node_executable(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("node must be a non-empty string")
    candidate = value.strip()
    explicit = Path(candidate).expanduser()
    if explicit.is_file():
        return str(explicit.resolve(strict=False))
    discovered = shutil.which(candidate)
    if discovered is None:
        raise Spine41StandaloneAcceptanceError(
            f"Node.js executable was not found: {candidate!r}"
        )
    return str(Path(discovered).resolve(strict=False))


def _dangerous_replace_target(path: Path) -> bool:
    """Return True for filesystem roots or paths too broad to delete safely."""

    resolved = path.resolve(strict=False)
    anchor = Path(resolved.anchor) if resolved.anchor else None
    if resolved.parent == resolved:
        return True
    if anchor is not None:
        try:
            relative_parts = resolved.relative_to(anchor).parts
        except ValueError:
            return True
        if len(relative_parts) < 2:
            return True
    return False


def prepare_output_root(value: Path, *, replace: bool) -> Path:
    """Create one dedicated output directory and never delete a broad path."""

    if not isinstance(value, Path):
        raise TypeError("output_root must be pathlib.Path")
    if not isinstance(replace, bool):
        raise TypeError("replace must be bool")
    resolved = value.expanduser().resolve(strict=False)
    if resolved.exists() and not resolved.is_dir():
        raise Spine41StandaloneAcceptanceError(
            f"Output root exists but is not a directory: {resolved}"
        )
    if resolved.exists() and any(resolved.iterdir()):
        if not replace:
            raise Spine41StandaloneAcceptanceError(
                f"Output root is not empty; pass --replace-output: {resolved}"
            )
        if _dangerous_replace_target(resolved):
            raise Spine41StandaloneAcceptanceError(
                f"Refusing to replace a broad or dangerous directory: {resolved}"
            )
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def build_blender_command(blender: Path, output_root: Path) -> tuple[str, ...]:
    """Return the exact Blender worker command used by the acceptance gate."""

    return (
        str(blender),
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        str(BLENDER_WORKER),
        "--",
        "--output",
        str(output_root),
    )


def build_oracle_command(
    node: str,
    json_path: Path,
    runtime_entry: Path,
) -> tuple[str, ...]:
    """Return the exact Node runtime-oracle command."""

    return (
        node,
        str(RUNTIME_ORACLE),
        str(json_path),
        str(runtime_entry),
    )


def _run_command(
    command: Sequence[str],
    *,
    label: str,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    LOGGER.info("%s: %s", label, subprocess.list2cmdline(tuple(command)))
    try:
        completed = subprocess.run(
            tuple(command),
            cwd=ROOT,
            env=None if environment is None else dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise Spine41StandaloneAcceptanceError(
            f"Unable to execute {label}: {exc}"
        ) from exc
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.returncode != 0:
        raise Spine41StandaloneAcceptanceError(
            f"{label} failed with exit code {completed.returncode}"
        )
    return completed


def parse_oracle_report(stdout: str) -> dict[str, object]:
    """Parse and validate the compact runtime-oracle JSON report."""

    if not isinstance(stdout, str):
        raise TypeError("stdout must be str")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise Spine41StandaloneAcceptanceError(
            f"Runtime oracle did not return valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise Spine41StandaloneAcceptanceError("Runtime oracle report root must be an object")
    if payload.get("ok") is not True:
        raise Spine41StandaloneAcceptanceError(
            f"Runtime oracle reported failure: {payload.get('message', payload)}"
        )
    if payload.get("version") != EXPECTED_VERSION:
        raise Spine41StandaloneAcceptanceError(
            f"Runtime oracle version mismatch: {payload.get('version')!r}"
        )

    counts = payload.get("counts")
    cache = payload.get("updateCache")
    matrices = payload.get("matrices")
    bounds = payload.get("bounds")
    if not isinstance(counts, dict):
        raise Spine41StandaloneAcceptanceError("Runtime oracle counts are missing")
    if int(counts.get("setupRenderableAttachments", 0)) <= 0:
        raise Spine41StandaloneAcceptanceError(
            "Runtime oracle found no setup-renderable attachments"
        )
    if not isinstance(cache, dict) or cache.get("everyConstraintScheduledExactlyOnce") is not True:
        raise Spine41StandaloneAcceptanceError(
            "Runtime oracle did not schedule every constraint exactly once"
        )
    if not isinstance(matrices, dict) or matrices.get("allFinite") is not True:
        raise Spine41StandaloneAcceptanceError("Runtime oracle found non-finite matrices")
    if not isinstance(bounds, dict):
        raise Spine41StandaloneAcceptanceError("Runtime oracle bounds are missing")
    for field_name in ("x", "y", "width", "height"):
        value = bounds.get(field_name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise Spine41StandaloneAcceptanceError(
                f"Runtime oracle bounds.{field_name} is not numeric"
            )
    if float(bounds["width"]) <= 0.0 or float(bounds["height"]) <= 0.0:
        raise Spine41StandaloneAcceptanceError(
            f"Runtime oracle bounds are not positive: {bounds}"
        )
    return payload


def run_acceptance(
    *,
    blender: Path,
    runtime_entry: Path,
    output_root: Path,
    node: str = "node",
    replace_output: bool = False,
) -> dict[str, object]:
    """Run Blender export, structural validation, and exact runtime validation."""

    blender = _resolve_required_file(blender, label="Blender executable")
    runtime_entry = _resolve_required_file(
        runtime_entry,
        label="Spine 4.1 runtime entry",
    )
    worker = _resolve_required_file(BLENDER_WORKER, label="Blender acceptance worker")
    oracle = _resolve_required_file(RUNTIME_ORACLE, label="Spine 4.1 runtime oracle")
    del worker, oracle
    node_executable = _resolve_node_executable(node)
    output_root = prepare_output_root(output_root, replace=replace_output)

    environment = dict(os.environ)
    _run_command(
        build_blender_command(blender, output_root),
        label="Blender Spine 4.1 standalone export",
        environment=environment,
    )

    json_path = output_root / EXPECTED_JSON_NAME
    if not json_path.is_file():
        raise Spine41StandaloneAcceptanceError(
            f"Blender acceptance JSON was not created: {json_path}"
        )
    blender_report_path = output_root / "blender_acceptance_report.json"
    if not blender_report_path.is_file():
        raise Spine41StandaloneAcceptanceError(
            f"Blender acceptance report was not created: {blender_report_path}"
        )
    blender_report = json.loads(blender_report_path.read_text(encoding="utf-8"))
    if not isinstance(blender_report, dict) or blender_report.get("status") != "passed":
        raise Spine41StandaloneAcceptanceError(
            f"Blender acceptance report did not pass: {blender_report}"
        )

    completed = _run_command(
        build_oracle_command(node_executable, json_path, runtime_entry),
        label="Spine 4.1 runtime oracle",
        environment=environment,
    )
    oracle_report = parse_oracle_report(completed.stdout)
    oracle_report_path = output_root / "runtime_oracle_report.json"
    oracle_report_path.write_text(
        json.dumps(oracle_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "status": "passed",
        "version": EXPECTED_VERSION,
        "jsonPath": str(json_path.resolve()),
        "blenderReportPath": str(blender_report_path.resolve()),
        "runtimeOracleReportPath": str(oracle_report_path.resolve()),
        "runtimeEntry": str(runtime_entry),
        "externalRuntimeReadOnly": True,
    }
    summary_path = output_root / "acceptance_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    arguments = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        summary = run_acceptance(
            blender=arguments.blender,
            runtime_entry=arguments.runtime_entry,
            output_root=arguments.output_root,
            node=arguments.node,
            replace_output=arguments.replace_output,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("SPINE41_STANDALONE_ACCEPTANCE=PASS")
        return 0
    except Spine41StandaloneAcceptanceError:
        LOGGER.exception("Spine 4.1 standalone acceptance failed")
        return 1
    except Exception:
        LOGGER.exception("Unexpected Spine 4.1 standalone acceptance failure")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
