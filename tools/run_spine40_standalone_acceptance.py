#!/usr/bin/env python3
"""Run Blender production export and the exact Spine 4.0 runtime oracle."""

from __future__ import annotations

import argparse
import json
import logging
from math import isfinite
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Mapping, Sequence


LOGGER = logging.getLogger("spine40_standalone_acceptance")
ROOT = Path(__file__).resolve().parents[1]
BLENDER_WORKER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_spine40_standalone_multi_object_integration.py"
)
RUNTIME_ORACLE = ROOT / "tools" / "spine40_runtime_oracle.mjs"
EXPECTED_JSON_NAME = "Spine40StandaloneMulti.json"
EXPECTED_VERSION = "4.0.64"


class Spine40StandaloneAcceptanceError(RuntimeError):
    """Raised when one Spine 4.0 acceptance step fails."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument(
        "--runtime-entry",
        type=Path,
        required=True,
        help="Read-only path to vendor/spine-webgl-40/index.js.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--node", default="node")
    parser.add_argument("--replace-output", action="store_true")
    return parser


def _resolve_required_file(value: Path, *, label: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{label} must be pathlib.Path")
    resolved = value.expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise Spine40StandaloneAcceptanceError(f"{label} does not exist: {resolved}")
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
        raise Spine40StandaloneAcceptanceError(
            f"Node.js executable was not found: {candidate!r}"
        )
    return str(Path(discovered).resolve(strict=False))


def _dangerous_replace_target(path: Path) -> bool:
    """Reject filesystem roots and broad first-level directories."""

    resolved = path.resolve(strict=False)
    if resolved.parent == resolved:
        return True
    if resolved.anchor:
        anchor = Path(resolved.anchor)
        try:
            relative_parts = resolved.relative_to(anchor).parts
        except ValueError:
            return True
        if len(relative_parts) < 2:
            return True
    return False


def prepare_output_root(value: Path, *, replace: bool) -> Path:
    """Prepare only the caller-selected output root after safety checks."""

    if not isinstance(value, Path):
        raise TypeError("output_root must be pathlib.Path")
    if not isinstance(replace, bool):
        raise TypeError("replace must be bool")
    resolved = value.expanduser().resolve(strict=False)
    if resolved.exists() and not resolved.is_dir():
        raise Spine40StandaloneAcceptanceError(
            f"Output root exists but is not a directory: {resolved}"
        )
    if resolved.exists() and any(resolved.iterdir()):
        if not replace:
            raise Spine40StandaloneAcceptanceError(
                f"Output root is not empty; pass --replace-output: {resolved}"
            )
        if _dangerous_replace_target(resolved):
            raise Spine40StandaloneAcceptanceError(
                f"Refusing to replace a broad or dangerous directory: {resolved}"
            )
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def build_blender_command(blender: Path, output_root: Path) -> tuple[str, ...]:
    """Return the fail-closed Blender command."""

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
    """Pass the runtime path only as read-only oracle input."""

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
        raise Spine40StandaloneAcceptanceError(
            f"Unable to execute {label}: {exc}"
        ) from exc
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.returncode != 0:
        raise Spine40StandaloneAcceptanceError(
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
        raise Spine40StandaloneAcceptanceError(
            f"Runtime oracle did not return valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise Spine40StandaloneAcceptanceError("Runtime oracle root must be an object")
    if payload.get("ok") is not True:
        raise Spine40StandaloneAcceptanceError(
            f"Runtime oracle reported failure: {payload.get('message', payload)}"
        )
    if payload.get("version") != EXPECTED_VERSION:
        raise Spine40StandaloneAcceptanceError(
            f"Runtime oracle version mismatch: {payload.get('version')!r}"
        )

    counts = payload.get("counts")
    cache = payload.get("updateCache")
    matrices = payload.get("matrices")
    bounds = payload.get("bounds")
    if not isinstance(counts, dict):
        raise Spine40StandaloneAcceptanceError("Runtime oracle counts are missing")
    attachment_count = counts.get("setupRenderableAttachments")
    if (
        not isinstance(attachment_count, int)
        or isinstance(attachment_count, bool)
        or attachment_count <= 0
    ):
        raise Spine40StandaloneAcceptanceError(
            "Runtime oracle found no setup-renderable attachments"
        )
    if (
        not isinstance(cache, dict)
        or cache.get("everyConstraintScheduledExactlyOnce") is not True
    ):
        raise Spine40StandaloneAcceptanceError(
            "Runtime oracle did not schedule every constraint exactly once"
        )
    if not isinstance(matrices, dict) or matrices.get("allFinite") is not True:
        raise Spine40StandaloneAcceptanceError("Runtime oracle found non-finite matrices")
    if not isinstance(bounds, dict):
        raise Spine40StandaloneAcceptanceError("Runtime oracle bounds are missing")

    numeric_bounds: dict[str, float] = {}
    for field_name in ("x", "y", "width", "height"):
        value = bounds.get(field_name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise Spine40StandaloneAcceptanceError(
                f"Runtime oracle bounds.{field_name} is not numeric"
            )
        numeric = float(value)
        if not isfinite(numeric):
            raise Spine40StandaloneAcceptanceError(
                f"Runtime oracle bounds.{field_name} is not finite"
            )
        numeric_bounds[field_name] = numeric
    if numeric_bounds["width"] <= 0.0 or numeric_bounds["height"] <= 0.0:
        raise Spine40StandaloneAcceptanceError(
            f"Runtime oracle bounds are not positive: {bounds}"
        )
    return payload


def _read_blender_report(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise Spine40StandaloneAcceptanceError(
            f"Blender acceptance report was not created: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Spine40StandaloneAcceptanceError(
            f"Unable to read Blender acceptance report: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("status") != "passed":
        raise Spine40StandaloneAcceptanceError(
            f"Blender acceptance report did not pass: {payload}"
        )
    if payload.get("version") != EXPECTED_VERSION:
        raise Spine40StandaloneAcceptanceError(
            f"Blender acceptance version mismatch: {payload.get('version')!r}"
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
        label="Spine 4.0 runtime entry",
    )
    _resolve_required_file(BLENDER_WORKER, label="Blender acceptance worker")
    _resolve_required_file(RUNTIME_ORACLE, label="Spine 4.0 runtime oracle")
    node_executable = _resolve_node_executable(node)
    output_root = prepare_output_root(output_root, replace=replace_output)
    environment = dict(os.environ)

    _run_command(
        build_blender_command(blender, output_root),
        label="Blender Spine 4.0 standalone export",
        environment=environment,
    )

    json_path = output_root / EXPECTED_JSON_NAME
    if not json_path.is_file():
        raise Spine40StandaloneAcceptanceError(
            f"Blender acceptance JSON was not created: {json_path}"
        )
    blender_report_path = output_root / "blender_acceptance_report.json"
    blender_report = _read_blender_report(blender_report_path)

    completed = _run_command(
        build_oracle_command(node_executable, json_path, runtime_entry),
        label="Spine 4.0 runtime oracle",
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
        "blenderReport": blender_report,
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
        print("SPINE40_STANDALONE_ACCEPTANCE=PASS")
        return 0
    except Spine40StandaloneAcceptanceError:
        LOGGER.exception("Spine 4.0 standalone acceptance failed")
        return 1
    except Exception:
        LOGGER.exception("Unexpected Spine 4.0 standalone acceptance failure")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
