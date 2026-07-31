#!/usr/bin/env python3
"""Validate both generated Spine 4.3 profile JSON files with official spine-core."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from math import isfinite
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Mapping, Sequence


LOGGER = logging.getLogger("spine43_standalone_acceptance")
ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ORACLE = ROOT / "tools" / "spine43_runtime_oracle.mjs"
SOURCE_LOADER = ROOT / "tools" / "spine43_ts_source_loader.mjs"
EXPECTED_VERSION = "4.3.23"
EXPECTED_CASES: Mapping[str, Mapping[str, object]] = {
    "TWO_AXIS_ROTATION_SCALE": {
        "directory": "two_axis",
        "stem": "Spine43TwoAxisStandaloneMulti",
        "ik": 3,
        "transform": 12,
    },
    "LEGACY_ROTATABLE_MESH": {
        "directory": "three_axis",
        "stem": "Spine43ThreeAxisStandaloneMulti",
        "ik": 3,
        "transform": 15,
    },
}


class Spine43RuntimeAcceptanceError(RuntimeError):
    """Raised when runtime discovery, execution, or evidence validation fails."""


@dataclass(frozen=True, slots=True)
class Spine43RuntimeEntry:
    """One discovered official runtime entry and its execution policy."""

    runtime_root: Path
    entry_path: Path
    mode: str
    package_version: str
    source_root: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_root, Path) or not self.runtime_root.is_dir():
            raise ValueError("runtime_root must be an existing directory")
        if not isinstance(self.entry_path, Path) or not self.entry_path.is_file():
            raise ValueError("entry_path must be an existing file")
        if self.mode not in {"BUILT_ESM", "SOURCE_TYPESCRIPT"}:
            raise ValueError(f"Unsupported runtime mode: {self.mode!r}")
        if not isinstance(self.package_version, str) or not self.package_version.startswith(
            "4.3."
        ):
            raise ValueError("package_version must belong to Spine 4.3")
        if self.mode == "SOURCE_TYPESCRIPT":
            if not isinstance(self.source_root, Path) or not self.source_root.is_dir():
                raise ValueError("source_root is required for SOURCE_TYPESCRIPT")
        elif self.source_root is not None:
            raise ValueError("source_root must be None for BUILT_ESM")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        required=True,
        help="Read-only root of the official spine-runtimes 4.3 checkout.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Existing output from run_spine43_standalone_acceptance.py.",
    )
    parser.add_argument(
        "--node",
        default="node",
        help="Node.js executable or command name. Defaults to node.",
    )
    return parser


def _resolve_required_file(value: Path, *, label: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{label} must be pathlib.Path")
    resolved = value.expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise Spine43RuntimeAcceptanceError(f"{label} does not exist: {resolved}")
    return resolved


def _resolve_required_directory(value: Path, *, label: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{label} must be pathlib.Path")
    resolved = value.expanduser().resolve(strict=False)
    if not resolved.is_dir():
        raise Spine43RuntimeAcceptanceError(f"{label} does not exist: {resolved}")
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
        raise Spine43RuntimeAcceptanceError(
            f"Node.js executable was not found: {candidate!r}"
        )
    return str(Path(discovered).resolve(strict=False))


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    if not path.is_file():
        raise Spine43RuntimeAcceptanceError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Spine43RuntimeAcceptanceError(
            f"Unable to read {label} {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise Spine43RuntimeAcceptanceError(f"{label} root must be a JSON object")
    return payload


def _read_package_version(runtime_root: Path) -> str:
    package_path = runtime_root / "spine-ts" / "spine-core" / "package.json"
    package = _load_json_object(package_path, label="spine-core package.json")
    version = package.get("version")
    if not isinstance(version, str) or not version.startswith("4.3."):
        raise Spine43RuntimeAcceptanceError(
            f"spine-core package version is not 4.3.x: {version!r}"
        )
    return version


def resolve_runtime_entry(runtime_root: Path) -> Spine43RuntimeEntry:
    """Prefer an existing built ESM entry, otherwise use the clean TypeScript source."""

    root = _resolve_required_directory(runtime_root, label="Spine 4.3 runtime root")
    package_version = _read_package_version(root)
    core_root = root / "spine-ts" / "spine-core"
    built_candidates = (
        core_root / "dist" / "esm" / "spine-core.mjs",
        core_root / "dist" / "esm" / "spine-core.min.mjs",
        core_root / "dist" / "index.js",
        core_root / "dist" / "index.mjs",
    )
    for candidate in built_candidates:
        if candidate.is_file():
            return Spine43RuntimeEntry(
                runtime_root=root,
                entry_path=candidate.resolve(),
                mode="BUILT_ESM",
                package_version=package_version,
            )

    source_root = core_root / "src"
    source_entry = source_root / "index.ts"
    if source_entry.is_file():
        return Spine43RuntimeEntry(
            runtime_root=root,
            entry_path=source_entry.resolve(),
            mode="SOURCE_TYPESCRIPT",
            package_version=package_version,
            source_root=source_root.resolve(),
        )

    searched = tuple(str(path) for path in (*built_candidates, source_entry))
    raise Spine43RuntimeAcceptanceError(
        "No usable spine-core 4.3 entry was found; searched=" + repr(searched)
    )


def build_runtime_command(
    node: str,
    json_path: Path,
    runtime: Spine43RuntimeEntry,
) -> tuple[str, ...]:
    """Build one read-only runtime command for built or source checkout mode."""

    if not isinstance(node, str) or not node.strip():
        raise TypeError("node must be a non-empty string")
    if not isinstance(json_path, Path):
        raise TypeError("json_path must be pathlib.Path")
    if not isinstance(runtime, Spine43RuntimeEntry):
        raise TypeError("runtime must be Spine43RuntimeEntry")

    if runtime.mode == "SOURCE_TYPESCRIPT":
        loader_url = SOURCE_LOADER.resolve(strict=False).as_uri()
        return (
            node,
            "--no-warnings",
            "--experimental-transform-types",
            "--experimental-loader",
            loader_url,
            str(RUNTIME_ORACLE),
            str(json_path),
            str(runtime.entry_path),
        )
    return (
        node,
        str(RUNTIME_ORACLE),
        str(json_path),
        str(runtime.entry_path),
    )


def build_runtime_environment(runtime: Spine43RuntimeEntry) -> dict[str, str]:
    """Return an isolated environment containing only the source-loader boundary."""

    if not isinstance(runtime, Spine43RuntimeEntry):
        raise TypeError("runtime must be Spine43RuntimeEntry")
    environment = dict(os.environ)
    if runtime.mode == "SOURCE_TYPESCRIPT":
        if runtime.source_root is None:
            raise Spine43RuntimeAcceptanceError("Source runtime has no source_root")
        environment["SPINE43_RUNTIME_SOURCE_ROOT"] = str(runtime.source_root)
    else:
        environment.pop("SPINE43_RUNTIME_SOURCE_ROOT", None)
    return environment


def _run_command(
    command: Sequence[str],
    *,
    label: str,
    environment: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    LOGGER.info("%s: %s", label, subprocess.list2cmdline(tuple(command)))
    try:
        completed = subprocess.run(
            tuple(command),
            cwd=ROOT,
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise Spine43RuntimeAcceptanceError(
            f"Unable to execute {label}: {exc}"
        ) from exc
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.returncode != 0:
        raise Spine43RuntimeAcceptanceError(
            f"{label} failed with exit code {completed.returncode}"
        )
    return completed


def parse_runtime_report(
    stdout: str,
    *,
    expected_ik: int,
    expected_transform: int,
) -> dict[str, object]:
    """Validate compact runtime evidence returned by the Node oracle."""

    if not isinstance(stdout, str):
        raise TypeError("stdout must be str")
    try:
        report = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise Spine43RuntimeAcceptanceError(
            f"Runtime oracle did not return valid JSON: {exc}"
        ) from exc
    if not isinstance(report, dict) or report.get("ok") is not True:
        raise Spine43RuntimeAcceptanceError(
            f"Runtime oracle reported failure: {report}"
        )
    if report.get("version") != EXPECTED_VERSION:
        raise Spine43RuntimeAcceptanceError(
            f"Runtime oracle version mismatch: {report.get('version')!r}"
        )

    counts = report.get("counts")
    cache = report.get("updateCache")
    matrices = report.get("matrices")
    bounds = report.get("bounds")
    if not isinstance(counts, dict):
        raise Spine43RuntimeAcceptanceError("Runtime counts are missing")
    expected_constraints = expected_ik + expected_transform
    expected_counts = {
        "constraints": expected_constraints,
        "ik": expected_ik,
        "transform": expected_transform,
    }
    for field_name, expected in expected_counts.items():
        if counts.get(field_name) != expected:
            raise Spine43RuntimeAcceptanceError(
                f"Runtime counts.{field_name} mismatch: "
                f"expected={expected}, actual={counts.get(field_name)!r}"
            )
    attachment_count = counts.get("setupRenderableAttachments")
    if (
        not isinstance(attachment_count, int)
        or isinstance(attachment_count, bool)
        or attachment_count <= 0
    ):
        raise Spine43RuntimeAcceptanceError(
            "Runtime found no setup-renderable attachments"
        )
    if (
        not isinstance(cache, dict)
        or cache.get("expectedConstraints") != expected_constraints
        or cache.get("scheduledConstraints") != expected_constraints
        or cache.get("everyConstraintScheduledExactlyOnce") is not True
    ):
        raise Spine43RuntimeAcceptanceError(
            f"Runtime update-cache evidence is incomplete: {cache}"
        )
    if not isinstance(matrices, dict) or matrices.get("allFinite") is not True:
        raise Spine43RuntimeAcceptanceError("Runtime found non-finite bone matrices")
    if not isinstance(bounds, dict):
        raise Spine43RuntimeAcceptanceError("Runtime bounds are missing")
    numeric_bounds: dict[str, float] = {}
    for field_name in ("x", "y", "width", "height"):
        value = bounds.get(field_name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise Spine43RuntimeAcceptanceError(
                f"Runtime bounds.{field_name} is not numeric"
            )
        numeric = float(value)
        if not isfinite(numeric):
            raise Spine43RuntimeAcceptanceError(
                f"Runtime bounds.{field_name} is not finite"
            )
        numeric_bounds[field_name] = numeric
    if numeric_bounds["width"] <= 0.0 or numeric_bounds["height"] <= 0.0:
        raise Spine43RuntimeAcceptanceError(
            f"Runtime bounds are not positive: {bounds}"
        )
    return report


def run_acceptance(
    *,
    runtime_root: Path,
    output_root: Path,
    node: str = "node",
) -> dict[str, object]:
    """Run the exact runtime parser and setup evaluation for both profile exports."""

    runtime = resolve_runtime_entry(runtime_root)
    output = _resolve_required_directory(output_root, label="Spine 4.3 output root")
    node_executable = _resolve_node_executable(node)
    _resolve_required_file(RUNTIME_ORACLE, label="Spine 4.3 runtime oracle")
    if runtime.mode == "SOURCE_TYPESCRIPT":
        _resolve_required_file(SOURCE_LOADER, label="Spine 4.3 TypeScript loader")
    environment = build_runtime_environment(runtime)

    profile_reports: dict[str, dict[str, object]] = {}
    for profile_name, expected in EXPECTED_CASES.items():
        json_path = (
            output
            / str(expected["directory"])
            / f"{expected['stem']}.json"
        )
        _resolve_required_file(json_path, label=f"{profile_name} Spine JSON")
        completed = _run_command(
            build_runtime_command(node_executable, json_path, runtime),
            label=f"Spine 4.3 runtime oracle {profile_name}",
            environment=environment,
        )
        report = parse_runtime_report(
            completed.stdout,
            expected_ik=int(expected["ik"]),
            expected_transform=int(expected["transform"]),
        )
        report_path = output / f"runtime_oracle_{expected['directory']}.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        profile_reports[profile_name] = {
            "jsonPath": str(json_path.resolve()),
            "runtimeReportPath": str(report_path.resolve()),
            "report": report,
        }

    summary = {
        "status": "passed",
        "version": EXPECTED_VERSION,
        "runtimePackageVersion": runtime.package_version,
        "runtimeMode": runtime.mode,
        "runtimeRoot": str(runtime.runtime_root),
        "runtimeEntry": str(runtime.entry_path),
        "externalRuntimeReadOnly": True,
        "profiles": profile_reports,
        "runtimeValidated": True,
        "manualEditorImportRequired": True,
    }
    summary_path = output / "runtime_acceptance_summary.json"
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
            runtime_root=arguments.runtime_root,
            output_root=arguments.output_root,
            node=arguments.node,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("SPINE43_RUNTIME_ACCEPTANCE=PASS")
        return 0
    except Spine43RuntimeAcceptanceError:
        LOGGER.exception("Spine 4.3 runtime acceptance failed")
        return 1
    except Exception:
        LOGGER.exception("Unexpected Spine 4.3 runtime acceptance failure")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
