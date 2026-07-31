#!/usr/bin/env python3
"""Run Blender production exports and exact Spine 3.8 runtime validation."""

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


LOGGER = logging.getLogger("spine38_standalone_acceptance")
ROOT = Path(__file__).resolve().parents[1]
BLENDER_WORKER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_spine38_standalone_profiles_integration.py"
)
RUNTIME_ORACLE = ROOT / "tools" / "spine38_runtime_oracle.mjs"
EXPECTED_VERSION = "3.8.99"
EXPECTED_CASES: Mapping[str, Mapping[str, object]] = {
    "TWO_AXIS_ROTATION_SCALE": {
        "directory": "two_axis",
        "stem": "Spine38TwoAxisStandaloneMulti",
        "bones": 55,
        "ik": 3,
        "transform": 12,
    },
    "LEGACY_ROTATABLE_MESH": {
        "directory": "three_axis",
        "stem": "Spine38ThreeAxisStandaloneMulti",
        "bones": 52,
        "ik": 3,
        "transform": 15,
    },
}


class Spine38StandaloneAcceptanceError(RuntimeError):
    """Raised when one Spine 3.8 acceptance stage fails."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument(
        "--runtime-entry",
        type=Path,
        required=True,
        help="Read-only path to vendor/spine-webgl-38/index.js.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--node", default="node")
    parser.add_argument("--replace-output", action="store_true")
    return parser


def _required_file(value: Path, *, label: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{label} must be pathlib.Path")
    resolved = value.expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise Spine38StandaloneAcceptanceError(f"{label} does not exist: {resolved}")
    return resolved


def _node_executable(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("node must be a non-empty string")
    explicit = Path(value.strip()).expanduser()
    if explicit.is_file():
        return str(explicit.resolve(strict=False))
    discovered = shutil.which(value.strip())
    if discovered is None:
        raise Spine38StandaloneAcceptanceError(
            f"Node.js executable was not found: {value!r}"
        )
    return str(Path(discovered).resolve(strict=False))


def _dangerous_replace_target(path: Path) -> bool:
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
    if not isinstance(value, Path):
        raise TypeError("output_root must be pathlib.Path")
    if not isinstance(replace, bool):
        raise TypeError("replace must be bool")
    resolved = value.expanduser().resolve(strict=False)
    if resolved.exists() and not resolved.is_dir():
        raise Spine38StandaloneAcceptanceError(
            f"Output root exists but is not a directory: {resolved}"
        )
    if resolved.exists() and any(resolved.iterdir()):
        if not replace:
            raise Spine38StandaloneAcceptanceError(
                f"Output root is not empty; pass --replace-output: {resolved}"
            )
        if _dangerous_replace_target(resolved):
            raise Spine38StandaloneAcceptanceError(
                f"Refusing to replace broad directory: {resolved}"
            )
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def build_blender_command(blender: Path, output_root: Path) -> tuple[str, ...]:
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
        raise Spine38StandaloneAcceptanceError(
            f"Unable to execute {label}: {exc}"
        ) from exc
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.returncode != 0:
        raise Spine38StandaloneAcceptanceError(
            f"{label} failed with exit code {completed.returncode}"
        )
    return completed


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    if not path.is_file():
        raise Spine38StandaloneAcceptanceError(f"{label} was not created: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Spine38StandaloneAcceptanceError(
            f"Unable to read {label} {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise Spine38StandaloneAcceptanceError(f"{label} root must be an object")
    return payload


def parse_oracle_report(
    stdout: str,
    *,
    expected: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(stdout, str):
        raise TypeError("stdout must be str")
    try:
        report = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise Spine38StandaloneAcceptanceError(
            f"Runtime oracle did not return valid JSON: {exc}"
        ) from exc
    if not isinstance(report, dict) or report.get("ok") is not True:
        raise Spine38StandaloneAcceptanceError(
            f"Runtime oracle reported failure: {report}"
        )
    if report.get("version") != EXPECTED_VERSION:
        raise Spine38StandaloneAcceptanceError(
            f"Runtime version mismatch: {report.get('version')!r}"
        )

    counts = report.get("counts")
    cache = report.get("updateCache")
    matrices = report.get("matrices")
    bounds = report.get("bounds")
    if not isinstance(counts, dict):
        raise Spine38StandaloneAcceptanceError("Runtime counts are missing")
    for field_name in ("bones", "ik", "transform"):
        if counts.get(field_name) != expected[field_name]:
            raise Spine38StandaloneAcceptanceError(
                f"Runtime counts.{field_name} mismatch: "
                f"expected={expected[field_name]}, actual={counts.get(field_name)!r}"
            )
    if counts.get("setupRenderableAttachments") != 3:
        raise Spine38StandaloneAcceptanceError(
            "Runtime must expose exactly three setup renderable attachments"
        )
    expected_constraints = int(expected["ik"]) + int(expected["transform"])
    if (
        not isinstance(cache, dict)
        or cache.get("expectedConstraints") != expected_constraints
        or cache.get("scheduledConstraints") != expected_constraints
        or cache.get("everyConstraintScheduledExactlyOnce") is not True
    ):
        raise Spine38StandaloneAcceptanceError(
            f"Runtime update cache evidence is incomplete: {cache}"
        )
    if (
        not isinstance(matrices, dict)
        or matrices.get("finiteBones") != expected["bones"]
        or matrices.get("allFinite") is not True
    ):
        raise Spine38StandaloneAcceptanceError(
            f"Runtime matrix evidence is incomplete: {matrices}"
        )
    if not isinstance(bounds, dict):
        raise Spine38StandaloneAcceptanceError("Runtime bounds are missing")
    numeric_bounds: dict[str, float] = {}
    for field_name in ("x", "y", "width", "height"):
        value = bounds.get(field_name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise Spine38StandaloneAcceptanceError(
                f"Runtime bounds.{field_name} is not numeric"
            )
        number = float(value)
        if not isfinite(number):
            raise Spine38StandaloneAcceptanceError(
                f"Runtime bounds.{field_name} is not finite"
            )
        numeric_bounds[field_name] = number
    if numeric_bounds["width"] <= 0.0 or numeric_bounds["height"] <= 0.0:
        raise Spine38StandaloneAcceptanceError(
            f"Runtime bounds are not positive: {bounds}"
        )
    return report


def validate_blender_report(
    report: dict[str, object],
    *,
    output_root: Path,
) -> dict[str, object]:
    if report.get("status") != "passed":
        raise Spine38StandaloneAcceptanceError(
            f"Blender report did not pass: {report}"
        )
    if report.get("version") != EXPECTED_VERSION:
        raise Spine38StandaloneAcceptanceError(
            f"Blender version mismatch: {report.get('version')!r}"
        )
    if report.get("mode") != "STANDALONE":
        raise Spine38StandaloneAcceptanceError(
            f"Blender mode mismatch: {report.get('mode')!r}"
        )
    profiles = report.get("profiles")
    if not isinstance(profiles, list):
        raise Spine38StandaloneAcceptanceError("Blender profiles must be an array")
    by_name: dict[str, dict[str, object]] = {}
    for index, value in enumerate(profiles):
        if not isinstance(value, dict):
            raise Spine38StandaloneAcceptanceError(
                f"profiles[{index}] must be an object"
            )
        name = value.get("profile")
        if not isinstance(name, str) or not name:
            raise Spine38StandaloneAcceptanceError(f"profiles[{index}] has no profile")
        by_name[name] = value
    if set(by_name) != set(EXPECTED_CASES):
        raise Spine38StandaloneAcceptanceError(
            f"Profile inventory differs: expected={tuple(EXPECTED_CASES)}, "
            f"actual={tuple(by_name)}"
        )

    for profile_name, expected in EXPECTED_CASES.items():
        profile = by_name[profile_name]
        for field_name in ("bones", "ik", "transform"):
            if profile.get(field_name) != expected[field_name]:
                raise Spine38StandaloneAcceptanceError(
                    f"{profile_name}.{field_name} mismatch: "
                    f"expected={expected[field_name]}, actual={profile.get(field_name)!r}"
                )
        for flag in (
            "legacyMixFieldsPresent",
        ):
            if profile.get(flag) is not True:
                raise Spine38StandaloneAcceptanceError(
                    f"{profile_name}.{flag} must be true"
                )
        for flag in (
            "newMixFieldsPresent",
            "connectedWrapperPresent",
            "crossObjectReferencesPresent",
            "sequencePresent",
        ):
            if profile.get(flag) is not False:
                raise Spine38StandaloneAcceptanceError(
                    f"{profile_name}.{flag} must be false"
                )
        json_path = (
            output_root
            / str(expected["directory"])
            / f"{expected['stem']}.json"
        ).resolve(strict=False)
        if Path(str(profile.get("jsonPath"))).resolve(strict=False) != json_path:
            raise Spine38StandaloneAcceptanceError(
                f"{profile_name}.jsonPath differs: {profile.get('jsonPath')!r}"
            )
        if not json_path.is_file():
            raise Spine38StandaloneAcceptanceError(
                f"{profile_name} JSON does not exist: {json_path}"
            )
    return report


def run_acceptance(
    *,
    blender: Path,
    runtime_entry: Path,
    output_root: Path,
    node: str = "node",
    replace_output: bool = False,
) -> dict[str, object]:
    blender = _required_file(blender, label="Blender executable")
    runtime_entry = _required_file(
        runtime_entry,
        label="Spine 3.8 runtime entry",
    )
    _required_file(BLENDER_WORKER, label="Blender worker")
    _required_file(RUNTIME_ORACLE, label="Spine 3.8 runtime oracle")
    node_executable = _node_executable(node)
    output_root = prepare_output_root(output_root, replace=replace_output)
    environment = dict(os.environ)

    _run_command(
        build_blender_command(blender, output_root),
        label="Blender Spine 3.8 standalone profile exports",
        environment=environment,
    )
    blender_report_path = output_root / "blender_acceptance_report.json"
    blender_report = validate_blender_report(
        _load_json(blender_report_path, label="Blender acceptance report"),
        output_root=output_root,
    )

    profile_reports: dict[str, dict[str, object]] = {}
    for profile_name, expected in EXPECTED_CASES.items():
        json_path = (
            output_root
            / str(expected["directory"])
            / f"{expected['stem']}.json"
        )
        completed = _run_command(
            build_oracle_command(node_executable, json_path, runtime_entry),
            label=f"Spine 3.8 runtime oracle {profile_name}",
            environment=environment,
        )
        oracle_report = parse_oracle_report(completed.stdout, expected=expected)
        report_path = output_root / f"runtime_oracle_{expected['directory']}.json"
        report_path.write_text(
            json.dumps(oracle_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        profile_reports[profile_name] = {
            "jsonPath": str(json_path.resolve()),
            "runtimeReportPath": str(report_path.resolve()),
            "report": oracle_report,
        }

    summary = {
        "status": "passed",
        "version": EXPECTED_VERSION,
        "mode": "STANDALONE",
        "runtimeEntry": str(runtime_entry),
        "externalRuntimeReadOnly": True,
        "runtimeValidated": True,
        "blenderReportPath": str(blender_report_path.resolve()),
        "profiles": profile_reports,
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
        print("SPINE38_STANDALONE_ACCEPTANCE=PASS")
        return 0
    except Spine38StandaloneAcceptanceError:
        LOGGER.exception("Spine 3.8 standalone acceptance failed")
        return 1
    except Exception:
        LOGGER.exception("Unexpected Spine 3.8 acceptance failure")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
