#!/usr/bin/env python3
"""Run the complete Blender and exact-runtime acceptance matrix for Spine 3.8-4.3."""

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

from run_spine43_runtime_acceptance import (
    build_runtime_command as build_spine43_runtime_command,
    build_runtime_environment as build_spine43_runtime_environment,
    resolve_runtime_entry as resolve_spine43_runtime_entry,
)
from spine_version_acceptance_matrix import (
    EXACT_VERSION_BY_TARGET,
    EXPECTED_CASE_COUNT_BY_TARGET,
    POSITIVE_CASES,
)


LOGGER = logging.getLogger("all_spine_versions_acceptance")
ROOT = Path(__file__).resolve().parents[1]
BLENDER_WORKER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_all_spine_versions_integration.py"
)
ORACLE_BY_TARGET = {
    "SPINE_3_8": ROOT / "tools" / "spine38_runtime_oracle.mjs",
    "SPINE_4_0": ROOT / "tools" / "spine40_runtime_oracle.mjs",
    "SPINE_4_1": ROOT / "tools" / "spine41_runtime_oracle.mjs",
    "SPINE_4_2": ROOT / "tools" / "spine42_runtime_oracle.mjs",
    "SPINE_4_3": ROOT / "tools" / "spine43_runtime_oracle.mjs",
}


class AllSpineVersionsAcceptanceError(RuntimeError):
    """Raised when any export, runtime, or report stage fails closed."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--runtime-38-entry", type=Path, required=True)
    parser.add_argument("--runtime-40-entry", type=Path, required=True)
    parser.add_argument("--runtime-41-entry", type=Path, required=True)
    parser.add_argument("--runtime-42-entry", type=Path, required=True)
    parser.add_argument(
        "--runtime-43-root",
        type=Path,
        required=True,
        help="Read-only root of an official Spine 4.3 spine-runtimes checkout.",
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
        raise AllSpineVersionsAcceptanceError(f"{label} does not exist: {resolved}")
    return resolved


def _required_directory(value: Path, *, label: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{label} must be pathlib.Path")
    resolved = value.expanduser().resolve(strict=False)
    if not resolved.is_dir():
        raise AllSpineVersionsAcceptanceError(f"{label} does not exist: {resolved}")
    return resolved


def _resolve_node(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("node must be a non-empty string")
    explicit = Path(value.strip()).expanduser()
    if explicit.is_file():
        return str(explicit.resolve(strict=False))
    discovered = shutil.which(value.strip())
    if discovered is None:
        raise AllSpineVersionsAcceptanceError(f"Node.js was not found: {value!r}")
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
        raise AllSpineVersionsAcceptanceError(
            f"Output root exists but is not a directory: {resolved}"
        )
    if resolved.exists() and any(resolved.iterdir()):
        if not replace:
            raise AllSpineVersionsAcceptanceError(
                f"Output root is not empty; pass --replace-output: {resolved}"
            )
        if _dangerous_replace_target(resolved):
            raise AllSpineVersionsAcceptanceError(
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
        raise AllSpineVersionsAcceptanceError(
            f"Unable to execute {label}: {exc}"
        ) from exc
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.returncode != 0:
        raise AllSpineVersionsAcceptanceError(
            f"{label} failed with exit code {completed.returncode}"
        )
    return completed


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    if not path.is_file():
        raise AllSpineVersionsAcceptanceError(f"{label} was not created: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AllSpineVersionsAcceptanceError(
            f"Unable to read {label} {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise AllSpineVersionsAcceptanceError(f"{label} root must be an object")
    return payload


def _parse_json_stdout(stdout: str, *, label: str) -> dict[str, object]:
    if not isinstance(stdout, str):
        raise TypeError("stdout must be str")
    stripped = stdout.strip()
    decoder = json.JSONDecoder()
    candidates = [index for index, character in enumerate(stripped) if character == "{"]
    for index in candidates:
        try:
            payload, end = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if stripped[index + end :].strip():
            continue
        if isinstance(payload, dict):
            return payload
    raise AllSpineVersionsAcceptanceError(f"{label} did not return one JSON object")


def _positive_number(value: object, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise AllSpineVersionsAcceptanceError(f"{label} must be numeric")
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise AllSpineVersionsAcceptanceError(f"{label} must be finite and positive")
    return number


def _validate_runtime_report(
    report: dict[str, object],
    *,
    target: str,
    profile: str,
    expected_version: str,
    object_count: int,
) -> dict[str, object]:
    if report.get("ok") is not True:
        raise AllSpineVersionsAcceptanceError(f"Runtime reported failure: {report}")
    if report.get("version") != expected_version:
        raise AllSpineVersionsAcceptanceError(
            f"Runtime version mismatch: expected={expected_version}, "
            f"actual={report.get('version')!r}"
        )
    counts = report.get("counts")
    cache = report.get("updateCache")
    matrices = report.get("matrices")
    bounds = report.get("bounds")
    if not isinstance(counts, dict):
        raise AllSpineVersionsAcceptanceError("Runtime counts are missing")
    attachment_count = counts.get("setupRenderableAttachments")
    if attachment_count != object_count:
        raise AllSpineVersionsAcceptanceError(
            "Runtime setup attachment count differs: "
            f"expected={object_count}, actual={attachment_count!r}"
        )
    bone_count = counts.get("bones")
    if not isinstance(bone_count, int) or isinstance(bone_count, bool) or bone_count <= 0:
        raise AllSpineVersionsAcceptanceError("Runtime bone count is invalid")
    if (
        not isinstance(cache, dict)
        or cache.get("everyConstraintScheduledExactlyOnce") is not True
        or cache.get("expectedConstraints") != cache.get("scheduledConstraints")
    ):
        raise AllSpineVersionsAcceptanceError(
            f"Runtime update-cache evidence is incomplete: {cache}"
        )
    if (
        not isinstance(matrices, dict)
        or matrices.get("allFinite") is not True
        or matrices.get("finiteBones") != bone_count
    ):
        raise AllSpineVersionsAcceptanceError(
            f"Runtime matrix evidence is incomplete: {matrices}"
        )
    if not isinstance(bounds, dict):
        raise AllSpineVersionsAcceptanceError("Runtime bounds are missing")
    _positive_number(bounds.get("width"), label="runtime bounds.width")
    _positive_number(bounds.get("height"), label="runtime bounds.height")
    for field_name in ("x", "y"):
        value = bounds.get(field_name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(float(value))
        ):
            raise AllSpineVersionsAcceptanceError(
                f"runtime bounds.{field_name} must be finite"
            )

    if target == "SPINE_3_8":
        scale = report.get("scaleResponse")
        if not isinstance(scale, dict):
            raise AllSpineVersionsAcceptanceError("Spine 3.8 scaleResponse is missing")
        if profile == "TWO_AXIS_ROTATION_SCALE":
            for field_name in (
                "applicable",
                "allControlsResponded",
                "boundsChanged",
                "constraintAffectsBounds",
                "matricesFinite",
            ):
                if scale.get(field_name) is not True:
                    raise AllSpineVersionsAcceptanceError(
                        f"Spine 3.8 scaleResponse.{field_name} must be true"
                    )
            if scale.get("controlCount") != object_count:
                raise AllSpineVersionsAcceptanceError(
                    "Spine 3.8 scale control count differs from object count"
                )
        elif scale.get("applicable") is not False:
            raise AllSpineVersionsAcceptanceError(
                "Spine 3.8 3-Axis case must not activate Scale response"
            )
    return report


def _validate_blender_report(
    report: dict[str, object],
    *,
    output_root: Path,
) -> tuple[dict[str, object], ...]:
    if report.get("status") != "passed" or report.get("caseCount") != len(POSITIVE_CASES):
        raise AllSpineVersionsAcceptanceError(
            f"Blender matrix report did not pass: {report}"
        )
    if report.get("expectedCaseCountByTarget") != dict(EXPECTED_CASE_COUNT_BY_TARGET):
        raise AllSpineVersionsAcceptanceError("Blender target case counts differ")
    capabilities = report.get("capabilities")
    if not isinstance(capabilities, dict):
        raise AllSpineVersionsAcceptanceError("Capability report is missing")
    if (
        len(capabilities.get("accepted", ())) != 20
        or len(capabilities.get("blocked", ())) != 20
    ):
        raise AllSpineVersionsAcceptanceError("Capability acceptance/block inventory differs")
    raw_cases = report.get("cases")
    if not isinstance(raw_cases, list):
        raise AllSpineVersionsAcceptanceError("Blender cases must be an array")
    by_key: dict[str, dict[str, object]] = {}
    for value in raw_cases:
        if not isinstance(value, dict) or not isinstance(value.get("key"), str):
            raise AllSpineVersionsAcceptanceError("Blender case record is invalid")
        by_key[str(value["key"])] = value
    expected_keys = {case.key for case in POSITIVE_CASES}
    if set(by_key) != expected_keys:
        raise AllSpineVersionsAcceptanceError("Blender case inventory differs")
    ordered: list[dict[str, object]] = []
    for case in POSITIVE_CASES:
        record = by_key[case.key]
        if (
            record.get("status") != "passed"
            or record.get("target") != case.target
            or record.get("version") != case.exact_version
            or record.get("profile") != case.profile
            or record.get("scope") != case.scope
            or record.get("objectCount") != case.object_count
        ):
            raise AllSpineVersionsAcceptanceError(f"Blender case metadata differs: {record}")
        json_path = Path(str(record.get("jsonPath"))).resolve(strict=False)
        try:
            json_path.relative_to(output_root)
        except ValueError as exc:
            raise AllSpineVersionsAcceptanceError(
                f"Blender case JSON escapes output root: {json_path}"
            ) from exc
        if not json_path.is_file():
            raise AllSpineVersionsAcceptanceError(
                f"Blender case JSON is missing: {json_path}"
            )
        ordered.append(record)
    return tuple(ordered)


def _standard_runtime_command(
    *,
    node: str,
    target: str,
    json_path: Path,
    runtime_entry: Path,
) -> tuple[str, ...]:
    return (
        node,
        str(ORACLE_BY_TARGET[target]),
        str(json_path),
        str(runtime_entry),
    )


def run_acceptance(
    *,
    blender: Path,
    runtime_38_entry: Path,
    runtime_40_entry: Path,
    runtime_41_entry: Path,
    runtime_42_entry: Path,
    runtime_43_root: Path,
    output_root: Path,
    node: str = "node",
    replace_output: bool = False,
) -> dict[str, object]:
    blender = _required_file(blender, label="Blender executable")
    runtime_entries = {
        "SPINE_3_8": _required_file(runtime_38_entry, label="Spine 3.8 runtime entry"),
        "SPINE_4_0": _required_file(runtime_40_entry, label="Spine 4.0 runtime entry"),
        "SPINE_4_1": _required_file(runtime_41_entry, label="Spine 4.1 runtime entry"),
        "SPINE_4_2": _required_file(runtime_42_entry, label="Spine 4.2 runtime entry"),
    }
    runtime_43_root = _required_directory(
        runtime_43_root,
        label="Spine 4.3 runtime root",
    )
    runtime_43 = resolve_spine43_runtime_entry(runtime_43_root)
    node_executable = _resolve_node(node)
    _required_file(BLENDER_WORKER, label="Blender matrix worker")
    for target, oracle in ORACLE_BY_TARGET.items():
        _required_file(oracle, label=f"{target} runtime oracle")
    output_root = prepare_output_root(output_root, replace=replace_output)
    environment = dict(os.environ)

    _run_command(
        build_blender_command(blender, output_root),
        label="Blender Spine 3.8-4.3 full export matrix",
        environment=environment,
    )
    blender_report_path = output_root / "blender_acceptance_report.json"
    blender_report = _load_json_object(
        blender_report_path,
        label="Blender acceptance report",
    )
    case_records = _validate_blender_report(blender_report, output_root=output_root)

    runtime_case_reports: dict[str, dict[str, object]] = {}
    target_passed = {target: 0 for target in EXACT_VERSION_BY_TARGET}
    for case, record in zip(POSITIVE_CASES, case_records, strict=True):
        json_path = Path(str(record["jsonPath"])).resolve()
        if case.target == "SPINE_4_3":
            command = build_spine43_runtime_command(
                node_executable,
                json_path,
                runtime_43,
            )
            case_environment = build_spine43_runtime_environment(runtime_43)
        else:
            command = _standard_runtime_command(
                node=node_executable,
                target=case.target,
                json_path=json_path,
                runtime_entry=runtime_entries[case.target],
            )
            case_environment = environment
        completed = _run_command(
            command,
            label=f"Runtime oracle {case.key}",
            environment=case_environment,
        )
        runtime_report = _validate_runtime_report(
            _parse_json_stdout(completed.stdout, label=f"Runtime oracle {case.key}"),
            target=case.target,
            profile=case.profile,
            expected_version=case.exact_version,
            object_count=case.object_count,
        )
        report_path = output_root / case.key / "runtime_oracle_report.json"
        report_path.write_text(
            json.dumps(runtime_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        runtime_case_reports[case.key] = {
            "target": case.target,
            "version": case.exact_version,
            "profile": case.profile,
            "scope": case.scope,
            "jsonPath": str(json_path),
            "runtimeReportPath": str(report_path.resolve()),
            "report": runtime_report,
        }
        target_passed[case.target] += 1

    if target_passed != dict(EXPECTED_CASE_COUNT_BY_TARGET):
        raise AllSpineVersionsAcceptanceError(
            f"Runtime target pass counts differ: {target_passed}"
        )
    targets = {
        target: {
            "version": EXACT_VERSION_BY_TARGET[target],
            "passed": target_passed[target],
            "failed": 0,
        }
        for target in EXACT_VERSION_BY_TARGET
    }
    summary = {
        "status": "passed",
        "caseCount": len(POSITIVE_CASES),
        "totalPassed": len(POSITIVE_CASES),
        "totalFailed": 0,
        "externalRuntimesReadOnly": True,
        "runtime43": {
            "root": str(runtime_43.runtime_root),
            "entry": str(runtime_43.entry_path),
            "mode": runtime_43.mode,
            "packageVersion": runtime_43.package_version,
        },
        "targets": targets,
        "blenderReportPath": str(blender_report_path.resolve()),
        "cases": runtime_case_reports,
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
            runtime_38_entry=arguments.runtime_38_entry,
            runtime_40_entry=arguments.runtime_40_entry,
            runtime_41_entry=arguments.runtime_41_entry,
            runtime_42_entry=arguments.runtime_42_entry,
            runtime_43_root=arguments.runtime_43_root,
            output_root=arguments.output_root,
            node=arguments.node,
            replace_output=arguments.replace_output,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("SPINE_ALL_VERSIONS_ACCEPTANCE=PASS")
        return 0
    except AllSpineVersionsAcceptanceError:
        LOGGER.exception("Spine 3.8-4.3 full acceptance matrix failed")
        return 1
    except Exception:
        LOGGER.exception("Unexpected Spine 3.8-4.3 acceptance failure")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
