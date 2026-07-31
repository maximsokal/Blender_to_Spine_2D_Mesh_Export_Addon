#!/usr/bin/env python3
"""Run every exact-runtime oracle against an existing Blender matrix export.

Unlike the full export runner, this diagnostic stage never stops at the first runtime
failure. It writes one report per case and a complete acceptance summary, then exits with
status 1 only after all version/profile/scope combinations have been attempted.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

from run_all_spine_versions_acceptance import (
    AllSpineVersionsAcceptanceError,
    ORACLE_BY_TARGET,
    ROOT,
    _load_json_object,
    _parse_json_stdout,
    _required_directory,
    _required_file,
    _resolve_node,
    _standard_runtime_command,
    _validate_blender_report,
    _validate_runtime_report,
)
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


LOGGER = logging.getLogger("all_spine_runtime_oracles")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-38-entry", type=Path, required=True)
    parser.add_argument("--runtime-40-entry", type=Path, required=True)
    parser.add_argument("--runtime-41-entry", type=Path, required=True)
    parser.add_argument("--runtime-42-entry", type=Path, required=True)
    parser.add_argument("--runtime-43-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--node", default="node")
    return parser


def _run_process(
    command: Sequence[str],
    *,
    label: str,
    environment: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a non-empty string")
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
        return subprocess.CompletedProcess(
            args=tuple(command),
            returncode=127,
            stdout=f"Unable to execute {label}: {exc}\n",
        )
    if completed.stdout:
        print(completed.stdout.rstrip())
    return completed


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path.resolve(strict=False)


def _failure_report(
    *,
    stage: str,
    message: str,
    exit_code: int | None,
    output: str,
) -> dict[str, object]:
    return {
        "ok": False,
        "stage": stage,
        "message": message,
        "exitCode": exit_code,
        "output": output,
    }


def run_runtime_matrix(
    *,
    runtime_38_entry: Path,
    runtime_40_entry: Path,
    runtime_41_entry: Path,
    runtime_42_entry: Path,
    runtime_43_root: Path,
    output_root: Path,
    node: str = "node",
) -> dict[str, object]:
    output_root = _required_directory(output_root, label="Matrix output root")
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
    for target, oracle in ORACLE_BY_TARGET.items():
        _required_file(oracle, label=f"{target} runtime oracle")

    blender_report_path = output_root / "blender_acceptance_report.json"
    blender_report = _load_json_object(
        blender_report_path,
        label="Blender acceptance report",
    )
    case_records = _validate_blender_report(blender_report, output_root=output_root)

    environment = dict(os.environ)
    cases: dict[str, dict[str, object]] = {}
    passed_by_target = {target: 0 for target in EXACT_VERSION_BY_TARGET}
    failed_by_target = {target: 0 for target in EXACT_VERSION_BY_TARGET}

    for case, record in zip(POSITIVE_CASES, case_records, strict=True):
        json_path = Path(str(record["jsonPath"])).resolve(strict=False)
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

        completed = _run_process(
            command,
            label=f"Runtime oracle {case.key}",
            environment=case_environment,
        )
        report_path = output_root / case.key / "runtime_oracle_report.json"

        if completed.returncode != 0:
            report = _failure_report(
                stage="EXECUTE_ORACLE",
                message=(
                    f"Runtime oracle exited with code {completed.returncode}"
                ),
                exit_code=completed.returncode,
                output=completed.stdout or "",
            )
            _write_json(report_path, report)
            failed_by_target[case.target] += 1
        else:
            try:
                report = _validate_runtime_report(
                    _parse_json_stdout(
                        completed.stdout or "",
                        label=f"Runtime oracle {case.key}",
                    ),
                    target=case.target,
                    profile=case.profile,
                    expected_version=case.exact_version,
                    object_count=case.object_count,
                )
            except Exception as exc:
                report = _failure_report(
                    stage="VALIDATE_ORACLE_REPORT",
                    message=f"{type(exc).__name__}: {exc}",
                    exit_code=completed.returncode,
                    output=completed.stdout or "",
                )
                _write_json(report_path, report)
                failed_by_target[case.target] += 1
            else:
                _write_json(report_path, report)
                passed_by_target[case.target] += 1

        cases[case.key] = {
            "target": case.target,
            "version": case.exact_version,
            "profile": case.profile,
            "scope": case.scope,
            "jsonPath": str(json_path),
            "runtimeReportPath": str(report_path.resolve(strict=False)),
            "report": report,
        }

    total_passed = sum(passed_by_target.values())
    total_failed = sum(failed_by_target.values())
    targets = {
        target: {
            "version": EXACT_VERSION_BY_TARGET[target],
            "expected": EXPECTED_CASE_COUNT_BY_TARGET[target],
            "passed": passed_by_target[target],
            "failed": failed_by_target[target],
        }
        for target in EXACT_VERSION_BY_TARGET
    }
    summary = {
        "status": "passed" if total_failed == 0 else "failed",
        "caseCount": len(POSITIVE_CASES),
        "totalPassed": total_passed,
        "totalFailed": total_failed,
        "externalRuntimesReadOnly": True,
        "runtime43": {
            "root": str(runtime_43.runtime_root),
            "entry": str(runtime_43.entry_path),
            "mode": runtime_43.mode,
            "packageVersion": runtime_43.package_version,
        },
        "targets": targets,
        "blenderReportPath": str(blender_report_path.resolve(strict=False)),
        "cases": cases,
    }
    _write_json(output_root / "acceptance_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    arguments = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        summary = run_runtime_matrix(
            runtime_38_entry=arguments.runtime_38_entry,
            runtime_40_entry=arguments.runtime_40_entry,
            runtime_41_entry=arguments.runtime_41_entry,
            runtime_42_entry=arguments.runtime_42_entry,
            runtime_43_root=arguments.runtime_43_root,
            output_root=arguments.output_root,
            node=arguments.node,
        )
    except AllSpineVersionsAcceptanceError:
        LOGGER.exception("Unable to initialize full runtime matrix")
        return 2
    except Exception:
        LOGGER.exception("Unexpected full runtime matrix failure")
        return 3

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["totalFailed"]:
        print("SPINE_ALL_RUNTIME_ORACLES=FAIL")
        return 1
    print("SPINE_ALL_RUNTIME_ORACLES=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
