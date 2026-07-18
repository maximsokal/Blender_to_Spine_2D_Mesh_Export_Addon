#!/usr/bin/env python3
"""Run one manifest case through Blender with per-file A1 pipeline tracing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from typing import Any, Mapping, Sequence
from uuid import uuid4


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIRECTORY = REPOSITORY_ROOT / "Blender_to_Spine2D_Mesh_Exporter"
PROBE_WORKER = REPOSITORY_ROOT / "tools" / "blender_a1_pipeline_probe.py"
STATIC_AUDIT = REPOSITORY_ROOT / "tools" / "audit_a1_pipeline.py"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Blender_to_Spine2D_Mesh_Exporter.infrastructure.pipeline_static_audit import (  # noqa: E402
    audit_pipeline_package,
)
from tools.a1_fixture_manifest import (  # noqa: E402
    A1FixtureCase,
    case_to_worker_payload,
    load_fixture_manifest,
)


class PipelineProbeRunnerError(RuntimeError):
    """Raised when a pipeline probe cannot be started or its report is incomplete."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--case", required=True, metavar="CASE_ID")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--blender", default=None)
    parser.add_argument("--backend", choices=("REWRITE", "LEGACY"), default="REWRITE")
    parser.add_argument("--focus-module", action="append", default=[])
    parser.add_argument("--focus-file", action="append", default=[])
    parser.add_argument("--max-events", type=int, default=250_000)
    parser.add_argument("--capture-values", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-source", action="store_true")
    parser.add_argument("--keep-source-copy", action="store_true")
    return parser


def _resolve_blender(candidate: str | None, manifest_value: str | None) -> str:
    value = candidate or manifest_value or "blender"
    resolved = shutil.which(value)
    if resolved is not None:
        return resolved
    path = Path(value).expanduser().resolve(strict=False)
    if path.is_file():
        return str(path)
    raise PipelineProbeRunnerError(f"Blender executable was not found: {value}")


def _selected_case(manifest: Any, case_id: str) -> A1FixtureCase:
    matches = [case for case in manifest.cases if case.case_id == case_id]
    if not matches:
        available = ", ".join(case.case_id for case in manifest.cases)
        raise PipelineProbeRunnerError(
            f"Case '{case_id}' was not found. Available cases: {available}"
        )
    return matches[0]


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve(strict=False)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(resolved)


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise PipelineProbeRunnerError(f"Unable to read report {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PipelineProbeRunnerError(
            f"Invalid report JSON at {exc.lineno}:{exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(value, Mapping):
        raise PipelineProbeRunnerError("Pipeline probe report root must be an object")
    return value


def _subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    current = environment.get("PYTHONPATH")
    values = [str(REPOSITORY_ROOT)]
    if current:
        values.append(current)
    environment["PYTHONPATH"] = os.pathsep.join(values)
    return environment


def _source_copy(source: Path) -> Path:
    destination = source.with_name(
        f".{source.stem}.a1-pipeline-probe-{uuid4().hex}.blend"
    )
    try:
        shutil.copy2(source, destination)
    except OSError as exc:
        raise PipelineProbeRunnerError(
            f"Unable to create a sibling probe copy next to {source}: {exc}"
        ) from exc
    return destination


def _command(
    *,
    blender: str,
    blend_file: Path,
    payload_json: Path,
    report_json: Path,
    backend: str,
    focus_modules: Sequence[str],
    focus_files: Sequence[str],
    max_events: int,
    capture_values: bool,
) -> list[str]:
    command = [
        blender,
        "--background",
        "--factory-startup",
        str(blend_file),
        "--python-exit-code",
        "1",
        "--python",
        str(PROBE_WORKER),
        "--",
        "--payload-json",
        str(payload_json),
        "--report-json",
        str(report_json),
        "--backend",
        backend,
        "--max-events",
        str(max_events),
    ]
    for value in focus_modules:
        command.extend(("--focus-module", value))
    for value in focus_files:
        command.extend(("--focus-file", value))
    if capture_values:
        command.append("--capture-values")
    return command


def _run_process(command: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("Command:\n")
        log.write(" ".join(json.dumps(part) for part in command) + "\n\n")
        log.flush()
        completed = subprocess.run(
            list(command),
            cwd=REPOSITORY_ROOT,
            env=_subprocess_environment(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(completed.returncode)


def run_probe(namespace: argparse.Namespace) -> Mapping[str, Any]:
    manifest = load_fixture_manifest(namespace.manifest)
    case = _selected_case(manifest, namespace.case)
    blender = _resolve_blender(namespace.blender, manifest.blender_executable)
    output_root = namespace.output_root.expanduser().resolve(strict=False)
    case_root = output_root / case.case_id
    if case_root.exists():
        if not namespace.overwrite:
            raise PipelineProbeRunnerError(
                f"Probe output exists: {case_root}; pass --overwrite"
            )
        shutil.rmtree(case_root)
    export_root = case_root / "exports"
    export_root.mkdir(parents=True, exist_ok=False)

    payload_json = case_root / "worker-payload.json"
    report_json = case_root / "pipeline-trace-report.json"
    static_json = case_root / "pipeline-static-audit.json"
    summary_json = case_root / "pipeline-probe-summary.json"
    blender_log = case_root / "blender.log"
    _write_json_atomic(payload_json, case_to_worker_payload(case, export_root))

    static_report = audit_pipeline_package(
        PACKAGE_DIRECTORY,
        package_name="Blender_to_Spine2D_Mesh_Exporter",
        focus_modules=tuple(namespace.focus_module) + tuple(namespace.focus_file),
    )
    _write_json_atomic(static_json, static_report)

    source = case.blend_file
    temporary_source: Path | None = None
    if not namespace.reuse_source:
        temporary_source = _source_copy(source)
        source = temporary_source
    try:
        command = _command(
            blender=blender,
            blend_file=source,
            payload_json=payload_json,
            report_json=report_json),
            backend=namespace.backend,
            focus_modules=namespace.focus_module,
            focus_files=namespace.focus_file,
            max_events=namespace.max_events,
            capture_values=namespace.capture_values,
        )
        return_code = _run_process(command, blender_log)
        runtime_report = (
            _read_json(report_json)
            if report_json.is_file()
            else {
                "success": False,
                "error": {"message": "Blender worker did not create a report"},
            }
        )
        summary = {
            "success": return_code == 0 and bool(runtime_report.get("success")),
            "case_id": case.case_id,
            "backend": namespace.backend,
            "source_blend": str(case.blend_file),
            "process_return_code": return_code,
            "blender_log": str(blender_log),
            "runtime_report": str(report_json),
            "static_audit": str(static_json),
            "runtime_summary": runtime_report.get("race", {}).get("summary"),
            "static_summary": static_report.get("summary"),
            "missing_expected_calls": runtime_report.get("trace", {}).get(
                "missing_expected_calls", []
            ),
        }
        _write_json_atomic(summary_json, summary)
        return summary
    finally:
        if temporary_source is not None and temporary_source.exists():
            if namespace.keep_source_copy:
                print(f"Kept probe source copy: {temporary_source}")
            else:
                temporary_source.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    namespace = _build_parser().parse_args(argv)
    try:
        summary = run_probe(namespace)
    except Exception as exc:
        traceback.print_exc()
        print(f"Pipeline probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
