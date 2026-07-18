#!/usr/bin/env python3
"""Run one A1 fixture inside Blender and record per-file runtime data flow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence

import bpy


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIRECTORY = REPOSITORY_ROOT / "Blender_to_Spine2D_Mesh_Exporter"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import Blender_to_Spine2D_Mesh_Exporter as addon  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.infrastructure.pipeline_trace import (  # noqa: E402
    PipelineTraceSession,
)
from tools import blender_a1_fixture_worker as worker  # noqa: E402


def _arguments_after_separator(argv: Sequence[str]) -> list[str]:
    try:
        separator = argv.index("--")
    except ValueError:
        return []
    return list(argv[separator + 1 :])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-json", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--backend", choices=("REWRITE", "LEGACY"), default="REWRITE")
    parser.add_argument(
        "--focus-module",
        action="append",
        default=[],
        help="Module or file substring to isolate in the focus section; repeatable",
    )
    parser.add_argument(
        "--focus-file",
        action="append",
        default=[],
        help="Addon-relative .py path to isolate; repeatable",
    )
    parser.add_argument("--max-events", type=int, default=250_000)
    parser.add_argument(
        "--capture-values",
        action="store_true",
        help="Include bounded scalar/path values in signatures; types/shapes are always recorded",
    )
    return parser


def _scenario(payload: Mapping[str, Any]) -> str:
    mode = str(payload.get("mode", "")).lower()
    if mode == "single":
        return "single"
    selected = payload.get("selected_objects")
    connected = payload.get("connected_objects")
    selected_count = len(selected) if isinstance(selected, list) else 0
    connected_count = len(connected) if isinstance(connected, list) else 0
    if connected_count >= 2 and connected_count < selected_count:
        return "mixed"
    if connected_count >= 2:
        return "multi_connected"
    return "multi_standalone"


def _expected_calls(scenario: str, backend: str) -> tuple[tuple[str, str], ...]:
    if backend == "LEGACY":
        return ()
    object_preparation = (
        ("blender_adapter.a1_object_preparation", "prepare_a1_object"),
        (
            "blender_adapter.a1_source_geometry_preparation",
            "prepare_a1_source_geometry",
        ),
        ("blender_adapter.a1_uv_preparation", "prepare_a1_uv"),
        ("blender_adapter.a1_texture_planning", "prepare_a1_texture_plan"),
        ("blender_adapter.a1_document_preparation", "prepare_a1_document"),
    )
    shared = (
        *object_preparation,
        ("blender_adapter.texture_executor", "stage_texture_plan_outputs"),
        (
            "blender_adapter.a1_projection_finalization",
            "finalize_prepared_camera_projection",
        ),
        ("infrastructure.atomic_files", "AtomicFileTransaction.commit"),
    )
    if scenario == "single":
        return (
            ("blender_adapter.a1_ui_bridge", "export_active_object_a1"),
            ("blender_adapter.a1_single_object_export", "export_a1_single_object"),
            *shared,
        )
    multi_output = (
        ("blender_adapter.a1_output_staging", "stage_and_finalize_a1_objects"),
        (
            "blender_adapter.a1_output_statistics",
            "record_final_document_statistics",
        ),
    )
    if scenario == "mixed":
        return (
            ("blender_adapter.a1_ui_bridge", "export_selected_objects_a1"),
            ("blender_adapter.a1_mixed_object_output", "export_a1_mixed_object"),
            ("blender_adapter.a1_mixed_object_export", "prepare_a1_mixed_object"),
            (
                "blender_adapter.a1_multi_object_composition",
                "compose_a1_multi_object_document",
            ),
            *multi_output,
            *shared,
        )
    return (
        ("blender_adapter.a1_ui_bridge", "export_selected_objects_a1"),
        ("blender_adapter.a1_multi_object_output", "export_a1_multi_object"),
        ("blender_adapter.a1_multi_object_export", "prepare_a1_multi_object"),
        (
            "blender_adapter.a1_multi_object_composition",
            "compose_a1_multi_object_document",
        ),
        *multi_output,
        *shared,
    )


def _normalize_focus(values: Sequence[str], files: Sequence[str]) -> tuple[str, ...]:
    result = [str(value).strip() for value in values if str(value).strip()]
    for raw in files:
        normalized = str(raw).replace("\\", "/").strip()
        prefix = "Blender_to_Spine2D_Mesh_Exporter/"
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
        if normalized.endswith(".py"):
            normalized = normalized[:-3]
        normalized = normalized.replace("/", ".")
        if normalized.endswith(".__init__"):
            normalized = normalized[: -len(".__init__")]
        if normalized:
            result.append(normalized)
    return tuple(dict.fromkeys(result))


def _run(namespace: argparse.Namespace) -> dict[str, Any]:
    payload = worker._load_payload(namespace.payload_json)
    scenario = _scenario(payload)
    focus = _normalize_focus(namespace.focus_module, namespace.focus_file)
    session = PipelineTraceSession(
        PACKAGE_DIRECTORY,
        package_name=addon.__name__,
        focus_modules=focus,
        max_events=namespace.max_events,
        capture_values=namespace.capture_values,
    )
    worker_report: Mapping[str, Any] | None = None
    run_error: dict[str, Any] | None = None
    try:
        with session:
            worker_report = worker._run(payload, namespace.backend)
    except Exception as exc:
        run_error = {
            "type": type(exc).__name__,
            "message": str(exc) or type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
    trace_report = session.build_report(
        run_success=run_error is None and bool(worker_report and worker_report.get("success")),
        run_error=run_error,
        scenario=scenario,
        expected_calls=_expected_calls(scenario, namespace.backend),
    )
    contract_success = not trace_report["missing_expected_calls"]
    report = {
        "success": trace_report["run_success"] and contract_success,
        "contract_success": contract_success,
        "backend": namespace.backend,
        "scenario": scenario,
        "blender_version": bpy.app.version_string,
        "source_blend": str(Path(bpy.data.filepath).resolve(strict=False)),
        "worker": None if worker_report is None else dict(worker_report),
        "trace": trace_report,
    }
    if run_error is not None:
        report["error"] = run_error
    return report


def main() -> None:
    namespace = _build_parser().parse_args(_arguments_after_separator(sys.argv))
    report_path = namespace.report_json.expanduser().resolve(strict=False)
    report: Mapping[str, Any] | None = None
    try:
        report = _run(namespace)
        worker._write_json_atomic(report_path, report)
        if not report.get("success"):
            raise RuntimeError("Pipeline probe completed with export/contract failures")
    except Exception:
        traceback.print_exc()
        if report is None:
            failure = {
                "success": False,
                "backend": namespace.backend,
                "blender_version": bpy.app.version_string,
                "error": {
                    "type": sys.exc_info()[0].__name__ if sys.exc_info()[0] else "Error",
                    "message": str(sys.exc_info()[1]),
                    "traceback": traceback.format_exc(),
                },
            }
            try:
                worker._write_json_atomic(report_path, failure)
            except Exception:
                traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
