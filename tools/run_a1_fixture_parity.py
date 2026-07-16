#!/usr/bin/env python3
"""Run Legacy and Rewrite in isolated Blender processes and compare their outputs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Mapping, Sequence
import uuid


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (  # noqa: E402
    A1ParitySettings,
    compare_a1_exports,
)
from tools.a1_fixture_manifest import (  # noqa: E402
    A1FixtureCase,
    A1FixtureManifest,
    FixtureManifestError,
    case_to_worker_payload,
    load_fixture_manifest,
)


EXIT_COMPATIBLE = 0
EXIT_INCOMPATIBLE = 1
EXIT_INVALID = 2
_EXPORT_WORKER = REPOSITORY_ROOT / "tools" / "blender_a1_fixture_worker.py"
_IMAGE_WORKER = REPOSITORY_ROOT / "tools" / "blender_a1_image_compare.py"


class FixtureRunnerError(RuntimeError):
    """Raised when fixture orchestration cannot be executed reliably."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--blender",
        default=None,
        help="Blender executable path or command; overrides the manifest",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="CASE_ID",
        help="Run only selected case IDs; may be supplied more than once",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing case output directory",
    )
    parser.add_argument(
        "--reuse-source",
        action="store_true",
        help=(
            "Open the original .blend in both processes instead of temporary sibling "
            "copies. Separate Blender processes are still used."
        ),
    )
    parser.add_argument(
        "--keep-source-copies",
        action="store_true",
        help="Keep temporary sibling .blend copies for debugging",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve(strict=False)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(resolved)


def _read_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise FixtureRunnerError(f"{label} does not exist: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise FixtureRunnerError(f"Unable to read {label}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise FixtureRunnerError(
            f"Invalid {label} JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(value, Mapping):
        raise FixtureRunnerError(f"{label} root must be an object")
    return value


def resolve_blender_executable(
    command_line_value: str | None,
    manifest: A1FixtureManifest,
) -> str:
    candidate = command_line_value or manifest.blender_executable or "blender"
    expanded = str(Path(candidate).expanduser()) if any(
        separator in candidate for separator in ("/", "\\")
    ) else candidate
    resolved = shutil.which(expanded)
    if resolved is not None:
        return resolved
    path = Path(expanded).resolve(strict=False)
    if path.is_file():
        return str(path)
    raise FixtureRunnerError(f"Blender executable was not found: {candidate}")


def build_export_command(
    blender_executable: str,
    blend_file: Path,
    payload_json: Path,
    backend: str,
    report_json: Path,
) -> list[str]:
    if backend not in {"LEGACY", "REWRITE"}:
        raise ValueError("backend must be LEGACY or REWRITE")
    return [
        blender_executable,
        "--background",
        "--factory-startup",
        str(blend_file),
        "--python-exit-code",
        "1",
        "--python",
        str(_EXPORT_WORKER),
        "--",
        "--payload-json",
        str(payload_json),
        "--backend",
        backend,
        "--report-json",
        str(report_json),
    ]


def build_image_compare_command(
    blender_executable: str,
    expected_directory: Path,
    actual_directory: Path,
    report_json: Path,
    case: A1FixtureCase,
) -> list[str]:
    parity = case.parity
    return [
        blender_executable,
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        str(_IMAGE_WORKER),
        "--",
        "--expected-dir",
        str(expected_directory),
        "--actual-dir",
        str(actual_directory),
        "--report-json",
        str(report_json),
        "--absolute-tolerance",
        str(parity.image_absolute_tolerance),
        "--max-differing-pixel-ratio",
        str(parity.image_max_differing_pixel_ratio),
        "--max-mean-absolute-delta",
        str(parity.image_max_mean_absolute_delta),
    ]


def _subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    entries = [str(REPOSITORY_ROOT)]
    if existing:
        entries.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment


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


def _copy_source_sibling(source: Path, backend: str) -> Path:
    token = uuid.uuid4().hex
    destination = source.with_name(
        f".{source.stem}.a1-parity-{backend.lower()}-{token}.blend"
    )
    try:
        shutil.copy2(source, destination)
    except OSError as exc:
        raise FixtureRunnerError(
            f"Unable to create sibling fixture copy next to {source}: {exc}. "
            "Use --reuse-source only when the fixture is immutable and backed up."
        ) from exc
    return destination


def _prepare_case_directory(
    root: Path,
    case: A1FixtureCase,
    *,
    overwrite: bool,
) -> Path:
    case_directory = root / case.case_id
    if case_directory.exists():
        if not overwrite:
            raise FixtureRunnerError(
                f"Case output already exists: {case_directory}; use --overwrite"
            )
        shutil.rmtree(case_directory)
    case_directory.mkdir(parents=True, exist_ok=False)
    return case_directory


def _worker_report_or_failure(
    path: Path,
    *,
    return_code: int,
    log_path: Path,
    backend: str,
) -> Mapping[str, Any]:
    if path.is_file():
        report = dict(_read_json_mapping(path, f"{backend} worker report"))
    else:
        report = {
            "success": False,
            "backend": backend,
            "error": "Worker did not create its report",
        }
    report["process_return_code"] = return_code
    report["log_file"] = str(log_path)
    if return_code != 0:
        report["success"] = False
    return report


def _parity_issue_payload(issue: Any) -> dict[str, Any]:
    return {
        "severity": issue.severity.value,
        "code": issue.code,
        "path": issue.path,
        "message": issue.message,
        "expected": issue.expected,
        "actual": issue.actual,
    }


def _compare_json_outputs(
    legacy_path: Path,
    rewrite_path: Path,
    case: A1FixtureCase,
) -> dict[str, Any]:
    legacy = _read_json_mapping(legacy_path, "Legacy export")
    rewrite = _read_json_mapping(rewrite_path, "Rewrite export")
    defaults = A1ParitySettings()
    settings = A1ParitySettings(
        absolute_tolerance=case.parity.absolute_tolerance,
        relative_tolerance=case.parity.relative_tolerance,
        ignored_paths=defaults.ignored_paths + case.parity.ignore_paths,
        compare_animations=case.parity.compare_animations,
        nonessential_mesh_edges_are_errors=case.parity.strict_edges,
    )
    report = compare_a1_exports(legacy, rewrite, settings)
    return {
        "compatible": report.compatible,
        "error_count": report.error_count,
        "warning_count": report.warning_count,
        "legacy_json": str(legacy_path),
        "rewrite_json": str(rewrite_path),
        "settings": asdict(report.settings),
        "legacy_fingerprint": (
            None
            if report.expected_fingerprint is None
            else report.expected_fingerprint.digest()
        ),
        "rewrite_fingerprint": (
            None
            if report.actual_fingerprint is None
            else report.actual_fingerprint.digest()
        ),
        "issues": [_parity_issue_payload(issue) for issue in report.issues],
    }


def _resolved_json_from_worker(report: Mapping[str, Any], backend: str) -> Path:
    value = report.get("expected_json")
    if not isinstance(value, str) or not value:
        raise FixtureRunnerError(f"{backend} worker did not report expected_json")
    path = Path(value).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FixtureRunnerError(f"{backend} final JSON is missing: {path}")
    return path


def _run_backend(
    case: A1FixtureCase,
    backend: str,
    blender_executable: str,
    blend_file: Path,
    case_directory: Path,
) -> Mapping[str, Any]:
    backend_directory = case_directory / backend.lower()
    export_directory = backend_directory / "exports"
    export_directory.mkdir(parents=True, exist_ok=False)
    payload_path = backend_directory / "worker-payload.json"
    report_path = backend_directory / "worker-report.json"
    log_path = backend_directory / "blender.log"
    _write_json_atomic(payload_path, case_to_worker_payload(case, export_directory))
    command = build_export_command(
        blender_executable,
        blend_file,
        payload_path,
        backend,
        report_path,
    )
    return_code = _run_process(command, log_path)
    report = _worker_report_or_failure(
        report_path,
        return_code=return_code,
        log_path=log_path,
        backend=backend,
    )
    _write_json_atomic(report_path, report)
    return report


def run_fixture_case(
    case: A1FixtureCase,
    *,
    blender_executable: str,
    output_root: Path,
    overwrite: bool,
    reuse_source: bool,
    keep_source_copies: bool,
) -> Mapping[str, Any]:
    case_directory = _prepare_case_directory(
        output_root,
        case,
        overwrite=overwrite,
    )
    legacy_source = rewrite_source = case.blend_file
    temporary_sources: list[Path] = []
    if not reuse_source:
        legacy_source = _copy_source_sibling(case.blend_file, "LEGACY")
        temporary_sources.append(legacy_source)
        rewrite_source = _copy_source_sibling(case.blend_file, "REWRITE")
        temporary_sources.append(rewrite_source)

    combined: dict[str, Any] = {
        "case_id": case.case_id,
        "source_blend": str(case.blend_file),
        "mode": case.mode.value,
        "compatible": False,
    }
    try:
        legacy_report = _run_backend(
            case,
            "LEGACY",
            blender_executable,
            legacy_source,
            case_directory,
        )
        rewrite_report = _run_backend(
            case,
            "REWRITE",
            blender_executable,
            rewrite_source,
            case_directory,
        )
        combined["legacy"] = legacy_report
        combined["rewrite"] = rewrite_report

        export_success = bool(legacy_report.get("success")) and bool(
            rewrite_report.get("success")
        )
        combined["export_success"] = export_success
        if export_success:
            json_report = _compare_json_outputs(
                _resolved_json_from_worker(legacy_report, "LEGACY"),
                _resolved_json_from_worker(rewrite_report, "REWRITE"),
                case,
            )
            json_report_path = case_directory / "json-parity-report.json"
            _write_json_atomic(json_report_path, json_report)
            combined["json_parity"] = json_report

            images_relative = Path(*case.settings.images_path.replace("\\", "/").split("/"))
            legacy_images = case_directory / "legacy" / "exports" / images_relative
            rewrite_images = case_directory / "rewrite" / "exports" / images_relative
            image_report_path = case_directory / "image-parity-report.json"
            image_log_path = case_directory / "image-parity-blender.log"
            image_command = build_image_compare_command(
                blender_executable,
                legacy_images,
                rewrite_images,
                image_report_path,
                case,
            )
            image_return_code = _run_process(image_command, image_log_path)
            if image_report_path.is_file():
                image_report = dict(
                    _read_json_mapping(image_report_path, "image parity report")
                )
            else:
                image_report = {
                    "compatible": False,
                    "error": "Image worker did not create a report",
                }
            image_report["process_return_code"] = image_return_code
            image_report["log_file"] = str(image_log_path)
            _write_json_atomic(image_report_path, image_report)
            combined["image_parity"] = image_report

            rewrite_clean = all(
                bool(rewrite_report.get(field_name))
                for field_name in (
                    "source_unchanged",
                    "context_restored",
                    "mesh_restored",
                    "temporary_datablocks_clean",
                )
            )
            combined["rewrite_state_clean"] = rewrite_clean
            combined["compatible"] = bool(json_report.get("compatible")) and bool(
                image_report.get("compatible")
            ) and rewrite_clean
        else:
            combined["json_parity"] = None
            combined["image_parity"] = None
            combined["rewrite_state_clean"] = False

        _write_json_atomic(case_directory / "parity-report.json", combined)
        return combined
    except Exception as exc:
        combined.update(
            {
                "compatible": False,
                "error_type": type(exc).__name__,
                "error": str(exc) or type(exc).__name__,
                "traceback": traceback.format_exc(),
            }
        )
        _write_json_atomic(case_directory / "parity-report.json", combined)
        return combined
    finally:
        if not keep_source_copies:
            for path in temporary_sources:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass


def _selected_cases(
    manifest: A1FixtureManifest,
    requested_ids: Sequence[str],
) -> tuple[A1FixtureCase, ...]:
    if not requested_ids:
        return manifest.cases
    requested = set(requested_ids)
    available = {case.case_id for case in manifest.cases}
    unknown = sorted(requested - available)
    if unknown:
        raise FixtureRunnerError("Unknown case IDs: " + ", ".join(unknown))
    return tuple(case for case in manifest.cases if case.case_id in requested)


def run(arguments: Sequence[str] | None = None) -> int:
    namespace = _build_parser().parse_args(arguments)
    try:
        manifest = load_fixture_manifest(namespace.manifest)
        blender = resolve_blender_executable(namespace.blender, manifest)
        output_root = namespace.output_root.expanduser().resolve(strict=False)
        output_root.mkdir(parents=True, exist_ok=True)
        cases = _selected_cases(manifest, namespace.case)
        reports: list[Mapping[str, Any]] = []
        for case in cases:
            if not namespace.quiet:
                print(f"[A1 PARITY] RUN {case.case_id}")
            report = run_fixture_case(
                case,
                blender_executable=blender,
                output_root=output_root,
                overwrite=namespace.overwrite,
                reuse_source=namespace.reuse_source,
                keep_source_copies=namespace.keep_source_copies,
            )
            reports.append(report)
            if not namespace.quiet:
                status = "COMPATIBLE" if report.get("compatible") else "INCOMPATIBLE"
                print(f"[A1 PARITY] {status} {case.case_id}")

        summary = {
            "schema_version": 1,
            "manifest": str(namespace.manifest.expanduser().resolve(strict=False)),
            "blender_executable": blender,
            "case_count": len(reports),
            "compatible_case_count": sum(
                1 for report in reports if report.get("compatible")
            ),
            "compatible": all(bool(report.get("compatible")) for report in reports),
            "cases": [
                {
                    "case_id": report.get("case_id"),
                    "compatible": bool(report.get("compatible")),
                    "report": str(
                        output_root
                        / str(report.get("case_id"))
                        / "parity-report.json"
                    ),
                }
                for report in reports
            ],
        }
        _write_json_atomic(output_root / "parity-summary.json", summary)
        print(
            "A1 fixture parity: "
            + ("COMPATIBLE" if summary["compatible"] else "INCOMPATIBLE")
            + f" ({summary['compatible_case_count']}/{summary['case_count']} cases)"
        )
        return EXIT_COMPATIBLE if summary["compatible"] else EXIT_INCOMPATIBLE
    except (FixtureManifestError, FixtureRunnerError, OSError, TypeError, ValueError) as exc:
        print(f"A1 fixture parity input error: {exc}", file=sys.stderr)
        return EXIT_INVALID


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
