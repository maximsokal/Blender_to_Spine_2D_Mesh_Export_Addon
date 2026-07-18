#!/usr/bin/env python3
"""Audit every production Python file in the A1 Rewrite pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIRECTORY = REPOSITORY_ROOT / "Blender_to_Spine2D_Mesh_Exporter"
PACKAGE_NAME = "Blender_to_Spine2D_Mesh_Exporter"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Blender_to_Spine2D_Mesh_Exporter.infrastructure.pipeline_static_audit import (  # noqa: E402
    audit_pipeline_package,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-directory",
        type=Path,
        default=PACKAGE_DIRECTORY,
        help="Addon package directory to scan",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON report path",
    )
    parser.add_argument(
        "--focus-module",
        action="append",
        default=[],
        help="Only audit module/file paths containing this text; repeatable",
    )
    parser.add_argument(
        "--fail-on",
        choices=("never", "error", "warning"),
        default="error",
        help="Process exit policy",
    )
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


def _print_summary(report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    print(
        "A1 pipeline audit: "
        f"modules={summary['module_count']} "
        f"errors={summary['error_count']} "
        f"warnings={summary['warning_count']} "
        f"info={summary['info_count']}"
    )
    for item in report["weak_spots"][:20]:
        print(
            f"- {item['relative_path']}: score={item['score']} "
            f"findings={item['finding_count']}"
        )
        for finding in item["top_findings"][:5]:
            function = "" if finding["function"] is None else f" {finding['function']}()"
            print(
                f"    {finding['severity']} {finding['code']} "
                f"line {finding['line']}{function}: {finding['message']}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    namespace = _build_parser().parse_args(argv)
    report = audit_pipeline_package(
        namespace.package_directory,
        package_name=PACKAGE_NAME,
        focus_modules=tuple(namespace.focus_module),
    )
    _print_summary(report)
    if namespace.output_json is not None:
        _write_json_atomic(namespace.output_json, report)

    summary = report["summary"]
    if namespace.fail_on == "warning" and (
        summary["error_count"] or summary["warning_count"]
    ):
        return 1
    if namespace.fail_on == "error" and summary["error_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
