"""Export selected production Spine acceptance cases without running the full matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Sequence

import bpy


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
TOOLS_DIRECTORY = REPOSITORY_ROOT / "tools"
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT, TOOLS_DIRECTORY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_all_spine_versions_integration import (  # noqa: E402
    _export_case,
    _prepare_output_directory,
)
from spine_version_acceptance_matrix import (  # noqa: E402
    POSITIVE_CASES,
    SpineVersionAcceptanceCase,
)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case",
        dest="case_keys",
        action="append",
        required=True,
        help="Exact acceptance case key. Repeat --case to export multiple cases.",
    )
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else ()
    return parser.parse_args(arguments)


def _select_cases(case_keys: Sequence[str]) -> tuple[SpineVersionAcceptanceCase, ...]:
    if not isinstance(case_keys, Sequence) or isinstance(case_keys, (str, bytes)):
        raise TypeError("case_keys must be a sequence of strings")
    requested = tuple(case_keys)
    if not requested:
        raise ValueError("At least one case key is required")
    if not all(isinstance(key, str) and key.strip() == key and key for key in requested):
        raise ValueError("Case keys must be non-empty canonical strings")
    if len(requested) != len(set(requested)):
        raise ValueError(f"Case keys cannot be repeated: {requested}")

    by_key = {case.key: case for case in POSITIVE_CASES}
    unknown = tuple(key for key in requested if key not in by_key)
    if unknown:
        raise ValueError(
            f"Unknown production acceptance case keys: {unknown}; "
            f"available={tuple(by_key)}"
        )
    return tuple(by_key[key] for key in requested)


def run(output_directory: Path, case_keys: Sequence[str]) -> Path:
    output_root = _prepare_output_directory(output_directory)
    selected = _select_cases(case_keys)
    cases = tuple(_export_case(output_root, case) for case in selected)
    report = {
        "status": "passed",
        "caseCount": len(cases),
        "selectedCaseKeys": [case.key for case in selected],
        "cases": cases,
    }
    report_path = output_root / "selected_blender_acceptance_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    arguments = _parse_arguments()
    selected = _select_cases(arguments.case_keys)
    print(f"Blender version: {bpy.app.version_string}")
    print(
        f"[SPINE_SELECTED_VERSIONS] RUN {len(selected)} production export cases: "
        + ", ".join(case.key for case in selected)
    )
    report_path = run(arguments.output, arguments.case_keys)
    print(f"[SPINE_SELECTED_VERSIONS] REPORT {report_path}")
    print(f"[SPINE_SELECTED_VERSIONS] PASS {len(selected)} production export cases")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
