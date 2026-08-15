#!/usr/bin/env python3
"""Save or verify installed Spine2D exact-version AddonPreferences in Blender."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import traceback

import bpy


def _arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True)
    parser.add_argument("--mode", required=True, choices=("save", "verify"))
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args(values)


def _write_report(path: Path, payload: dict[str, object]) -> None:
    resolved = path.resolve(strict=False)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(resolved)


def _custom_exact_version(default_version: str) -> str:
    major_text, minor_text, patch_text = default_version.split(".")
    patch = int(patch_text)
    custom_patch = patch - 1 if patch > 0 else patch + 1
    return f"{int(major_text)}.{int(minor_text)}.{custom_patch}"


def _run(namespace: argparse.Namespace) -> dict[str, object]:
    module = importlib.import_module(namespace.module)
    helper = importlib.import_module(
        f"{namespace.module}.blender_adapter.spine_version_preferences"
    )
    preferences = helper.get_spine_addon_preferences(required=True)
    expected = {
        spec.target.value: _custom_exact_version(spec.default_version)
        for spec in helper.SPINE_EXACT_VERSION_PREFERENCE_SPECS
    }

    if namespace.mode == "save":
        for spec in helper.SPINE_EXACT_VERSION_PREFERENCE_SPECS:
            helper.assign_spine_project_exact_version(
                preferences,
                spec.target,
                expected[spec.target.value],
            )
        save_result = set(bpy.ops.wm.save_userpref())
        if "FINISHED" not in save_result:
            raise RuntimeError(
                f"bpy.ops.wm.save_userpref returned {sorted(save_result)!r}"
            )
    else:
        save_result = set()

    actual = {
        spec.target.value: str(getattr(preferences, spec.property_name))
        for spec in helper.SPINE_EXACT_VERSION_PREFERENCE_SPECS
    }
    if actual != expected:
        raise RuntimeError(
            "Exact Spine project preferences differ after "
            f"{namespace.mode}: expected={expected!r}, actual={actual!r}"
        )

    return {
        "status": "passed",
        "mode": namespace.mode,
        "module": module.__name__,
        "expected": expected,
        "actual": actual,
        "save_operator_result": sorted(save_result),
    }


def main() -> None:
    namespace = _arguments()
    try:
        payload = _run(namespace)
    except Exception as exc:
        payload = {
            "status": "failed",
            "mode": namespace.mode,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_report(namespace.report_json, payload)
        raise
    _write_report(namespace.report_json, payload)
    print(
        "[SPINE-VERSION-PREFERENCES] PASS "
        f"mode={namespace.mode} values={payload['actual']!r}",
        flush=True,
    )


if __name__ == "__main__":
    main()
