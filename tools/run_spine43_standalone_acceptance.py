#!/usr/bin/env python3
"""Run real Blender 5.2 Spine 4.3 standalone exports for both rig profiles."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Mapping, Sequence


LOGGER = logging.getLogger("spine43_standalone_acceptance")
ROOT = Path(__file__).resolve().parents[1]
BLENDER_WORKER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_spine43_standalone_profiles_integration.py"
)
EXPECTED_VERSION = "4.3.23"
EXPECTED_CASES: Mapping[str, Mapping[str, object]] = {
    "TWO_AXIS_ROTATION_SCALE": {
        "directory": "two_axis",
        "stem": "Spine43TwoAxisStandaloneMulti",
        "bones": 52,
        "ik": 3,
        "transform": 12,
    },
    "LEGACY_ROTATABLE_MESH": {
        "directory": "three_axis",
        "stem": "Spine43ThreeAxisStandaloneMulti",
        "bones": 46,
        "ik": 3,
        "transform": 15,
    },
}


class Spine43StandaloneAcceptanceError(RuntimeError):
    """Raised when one structural acceptance step is invalid or fails."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blender",
        type=Path,
        required=True,
        help="Path to the Blender 5.2+ executable.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory that receives both 4.3 profile exports and reports.",
    )
    parser.add_argument(
        "--replace-output",
        action="store_true",
        help="Replace a non-empty dedicated output directory after safety checks.",
    )
    return parser


def _resolve_required_file(value: Path, *, label: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{label} must be pathlib.Path")
    resolved = value.expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise Spine43StandaloneAcceptanceError(f"{label} does not exist: {resolved}")
    return resolved


def _dangerous_replace_target(path: Path) -> bool:
    """Return True for filesystem roots or paths too broad to delete safely."""

    resolved = path.resolve(strict=False)
    anchor = Path(resolved.anchor) if resolved.anchor else None
    if resolved.parent == resolved:
        return True
    if anchor is not None:
        try:
            relative_parts = resolved.relative_to(anchor).parts
        except ValueError:
            return True
        if len(relative_parts) < 2:
            return True
    return False


def prepare_output_root(value: Path, *, replace: bool) -> Path:
    """Create one dedicated output directory and never delete a broad path."""

    if not isinstance(value, Path):
        raise TypeError("output_root must be pathlib.Path")
    if not isinstance(replace, bool):
        raise TypeError("replace must be bool")
    resolved = value.expanduser().resolve(strict=False)
    if resolved.exists() and not resolved.is_dir():
        raise Spine43StandaloneAcceptanceError(
            f"Output root exists but is not a directory: {resolved}"
        )
    if resolved.exists() and any(resolved.iterdir()):
        if not replace:
            raise Spine43StandaloneAcceptanceError(
                f"Output root is not empty; pass --replace-output: {resolved}"
            )
        if _dangerous_replace_target(resolved):
            raise Spine43StandaloneAcceptanceError(
                f"Refusing to replace a broad or dangerous directory: {resolved}"
            )
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def build_blender_command(blender: Path, output_root: Path) -> tuple[str, ...]:
    """Return the exact Blender worker command used by this gate."""

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
) -> subprocess.CompletedProcess[str]:
    LOGGER.info("%s: %s", label, subprocess.list2cmdline(tuple(command)))
    try:
        completed = subprocess.run(
            tuple(command),
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise Spine43StandaloneAcceptanceError(
            f"Unable to execute {label}: {exc}"
        ) from exc
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.returncode != 0:
        raise Spine43StandaloneAcceptanceError(
            f"{label} failed with exit code {completed.returncode}"
        )
    return completed


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    if not path.is_file():
        raise Spine43StandaloneAcceptanceError(f"{label} was not created: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Spine43StandaloneAcceptanceError(
            f"Unable to read {label} {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise Spine43StandaloneAcceptanceError(f"{label} root must be a JSON object")
    return payload


def _require_non_negative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Spine43StandaloneAcceptanceError(f"{label} must be a non-negative integer")
    return value


def _validate_generated_json(
    path: Path,
    *,
    expected_constraint_count: int,
) -> None:
    """Recheck the persisted target boundary independently of Blender worker state."""

    payload = _load_json_object(path, label="Spine 4.3 JSON")
    skeleton = payload.get("skeleton")
    if not isinstance(skeleton, dict) or skeleton.get("spine") != EXPECTED_VERSION:
        raise Spine43StandaloneAcceptanceError(
            f"Generated JSON has unexpected skeleton metadata: {skeleton}"
        )
    for legacy_field in ("ik", "transform", "path", "physics", "slider"):
        if legacy_field in payload:
            raise Spine43StandaloneAcceptanceError(
                f"Generated JSON retained legacy root field {legacy_field!r}"
            )
    constraints = payload.get("constraints")
    if not isinstance(constraints, list) or len(constraints) != expected_constraint_count:
        raise Spine43StandaloneAcceptanceError(
            "Generated JSON unified constraint count differs: "
            f"expected={expected_constraint_count}, actual="
            f"{len(constraints) if isinstance(constraints, list) else constraints!r}"
        )
    for index, constraint in enumerate(constraints):
        if not isinstance(constraint, dict):
            raise Spine43StandaloneAcceptanceError(
                f"constraints[{index}] must be a JSON object"
            )
        if constraint.get("type") not in {"ik", "transform"}:
            raise Spine43StandaloneAcceptanceError(
                f"constraints[{index}] has unsupported type {constraint.get('type')!r}"
            )
        if "order" in constraint:
            raise Spine43StandaloneAcceptanceError(
                f"constraints[{index}] retained legacy order"
            )


def validate_blender_report(
    report: dict[str, object],
    *,
    output_root: Path,
) -> dict[str, object]:
    """Validate worker evidence and every persisted artifact for both profiles."""

    if not isinstance(report, dict):
        raise TypeError("report must be dict")
    if not isinstance(output_root, Path):
        raise TypeError("output_root must be pathlib.Path")
    if report.get("status") != "passed":
        raise Spine43StandaloneAcceptanceError(
            f"Blender acceptance report did not pass: {report}"
        )
    if report.get("version") != EXPECTED_VERSION:
        raise Spine43StandaloneAcceptanceError(
            f"Blender report version mismatch: {report.get('version')!r}"
        )
    if report.get("mode") != "STANDALONE":
        raise Spine43StandaloneAcceptanceError(
            f"Blender report mode mismatch: {report.get('mode')!r}"
        )
    if report.get("runtimeValidated") is not False:
        raise Spine43StandaloneAcceptanceError(
            "Structural gate must not claim unavailable Spine 4.3 runtime validation"
        )
    if report.get("manualEditorImportRequired") is not True:
        raise Spine43StandaloneAcceptanceError(
            "Blender report must require manual Spine Editor 4.3 import"
        )

    raw_profiles = report.get("profiles")
    if not isinstance(raw_profiles, list):
        raise Spine43StandaloneAcceptanceError("Blender report profiles must be an array")
    profiles_by_name: dict[str, dict[str, object]] = {}
    for index, value in enumerate(raw_profiles):
        if not isinstance(value, dict):
            raise Spine43StandaloneAcceptanceError(
                f"Blender report profiles[{index}] must be an object"
            )
        profile_name = value.get("profile")
        if not isinstance(profile_name, str) or not profile_name:
            raise Spine43StandaloneAcceptanceError(
                f"Blender report profiles[{index}] has no profile"
            )
        if profile_name in profiles_by_name:
            raise Spine43StandaloneAcceptanceError(
                f"Duplicate Blender report profile: {profile_name}"
            )
        profiles_by_name[profile_name] = value

    if set(profiles_by_name) != set(EXPECTED_CASES):
        raise Spine43StandaloneAcceptanceError(
            "Blender report profile inventory differs: "
            f"expected={tuple(EXPECTED_CASES)}, actual={tuple(profiles_by_name)}"
        )

    for profile_name, expected in EXPECTED_CASES.items():
        profile_report = profiles_by_name[profile_name]
        if profile_report.get("status") != "passed":
            raise Spine43StandaloneAcceptanceError(
                f"Profile {profile_name} did not pass: {profile_report}"
            )
        for field_name in ("bones", "ik", "transform"):
            actual = _require_non_negative_int(
                profile_report.get(field_name),
                label=f"{profile_name}.{field_name}",
            )
            if actual != expected[field_name]:
                raise Spine43StandaloneAcceptanceError(
                    f"{profile_name}.{field_name} mismatch: "
                    f"expected={expected[field_name]}, actual={actual}"
                )
        expected_constraints = int(expected["ik"]) + int(expected["transform"])
        if profile_report.get("constraints") != expected_constraints:
            raise Spine43StandaloneAcceptanceError(
                f"{profile_name}.constraints mismatch: {profile_report.get('constraints')!r}"
            )
        if profile_report.get("connectedWrapperPresent") is not False:
            raise Spine43StandaloneAcceptanceError(
                f"{profile_name} contains a connected wrapper"
            )
        if profile_report.get("crossObjectReferencesPresent") is not False:
            raise Spine43StandaloneAcceptanceError(
                f"{profile_name} contains cross-object references"
            )
        if profile_report.get("legacyRootConstraintCollectionsPresent") is not False:
            raise Spine43StandaloneAcceptanceError(
                f"{profile_name} retained legacy root constraint collections"
            )
        if profile_report.get("legacyConstraintOrderPresent") is not False:
            raise Spine43StandaloneAcceptanceError(
                f"{profile_name} retained legacy constraint order"
            )

        case_directory = output_root / str(expected["directory"])
        json_path = case_directory / f"{expected['stem']}.json"
        reported_json_path = profile_report.get("jsonPath")
        if not isinstance(reported_json_path, str):
            raise Spine43StandaloneAcceptanceError(
                f"{profile_name}.jsonPath must be a string"
            )
        if Path(reported_json_path).resolve(strict=False) != json_path.resolve(strict=False):
            raise Spine43StandaloneAcceptanceError(
                f"{profile_name}.jsonPath differs: {reported_json_path!r}"
            )
        _validate_generated_json(
            json_path,
            expected_constraint_count=expected_constraints,
        )

        texture_paths = profile_report.get("texturePaths")
        if not isinstance(texture_paths, list) or len(texture_paths) != 3:
            raise Spine43StandaloneAcceptanceError(
                f"{profile_name} must report exactly three textures"
            )
        for texture_index, raw_path in enumerate(texture_paths):
            if not isinstance(raw_path, str) or not Path(raw_path).is_file():
                raise Spine43StandaloneAcceptanceError(
                    f"{profile_name} texture[{texture_index}] does not exist: {raw_path!r}"
                )

    return report


def run_acceptance(
    *,
    blender: Path,
    output_root: Path,
    replace_output: bool = False,
) -> dict[str, object]:
    """Run Blender production exports and validate the structural evidence."""

    blender = _resolve_required_file(blender, label="Blender executable")
    _resolve_required_file(BLENDER_WORKER, label="Blender acceptance worker")
    output_root = prepare_output_root(output_root, replace=replace_output)

    _run_command(
        build_blender_command(blender, output_root),
        label="Blender Spine 4.3 standalone profile exports",
    )

    blender_report_path = output_root / "blender_acceptance_report.json"
    blender_report = validate_blender_report(
        _load_json_object(blender_report_path, label="Blender acceptance report"),
        output_root=output_root,
    )

    summary = {
        "status": "passed",
        "version": EXPECTED_VERSION,
        "mode": "STANDALONE",
        "profiles": tuple(EXPECTED_CASES),
        "blenderReportPath": str(blender_report_path.resolve()),
        "runtimeValidated": False,
        "manualEditorImportRequired": True,
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
            output_root=arguments.output_root,
            replace_output=arguments.replace_output,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("SPINE43_STANDALONE_STRUCTURAL_ACCEPTANCE=PASS")
        return 0
    except Spine43StandaloneAcceptanceError:
        LOGGER.exception("Spine 4.3 standalone structural acceptance failed")
        return 1
    except Exception:
        LOGGER.exception("Unexpected Spine 4.3 structural acceptance failure")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
