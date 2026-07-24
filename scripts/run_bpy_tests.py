"""Run real Blender 5.2 integration tests through the official bpy wheel.

This runner intentionally refuses unsupported Python or bpy versions. It prevents a
misconfigured environment from reporting a successful suite after importing test mocks.
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
import sys


EXPECTED_PYTHON = (3, 13)
EXPECTED_BPY_DISTRIBUTION = "5.2.0"
EXPECTED_BLENDER = (5, 2, 0)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = REPOSITORY_ROOT / "tests_bpy"


def _fail(message: str) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return 2


def main(arguments: list[str] | None = None) -> int:
    """Validate the runtime, then execute only the real-bpy test root."""

    if sys.version_info[:2] != EXPECTED_PYTHON:
        return _fail(
            "bpy 5.2.0 tests require CPython 3.13; "
            f"current interpreter is {sys.version_info.major}.{sys.version_info.minor}"
        )

    try:
        installed_bpy = metadata.version("bpy")
    except metadata.PackageNotFoundError:
        return _fail(
            "official bpy package is not installed; run "
            "'python -m pip install -r requirements-bpy.txt'"
        )
    if installed_bpy != EXPECTED_BPY_DISTRIBUTION:
        return _fail(
            f"expected bpy distribution {EXPECTED_BPY_DISTRIBUTION}, got {installed_bpy}"
        )

    try:
        import bpy
        import bmesh  # noqa: F401 - import is part of the runtime gate
        import pytest
    except Exception as exc:
        return _fail(f"unable to import the real Blender Python runtime: {exc}")

    runtime_version = tuple(int(value) for value in bpy.app.version[:3])
    if runtime_version != EXPECTED_BLENDER:
        return _fail(
            f"expected Blender runtime {EXPECTED_BLENDER}, got {runtime_version}"
        )
    if not TEST_ROOT.is_dir():
        return _fail(f"test directory does not exist: {TEST_ROOT}")

    pytest_arguments = [
        str(TEST_ROOT),
        "-q",
        "--strict-markers",
        "--maxfail=1",
    ]
    if arguments:
        pytest_arguments.extend(arguments)
    return int(pytest.main(pytest_arguments))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
