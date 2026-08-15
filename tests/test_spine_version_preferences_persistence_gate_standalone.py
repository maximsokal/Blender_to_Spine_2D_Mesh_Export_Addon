"""Standalone bootstrap contract for the Blender preference persistence gate."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "run_spine_version_preferences_persistence_gate.py"


def test_spine_version_preferences_persistence_gate_direct_help_bootstraps() -> None:
    completed = subprocess.run(
        [sys.executable, str(TOOL), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--output-root" in completed.stdout
    assert "--blender" in completed.stdout
    assert "ModuleNotFoundError" not in completed.stderr
