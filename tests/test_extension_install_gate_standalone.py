"""The extension lifecycle gate must run both as a module and a direct script."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_direct_script_help_bootstraps_prepare_package_import(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "tools" / "run_extension_install_gate.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout
    assert "--output-root" in completed.stdout
    assert "ModuleNotFoundError" not in completed.stdout
