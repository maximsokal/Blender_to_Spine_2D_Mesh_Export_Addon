from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _run_help(script_name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "tools" / script_name), "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_memory_plateau_gate_direct_script_bootstraps_tools_import():
    completed = _run_help("run_memory_plateau_gate.py")

    assert completed.returncode == 0, completed.stdout
    assert "--fixture" in completed.stdout
    assert "--work-root" in completed.stdout


def test_public_fixture_runner_direct_script_bootstraps_tools_import():
    completed = _run_help("run_public_blend_fixture_generation.py")

    assert completed.returncode == 0, completed.stdout
    assert "--output-root" in completed.stdout
