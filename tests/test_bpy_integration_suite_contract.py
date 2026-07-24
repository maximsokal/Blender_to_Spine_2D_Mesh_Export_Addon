"""Fast checks keeping the optional real-bpy suite aligned with the extension runtime."""

from __future__ import annotations

import ast
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
BPY_TESTS = ROOT / "tests_bpy"
RUNNER = ROOT / "scripts" / "run_bpy_tests.py"
REQUIREMENTS = ROOT / "requirements-bpy.txt"


def test_bpy_suite_versions_match_manifest_runtime():
    with (PACKAGE / "blender_manifest.toml").open("rb") as stream:
        manifest = tomllib.load(stream)

    requirements = REQUIREMENTS.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert manifest["blender_version_min"] == "5.2.0"
    assert "bpy==5.2.0" in requirements
    assert 'EXPECTED_BPY_DISTRIBUTION = "5.2.0"' in runner
    assert "EXPECTED_BLENDER = (5, 2, 0)" in runner
    assert "EXPECTED_PYTHON = (3, 13)" in runner


def test_real_bpy_suite_is_isolated_from_legacy_global_mocks():
    assert BPY_TESTS.is_dir()
    assert BPY_TESTS.parent == ROOT
    assert BPY_TESTS != ROOT / "tests"

    conftest_source = (BPY_TESTS / "conftest.py").read_text(encoding="utf-8")
    assert "tests/conftest.py" in conftest_source
    assert "pytest.importorskip" in conftest_source
    assert "tests_bpy imported mocked Blender modules" in conftest_source
    assert 'sys.modules["bpy"]' not in conftest_source
    assert 'sys.modules["bmesh"]' not in conftest_source


def test_runner_executes_only_real_bpy_test_root_and_fails_closed():
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RUNNER))
    string_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "tests_bpy" in string_literals
    assert "tests" not in string_literals
    assert "--strict-markers" in string_literals
    assert "--maxfail=1" in string_literals
    assert "official bpy package is not installed; run " in string_literals


def test_real_bpy_files_never_import_legacy_test_fixtures():
    for path in BPY_TESTS.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "from tests.conftest" not in source, path.name
        assert "import tests.conftest" not in source, path.name
        ast.parse(source, filename=str(path))
