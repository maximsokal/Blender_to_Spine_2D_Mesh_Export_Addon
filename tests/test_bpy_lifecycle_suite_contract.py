"""Fast contract checks for lifecycle and resource-stability real-bpy regressions."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BPY_TESTS = ROOT / "tests_bpy"


def _source(filename: str) -> str:
    path = BPY_TESTS / filename
    assert path.is_file(), filename
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path))
    return source


def test_lifecycle_suite_keeps_import_reload_and_twenty_cycle_stress():
    source = _source("test_extension_lifecycle_real_bpy.py")
    for required in (
        "range(20)",
        "pkgutil.walk_packages",
        "importlib.reload",
        "subprocess.run",
        "depsgraph_update_post",
        "extension.register()",
        "extension.unregister()",
        "OBJECT_PT_spine2d_mesh",
        "_deferred_view3d_redraw",
    ):
        assert required in source
    assert "REGISTRATION_STEPS" not in source
    assert "repolish_ui" not in source
    assert "OBJECT_PT_Spine2DOrderedMeshPanel" not in source


def test_resource_suite_keeps_fault_user_map_growth_and_blend_roundtrip():
    source = _source("test_resource_stability_real_bpy.py")
    for required in (
        "bpy.data.user_map()",
        "forced snapshot-build failure",
        "forced UV materialization failure",
        "range(25)",
        "range(10)",
        "bpy.ops.wm.save_as_mainfile",
        "bpy.ops.wm.open_mainfile",
        "Юнікод_日本語",
    ):
        assert required in source
