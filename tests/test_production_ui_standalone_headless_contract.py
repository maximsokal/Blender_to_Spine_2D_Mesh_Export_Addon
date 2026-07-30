"""Source contract for the real-Blender stale Connect routing regression."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_production_ui_standalone_policy_integration.py"
)


def _source() -> str:
    """Return the complete runner source or fail with the exact missing path."""

    assert RUNNER.is_file(), RUNNER
    return RUNNER.read_text(encoding="utf-8")


def test_runner_compiles() -> None:
    source = _source()
    compile(source, str(RUNNER), "exec")


def test_runner_uses_real_bpy_and_the_production_ui_plan() -> None:
    source = _source()

    assert "import bpy" in source
    assert "addon.register()" in source
    assert "build_selected_ui_export_plan(bpy.context)" in source
    assert "production.settings.mode is A1MultiObjectMode.STANDALONE" in source
    assert "production.connected_sources == ()" in source
    assert "resolve_a1_multi_object_preparation_settings" in source
    assert "SpineJsonTarget.SPINE_4_1" in source


def test_runner_preserves_but_isolates_development_connect_state() -> None:
    source = _source()

    assert "spine2d_connect_settings.enabled = True" in source
    assert "build_development_connected_ui_export_plan(bpy.context)" in source
    assert "development.settings.mode is A1MultiObjectMode.MIXED" in source
    assert "len(development.connected_sources) == 2" in source
    assert "len(development.standalone_sources) == 1" in source


def test_runner_avoids_operator_driven_scene_setup() -> None:
    source = _source()

    assert "bpy.ops." not in source
    assert "[PRODUCTION_UI_STANDALONE] PASS" in source
