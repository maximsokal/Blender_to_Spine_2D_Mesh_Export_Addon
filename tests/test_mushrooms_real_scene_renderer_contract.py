"""Static contract for renderer-safe direct mushrooms asset regressions."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REAL_GEOMETRY_RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_mushrooms_real_blend_integration.py"
)
REAL_PARALLAX_RUNNER = (
    ROOT
    / "tests"
    / "blender_headless"
    / "run_mushrooms_real_blend_parallax_budget_integration.py"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_real_geometry_runner_uses_and_preserves_loaded_scene_renderer() -> None:
    source = _read(REAL_GEOMETRY_RUNNER)

    assert "from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.render_engine_contract import" in source
    assert "def _loaded_scene_render_engine() -> str:" in source
    assert "return render_engine_contract(value).blender_engine" in source
    assert "render_engine=_loaded_scene_render_engine()" in source
    assert "render_engine_before = _loaded_scene_render_engine()" in source
    assert "settings.bake_execution.render_engine != render_engine_before" in source
    assert "_loaded_scene_render_engine() == render_engine_before" in source
    assert 'f"render_engine={render_engine_before}' in source
    assert "bpy.context.scene.render.engine =" not in source
    assert "scene.render.engine =" not in source


def test_real_parallax_runner_inherits_and_preserves_loaded_scene_renderer() -> None:
    source = _read(REAL_PARALLAX_RUNNER)

    assert "_loaded_scene_render_engine," in source
    assert "render_engine_before = _loaded_scene_render_engine()" in source
    assert "settings.bake_execution.render_engine == render_engine_before" in source
    assert "_loaded_scene_render_engine() == render_engine_before" in source
    assert 'f"render_engine={render_engine_before}' in source
    assert "bpy.context.scene.render.engine =" not in source
    assert "scene.render.engine =" not in source
