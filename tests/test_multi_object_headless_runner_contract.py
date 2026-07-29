from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests" / "blender_headless" / "run_multi_object_export_integration.py"


def test_multi_object_headless_runner_uses_current_semantic_bake_owner():
    source = RUNNER.read_text(encoding="utf-8")

    assert "blender_adapter.semantic_bake_execution as bake_module" in source
    assert "bake_executor as bake_module" not in source
    assert "def fail_second_bake(" in source
    assert "uv_layer_name" in source
    assert "uv_layer_name=uv_layer_name" in source
