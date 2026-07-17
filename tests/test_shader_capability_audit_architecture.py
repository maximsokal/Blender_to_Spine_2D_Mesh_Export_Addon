from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_capability_gate_remains_at_the_blender_adapter_boundary():
    strategies = _source(
        "Blender_to_Spine2D_Mesh_Exporter/domain/baking/strategies.py"
    )
    camera_projection = _source(
        "Blender_to_Spine2D_Mesh_Exporter/domain/baking/camera_projection.py"
    )
    preparation = _source(
        "Blender_to_Spine2D_Mesh_Exporter/blender_adapter/a1_object_preparation.py"
    )
    production_gate = _source(
        "Blender_to_Spine2D_Mesh_Exporter/blender_adapter/"
        "production_shader_capabilities.py"
    )

    assert "shader_capability_audit" not in strategies
    assert "production_shader_capabilities" not in strategies
    assert "shader_capability_audit" not in camera_projection
    assert "production_shader_capabilities" not in camera_projection
    assert "audit_object_material_capabilities" in preparation
    assert "build_capability_checked_texture_plan" in preparation
    assert "analyse_material_graph_detailed" in production_gate
    assert "GROUP_RENDER_REQUIRED" in production_gate
    assert "UNSUPPORTED" in production_gate


def test_blender_capability_fixtures_remain_manual_only():
    workflow = _source(".github/workflows/blender-camera-projection.yml")

    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "push:" not in workflow
    assert "run_shader_capability_audit_integration.py" in workflow
    assert "run_render_engine_contract_integration.py" in workflow
