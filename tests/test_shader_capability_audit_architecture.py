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
    texture_planning = _source(
        "Blender_to_Spine2D_Mesh_Exporter/blender_adapter/a1_texture_planning.py"
    )
    object_audit = _source(
        "Blender_to_Spine2D_Mesh_Exporter/blender_adapter/"
        "production_shader_capability_object_audit.py"
    )
    routing = _source(
        "Blender_to_Spine2D_Mesh_Exporter/blender_adapter/"
        "production_shader_capability_routing.py"
    )
    facade = _source(
        "Blender_to_Spine2D_Mesh_Exporter/blender_adapter/"
        "production_shader_capabilities.py"
    )

    assert "shader_capability_audit" not in strategies
    assert "production_shader_capabil" not in strategies
    assert "shader_capability_audit" not in camera_projection
    assert "production_shader_capabil" not in camera_projection

    assert "production_shader_capability_object_audit" in texture_planning
    assert "production_shader_capability_routing" in texture_planning
    assert "production_shader_capabilities" not in texture_planning

    assert "analyse_production_material_graph" in object_audit
    assert "audit_material_graph_capabilities" in object_audit
    assert "build_texture_plan" not in object_audit
    assert "build_camera_projection_plan" not in object_audit

    assert "GROUP_RENDER_REQUIRED" in routing
    assert "UNSUPPORTED" in routing
    assert "build_texture_plan" in routing
    assert "build_camera_projection_plan" in routing

    assert "def " not in facade
    assert "class " not in facade


def test_blender_capability_fixtures_remain_manual_only():
    workflow = _source(".github/workflows/blender-camera-projection.yml")

    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "push:" not in workflow
    assert "run_shader_capability_audit_integration.py" in workflow
    assert "run_render_engine_contract_integration.py" in workflow
