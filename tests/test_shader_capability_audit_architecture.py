from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_capability_audit_is_diagnostic_not_a_production_router():
    production_files = (
        "Blender_to_Spine2D_Mesh_Exporter/domain/baking/strategies.py",
        "Blender_to_Spine2D_Mesh_Exporter/domain/baking/camera_projection.py",
        "Blender_to_Spine2D_Mesh_Exporter/blender_adapter/a1_object_preparation.py",
    )

    for relative_path in production_files:
        source = _source(relative_path)
        assert "audit_material_graph_capabilities" not in source
        assert "shader_capability_audit" not in source


def test_blender_capability_fixture_remains_manual_only():
    workflow = _source(".github/workflows/blender-camera-projection.yml")

    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "push:" not in workflow
    assert "run_shader_capability_audit_integration.py" in workflow
