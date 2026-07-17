from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_b4_captures_disables_and_restores_postprocess_switches():
    source = _source(
        "Blender_to_Spine2D_Mesh_Exporter/blender_adapter/"
        "camera_projection_state.py"
    )

    assert '"render.use_compositing"' in source
    assert '"render.use_sequencer"' in source
    assert "scene.render.use_compositing = False" in source
    assert "scene.render.use_sequencer = False" in source


def test_b4_execution_revalidates_renderer_contract():
    source = _source(
        "Blender_to_Spine2D_Mesh_Exporter/blender_adapter/"
        "camera_projection_state.py"
    )

    assert "render_engine_contract_from_execution" in source
    assert "plan.scene_context.render_engine" in source
    assert "execution engine differs from the analyzed renderer" in source
