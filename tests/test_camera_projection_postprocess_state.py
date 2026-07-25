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


def test_b4_validation_revalidates_renderer_contract():
    source = _source(
        "Blender_to_Spine2D_Mesh_Exporter/blender_adapter/"
        "camera_projection_validation.py"
    )

    assert "render_engine_contract_from_execution" in source
    assert "plan.scene_context.render_engine" in source
    assert "execution engine differs from the analyzed renderer" in source


def test_b4_postprocess_runs_outside_reversible_scene_scope():
    execution_source = _source(
        "Blender_to_Spine2D_Mesh_Exporter/blender_adapter/"
        "camera_projection_execution.py"
    )
    output_source = _source(
        "Blender_to_Spine2D_Mesh_Exporter/blender_adapter/"
        "camera_projection_output.py"
    )

    assert "process_projection_outputs" not in execution_source
    render_index = output_source.index(
        "render_camera_projection_frames"
    )
    postprocess_index = output_source.index(
        "process_projection_outputs"
    )
    assert render_index < postprocess_index
