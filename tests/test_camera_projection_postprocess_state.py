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


def test_b4_image_settings_restore_format_before_dependent_mode_and_depth():
    source = _source(
        "Blender_to_Spine2D_Mesh_Exporter/blender_adapter/"
        "camera_projection_state.py"
    )

    assert "for entry in _ordered_scene_restore_entries(self.scene_values)" in source
    marker = "_IMAGE_SETTING_RESTORE_ORDER = ("
    start = source.index(marker)
    end = source.index(")", start)
    restore_contract = source[start:end]

    file_format_index = restore_contract.index(
        '"render.image_settings.file_format"'
    )
    color_mode_index = restore_contract.index(
        '"render.image_settings.color_mode"'
    )
    color_depth_index = restore_contract.index(
        '"render.image_settings.color_depth"'
    )
    assert file_format_index < color_mode_index < color_depth_index


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
