"""Static regressions for the strict Blender 5.2 projection surface."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
)


def _source(name: str) -> str:
    return (ADAPTER / name).read_text(encoding="utf-8")


def test_projection_execution_uses_one_physical_render_operator_owner():
    camera = _source("camera_projection_execution.py")
    grouped = _source("grouped_camera_projection_execution.py")

    assert "def _call_render_operator(" in camera
    assert "bpy_module.ops.render.render" in camera
    assert "call_public_render_operator" not in camera
    assert "call_public_render_operator" not in grouped
    assert "from .camera_projection_execution import _call_render_operator" in grouped
    assert "_call_render_operator(runtime.bpy_module)" in grouped


def test_projection_postprocess_has_one_request_and_no_compatibility_wrapper():
    postprocess = _source("camera_projection_postprocess.py")
    camera_output = _source("camera_projection_output.py")
    grouped_postprocess = _source("grouped_camera_projection_postprocess.py")

    assert "def process_projection_outputs(" in postprocess
    assert "def process_camera_projection_outputs(" not in postprocess
    assert "def log_camera_projection_layout(" not in postprocess
    assert "def _coerce_request(" not in postprocess
    assert "apply_crop" not in postprocess
    assert "build_full_frame_layout" not in postprocess
    assert "build_camera_projection_postprocess_request" in camera_output
    assert "process_projection_outputs" in camera_output
    assert "process_projection_outputs" in grouped_postprocess
    assert "apply_crop" not in grouped_postprocess


def test_projection_image_has_no_binary_mask_compatibility_api():
    image = _source("camera_projection_image.py")

    assert "def read_staged_alpha_mask(" not in image
    assert "Compatibility wrapper" not in image
    assert "data.images.remove(image, do_unlink=True)" in image
    assert "Unable to restore cropped image color space" in image


def test_camera_output_has_only_detailed_layout_staging_contract():
    output = _source("camera_projection_output.py")

    assert "class CameraProjectionStageResult" in output
    assert "def stage_camera_projection_outputs_detailed(" in output
    assert "def stage_camera_projection_outputs(" not in output
    assert "def _render_to_reservations(" not in output
    assert "_reserve =" not in output
    assert "_build_execution_result =" not in output


def test_grouped_projection_contains_no_private_compatibility_aliases():
    output = _source("grouped_camera_projection_output.py")
    validation = _source("grouped_camera_projection_validation.py")
    visibility = _source("grouped_camera_projection_visibility.py")

    assert "_reserve_group_outputs" not in output
    assert "validate_grouped_projection_runtime" not in validation
    assert "Compatibility wrapper" not in validation
    assert 'hasattr(obj, "visible_camera")' not in visibility
    assert "obj.visible_camera = True" in visibility
    assert "obj.visible_camera = False" in visibility


def test_active_projection_adapter_contains_no_b4_runtime_names():
    names = (
        "camera_projection_execution.py",
        "camera_projection_image.py",
        "camera_projection_output.py",
        "camera_projection_postprocess.py",
        "camera_projection_state.py",
        "camera_projection_validation.py",
        "grouped_camera_projection_execution.py",
        "grouped_camera_projection_output.py",
        "grouped_camera_projection_postprocess.py",
        "grouped_camera_projection_validation.py",
        "grouped_camera_projection_visibility.py",
    )

    for name in names:
        assert "B4" not in _source(name), name
