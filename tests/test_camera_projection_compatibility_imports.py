"""Keep the camera-projection compatibility facade importable after owner splits."""

from pathlib import Path


ADAPTER = (
    Path(__file__).resolve().parents[1]
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
)


def test_camera_projection_core_aliases_current_physical_owners():
    source = (ADAPTER / "camera_projection_executor_core.py").read_text(
        encoding="utf-8"
    )

    assert "_reserve_camera_projection_outputs as _reserve" in source
    assert "_stage_validated_camera_projection as _render_to_reservations" in source
    assert (
        "build_camera_projection_execution_result as _build_execution_result"
        in source
    )
    assert "stage_camera_projection_outputs," not in source


def test_camera_projection_core_imports_without_retired_symbols():
    from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (
        camera_projection_executor_core as facade,
    )

    assert facade._reserve is not None
    assert facade._render_to_reservations is not None
    assert facade._build_execution_result is facade.build_camera_projection_execution_result
