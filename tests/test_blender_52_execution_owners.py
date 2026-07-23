"""Regressions for direct Blender 5.2 bake/render operator ownership."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.bake_execution_error import (
    BakeExecutionError,
)
from Blender_to_Spine2D_Mesh_Export_Addon.blender_adapter.camera_projection_error import (
    CameraProjectionExecutionError,
)
from Blender_to_Spine2D_Mesh_Export_Addon.blender_adapter.camera_projection_execution import (
    _call_render_operator,
)
from Blender_to_Spine2D_Mesh_Export_Addon.blender_adapter.semantic_bake_execution import (
    _call_bake_operator,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
ADAPTER = PACKAGE / "blender_adapter"
_DEFAULT_RESULT = object()


class _Operator:
    def __init__(self, *, poll=True, result=_DEFAULT_RESULT, error=None):
        self.poll_result = poll
        self.result = {"FINISHED"} if result is _DEFAULT_RESULT else result
        self.error = error
        self.calls: list[dict[str, object]] = []

    def poll(self):
        return self.poll_result

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        return self.result


def _bpy(*, bake=None, render=None):
    return SimpleNamespace(
        ops=SimpleNamespace(
            object=SimpleNamespace(bake=bake),
            render=SimpleNamespace(render=render),
        )
    )


def test_semantic_execution_owns_blender_bake_operator():
    operator = _Operator()

    _call_bake_operator(_bpy(bake=operator), "diffuse")

    assert operator.calls == [{"type": "DIFFUSE"}]


def test_bake_operator_rejects_poll_cancel_and_invalid_result():
    with pytest.raises(BakeExecutionError, match=r"poll\(\) returned False"):
        _call_bake_operator(_bpy(bake=_Operator(poll=False)), "DIFFUSE")

    with pytest.raises(BakeExecutionError, match="did not finish"):
        _call_bake_operator(
            _bpy(bake=_Operator(result={"CANCELLED"})),
            "DIFFUSE",
        )

    with pytest.raises(BakeExecutionError, match="invalid result"):
        _call_bake_operator(
            _bpy(bake=_Operator(result=None)),
            "DIFFUSE",
        )


def test_camera_execution_owns_blender_render_operator():
    operator = _Operator()

    _call_render_operator(_bpy(render=operator))

    assert operator.calls == [{"write_still": True}]


def test_render_operator_rejects_poll_cancel_and_exception():
    with pytest.raises(
        CameraProjectionExecutionError,
        match=r"poll\(\) returned False",
    ):
        _call_render_operator(_bpy(render=_Operator(poll=False)))

    with pytest.raises(CameraProjectionExecutionError, match="did not finish"):
        _call_render_operator(
            _bpy(render=_Operator(result={"CANCELLED"})),
        )

    with pytest.raises(CameraProjectionExecutionError, match="write_still=True"):
        _call_render_operator(
            _bpy(render=_Operator(error=RuntimeError("render failed"))),
        )


def test_active_adapter_contains_no_execution_compatibility_facades():
    retired = (
        "bake_executor.py",
        "bake_executor_core.py",
        "semantic_bake_executor.py",
        "camera_projection_executor.py",
    )
    for name in retired:
        assert not (ADAPTER / name).exists(), name

    semantic = (ADAPTER / "semantic_bake_execution.py").read_text(
        encoding="utf-8"
    )
    camera = (ADAPTER / "camera_projection_execution.py").read_text(
        encoding="utf-8"
    )
    texture = (ADAPTER / "texture_executor.py").read_text(encoding="utf-8")
    package = (ADAPTER / "__init__.py").read_text(encoding="utf-8")

    assert "bpy_module.ops.object.bake" in semantic
    assert "from . import bake_executor" not in semantic
    assert "bpy_module.ops.render.render" in camera
    assert "from . import bake_executor" not in camera
    assert "semantic_bake_executor" not in texture
    assert "camera_projection_executor" not in texture
    assert "from .bake_executor import" not in package
    assert "from .camera_projection_executor import" not in package


def test_bake_scene_configuration_has_no_optional_legacy_mode():
    source = (ADAPTER / "bake_scene_state.py").read_text(encoding="utf-8")

    assert "bake_mode: BakeMode," in source
    assert "bake_mode: BakeMode | None" not in source
    assert "plan.bake_mode if bake_mode is None" not in source
    assert "preserves the legacy" not in source


def test_camera_output_contains_no_historical_staging_api():
    output = (ADAPTER / "camera_projection_output.py").read_text(
        encoding="utf-8"
    )
    validation = (ADAPTER / "camera_projection_validation.py").read_text(
        encoding="utf-8"
    )

    assert "def _render_to_reservations(" not in output
    assert "def stage_camera_projection_outputs(" not in output
    assert "_reserve =" not in output
    assert "_build_execution_result =" not in output
    assert "Historical private names" not in output
    assert "def validate_projection_runtime(" not in validation
    assert "Compatibility runtime validation" not in validation
    assert "B4" not in output
    assert "B4" not in validation
