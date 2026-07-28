from __future__ import annotations

from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.bake_execution_error import (
    BakeExecutionError,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.semantic_bake_execution import (
    _call_bake_operator,
)


class _BakeOperator:
    def __init__(self, result=None) -> None:
        self.calls: list[dict[str, str]] = []
        self.result = {"FINISHED"} if result is None else result

    def poll(self) -> bool:
        return True

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        return self.result


def _bpy_with_operator(operator: _BakeOperator):
    return SimpleNamespace(
        ops=SimpleNamespace(
            object=SimpleNamespace(
                bake=operator,
            )
        )
    )


def test_bake_operator_receives_explicit_destination_uv_layer():
    operator = _BakeOperator()

    _call_bake_operator(
        _bpy_with_operator(operator),
        "emission",
        uv_layer_name="SpineBakeUV",
    )

    assert operator.calls == [
        {
            "type": "EMISSION",
            "uv_layer": "SpineBakeUV",
        }
    ]


def test_bake_operator_rejects_empty_destination_uv_layer():
    with pytest.raises(ValueError, match="uv_layer_name"):
        _call_bake_operator(
            _bpy_with_operator(_BakeOperator()),
            "EMIT",
            uv_layer_name=" ",
        )


def test_bake_operator_failure_reports_destination_uv_layer():
    operator = _BakeOperator(result={"CANCELLED"})

    with pytest.raises(BakeExecutionError, match="SpineBakeUV"):
        _call_bake_operator(
            _bpy_with_operator(operator),
            "EMIT",
            uv_layer_name="SpineBakeUV",
        )
