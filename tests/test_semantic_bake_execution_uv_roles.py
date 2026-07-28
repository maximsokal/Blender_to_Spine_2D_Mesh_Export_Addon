from __future__ import annotations

from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import semantic_bake_execution
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.bake_execution_error import (
    BakeExecutionError,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.material_uv_binding import (
    MaterialUvBindingError,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.semantic_bake_execution import (
    _bind_temporary_source_uv_sampling,
    _call_bake_operator,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.semantic_bake_validation import (
    SemanticBakeRuntime,
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


def _runtime(
    *,
    render_uv_layer: str | None = "SourceUV",
    uv_layer_names: tuple[str, ...] = ("SourceUV", "SpineBakeUV"),
    selected_to_active: bool = False,
) -> SemanticBakeRuntime:
    """Build the minimal valid runtime surface used by the internal policy helper."""

    runtime = object.__new__(SemanticBakeRuntime)
    object.__setattr__(
        runtime,
        "plan",
        SimpleNamespace(
            source_object_id="Object",
            settings=SimpleNamespace(selected_to_active=selected_to_active),
        ),
    )
    object.__setattr__(
        runtime,
        "target_snapshot",
        SimpleNamespace(
            render_uv_layer=render_uv_layer,
            uv_layer_names=uv_layer_names,
        ),
    )
    return runtime


def _prepared_materials():
    return SimpleNamespace(
        materials=("temporary-material",),
        used_material_indices=(0,),
        image_nodes=("temporary-bake-node",),
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


def test_local_semantic_bake_binds_owned_materials_to_source_uv(monkeypatch):
    calls = []

    def fake_bind(materials, uv_layer_name, *, used_material_indices, excluded_nodes):
        calls.append(
            (
                materials,
                uv_layer_name,
                used_material_indices,
                excluded_nodes,
            )
        )
        return (SimpleNamespace(consumer_count=2),)

    monkeypatch.setattr(
        semantic_bake_execution,
        "bind_materials_implicit_uv_sampling",
        fake_bind,
    )
    prepared = _prepared_materials()

    _bind_temporary_source_uv_sampling(
        _runtime(),
        prepared,
        generated_material=None,
    )

    assert calls == [
        (
            prepared.materials,
            "SourceUV",
            prepared.used_material_indices,
            prepared.image_nodes,
        )
    ]


@pytest.mark.parametrize(
    ("selected_to_active", "generated_material"),
    (
        (True, None),
        (False, object()),
    ),
)
def test_non_local_sampling_paths_skip_temporary_material_binding(
    monkeypatch,
    selected_to_active,
    generated_material,
):
    def unexpected_bind(*_args, **_kwargs):
        raise AssertionError("temporary material binding must be skipped")

    monkeypatch.setattr(
        semantic_bake_execution,
        "bind_materials_implicit_uv_sampling",
        unexpected_bind,
    )

    _bind_temporary_source_uv_sampling(
        _runtime(selected_to_active=selected_to_active),
        _prepared_materials(),
        generated_material=generated_material,
    )


@pytest.mark.parametrize(
    ("render_uv_layer", "uv_layer_names", "message"),
    (
        (None, ("SpineBakeUV",), "no source render UV"),
        ("SourceUV", ("SpineBakeUV",), "absent from target UV layers"),
    ),
)
def test_local_binding_rejects_missing_source_uv_contract(
    render_uv_layer,
    uv_layer_names,
    message,
):
    with pytest.raises(BakeExecutionError, match=message):
        _bind_temporary_source_uv_sampling(
            _runtime(
                render_uv_layer=render_uv_layer,
                uv_layer_names=uv_layer_names,
            ),
            _prepared_materials(),
            generated_material=None,
        )


def test_material_uv_binding_error_is_wrapped_at_execution_boundary(monkeypatch):
    def fail_binding(*_args, **_kwargs):
        raise MaterialUvBindingError("forced binding failure")

    monkeypatch.setattr(
        semantic_bake_execution,
        "bind_materials_implicit_uv_sampling",
        fail_binding,
    )

    with pytest.raises(BakeExecutionError, match="SourceUV") as exc_info:
        _bind_temporary_source_uv_sampling(
            _runtime(),
            _prepared_materials(),
            generated_material=None,
        )

    assert isinstance(exc_info.value.__cause__, MaterialUvBindingError)
