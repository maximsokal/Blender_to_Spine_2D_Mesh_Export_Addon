"""Unit regressions for evaluated-mesh UV role restoration."""

from types import SimpleNamespace

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.evaluated_mesh_reader import (
    _resolve_evaluated_uv_roles,
)


class _UvLayers(list):
    def __init__(self, layers, *, active=None):
        super().__init__(layers)
        self.active = active


def _mesh(*layers, active):
    return SimpleNamespace(uv_layers=_UvLayers(layers, active=active))


def _layer(name: str, *, active_render: bool = False):
    return SimpleNamespace(name=name, active_render=active_render)


def test_source_uv_roles_override_evaluated_role_drift():
    evaluated_uv = _layer("UVMap", active_render=True)
    evaluated_source_uv = _layer("SourceUV", active_render=False)
    source_uv = _layer("UVMap", active_render=False)
    source_render_uv = _layer("SourceUV", active_render=True)

    active_name, render_name = _resolve_evaluated_uv_roles(
        evaluated_mesh=_mesh(
            evaluated_uv,
            evaluated_source_uv,
            active=evaluated_uv,
        ),
        source_mesh=_mesh(
            source_uv,
            source_render_uv,
            active=source_render_uv,
        ),
        resolved_uv_layers=(evaluated_uv, evaluated_source_uv),
    )

    assert active_name == "SourceUV"
    assert render_name == "SourceUV"


def test_evaluated_roles_remain_fallback_for_modifier_generated_layers():
    evaluated_uv = _layer("GeneratedUV", active_render=True)
    source_uv = _layer("UVMap", active_render=True)

    active_name, render_name = _resolve_evaluated_uv_roles(
        evaluated_mesh=_mesh(evaluated_uv, active=evaluated_uv),
        source_mesh=_mesh(source_uv, active=source_uv),
        resolved_uv_layers=(evaluated_uv,),
    )

    assert active_name == "GeneratedUV"
    assert render_name == "GeneratedUV"
