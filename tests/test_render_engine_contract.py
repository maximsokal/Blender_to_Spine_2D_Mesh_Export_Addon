from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.render_engine_contract import (
    RenderEngineContract,
    RenderEngineContractError,
    render_engine_contract,
    render_engine_contract_from_execution,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeExecutionSettings,
    ColorManagementSnapshot,
    SceneBakeContext,
)


ROOT = Path(__file__).resolve().parents[1]


def _scene_context(render_engine: str) -> SceneBakeContext:
    return SceneBakeContext(
        scene_name="Scene",
        render_engine=render_engine,
        analysis_frame=1,
        world=None,
        camera=None,
        lights=(),
        visible_object_ids=(),
        shadow_caster_ids=(),
        color_management=ColorManagementSnapshot(
            view_transform="Standard",
            look="",
            exposure=0.0,
            gamma=1.0,
            working_space_interop_id="lin_rec709_scene",
        ),
    )


def test_renderer_aliases_normalize_to_blender_52_contract():
    assert render_engine_contract("CYCLES") == RenderEngineContract(
        "CYCLES",
        "CYCLES",
    )
    assert render_engine_contract("BLENDER_EEVEE") == RenderEngineContract(
        "BLENDER_EEVEE",
        "EEVEE",
    )
    assert render_engine_contract("EEVEE") == RenderEngineContract(
        "BLENDER_EEVEE",
        "EEVEE",
    )


def test_removed_eevee_next_alias_never_leaks_back_to_blender_runtime():
    contract = render_engine_contract("BLENDER_EEVEE_NEXT")

    assert contract.blender_engine == "BLENDER_EEVEE"
    assert contract.shader_target == "EEVEE"


def test_execution_settings_resolve_the_shader_output_target():
    contract = render_engine_contract_from_execution(
        BakeExecutionSettings(render_engine="BLENDER_EEVEE")
    )

    assert contract.blender_engine == "BLENDER_EEVEE"
    assert contract.shader_target == "EEVEE"
    assert contract.uses_eevee


def test_renderer_contract_rejects_scene_mismatch():
    contract = render_engine_contract("CYCLES")

    with pytest.raises(RenderEngineContractError, match="requested=CYCLES"):
        contract.validate_scene(_scene_context("BLENDER_EEVEE"))


def test_renderer_contract_accepts_scene_aliases():
    render_engine_contract("EEVEE").validate_scene(
        _scene_context("BLENDER_EEVEE")
    )


def test_production_preparation_uses_one_renderer_contract():
    source = (
        ROOT
        / "Blender_to_Spine2D_Mesh_Exporter"
        / "blender_adapter"
        / "a1_object_preparation.py"
    ).read_text(encoding="utf-8")

    assert "render_target=renderer.shader_target" in source
    assert "renderer.validate_scene(scene_bake_context)" in source
    assert "if renderer.uses_eevee:" in source
    assert "build_camera_projection_plan(" in source
