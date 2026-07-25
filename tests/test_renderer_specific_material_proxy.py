from contextlib import nullcontext
from types import SimpleNamespace

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.bake_materials import (
    PreparedBakeMaterials,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.render_engine_contract import (
    render_engine_contract,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeMode,
    BakePassPlan,
    BakeStrategyId,
    MaterialSemanticChannel,
)


class FakeOutput:
    type = "OUTPUT_MATERIAL"

    def __init__(self, name, target, active):
        self.name = name
        self.target = target
        self.is_active_output = active


class FakeNodes(tuple):
    pass


class FakeNodeTree:
    def __init__(self, outputs):
        self.nodes = FakeNodes(outputs)

    def get_output_node(self, target):
        candidates = [node for node in self.nodes if node.target == target]
        if not candidates:
            candidates = [node for node in self.nodes if node.target == "ALL"]
        active = [node for node in candidates if node.is_active_output]
        return (active or candidates or [None])[0]


def test_renderer_contract_normalizes_only_supported_blender_52_engines():
    assert render_engine_contract("CYCLES").shader_target == "CYCLES"
    assert render_engine_contract("BLENDER_EEVEE").shader_target == "EEVEE"


def test_prepared_materials_normalize_and_forward_renderer_target(monkeypatch):
    calls = []

    def fake_prepare(materials, pass_plan, *, used_material_indices, render_target):
        calls.append((materials, pass_plan, used_material_indices, render_target))
        return nullcontext()

    monkeypatch.setattr(
        "Blender_to_Spine2D_Mesh_Exporter.blender_adapter.scene_material_preparation."
        "temporary_prepare_scene_material_pass",
        fake_prepare,
    )
    material = object()
    prepared = PreparedBakeMaterials(
        materials=(material,),
        image_nodes=(),
        placeholder_slot_indices=(),
        used_material_indices=(0,),
        render_target="BLENDER_EEVEE",
    )
    pass_plan = BakePassPlan(
        pass_index=0,
        strategy_id=BakeStrategyId.ALPHA,
        bake_mode=BakeMode.EMIT,
        material_slot_indices=(0,),
        semantic_channels=(MaterialSemanticChannel.ALPHA,),
    )

    with prepared.prepare_pass(pass_plan):
        pass

    assert prepared.render_target == "EEVEE"
    assert calls == [((material,), pass_plan, (0,), "EEVEE")]
