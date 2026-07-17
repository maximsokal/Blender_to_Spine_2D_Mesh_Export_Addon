from contextlib import nullcontext
from types import SimpleNamespace

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.bake_materials import (
    PreparedBakeMaterials,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.scene_material_preparation import (
    _renderer_output,
    _temporary_renderer_output_selection,
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


def test_renderer_output_resolves_exact_target_before_generic():
    generic = FakeOutput("Generic", "ALL", True)
    cycles = FakeOutput("Cycles", "CYCLES", True)
    eevee = FakeOutput("Eevee", "EEVEE", True)
    tree = FakeNodeTree((generic, eevee, cycles))

    assert _renderer_output(tree, "CYCLES") is cycles
    assert _renderer_output(tree, "BLENDER_EEVEE_NEXT") is eevee


def test_temporary_output_selection_is_exact_and_restored():
    eevee = FakeOutput("Eevee", "EEVEE", True)
    cycles = FakeOutput("Cycles", "CYCLES", True)
    material = SimpleNamespace(node_tree=FakeNodeTree((eevee, cycles)))
    original = (eevee.is_active_output, cycles.is_active_output)

    with _temporary_renderer_output_selection((material,), "CYCLES"):
        assert not eevee.is_active_output
        assert cycles.is_active_output

    assert (eevee.is_active_output, cycles.is_active_output) == original


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
    prepared = PreparedBakeMaterials(
        materials=(object(),),
        image_nodes=(),
        placeholder_slot_indices=(),
        used_material_indices=(0,),
        render_target="BLENDER_EEVEE_NEXT",
    )
    pass_plan = object()

    try:
        with prepared.prepare_pass(pass_plan):
            pass
    except TypeError:
        # PreparedBakeMaterials intentionally validates the public pass type before delegation.
        pass

    assert prepared.render_target == "EEVEE"
    assert not calls
