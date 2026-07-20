import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import material_analyzer
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import material_graph_resolution
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.material_analysis_error import (
    MaterialAnalysisError,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.material_node_classification import (
    classify_material_nodes,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.material_object_analysis import (
    analyse_object_materials,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.material_slot_analysis import (
    analyse_material_slot,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.shader_graph_error import (
    MaterialGraphAnalysisError,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    MaterialGraphSnapshot,
    MaterialKind,
    MaterialSemanticChannel,
    ShaderNodeSnapshot,
)


ADAPTER = (
    Path(__file__).resolve().parents[1]
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
)


def _source(name: str) -> str:
    return (ADAPTER / name).read_text(encoding="utf-8")


def _top_level_definitions(name: str):
    tree = ast.parse(_source(name), filename=name)
    return tuple(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )


class FakeImage:
    def __init__(
        self,
        name="Image",
        *,
        source="FILE",
        filepath=None,
        frame_duration=1,
    ):
        self.name = name
        self.source = source
        self.filepath_raw = filepath
        self.filepath = filepath
        self.frame_duration = frame_duration


class FakeNode:
    def __init__(
        self,
        node_type,
        *,
        name=None,
        image=None,
        muted=False,
    ):
        self.type = node_type
        self.name = name or node_type
        self.image = image
        self.mute = muted


class FakeMaterial:
    def __init__(self, name, nodes=None, *, use_nodes=True):
        self.name = name
        self.use_nodes = use_nodes
        self.node_tree = None if nodes is None else SimpleNamespace(nodes=tuple(nodes))


class FakeSlot:
    def __init__(self, material):
        self.material = material


class FakeObject:
    type = "MESH"

    def __init__(self, name, materials):
        self.name = name
        self.material_slots = tuple(FakeSlot(material) for material in materials)


def _missing_output_graph(*issues):
    return MaterialGraphSnapshot(
        material_name="Material",
        active_output_node_id=None,
        reachable_nodes=(),
        reachable_links=(),
        semantic_channels=(),
        dependencies=(),
        issues=tuple(issues),
    )


def _effective_graph(*issues):
    output = ShaderNodeSnapshot(
        node_id="Material Output",
        node_type="OUTPUT_MATERIAL",
        node_name="Material Output",
    )
    return MaterialGraphSnapshot(
        material_name="Material",
        active_output_node_id=output.node_id,
        reachable_nodes=(output,),
        reachable_links=(),
        semantic_channels=(MaterialSemanticChannel.SURFACE_COLOR,),
        dependencies=(),
        issues=tuple(issues),
    )


def test_material_analyzer_is_compatibility_only():
    assert _top_level_definitions("material_analyzer.py") == ()
    source = _source("material_analyzer.py")
    for owner in (
        "material_analysis_error",
        "material_analysis_rna",
        "material_node_classification",
        "material_object_analysis",
        "material_slot_analysis",
    ):
        assert owner in source


def test_material_physical_owners_have_separate_responsibilities():
    rna = _source("material_analysis_rna.py")
    classification = _source("material_node_classification.py")
    graph = _source("material_graph_resolution.py")
    slot = _source("material_slot_analysis.py")
    object_owner = _source("material_object_analysis.py")

    assert "MaterialKind" not in rna
    assert "analyse_material_graph_detailed" not in classification
    assert "MaterialAnalysis(" not in graph
    assert "material_slots" not in slot
    assert "PROCEDURAL_NODE_TYPES" not in object_owner
    assert "build_texture_plan" not in object_owner


def test_production_and_public_package_use_physical_material_owners():
    planning = _source("a1_texture_planning.py")
    package = _source("__init__.py")

    assert "from .material_object_analysis import analyse_object_materials" in planning
    assert "from .material_analyzer import" not in planning
    assert "from .material_analysis_error import MaterialAnalysisError" in package
    assert "from .material_object_analysis import analyse_object_materials" in package
    assert "from .material_slot_analysis import analyse_material_slot" in package
    assert "from .material_analyzer import" not in package


def test_optional_image_filepaths_have_total_order_independent_of_node_order():
    nodes = (
        FakeNode(
            "TEX_IMAGE",
            name="Packed",
            image=FakeImage("Shared", filepath=None),
        ),
        FakeNode(
            "TEX_IMAGE",
            name="File",
            image=FakeImage("Shared", filepath="//shared.png"),
        ),
    )

    forward = classify_material_nodes(nodes)
    reverse = classify_material_nodes(tuple(reversed(nodes)))

    assert forward.kind is MaterialKind.IMAGE
    assert forward.image_dependencies == reverse.image_dependencies
    assert tuple(value.filepath for value in forward.image_dependencies) == (
        None,
        "//shared.png",
    )


def test_duplicate_image_dependencies_are_deduplicated():
    image_a = FakeImage("Atlas", filepath="//atlas.png")
    image_b = FakeImage("Atlas", filepath="//atlas.png")
    result = classify_material_nodes(
        (
            FakeNode("TEX_IMAGE", name="A", image=image_a),
            FakeNode("TEX_IMAGE", name="B", image=image_b),
        )
    )

    assert len(result.image_dependencies) == 1
    assert result.image_dependencies[0].image_name == "Atlas"


def test_muted_and_temporary_nodes_do_not_change_classification():
    result = classify_material_nodes(
        (
            FakeNode("TEX_NOISE", muted=True),
            FakeNode(
                "TEX_IMAGE",
                name="TEMP_BAKE_Output",
                image=FakeImage("Bake"),
            ),
            FakeNode("BSDF_PRINCIPLED"),
        )
    )

    assert result.kind is MaterialKind.SOLID_COLOR
    assert result.node_types == ("BSDF_PRINCIPLED",)
    assert result.image_dependencies == ()


def test_effective_graph_nodes_override_unreachable_root_nodes(monkeypatch):
    root_noise = FakeNode("TEX_NOISE")
    reachable = FakeNode("BSDF_PRINCIPLED")
    material = FakeMaterial("Material", (root_noise,))
    monkeypatch.setattr(
        material_graph_resolution,
        "analyse_material_graph_detailed",
        lambda *_args, **_kwargs: SimpleNamespace(
            snapshot=_effective_graph(),
            reachable_nodes=(reachable,),
        ),
    )

    result = analyse_material_slot(0, material, render_target="CYCLES")

    assert result.kind is MaterialKind.SOLID_COLOR
    assert result.node_types == ("BSDF_PRINCIPLED",)
    assert result.graph is not None


def test_missing_output_uses_root_nodes_and_preserves_graph_issues(monkeypatch):
    root_noise = FakeNode("TEX_NOISE")
    unreachable = FakeNode(
        "TEX_IMAGE",
        image=FakeImage("Unreachable", filepath="//unused.png"),
    )
    material = FakeMaterial("Material", (root_noise,))
    monkeypatch.setattr(
        material_graph_resolution,
        "analyse_material_graph_detailed",
        lambda *_args, **_kwargs: SimpleNamespace(
            snapshot=_missing_output_graph("Missing output"),
            reachable_nodes=(unreachable,),
        ),
    )

    result = analyse_material_slot(0, material, render_target="CYCLES")

    assert result.kind is MaterialKind.PROCEDURAL
    assert result.graph is None
    assert result.issues == ("Missing output",)


def test_graph_analysis_error_uses_root_nodes_and_records_diagnostic(monkeypatch):
    material = FakeMaterial("Material", (FakeNode("TEX_NOISE"),))

    def fail(*_args, **_kwargs):
        raise MaterialGraphAnalysisError("broken graph")

    monkeypatch.setattr(
        material_graph_resolution,
        "analyse_material_graph_detailed",
        fail,
    )

    result = analyse_material_slot(0, material, render_target="CYCLES")

    assert result.kind is MaterialKind.PROCEDURAL
    assert result.graph is None
    assert result.issues == ("Shader graph analysis failed: broken graph",)


def test_classification_issues_precede_graph_issues_and_deduplicate(monkeypatch):
    broken = FakeNode("TEX_IMAGE", name="Broken", image=None)
    material = FakeMaterial("Material", (broken,))
    classification_issue = "Image Texture node 'Broken' has no image"
    monkeypatch.setattr(
        material_graph_resolution,
        "analyse_material_graph_detailed",
        lambda *_args, **_kwargs: SimpleNamespace(
            snapshot=_missing_output_graph(classification_issue, "Graph note"),
            reachable_nodes=(),
        ),
    )

    result = analyse_material_slot(0, material, render_target="CYCLES")

    assert result.kind is MaterialKind.UNSUPPORTED
    assert result.issues == (classification_issue, "Graph note")


def test_object_analysis_preserves_dense_slots_and_source_id_override(monkeypatch):
    monkeypatch.setattr(
        material_graph_resolution,
        "analyse_material_graph_detailed",
        lambda *_args, **_kwargs: SimpleNamespace(
            snapshot=_missing_output_graph(),
            reachable_nodes=(),
        ),
    )
    result = analyse_object_materials(
        FakeObject(
            "Cube",
            (
                FakeMaterial("A", None, use_nodes=False),
                None,
                FakeMaterial("B", None, use_nodes=False),
            ),
        ),
        source_object_id="SourceId",
        render_target="CYCLES",
    )

    assert result.source_object_id == "SourceId"
    assert tuple(slot.slot_index for slot in result.slots) == (0, 1, 2)
    assert tuple(slot.kind for slot in result.slots) == (
        MaterialKind.SOLID_COLOR,
        MaterialKind.EMPTY,
        MaterialKind.SOLID_COLOR,
    )


def test_material_slot_iteration_failure_is_wrapped():
    class BrokenObject:
        type = "MESH"
        name = "Broken"

        @property
        def material_slots(self):
            raise RuntimeError("slot collection failed")

    with pytest.raises(
        MaterialAnalysisError,
        match="Failed to analyze materials for 'Broken': slot collection failed",
    ):
        analyse_object_materials(BrokenObject(), render_target="CYCLES")


def test_facade_retains_historical_return_types_and_alias_identity():
    nodes = (FakeNode("BSDF_PRINCIPLED"),)
    legacy = material_analyzer._classify_nodes(nodes)

    assert isinstance(legacy, tuple)
    assert len(legacy) == 4
    assert legacy[0] is MaterialKind.SOLID_COLOR
    assert material_analyzer.analyse_material_slot is analyse_material_slot
    assert material_analyzer.analyse_object_materials is analyse_object_materials
