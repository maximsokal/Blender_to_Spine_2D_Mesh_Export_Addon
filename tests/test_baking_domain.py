from pathlib import Path
from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (
    analyse_material_slot,
    analyse_object_materials,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeMaterialPolicy,
    BakeMode,
    BakePlanError,
    BakeSettings,
    MaterialKind,
    ObjectMaterialAnalysis,
    TextureFormat,
    build_bake_plan,
    sanitize_filename_stem,
)


class FakeNodes(tuple):
    pass


class FakeNodeTree:
    def __init__(self, nodes):
        self.nodes = FakeNodes(nodes)


class FakeImage:
    def __init__(
        self,
        name,
        *,
        source="FILE",
        filepath="",
        frame_duration=1,
    ):
        self.name = name
        self.source = source
        self.filepath_raw = filepath
        self.frame_duration = frame_duration


class FakeNode:
    def __init__(self, node_type, *, name=None, image=None):
        self.type = node_type
        self.name = name or node_type
        self.image = image


class FakeMaterial:
    def __init__(self, name, nodes=None, *, use_nodes=True):
        self.name = name
        self.use_nodes = use_nodes
        self.node_tree = None if nodes is None else FakeNodeTree(nodes)


class FakeSlot:
    def __init__(self, material):
        self.material = material


class FakeObject:
    type = "MESH"

    def __init__(self, name, materials):
        self.name = name
        self.material_slots = tuple(FakeSlot(material) for material in materials)


def test_empty_and_legacy_non_node_material_classification():
    empty = analyse_material_slot(0, None)
    solid = analyse_material_slot(
        1,
        FakeMaterial("Solid", nodes=None, use_nodes=False),
    )

    assert empty.kind is MaterialKind.EMPTY
    assert solid.kind is MaterialKind.SOLID_COLOR
    assert "diffuse_color fallback" in solid.issues[0]


def test_image_sequence_dependency_is_preserved():
    material = FakeMaterial(
        "Sequence",
        [
            FakeNode(
                "TEX_IMAGE",
                image=FakeImage(
                    "Walk",
                    source="SEQUENCE",
                    filepath="//walk_0001.png",
                    frame_duration=24,
                ),
            ),
            FakeNode("BSDF_PRINCIPLED"),
        ],
    )

    analysis = analyse_material_slot(0, material)

    assert analysis.kind is MaterialKind.IMAGE
    assert analysis.animated
    assert analysis.image_dependencies[0].image_name == "Walk"
    assert analysis.image_dependencies[0].frame_duration == 24


def test_mixed_material_records_image_and_procedural_nodes():
    material = FakeMaterial(
        "Mixed",
        [
            FakeNode("TEX_IMAGE", image=FakeImage("Atlas")),
            FakeNode("TEX_NOISE"),
            FakeNode("BSDF_PRINCIPLED"),
        ],
    )

    analysis = analyse_material_slot(0, material)

    assert analysis.kind is MaterialKind.MIXED
    assert analysis.has_image_dependency
    assert analysis.node_types == ("BSDF_PRINCIPLED", "TEX_IMAGE", "TEX_NOISE")


def test_missing_image_reference_is_unsupported():
    analysis = analyse_material_slot(
        0,
        FakeMaterial("Broken", [FakeNode("TEX_IMAGE", image=None)]),
    )
    assert analysis.kind is MaterialKind.UNSUPPORTED
    assert "has no image" in analysis.issues[0]


def test_temporary_bake_nodes_do_not_change_analysis():
    analysis = analyse_material_slot(
        0,
        FakeMaterial(
            "Solid",
            [
                FakeNode("TEX_IMAGE", name="TEMP_BAKE_Output", image=FakeImage("Bake")),
                FakeNode("UVMAP", name="TEMP_UV_SpineBakeUV"),
                FakeNode("BSDF_PRINCIPLED"),
            ],
        ),
    )
    assert analysis.kind is MaterialKind.SOLID_COLOR
    assert analysis.image_dependencies == ()


def test_object_analysis_keeps_dense_slot_order():
    result = analyse_object_materials(
        FakeObject(
            "Cube",
            [
                FakeMaterial("A", [FakeNode("TEX_IMAGE", image=FakeImage("A"))]),
                None,
                FakeMaterial("B", [FakeNode("TEX_NOISE")]),
            ],
        )
    )
    assert result.source_object_id == "Cube"
    assert tuple(slot.slot_index for slot in result.slots) == (0, 1, 2)
    assert tuple(slot.kind for slot in result.slots) == (
        MaterialKind.IMAGE,
        MaterialKind.EMPTY,
        MaterialKind.PROCEDURAL,
    )


def test_legacy_any_image_policy_preserves_current_mixed_material_choice(tmp_path):
    mixed = analyse_material_slot(
        0,
        FakeMaterial(
            "Mixed",
            [
                FakeNode("TEX_IMAGE", image=FakeImage("Atlas")),
                FakeNode("TEX_NOISE"),
            ],
        ),
    )
    analysis = ObjectMaterialAnalysis("Cube", (mixed,))

    legacy = build_bake_plan(
        analysis,
        BakeSettings(
            width=256,
            height=256,
            output_directory=tmp_path,
            output_stem="Cube",
            material_policy=BakeMaterialPolicy.LEGACY_ANY_IMAGE,
        ),
    )
    strict = build_bake_plan(
        analysis,
        BakeSettings(
            width=256,
            height=256,
            output_directory=tmp_path,
            output_stem="Cube",
            material_policy=BakeMaterialPolicy.CONSERVATIVE_MIXED,
        ),
    )

    assert legacy.bake_mode is BakeMode.DIFFUSE
    assert strict.bake_mode is BakeMode.COMBINED


def test_any_non_image_material_selects_procedural_mode_for_multiple_slots(tmp_path):
    image = analyse_material_slot(
        0,
        FakeMaterial("Image", [FakeNode("TEX_IMAGE", image=FakeImage("Atlas"))]),
    )
    solid = analyse_material_slot(
        1,
        FakeMaterial("Solid", [FakeNode("BSDF_PRINCIPLED")]),
    )
    plan = build_bake_plan(
        ObjectMaterialAnalysis("Cube", (image, solid)),
        BakeSettings(
            width=512,
            height=256,
            output_directory=tmp_path,
            output_stem="Cube",
        ),
    )
    assert plan.bake_mode is BakeMode.COMBINED


def test_sequence_plan_contains_exact_frame_range_and_representative(tmp_path):
    image = analyse_material_slot(
        0,
        FakeMaterial("Image", [FakeNode("TEX_IMAGE", image=FakeImage("Atlas"))]),
    )
    plan = build_bake_plan(
        ObjectMaterialAnalysis("Cube", (image,)),
        BakeSettings(
            width=128,
            height=128,
            output_directory=tmp_path,
            output_stem="Cube",
            texture_format=TextureFormat.PNG,
            sequence_start_frame=7,
            sequence_frame_count=3,
            sequence_frame_digits=4,
        ),
    )

    assert plan.sequence
    assert tuple(task.timeline_frame for task in plan.frame_tasks) == (7, 8, 9)
    assert tuple(task.image_name for task in plan.frame_tasks) == (
        "Cube_Baked_0007",
        "Cube_Baked_0008",
        "Cube_Baked_0009",
    )
    assert plan.representative_task is plan.frame_tasks[0]
    assert plan.frame_tasks[0].output_path == tmp_path / "Cube_Baked_0007.png"


def test_single_frame_plan_uses_none_timeline_frame(tmp_path):
    image = analyse_material_slot(
        0,
        FakeMaterial("Image", [FakeNode("TEX_IMAGE", image=FakeImage("Atlas"))]),
    )
    plan = build_bake_plan(
        ObjectMaterialAnalysis("Cube", (image,)),
        BakeSettings(
            width=64,
            height=64,
            output_directory=tmp_path,
            output_stem="Cube",
        ),
    )
    assert not plan.sequence
    assert plan.frame_tasks[0].timeline_frame is None
    assert plan.frame_tasks[0].image_name == "Cube_Baked"


def test_unsupported_material_blocks_plan(tmp_path):
    broken = analyse_material_slot(
        0,
        FakeMaterial("Broken", [FakeNode("TEX_IMAGE", image=None)]),
    )
    with pytest.raises(BakePlanError):
        build_bake_plan(
            ObjectMaterialAnalysis("Cube", (broken,)),
            BakeSettings(
                width=64,
                height=64,
                output_directory=tmp_path,
                output_stem="Cube",
            ),
        )


def test_filename_sanitization_is_windows_safe():
    assert sanitize_filename_stem(' Cube:Part/"A". ') == "Cube_Part__A_"
    with pytest.raises(BakePlanError):
        sanitize_filename_stem("...")
