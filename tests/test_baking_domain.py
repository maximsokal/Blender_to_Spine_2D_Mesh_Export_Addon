from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import analyse_material_slot
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.material_analysis_error import (
    MaterialAnalysisError,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeMaterialPolicy,
    BakeMode,
    BakePlanError,
    BakeSettings,
    ImageDependency,
    MaterialAnalysis,
    MaterialKind,
    ObjectMaterialAnalysis,
    TextureFormat,
    build_bake_plan,
    sanitize_filename_stem,
)


class FakeMaterial:
    def __init__(self, name: str, node_tree=None):
        self.name = name
        self.node_tree = node_tree


def _material(
    slot_index: int,
    kind: MaterialKind,
    *,
    node_types: tuple[str, ...] = (),
    images: tuple[ImageDependency, ...] = (),
    issues: tuple[str, ...] = (),
) -> MaterialAnalysis:
    return MaterialAnalysis(
        slot_index=slot_index,
        material_name=None if kind is MaterialKind.EMPTY else f"Material{slot_index}",
        kind=kind,
        node_types=node_types,
        image_dependencies=images,
        issues=issues,
    )


def _image_dependency(
    name: str = "Atlas",
    *,
    source: str = "FILE",
    frame_duration: int = 1,
) -> ImageDependency:
    return ImageDependency(
        image_name=name,
        source=source,
        filepath=f"//{name}.png",
        frame_duration=frame_duration,
    )


def test_empty_material_slot_is_supported_with_explicit_renderer_target():
    empty = analyse_material_slot(0, None, render_target="CYCLES")

    assert empty.kind is MaterialKind.EMPTY
    assert empty.material_name is None
    assert empty.issues == ("Material slot is empty",)


def test_non_node_material_fails_closed_in_blender_52_adapter():
    with pytest.raises(MaterialAnalysisError, match="has no node tree"):
        analyse_material_slot(
            0,
            FakeMaterial("Solid", node_tree=None),
            render_target="EEVEE",
        )


def test_image_sequence_dependency_is_preserved_by_domain_contract():
    dependency = _image_dependency("Walk", source="SEQUENCE", frame_duration=24)
    analysis = _material(
        0,
        MaterialKind.IMAGE,
        node_types=("BSDF_PRINCIPLED", "TEX_IMAGE"),
        images=(dependency,),
    )

    assert analysis.animated
    assert analysis.image_dependencies[0].image_name == "Walk"
    assert analysis.image_dependencies[0].frame_duration == 24


def test_mixed_material_records_image_and_procedural_nodes():
    analysis = _material(
        0,
        MaterialKind.MIXED,
        node_types=("BSDF_PRINCIPLED", "TEX_IMAGE", "TEX_NOISE"),
        images=(_image_dependency(),),
    )

    assert analysis.kind is MaterialKind.MIXED
    assert analysis.has_image_dependency
    assert analysis.node_types == ("BSDF_PRINCIPLED", "TEX_IMAGE", "TEX_NOISE")


def test_unsupported_material_carries_explicit_issue():
    analysis = _material(
        0,
        MaterialKind.UNSUPPORTED,
        issues=("Image texture node has no image",),
    )
    assert analysis.kind is MaterialKind.UNSUPPORTED
    assert "has no image" in analysis.issues[0]


def test_object_analysis_keeps_dense_slot_order():
    result = ObjectMaterialAnalysis(
        "Cube",
        (
            _material(
                0,
                MaterialKind.IMAGE,
                node_types=("BSDF_PRINCIPLED", "TEX_IMAGE"),
                images=(_image_dependency("A"),),
            ),
            _material(1, MaterialKind.EMPTY),
            _material(2, MaterialKind.PROCEDURAL, node_types=("TEX_NOISE",)),
        ),
    )
    assert tuple(slot.slot_index for slot in result.slots) == (0, 1, 2)
    assert tuple(slot.kind for slot in result.slots) == (
        MaterialKind.IMAGE,
        MaterialKind.EMPTY,
        MaterialKind.PROCEDURAL,
    )


def test_image_dependency_policy_preserves_mixed_material_diffuse_choice(tmp_path):
    mixed = _material(
        0,
        MaterialKind.MIXED,
        node_types=("BSDF_PRINCIPLED", "TEX_IMAGE", "TEX_NOISE"),
        images=(_image_dependency(),),
    )
    analysis = ObjectMaterialAnalysis("Cube", (mixed,))

    image_policy = build_bake_plan(
        analysis,
        BakeSettings(
            width=256,
            height=256,
            output_directory=tmp_path,
            output_stem="Cube",
            material_policy=BakeMaterialPolicy.IMAGE_DEPENDENCY_DIFFUSE,
        ),
    )
    conservative = build_bake_plan(
        analysis,
        BakeSettings(
            width=256,
            height=256,
            output_directory=tmp_path,
            output_stem="Cube",
            material_policy=BakeMaterialPolicy.CONSERVATIVE_MIXED,
        ),
    )

    assert image_policy.bake_mode is BakeMode.DIFFUSE
    assert conservative.bake_mode is BakeMode.COMBINED


def test_any_non_image_material_selects_combined_mode_for_multiple_slots(tmp_path):
    image = _material(
        0,
        MaterialKind.IMAGE,
        node_types=("BSDF_PRINCIPLED", "TEX_IMAGE"),
        images=(_image_dependency(),),
    )
    solid = _material(1, MaterialKind.SOLID_COLOR, node_types=("BSDF_PRINCIPLED",))
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
    image = _material(
        0,
        MaterialKind.IMAGE,
        node_types=("BSDF_PRINCIPLED", "TEX_IMAGE"),
        images=(_image_dependency(),),
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
    image = _material(
        0,
        MaterialKind.IMAGE,
        node_types=("BSDF_PRINCIPLED", "TEX_IMAGE"),
        images=(_image_dependency(),),
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
    broken = _material(
        0,
        MaterialKind.UNSUPPORTED,
        issues=("Image texture node has no image",),
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
