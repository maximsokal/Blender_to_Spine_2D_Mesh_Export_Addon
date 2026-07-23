from types import SimpleNamespace

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_texture_planning import (
    _should_generate_material,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    A1MaterialSourcePolicy,
    MaterialAnalysis,
    MaterialKind,
    ObjectMaterialAnalysis,
)


def _uv(policy: A1MaterialSourcePolicy, *material_indices: int):
    return SimpleNamespace(
        source=SimpleNamespace(
            settings=SimpleNamespace(material_source_policy=policy),
        ),
        texturing_topology=SimpleNamespace(
            snapshot=SimpleNamespace(
                faces=tuple(
                    SimpleNamespace(material_index=index)
                    for index in material_indices
                ),
            ),
        ),
    )


def _analysis(*kinds: MaterialKind) -> ObjectMaterialAnalysis:
    return ObjectMaterialAnalysis(
        source_object_id="Hero",
        slots=tuple(
            MaterialAnalysis(
                slot_index=index,
                material_name=(None if kind is MaterialKind.EMPTY else f"Material{index}"),
                kind=kind,
            )
            for index, kind in enumerate(kinds)
        ),
    )


def test_require_source_never_generates_even_when_object_has_no_material_slots():
    enabled, reason = _should_generate_material(
        _uv(A1MaterialSourcePolicy.REQUIRE_SOURCE, 0),
        _analysis(),
    )

    assert not enabled
    assert reason == ""


def test_force_generated_always_ignores_usable_source_materials():
    enabled, reason = _should_generate_material(
        _uv(A1MaterialSourcePolicy.FORCE_GENERATED, 0),
        _analysis(MaterialKind.SOLID_COLOR),
    )

    assert enabled
    assert reason == "forced by Rewrite material source policy"


def test_generate_if_missing_handles_object_without_any_usable_material():
    enabled, reason = _should_generate_material(
        _uv(A1MaterialSourcePolicy.GENERATE_IF_MISSING, 0),
        _analysis(),
    )

    assert enabled
    assert reason == "source object has no usable materials"


def test_generate_if_missing_handles_used_empty_slot():
    enabled, reason = _should_generate_material(
        _uv(A1MaterialSourcePolicy.GENERATE_IF_MISSING, 1),
        _analysis(MaterialKind.SOLID_COLOR, MaterialKind.EMPTY),
    )

    assert enabled
    assert "(1,)" in reason


def test_generate_if_missing_handles_used_slot_outside_source_slot_range():
    enabled, reason = _should_generate_material(
        _uv(A1MaterialSourcePolicy.GENERATE_IF_MISSING, 3),
        _analysis(MaterialKind.SOLID_COLOR),
    )

    assert enabled
    assert "(3,)" in reason


def test_generate_if_missing_does_not_replace_usable_material_for_unused_empty_slot():
    enabled, reason = _should_generate_material(
        _uv(A1MaterialSourcePolicy.GENERATE_IF_MISSING, 0),
        _analysis(MaterialKind.SOLID_COLOR, MaterialKind.EMPTY),
    )

    assert not enabled
    assert reason == ""
