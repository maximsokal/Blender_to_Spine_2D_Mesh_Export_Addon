from dataclasses import replace
from math import isfinite

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    LegacyRigBuildError,
    LegacyRigBuildRequest,
    LegacyRigProfile,
    LegacyZGroup,
    UniformScaleMode,
    build_legacy_rig,
    calculate_uniform_scale,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_rig_builder import (
    _build_constraints,
    _build_z_group_bones,
    _main_position,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_rig_validation import (
    validate_legacy_rig_numeric_payload,
)


def _request(**overrides):
    values = {
        "prefix": "Mesh",
        "texture_width": 100,
        "texture_height": 100,
        "z_groups": (LegacyZGroup(0.0),),
    }
    values.update(overrides)
    return LegacyRigBuildRequest(**values)


@pytest.mark.parametrize(
    ("factory", "match"),
    (
        (lambda: LegacyZGroup(True), "z_value"),
        (lambda: LegacyZGroup(0.0, False), "height_real_pixels"),
        (lambda: _request(texture_width=True), "texture_width"),
        (lambda: _request(texture_height=False), "texture_height"),
        (lambda: _request(average_y_pixels=True), "average_y_pixels"),
        (lambda: _request(main_position_pixels=(True, 0.0)), "main_position_pixels"),
        (lambda: calculate_uniform_scale(True, 100), "texture_width"),
    ),
)
def test_bool_is_not_accepted_as_legacy_rig_numeric_data(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()


def test_request_prefix_is_canonical_and_internal_spaces_remain_valid():
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        _request(prefix=" Mesh ")

    result = build_legacy_rig(_request(prefix="Boss Head"))
    assert result.request.prefix == "Boss Head"
    assert result.info.prefix == "Boss Head"
    assert result.info.main_bone_name == "Boss Head_main"


def test_explicit_falsy_profile_is_not_replaced_by_default():
    with pytest.raises(TypeError, match="profile must be LegacyRigProfile"):
        build_legacy_rig(_request(), profile=0)


def test_internal_bone_namespace_collisions_fail_during_plan():
    with pytest.raises(LegacyRigBuildError, match="bone namespace"):
        build_legacy_rig(_request(prefix="root"))

    with pytest.raises(LegacyRigBuildError, match="bone namespace"):
        build_legacy_rig(
            _request(prefix="Mesh"),
            LegacyRigProfile(root_name="Mesh"),
        )


def test_derived_scale_and_z_overflow_fail_before_model_export():
    with pytest.raises(ValueError, match="too large"):
        calculate_uniform_scale(10**10000, 1)

    with pytest.raises(LegacyRigBuildError, match="delta must be finite"):
        build_legacy_rig(
            _request(
                z_groups=(
                    LegacyZGroup(-1e308),
                    LegacyZGroup(1e308),
                )
            )
        )


def test_scale_modes_preserve_historical_results():
    assert calculate_uniform_scale(200, 100) == (float(200) + float(100)) / 2.0
    assert calculate_uniform_scale(200, 100, UniformScaleMode.MAXIMUM) == 200.0
    assert calculate_uniform_scale(200, 100, UniformScaleMode.MINIMUM) == 100.0


def test_custom_zero_index_base_and_negative_height_override_remain_supported():
    result = build_legacy_rig(
        _request(z_groups=(LegacyZGroup(0.0, height_real_pixels=-25.0),)),
        LegacyRigProfile(z_index_base=0),
    )
    group = result.info.z_groups[0]
    assert (group.index, group.y_offset_pixels, group.calculation_method) == (
        0,
        -25.0,
        "height_real_pixels",
    )
    assert group.bone_name == "Mesh_0"


def test_profile_rejects_bool_indices_without_changing_valid_names():
    with pytest.raises(ValueError):
        LegacyRigProfile(z_index_base=True)

    profile = LegacyRigProfile()
    with pytest.raises(ValueError):
        profile.z_bone("Mesh", True)
    with pytest.raises(ValueError):
        profile.z_scale_bone("Mesh", True)
    with pytest.raises(ValueError):
        profile.segment_slot("Mesh", False)
    with pytest.raises(ValueError):
        profile.vertex_bone("Mesh", True)

    assert profile.z_bone("Mesh", 1) == "Mesh_1"
    assert profile.segment_slot("Mesh", 0) == "Mesh_Segment_0"


def test_bone_for_z_rejects_bool_and_preserves_lookup_contract():
    result = build_legacy_rig(
        _request(z_groups=(LegacyZGroup(-1.0), LegacyZGroup(1.0)))
    )
    assert result.info.bone_for_z(1.0) == "Mesh_2"
    with pytest.raises(ValueError, match="z_value"):
        result.info.bone_for_z(True)
    with pytest.raises(ValueError, match="tolerance"):
        result.info.bone_for_z(1.0, tolerance=False)
    with pytest.raises(KeyError):
        result.info.bone_for_z(0.0)


def test_result_validation_rejects_tampered_structure_and_nonfinite_payload():
    result = build_legacy_rig(_request())
    changed_bone = replace(result.bones[1], x=1.0)
    tampered = replace(result, bones=(result.bones[0], changed_bone, *result.bones[2:]))
    with pytest.raises(LegacyRigBuildError, match="bones"):
        tampered.validate()

    infinite_bone = replace(result.bones[1], x=float("inf"))
    nonfinite = replace(
        result,
        bones=(result.bones[0], infinite_bone, *result.bones[2:]),
    )
    with pytest.raises(LegacyRigBuildError, match="must be finite"):
        validate_legacy_rig_numeric_payload(nonfinite)


def test_all_generated_numeric_payload_is_finite():
    result = build_legacy_rig(
        _request(
            texture_width=200,
            texture_height=100,
            z_groups=(
                LegacyZGroup(-1.0),
                LegacyZGroup(0.5),
                LegacyZGroup(2.0),
            ),
            main_position_pixels=(10.126, -20.555),
        )
    )
    assert (result.bones[1].x, result.bones[1].y) == (10.13, -20.55)

    for bone in result.bones:
        for value in (
            bone.length,
            bone.x,
            bone.y,
            bone.rotation,
            bone.scale_x,
            bone.scale_y,
        ):
            assert value is None or isfinite(float(value))
    for constraint in (*result.ik, *result.transform):
        for value in constraint.extras.values():
            assert (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or isfinite(float(value))
            )


def test_historical_private_helpers_keep_signatures_and_return_shapes():
    request = _request(
        z_groups=(LegacyZGroup(-1.0), LegacyZGroup(1.0)),
    )
    profile = LegacyRigProfile()
    bones, metadata = _build_z_group_bones(
        request,
        profile,
        parent_bone_name="Mesh_rotate_X",
        uniform_scale=100.0,
        half_scale=50.0,
    )
    assert len(bones) == 4
    assert tuple(item.index for item in metadata) == (1, 2)
    assert _main_position(request) == (0.0, 0.0)

    result = build_legacy_rig(request, profile)
    ik, transform = _build_constraints(request, profile, result.info)
    assert ik == result.ik
    assert transform == result.transform


def test_request_positional_field_order_is_unchanged():
    request = LegacyRigBuildRequest(
        "Mesh",
        100,
        200,
        (LegacyZGroup(0.0),),
        3.0,
        (1.0, 2.0),
        UniformScaleMode.MAXIMUM,
    )
    assert request.prefix == "Mesh"
    assert request.average_y_pixels == 3.0
    assert request.main_position_pixels == (1.0, 2.0)
    assert request.scale_mode is UniformScaleMode.MAXIMUM
