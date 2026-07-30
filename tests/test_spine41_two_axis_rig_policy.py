"""Target-aware two-axis rig construction contracts for Spine 4.1."""

from __future__ import annotations

from dataclasses import replace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_rig_contracts import (
    LegacyRigBuildRequest,
    LegacyZGroup,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_builder import build_rig
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import (
    A1RigProfile,
    A1RigSetupPoseMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.spine41_setup_safety import (
    find_spine41_unsafe_world_constraints,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.two_axis_scale_rig import (
    build_two_axis_scale_rig,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.two_axis_scale_spine41 import (
    adapt_two_axis_scale_rig_for_spine41,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import SpineJsonTarget


def _request() -> LegacyRigBuildRequest:
    return LegacyRigBuildRequest(
        prefix="Cone",
        texture_width=256,
        texture_height=256,
        z_groups=(
            LegacyZGroup(z_value=0.0, height_real_pixels=0.0),
            LegacyZGroup(z_value=1.0, height_real_pixels=128.0),
        ),
        main_position_pixels=(0.0, 0.0),
        setup_pose_mode=A1RigSetupPoseMode.NORMALIZED_SINGLE,
    )


def _minimal_document(rig):
    from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import SpineDocument

    return SpineDocument(
        skeleton={"spine": "4.1.24"},
        bones=rig.bones,
        slots=(),
        skins=(),
        ik=rig.ik,
        transform=rig.transform,
    )


def test_spine_four_two_builder_remains_identical_to_existing_two_axis_builder() -> None:
    request = _request()

    expected = build_two_axis_scale_rig(request)
    actual = build_rig(
        request,
        A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        spine_target=SpineJsonTarget.SPINE_4_2,
    )

    assert actual == expected


def test_spine_four_one_builder_keeps_canonical_rig_for_projection_validation() -> None:
    request = _request()
    expected = build_two_axis_scale_rig(request)

    actual = build_rig(
        request,
        A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        spine_target=SpineJsonTarget.SPINE_4_1,
    )

    assert actual == expected
    # Attachment projection and mesh-document assembly call this exact method. The target
    # variant must not be applied until those canonical plan checks have completed.
    actual.validate()


def test_spine_four_one_research_view_preserves_world_scale_constraint_semantics() -> None:
    source = build_two_axis_scale_rig(_request())

    adapted = adapt_two_axis_scale_rig_for_spine41(source)

    assert adapted.request == source.request
    assert adapted.profile == source.profile
    assert adapted.info == source.info
    assert adapted.ik == source.ik
    assert len(adapted.bones) == len(source.bones) + len(source.info.z_groups)

    source_by_name = {item.name: item for item in source.transform}
    adapted_by_name = {item.name: item for item in adapted.transform}
    assert tuple(adapted_by_name) == tuple(source_by_name)

    scale_name = adapted.profile.scale_constraint(source.request.prefix)
    depth_name = adapted.profile.scale_depth_constraint(source.request.prefix)
    unchanged_names = set(source_by_name) - {scale_name}
    for name in unchanged_names:
        assert adapted_by_name[name] == source_by_name[name]

    source_scale = source_by_name[scale_name]
    adapted_scale = adapted_by_name[scale_name]
    assert source_scale.extras.get("local") is None
    assert adapted_scale.extras.get("local") is None
    assert adapted_scale.extras["relative"] is True
    assert source.info.main_rotation_bone_name in source_scale.bones
    assert source.info.main_rotation_bone_name not in adapted_scale.bones
    assert source.info.scale_bone_name in adapted_scale.bones
    assert tuple(
        name for name in adapted_scale.bones if name in source.info.sub_bone_names
    ) == tuple(
        name for name in source_scale.bones if name in source.info.sub_bone_names
    )

    assert source_by_name[depth_name].bones == source.info.sub_bone_scale_names
    assert adapted_by_name[depth_name].bones == source.info.sub_bone_scale_names

    adapted_bones = {bone.name: bone for bone in adapted.bones}
    for wrapper_name in source.info.sub_bone_scale_names:
        bridge_name = f"{wrapper_name}_spine41_bridge"
        assert adapted_bones[bridge_name].parent == source.info.main_rotation_bone_name
        assert adapted_bones[wrapper_name].parent == bridge_name


def test_spine_four_one_two_axis_variant_has_no_singular_world_constraint_parent() -> None:
    source = build_two_axis_scale_rig(_request())
    adapted = adapt_two_axis_scale_rig_for_spine41(source)

    assert find_spine41_unsafe_world_constraints(_minimal_document(adapted)) == ()


def test_spine_four_one_adapter_fails_if_source_constraint_schema_drifted() -> None:
    source = build_two_axis_scale_rig(_request())
    scale_name = source.profile.scale_constraint(source.request.prefix)
    changed = tuple(
        replace(item, extras={**item.extras, "mixRotate": 1})
        if item.name == scale_name
        else item
        for item in source.transform
    )
    broken = replace(source, transform=changed)

    with pytest.raises(ValueError, match="mixRotate=0"):
        adapt_two_axis_scale_rig_for_spine41(broken)


def test_spine_four_one_three_axis_remains_fail_closed() -> None:
    with pytest.raises(ValueError, match="not yet runtime-validated"):
        build_rig(
            _request(),
            A1RigProfile.THREE_AXIS_ROTATION,
            spine_target=SpineJsonTarget.SPINE_4_1,
        )
