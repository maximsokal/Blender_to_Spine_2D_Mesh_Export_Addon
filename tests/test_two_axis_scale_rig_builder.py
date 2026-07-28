from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    A1RigProfile,
    LegacyRigBuildRequest,
    LegacyZGroup,
    SpineDocument,
    SpineValidator,
    build_rig,
    build_two_axis_scale_rig,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_visuals import (
    apply_rig_visual_options,
)


def _request(z_groups=None):
    return LegacyRigBuildRequest(
        prefix="Box",
        texture_width=500,
        texture_height=500,
        z_groups=z_groups
        or (
            LegacyZGroup(-1.0, height_real_pixels=-200.0),
            LegacyZGroup(1.0, height_real_pixels=300.0),
        ),
    )


def test_two_axis_scale_rig_matches_reference_semantic_hierarchy():
    result = build_two_axis_scale_rig(_request())

    assert result.profile.profile_id == A1RigProfile.TWO_AXIS_ROTATION_SCALE.value
    assert tuple(bone.name for bone in result.bones) == (
        "root",
        "Box_main",
        "Box",
        "Box_scale_rotate_X",
        "Box_rotate_X",
        "Box_1_scale",
        "Box_1",
        "Box_2_scale",
        "Box_2",
        "Box_rotation_Y",
        "Box_rotate_X_constraint",
        "Box_rotate_X_constraint_scale_IK",
        "Box_rotate_X_constraint_rotate_IK",
        "Box_rotate_X_constraint_IK",
        "Box_rotation_X",
        "Box_scale",
    )
    assert "Box_rotation_Z" not in {bone.name for bone in result.bones}
    assert result.info.control_bone_names == (
        "Box_rotation_X",
        "Box_rotation_Y",
        "Box_scale",
    )
    bone_by_name = {bone.name: bone for bone in result.bones}
    assert bone_by_name["Box_rotation_X"].rotation == -134.67
    assert bone_by_name["Box_rotation_Y"].rotation == -17.43
    result.validate()


def test_two_axis_constraint_schedule_and_scale_targets_are_exact():
    result = build_two_axis_scale_rig(_request())

    assert tuple(constraint.order for constraint in result.transform) == (0, 4, 2, 3)
    assert tuple(constraint.order for constraint in result.ik) == (1,)

    rotate_x, rotate_y, scale, depth = result.transform
    assert rotate_x.bones == (
        "Box_rotate_X_constraint_rotate_IK",
        "Box_rotate_X",
    )
    assert rotate_x.target == "Box_rotation_X"
    assert rotate_y.bones == ("Box_2", "Box_1")
    assert rotate_y.target == "Box_rotation_Y"
    assert scale.bones == ("Box_rotate_X", "Box_2", "Box_1")
    assert scale.target == "Box_scale"
    assert scale.extras["relative"] is True
    assert depth.bones == ("Box_1_scale", "Box_2_scale")
    assert depth.target == "Box_rotate_X_constraint"
    assert depth.extras["x"] == -200.0


def test_two_axis_builder_supports_arbitrary_depth_group_counts():
    groups = tuple(
        LegacyZGroup(float(index), height_real_pixels=float(index * 125 - 250))
        for index in range(5)
    )
    result = build_two_axis_scale_rig(_request(groups))
    reversed_rotation_bones = tuple(reversed(result.info.sub_bone_names))

    assert len(result.info.z_groups) == 5
    assert len(result.info.sub_bone_names) == 5
    assert result.transform[1].bones == reversed_rotation_bones
    assert result.transform[2].bones == (
        result.info.main_rotation_bone_name,
        *reversed_rotation_bones,
    )
    assert result.transform[3].bones == result.info.sub_bone_scale_names
    result.validate()


def test_rig_router_preserves_three_axis_default_and_selects_two_axis():
    legacy = build_rig(_request())
    two_axis = build_rig(_request(), A1RigProfile.TWO_AXIS_ROTATION_SCALE)

    assert legacy.profile.profile_id == A1RigProfile.THREE_AXIS_ROTATION.value
    assert two_axis.profile.profile_id == A1RigProfile.TWO_AXIS_ROTATION_SCALE.value
    assert "Box_rotation_Z" in {bone.name for bone in legacy.bones}
    assert "Box_rotation_Z" not in {bone.name for bone in two_axis.bones}


def test_two_axis_visuals_reference_only_existing_controls():
    result = build_two_axis_scale_rig(_request())
    document = SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=result.bones,
        slots=(),
        skins=(),
        ik=result.ik,
        transform=result.transform,
    )

    # A default skin is required only when control attachments are requested.
    preview_only = apply_rig_visual_options(
        document,
        prefix="Box",
        rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        include_control_icons=False,
        include_preview_animation=True,
    )
    preview_bones = set(preview_only.animations["preview"]["bones"])
    assert preview_bones == {"Box_rotation_X", "Box_rotation_Y", "Box_scale"}
    assert SpineValidator().validate(preview_only) == ()
