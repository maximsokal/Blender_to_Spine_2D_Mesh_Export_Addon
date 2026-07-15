import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    LegacyRigBuildRequest,
    LegacyRigProfile,
    LegacyZGroup,
    SpineDocument,
    SpineSerializer,
    SpineValidator,
    UniformScaleMode,
    build_legacy_rig,
    calculate_uniform_scale,
)


def build_cone_rig():
    return build_legacy_rig(
        LegacyRigBuildRequest(
            prefix="Cone",
            texture_width=128,
            texture_height=128,
            z_groups=(
                LegacyZGroup(-1.0, height_real_pixels=-128.0),
                LegacyZGroup(1.0, height_real_pixels=128.0),
            ),
            average_y_pixels=0.0,
        )
    )


def test_cone_hierarchy_matches_golden_order_and_parents():
    result = build_cone_rig()

    assert tuple(bone.name for bone in result.bones) == (
        "root",
        "Cone_main",
        "Cone",
        "Cone_scale_rotate_X",
        "Cone_rotate_X",
        "Cone_1_scale",
        "Cone_1",
        "Cone_2_scale",
        "Cone_2",
        "Cone_rotation_X",
        "Cone_rotation_Y",
        "Cone_rotation_Z",
        "Cone_rotate_X_constraint",
        "Cone_rotate_X_constraint_scale_IK",
        "Cone_rotate_X_constraint_rotate_IK",
        "Cone_rotate_X_constraint_IK",
    )
    assert tuple(bone.parent for bone in result.bones) == (
        None,
        "root",
        "Cone_main",
        "Cone",
        "Cone_scale_rotate_X",
        "Cone_rotate_X",
        "Cone_1_scale",
        "Cone_rotate_X",
        "Cone_2_scale",
        "Cone_main",
        "Cone_main",
        "Cone_main",
        "Cone",
        "Cone",
        "Cone_rotate_X_constraint_scale_IK",
        "Cone_rotate_X_constraint_rotate_IK",
    )


def test_cone_bone_payload_matches_stable_golden_values():
    result = build_cone_rig()
    serializer = SpineSerializer()
    bones = [serializer.bone_to_dict(bone) for bone in result.bones]

    assert bones == [
        {"name": "root"},
        {"name": "Cone_main", "parent": "root", "x": 0.0, "y": 0.0},
        {"name": "Cone", "parent": "Cone_main"},
        {
            "name": "Cone_scale_rotate_X",
            "parent": "Cone",
            "length": 64.0,
            "y": -0.5,
            "scaleX": 0.0,
        },
        {
            "name": "Cone_rotate_X",
            "parent": "Cone_scale_rotate_X",
            "color": "ff0000ff",
        },
        {
            "name": "Cone_1_scale",
            "parent": "Cone_rotate_X",
            "length": 64.0,
            "y": -128.0,
            "rotation": 90.0,
            "color": "abe323ff",
            "inherit": "onlyTranslation",
        },
        {"name": "Cone_1", "parent": "Cone_1_scale", "rotation": -90.0},
        {
            "name": "Cone_2_scale",
            "parent": "Cone_rotate_X",
            "length": 64.0,
            "y": 128.0,
            "rotation": 90.0,
            "color": "abe323ff",
            "inherit": "onlyTranslation",
        },
        {"name": "Cone_2", "parent": "Cone_2_scale", "rotation": -90.0},
        {
            "name": "Cone_rotation_X",
            "parent": "Cone_main",
            "length": 64.0,
            "x": 128.0,
            "y": 64.0,
            "color": "ff0000ff",
        },
        {
            "name": "Cone_rotation_Y",
            "parent": "Cone_main",
            "length": 64.0,
            "x": 128.0,
            "color": "00ff18ff",
        },
        {
            "name": "Cone_rotation_Z",
            "parent": "Cone_main",
            "length": 64.0,
            "x": 128.0,
            "y": -64.0,
            "color": "002cffff",
        },
        {
            "name": "Cone_rotate_X_constraint",
            "parent": "Cone",
            "length": 64.0,
            "y": -0.5,
            "rotation": 90.0,
            "color": "abe323ff",
        },
        {
            "name": "Cone_rotate_X_constraint_scale_IK",
            "parent": "Cone",
            "y": 63.5,
            "scaleX": 0.0,
        },
        {
            "name": "Cone_rotate_X_constraint_rotate_IK",
            "parent": "Cone_rotate_X_constraint_scale_IK",
            "x": -64.0,
        },
        {
            "name": "Cone_rotate_X_constraint_IK",
            "parent": "Cone_rotate_X_constraint_rotate_IK",
            "x": 64.0,
            "rotation": 90.0,
            "color": "ff3f00ff",
            "icon": "ik",
        },
    ]


def test_constraints_match_golden_serialization_order_and_parameters():
    result = build_cone_rig()
    serializer = SpineSerializer()

    assert [serializer.ik_to_dict(item) for item in result.ik] == [
        {
            "name": "Cone_scale_constraint_IK",
            "order": 3,
            "bones": ["Cone_rotate_X_constraint"],
            "target": "Cone_rotate_X_constraint_IK",
            "compress": True,
            "stretch": True,
        }
    ]
    assert [serializer.transform_to_dict(item) for item in result.transform] == [
        {
            "name": "Cone_rotation_X",
            "order": 1,
            "bones": ["Cone_1_scale", "Cone_2_scale", "Cone"],
            "target": "Cone_rotation_X",
            "rotation": 90,
            "local": True,
            "relative": True,
            "x": -256.0,
            "y": -64.0,
            "mixX": 0,
            "mixScaleX": 0,
            "mixShearY": 0,
        },
        {
            "name": "Cone_rotation_Y",
            "order": 2,
            "bones": [
                "Cone_rotate_X",
                "Cone_rotate_X_constraint_rotate_IK",
            ],
            "target": "Cone_rotation_Y",
            "local": True,
            "relative": True,
            "x": 128.0,
            "scaleX": -1,
            "mixX": 0,
            "mixScaleX": 0,
            "mixShearY": 0,
        },
        {
            "name": "Cone_rotation_Z",
            "order": 5,
            "bones": ["Cone_1", "Cone_2"],
            "target": "Cone_rotation_Z",
            "local": True,
            "mixX": 0,
            "mixScaleX": 0,
            "mixShearY": 0,
        },
        {
            "name": "Cone_scale_constraint",
            "order": 4,
            "bones": ["Cone_1_scale", "Cone_2_scale"],
            "target": "Cone_rotate_X_constraint",
            "scaleX": -1,
            "mixRotate": 0,
            "mixX": 0,
            "mixShearY": 0,
        },
        {
            "name": "Cone_scale_compensator",
            "order": 6,
            "bones": ["Cone_2_scale", "Cone_1_scale"],
            "target": "Cone",
            "mixRotate": 0,
            "mixX": 0,
            "mixScaleX": 0,
            "mixScaleY": 0,
            "mixShearY": 0,
        },
    ]


def test_builder_output_passes_full_spine_cross_reference_validation():
    result = build_cone_rig()
    document = SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=result.bones,
        slots=(),
        skins=(),
        ik=result.ik,
        transform=result.transform,
    )
    assert SpineValidator().validate(document) == ()


def test_z_groups_are_sorted_and_numbered_from_one():
    result = build_legacy_rig(
        LegacyRigBuildRequest(
            prefix="Mesh",
            texture_width=200,
            texture_height=100,
            z_groups=(LegacyZGroup(2.0), LegacyZGroup(-1.0), LegacyZGroup(0.5)),
        )
    )

    assert tuple(item.z_value for item in result.info.z_groups) == (-1.0, 0.5, 2.0)
    assert tuple(item.index for item in result.info.z_groups) == (1, 2, 3)
    assert tuple(item.y_offset_pixels for item in result.info.z_groups) == (
        0.0,
        225.0,
        450.0,
    )
    assert result.info.bone_for_z(0.5) == "Mesh_2"


def test_main_pixel_position_overrides_average_y_and_is_rounded():
    result = build_legacy_rig(
        LegacyRigBuildRequest(
            prefix="Mesh",
            texture_width=100,
            texture_height=100,
            z_groups=(LegacyZGroup(0.0),),
            average_y_pixels=123.0,
            main_position_pixels=(10.126, -20.555),
        )
    )
    main = result.bones[1]
    assert main.x == 10.13
    assert main.y == -20.55


def test_uniform_scale_modes_match_legacy_config_contract():
    assert calculate_uniform_scale(200, 100) == 150.0
    assert calculate_uniform_scale(200, 100, UniformScaleMode.MAXIMUM) == 200.0
    assert calculate_uniform_scale(200, 100, UniformScaleMode.MINIMUM) == 100.0


def test_request_rejects_ambiguous_or_invalid_z_groups():
    with pytest.raises(ValueError):
        LegacyRigBuildRequest(
            prefix="Mesh",
            texture_width=100,
            texture_height=100,
            z_groups=(),
        )
    with pytest.raises(ValueError):
        LegacyRigBuildRequest(
            prefix="Mesh",
            texture_width=100,
            texture_height=100,
            z_groups=(LegacyZGroup(0.0), LegacyZGroup(0.0)),
        )


def test_profile_centralizes_one_based_z_and_vertex_names():
    profile = LegacyRigProfile()
    assert profile.z_scale_bone("Cone", 1) == "Cone_1_scale"
    assert profile.z_bone("Cone", 2) == "Cone_2"
    assert profile.segment_slot("Cone", 3) == "Cone_Segment_3"
    assert profile.vertex_bone("Cone_Segment_0", 4) == "Cone_Segment_0_vertex_4"
    with pytest.raises(ValueError):
        profile.z_bone("Cone", 0)
