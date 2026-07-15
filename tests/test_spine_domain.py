import json

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    Bone,
    IKConstraint,
    LegacyRigProfile,
    MeshAttachment,
    Skin,
    Slot,
    SpineDocument,
    SpineSerializer,
    SpineValidationError,
    SpineValidator,
    TransformConstraint,
    build_legacy_fingerprint,
)


def make_valid_document() -> SpineDocument:
    attachment = MeshAttachment(
        name="Cube",
        uvs=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        triangles=(0, 1, 2),
        vertices=(
            1,
            1,
            0.0,
            0.0,
            1.0,
            1,
            1,
            1.0,
            0.0,
            1.0,
            1,
            1,
            0.0,
            1.0,
            1.0,
        ),
        hull=3,
    )
    return SpineDocument(
        skeleton={"spine": "4.2.43", "hash": "legacy"},
        bones=(
            Bone("root"),
            Bone("Cube_main", parent="root"),
            Bone("Cube", parent="Cube_main"),
            Bone("Cube_rotation_X", parent="Cube_main"),
        ),
        slots=(Slot("Cube", bone="Cube", attachment="Cube"),),
        skins=(Skin("default", {"Cube": {"Cube": attachment}}),),
        ik=(
            IKConstraint(
                "Cube_scale_constraint_IK",
                order=0,
                bones=("Cube",),
                target="Cube_rotation_X",
            ),
        ),
        transform=(
            TransformConstraint(
                "Cube_rotation_X",
                order=1,
                bones=("Cube",),
                target="Cube_rotation_X",
                extras={"local": True},
            ),
        ),
        animations={"preview": {}},
    )


def test_valid_document_serializes_deterministically():
    document = make_valid_document()
    validator = SpineValidator()
    serializer = SpineSerializer()

    assert validator.validate(document) == ()
    data = serializer.to_dict(document)

    assert [bone["name"] for bone in data["bones"]] == [
        "root",
        "Cube_main",
        "Cube",
        "Cube_rotation_X",
    ]
    assert data["transform"][0]["local"] is True
    assert json.loads(serializer.to_json(document)) == data


def test_validator_rejects_parent_after_child_and_missing_slot_bone():
    document = SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("Child", parent="Parent"), Bone("Parent")),
        slots=(Slot("slot", bone="Missing"),),
        skins=(),
    )

    issues = SpineValidator().validate(document)
    codes = {issue.code for issue in issues}

    assert "PARENT_AFTER_CHILD" in codes
    assert "MISSING_SLOT_BONE" in codes
    with pytest.raises(SpineValidationError):
        SpineValidator().validate_or_raise(document)


def test_legacy_profile_is_single_source_of_names():
    profile = LegacyRigProfile()

    assert profile.main_bone("Cube") == "Cube_main"
    assert profile.base_bone("Cube") == "Cube"
    assert profile.control_bones("Cube") == (
        "Cube_rotation_X",
        "Cube_rotation_Y",
        "Cube_rotation_Z",
    )
    assert profile.z_scale_bone("Cube", 2) == "Cube_2_scale"
    assert profile.z_bone("Cube", 2) == "Cube_2"


def test_legacy_fingerprint_ignores_volatile_skeleton_metadata():
    serializer = SpineSerializer()
    first = serializer.to_dict(make_valid_document())
    second = json.loads(json.dumps(first))
    second["skeleton"]["hash"] = "different"
    second["skeleton"]["images"] = "other/path"

    first_fingerprint = build_legacy_fingerprint(first)
    second_fingerprint = build_legacy_fingerprint(second)

    assert first_fingerprint == second_fingerprint
    assert first_fingerprint.digest() == second_fingerprint.digest()
