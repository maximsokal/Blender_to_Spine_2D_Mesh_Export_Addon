from copy import deepcopy

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    Bone,
    MeshAttachment,
    Skin,
    Slot,
    SpineDocument,
    SpineSerializer,
    SpineValidationError,
    SpineValidator,
)


def typed_parent():
    return MeshAttachment(
        name="parent",
        uvs=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        triangles=(0, 1, 2),
        vertices=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        hull=3,
    )


def build_document(child):
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=(Slot("slot", "root", attachment="child"),),
        skins=(
            Skin(
                "default",
                {
                    "slot": {
                        "parent": typed_parent(),
                        "child": child,
                    }
                },
            ),
        ),
        animations={"animation": {}},
    )


def test_truthy_parent_mesh_does_not_require_geometry_arrays():
    child = {
        "type": "mesh",
        "parent": "parent",
        "timelines": False,
        "futureField": True,
    }
    source = deepcopy(child)
    document = build_document(child)

    assert SpineValidator().validate(document) == ()
    serialized = SpineSerializer().to_dict(document)

    assert serialized["skins"][0]["attachments"]["slot"]["child"] == source


@pytest.mark.parametrize("parent", ("", False, 0, None))
def test_falsy_parent_mesh_remains_an_unlinked_mesh(parent):
    child = {"type": "mesh", "parent": parent}
    codes = {
        issue.code for issue in SpineValidator().validate(build_document(child))
    }

    assert codes == {"MISSING_MESH_FIELD"}


def test_parent_bearing_mesh_sequence_is_validated_before_early_return():
    child = {
        "type": "mesh",
        "parent": "parent",
        "sequence": {"count": 0},
    }
    document = build_document(child)

    with pytest.raises(SpineValidationError) as error:
        SpineSerializer().to_dict(document)

    assert {item.code for item in error.value.issues} == {
        "INVALID_SEQUENCE_COUNT"
    }


def test_non_string_truthy_parent_reaches_linked_mesh_name_contract():
    child = {"type": "mesh", "parent": True}
    document = build_document(child)

    assert SpineValidator().validate(document) == ()
    with pytest.raises(TypeError, match="parent must be str"):
        SpineSerializer().to_dict(document)
