import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    Bone,
    Skin,
    Slot,
    SpineDocument,
    SpineSerializer,
    SpineValidator,
)


def build_document(skeleton, *, bone_icon=None):
    return SpineDocument(
        skeleton=skeleton,
        bones=(Bone("root", icon=bone_icon),),
        slots=(Slot("slot", "root"),),
        skins=(Skin("default", {}),),
        animations={"animation": {}},
    )


@pytest.mark.parametrize(
    "skeleton",
    (
        {},
        {"spine": "4.2.43"},
        {
            "hash": "",
            "spine": "3.7.94",
            "x": -12.5,
            "y": 0,
            "width": 0,
            "height": 512.25,
            "referenceScale": 100,
            "fps": 30.0,
            "images": "",
            "audio": "./audio",
            "futureField": True,
        },
    ),
)
def test_valid_skeleton_metadata_and_future_fields_are_accepted(skeleton):
    document = build_document(skeleton)
    assert SpineValidator().validate(document) == ()


@pytest.mark.parametrize(
    "field_name, value, exception_type",
    (
        ("spine", True, TypeError),
        ("spine", 42, TypeError),
        ("spine", "", ValueError),
        ("spine", "   ", ValueError),
        ("hash", True, TypeError),
        ("hash", 42, TypeError),
        ("images", None, TypeError),
        ("images", 42, TypeError),
        ("audio", False, TypeError),
        ("audio", 42, TypeError),
    ),
)
def test_skeleton_string_fields_reject_invalid_values(
    field_name,
    value,
    exception_type,
):
    with pytest.raises(exception_type) as error:
        build_document({field_name: value})
    assert f"document.skeleton.{field_name}" in str(error.value)


@pytest.mark.parametrize(
    "field_name",
    ("x", "y", "width", "height", "referenceScale", "fps"),
)
@pytest.mark.parametrize(
    "value, exception_type",
    ((True, TypeError), ("1", TypeError), (None, TypeError)),
)
def test_skeleton_numeric_fields_require_numbers(
    field_name,
    value,
    exception_type,
):
    with pytest.raises(exception_type) as error:
        build_document({field_name: value})
    assert f"document.skeleton.{field_name}" in str(error.value)


@pytest.mark.parametrize("icon", (None, "", "star", "custom/icon"))
def test_bone_icon_remains_an_optional_string(icon):
    document = build_document({"spine": "4.2.43"}, bone_icon=icon)
    assert SpineValidator().validate(document) == ()


@pytest.mark.parametrize("icon", (True, 1, (), {}))
def test_bone_icon_rejects_non_string_values(icon):
    with pytest.raises(TypeError, match="icon must be str or None"):
        build_document({"spine": "4.2.43"}, bone_icon=icon)


def test_serializer_preserves_skeleton_mapping_without_inserting_defaults():
    skeleton = {
        "spine": "4.2.43",
        "hash": "custom-hash",
        "x": -5,
        "width": 0,
        "images": "",
        "futureField": {"enabled": True},
    }
    serialized = SpineSerializer().to_dict(build_document(skeleton))
    assert serialized["skeleton"] == skeleton
    assert "audio" not in serialized["skeleton"]
    assert "fps" not in serialized["skeleton"]
