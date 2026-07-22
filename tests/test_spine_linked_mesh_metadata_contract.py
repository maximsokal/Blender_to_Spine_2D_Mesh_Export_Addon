from copy import deepcopy

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    Bone,
    Skin,
    Slot,
    SpineDocument,
    SpineSerializer,
    SpineValidationError,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.linked_mesh_contract import (
    validate_setup_linked_meshes,
)


RAW_MESH = {
    "type": "mesh",
    "uvs": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    "triangles": [0, 1, 2],
    "vertices": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    "hull": 3,
}


def linked(**metadata):
    return {
        "type": "linkedmesh",
        "parent": "parent",
        **metadata,
    }


def build_skins(child):
    return (
        Skin(
            "default",
            {
                "slot": {
                    "parent": deepcopy(RAW_MESH),
                    "child": child,
                }
            },
        ),
    )


def build_document(child):
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=(Slot("slot", "root", attachment="child"),),
        skins=build_skins(child),
        animations={"animation": {}},
    )


@pytest.mark.parametrize(
    "metadata",
    (
        {},
        {"timelines": True},
        {"timelines": False},
        {"name": "runtime name"},
        {"name": ""},
        {"path": "images/child"},
        {"path": ""},
        {"color": None},
        {"color": ""},
        {"color": "ffffff"},
        {"color": "ffffffff"},
        {"color": "#ffffff"},
        {"color": "#ffffffff"},
        {"color": "Aa01fF80"},
        {"sequence": None},
        {"sequence": {"count": 2}},
        {
            "timelines": False,
            "name": "child override",
            "path": "images/child_",
            "color": "80ff00cc",
            "sequence": {"count": 3, "digits": 0},
            "futureField": {"enabled": True},
        },
    ),
)
def test_runtime_valid_linked_metadata_is_preserved_without_defaults(metadata):
    child = linked(**deepcopy(metadata))
    source = deepcopy(child)

    serialized = SpineSerializer().to_dict(build_document(child))

    assert serialized["skins"][0]["attachments"]["slot"]["child"] == source
    assert "timelines" in source or "timelines" not in serialized["skins"][0][
        "attachments"
    ]["slot"]["child"]


@pytest.mark.parametrize("value", (None, 0, 1, "true", [], {}, ()))
def test_timelines_must_be_an_explicit_boolean(value):
    child = linked(timelines=value)

    with pytest.raises(TypeError, match=r"child\.timelines must be bool"):
        validate_setup_linked_meshes(build_skins(child))


@pytest.mark.parametrize("field_name", ("name", "path"))
@pytest.mark.parametrize("value", (None, True, 1, 1.5, [], {}, ()))
def test_name_and_path_metadata_must_be_strings(field_name, value):
    child = linked(**{field_name: value})

    with pytest.raises(TypeError, match=rf"child\.{field_name} must be str"):
        validate_setup_linked_meshes(build_skins(child))


@pytest.mark.parametrize("value", (True, 1, 1.5, [], {}, ()))
def test_color_metadata_requires_string_or_none(value):
    child = linked(color=value)

    with pytest.raises(TypeError, match=r"child\.color must be str or None"):
        validate_setup_linked_meshes(build_skins(child))


@pytest.mark.parametrize(
    "value",
    (
        "fff",
        "fffffff",
        "fffffffff",
        "gggggg",
        "#fffff",
        "##ffffff",
        "ffffff00ff",
        "white",
    ),
)
def test_non_runtime_hex_colors_are_rejected(value):
    child = linked(color=value)

    with pytest.raises(ValueError, match="6 or 8 hexadecimal digits"):
        validate_setup_linked_meshes(build_skins(child))


@pytest.mark.parametrize("value", (True, 1, "sequence", [], ()))
def test_sequence_metadata_requires_mapping_or_none_in_direct_resolver(value):
    child = linked(sequence=value)

    with pytest.raises(TypeError, match=r"child\.sequence must be a mapping or None"):
        validate_setup_linked_meshes(build_skins(child))


def test_detailed_sequence_scalars_remain_owned_by_spine_validator():
    child = linked(sequence={"count": 0})

    with pytest.raises(SpineValidationError) as error:
        SpineSerializer().to_dict(build_document(child))

    assert {issue.code for issue in error.value.issues} == {
        "INVALID_SEQUENCE_COUNT"
    }


def test_invalid_metadata_on_an_intermediate_link_is_not_skipped():
    skins = (
        Skin(
            "default",
            {
                "slot": {
                    "parent": deepcopy(RAW_MESH),
                    "middle": linked(timelines="yes"),
                    "child": {
                        "type": "linkedmesh",
                        "parent": "middle",
                    },
                }
            },
        ),
    )

    with pytest.raises(TypeError, match=r"middle\.timelines must be bool"):
        validate_setup_linked_meshes(skins)


def test_unlinked_mesh_does_not_consume_linked_only_metadata():
    mesh = {
        **deepcopy(RAW_MESH),
        "timelines": "future-runtime-field",
    }
    document = SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=(Slot("slot", "root", attachment="mesh"),),
        skins=(Skin("default", {"slot": {"mesh": mesh}}),),
        animations={"animation": {}},
    )
    source = deepcopy(mesh)

    serialized = SpineSerializer().to_dict(document)

    assert serialized["skins"][0]["attachments"]["slot"]["mesh"] == source
