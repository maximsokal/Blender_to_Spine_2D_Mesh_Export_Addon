from copy import deepcopy

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    Bone,
    MeshAttachment,
    Skin,
    Slot,
    SpineDocument,
    SpineSerializer,
    SpineValidator,
)


VALID_RAW_MESH = {
    "type": "mesh",
    "uvs": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    "triangles": [0, 1, 2],
    "vertices": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    "hull": 0,
    "width": 0,
    "height": 0,
}


def build_typed_mesh(
    *,
    path: str | None = None,
    extras: dict[str, object] | None = None,
) -> MeshAttachment:
    return MeshAttachment(
        name="mesh",
        uvs=tuple(VALID_RAW_MESH["uvs"]),
        triangles=tuple(VALID_RAW_MESH["triangles"]),
        vertices=tuple(VALID_RAW_MESH["vertices"]),
        hull=0,
        path=path,
        width=0,
        height=0,
        extras={} if extras is None else extras,
    )


def build_document(attachment: MeshAttachment | dict[str, object]) -> SpineDocument:
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=(Slot("slot", "root", attachment="mesh"),),
        skins=(Skin("default", {"slot": {"mesh": attachment}}),),
        animations={"animation": {}},
    )


@pytest.mark.parametrize("typed", (False, True))
def test_runtime_default_zero_dimensions_and_hull_remain_valid(typed):
    attachment = build_typed_mesh() if typed else deepcopy(VALID_RAW_MESH)
    document = build_document(attachment)

    assert SpineValidator().validate(document) == ()

    serialized = SpineSerializer().to_dict(document)
    mesh = serialized["skins"][0]["attachments"]["slot"]["mesh"]
    assert mesh["hull"] == 0
    assert mesh["width"] == 0
    assert mesh["height"] == 0


@pytest.mark.parametrize(
    "color",
    ("FFFFFFFF", "AABBCCDD", "aabbccdd"),
)
def test_official_rgba_attachment_colors_are_accepted(color):
    raw = deepcopy(VALID_RAW_MESH)
    raw.update({"name": "actual-mesh", "path": "", "color": color})

    document = build_document(raw)

    assert SpineValidator().validate(document) == ()


@pytest.mark.parametrize(
    "field_name, value, exception_type",
    (
        ("name", True, TypeError),
        ("name", "   ", ValueError),
        ("path", None, TypeError),
        ("path", 7, TypeError),
        ("color", False, TypeError),
        ("color", "FFFFFF", ValueError),
        ("color", "#AABBCCDD", ValueError),
        ("color", "GGGGGGGG", ValueError),
        ("color", "123456789", ValueError),
    ),
)
def test_raw_attachment_metadata_is_rejected_at_skin_boundary(
    field_name,
    value,
    exception_type,
):
    raw = deepcopy(VALID_RAW_MESH)
    raw[field_name] = value

    with pytest.raises(exception_type) as error:
        build_document(raw)

    assert f".{field_name}" in str(error.value)


@pytest.mark.parametrize(
    "extras, exception_type",
    (
        ({"name": 7}, TypeError),
        ({"name": ""}, ValueError),
        ({"color": None}, TypeError),
        ({"color": "12XZ5678"}, ValueError),
        ({"color": "AABBCC"}, ValueError),
    ),
)
def test_typed_mesh_extras_share_attachment_metadata_contract(
    extras,
    exception_type,
):
    with pytest.raises(exception_type):
        build_typed_mesh(extras=extras)


def test_typed_path_still_uses_existing_optional_string_contract():
    with pytest.raises(TypeError, match="path must be str or None"):
        build_typed_mesh(path=True)


def test_serializer_preserves_valid_typed_metadata_without_normalization():
    attachment = build_typed_mesh(
        path="images/body",
        extras={"name": "actual-body", "color": "AABBCCDD"},
    )

    serialized = SpineSerializer().to_dict(build_document(attachment))
    mesh = serialized["skins"][0]["attachments"]["slot"]["mesh"]

    assert mesh["name"] == "actual-body"
    assert mesh["path"] == "images/body"
    assert mesh["color"] == "AABBCCDD"
