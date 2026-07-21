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


VALID_RAW_MESH = {
    "type": "mesh",
    "uvs": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    "triangles": [0, 1, 2, 0, 2, 3],
    "vertices": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    "hull": 4,
    "edges": [0, 1, 1, 2, 2, 3, 3, 0, 0, 2],
    "width": 64.0,
    "height": 64.0,
}


def build_typed_mesh(payload: dict[str, object]) -> MeshAttachment:
    return MeshAttachment(
        name="mesh",
        uvs=tuple(payload["uvs"]),
        triangles=tuple(payload["triangles"]),
        vertices=tuple(payload["vertices"]),
        hull=payload["hull"],
        edges=tuple(payload.get("edges", ())),
        width=payload.get("width"),
        height=payload.get("height"),
    )


def build_document(attachment: MeshAttachment | dict[str, object]) -> SpineDocument:
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=(Slot("slot", "root", attachment="mesh"),),
        skins=(Skin("default", {"slot": {"mesh": attachment}}),),
        animations={"animation": {}},
    )


def issue_codes(
    payload: dict[str, object],
    *,
    typed: bool,
) -> tuple[str, ...]:
    attachment = build_typed_mesh(payload) if typed else payload
    return tuple(
        issue.code for issue in SpineValidator().validate(build_document(attachment))
    )


@pytest.mark.parametrize("typed", (False, True))
def test_valid_raw_and_typed_mesh_topology_have_no_validation_issues(typed):
    payload = deepcopy(VALID_RAW_MESH)

    assert issue_codes(payload, typed=typed) == ()


@pytest.mark.parametrize("typed", (False, True))
@pytest.mark.parametrize(
    "field_name, value, expected_code",
    (
        ("triangles", [], "EMPTY_TRIANGLE_ARRAY"),
        ("triangles", [0, 0, 1], "DEGENERATE_TRIANGLE"),
        (
            "triangles",
            [0, 1, 2, 2, 1, 0, 0, 2, 3],
            "DUPLICATE_TRIANGLE",
        ),
        ("triangles", [0, 1, 2], "UNUSED_MESH_VERTEX"),
        ("edges", [0, 1, 1, 1, 1, 2, 2, 3, 3, 0], "SELF_EDGE"),
        ("edges", [0, 1, 1, 0, 1, 2, 2, 3, 3, 0], "DUPLICATE_EDGE"),
    ),
)
def test_raw_and_typed_meshes_share_topology_issue_codes(
    typed,
    field_name,
    value,
    expected_code,
):
    payload = deepcopy(VALID_RAW_MESH)
    payload[field_name] = value

    assert expected_code in issue_codes(payload, typed=typed)


@pytest.mark.parametrize(
    "field_name, value, expected_code, forbidden_codes",
    (
        (
            "triangles",
            [0, True, 2],
            "NON_INTEGER_TRIANGLE_INDEX",
            {"DEGENERATE_TRIANGLE", "UNUSED_MESH_VERTEX"},
        ),
        (
            "triangles",
            [0, 1, 9],
            "TRIANGLE_INDEX_OUT_OF_RANGE",
            {"DEGENERATE_TRIANGLE", "UNUSED_MESH_VERTEX"},
        ),
        (
            "edges",
            [0, False],
            "NON_INTEGER_EDGE_INDEX",
            {"SELF_EDGE", "DUPLICATE_EDGE"},
        ),
        (
            "edges",
            [0, 9],
            "EDGE_INDEX_OUT_OF_RANGE",
            {"SELF_EDGE", "DUPLICATE_EDGE"},
        ),
    ),
)
def test_raw_scalar_index_errors_do_not_trigger_secondary_topology_errors(
    field_name,
    value,
    expected_code,
    forbidden_codes,
):
    payload = deepcopy(VALID_RAW_MESH)
    payload[field_name] = value

    codes = set(issue_codes(payload, typed=False))

    assert expected_code in codes
    assert codes.isdisjoint(forbidden_codes)


def test_degenerate_triangle_does_not_add_unused_vertex_noise():
    payload = deepcopy(VALID_RAW_MESH)
    payload["triangles"] = [0, 0, 1]

    codes = issue_codes(payload, typed=False)

    assert codes.count("DEGENERATE_TRIANGLE") == 1
    assert "UNUSED_MESH_VERTEX" not in codes


def test_validate_or_raise_preserves_duplicate_edge_code_and_pair_path():
    payload = deepcopy(VALID_RAW_MESH)
    payload["edges"] = [0, 1, 1, 0]

    with pytest.raises(SpineValidationError) as error:
        SpineValidator().validate_or_raise(build_document(payload))

    issue = next(
        item for item in error.value.issues if item.code == "DUPLICATE_EDGE"
    )
    assert issue.path.endswith(".edges[1]")


def test_serializer_rejects_raw_mesh_that_bypasses_typed_request_contract():
    payload = deepcopy(VALID_RAW_MESH)
    payload["triangles"] = [0, 0, 1]

    with pytest.raises(SpineValidationError) as error:
        SpineSerializer().to_dict(build_document(payload))

    assert {item.code for item in error.value.issues} == {"DEGENERATE_TRIANGLE"}


def test_valid_typed_mesh_serialization_preserves_topology_arrays():
    payload = deepcopy(VALID_RAW_MESH)
    document = build_document(build_typed_mesh(payload))

    serialized = SpineSerializer().to_dict(document)
    mesh = serialized["skins"][0]["attachments"]["slot"]["mesh"]

    assert mesh["uvs"] == payload["uvs"]
    assert mesh["triangles"] == payload["triangles"]
    assert mesh["hull"] == payload["hull"]
    assert mesh["edges"] == payload["edges"]
