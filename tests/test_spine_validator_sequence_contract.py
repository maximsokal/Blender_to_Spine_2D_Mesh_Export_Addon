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


def build_typed_mesh(sequence: dict[str, object] | None) -> MeshAttachment:
    return MeshAttachment(
        name="mesh",
        uvs=tuple(VALID_RAW_MESH["uvs"]),
        triangles=tuple(VALID_RAW_MESH["triangles"]),
        vertices=tuple(VALID_RAW_MESH["vertices"]),
        hull=VALID_RAW_MESH["hull"],
        edges=tuple(VALID_RAW_MESH["edges"]),
        width=VALID_RAW_MESH["width"],
        height=VALID_RAW_MESH["height"],
        sequence=sequence,
    )


def build_raw_mesh(sequence: object) -> dict[str, object]:
    payload = deepcopy(VALID_RAW_MESH)
    payload["sequence"] = sequence
    return payload


def build_document(attachment: MeshAttachment | dict[str, object]) -> SpineDocument:
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=(Slot("slot", "root", attachment="mesh"),),
        skins=(Skin("default", {"slot": {"mesh": attachment}}),),
        animations={"animation": {}},
    )


def sequence_issue_codes(
    sequence: dict[str, object],
    *,
    typed: bool,
) -> tuple[str, ...]:
    attachment = build_typed_mesh(sequence) if typed else build_raw_mesh(sequence)
    return tuple(
        issue.code for issue in SpineValidator().validate(build_document(attachment))
    )


@pytest.mark.parametrize("typed", (False, True))
@pytest.mark.parametrize(
    "sequence",
    (
        {"count": 10, "start": 0},
        {"count": 3, "start": 7, "digits": 4, "setup": 1},
        {"count": 2, "start": 0, "futureField": True},
    ),
)
def test_valid_raw_and_typed_sequences_have_no_validation_issues(typed, sequence):
    assert sequence_issue_codes(sequence, typed=typed) == ()


def test_raw_sequence_must_be_a_mapping():
    issues = SpineValidator().validate(build_document(build_raw_mesh([1, 2, 3])))

    issue = next(item for item in issues if item.code == "INVALID_SEQUENCE_MAPPING")
    assert issue.path.endswith(".sequence")


@pytest.mark.parametrize("typed", (False, True))
@pytest.mark.parametrize("missing_field", ("count", "start"))
def test_sequence_requires_count_and_start(typed, missing_field):
    sequence = {"count": 3, "start": 0}
    del sequence[missing_field]

    issues = SpineValidator().validate(
        build_document(build_typed_mesh(sequence) if typed else build_raw_mesh(sequence))
    )
    matching = [item for item in issues if item.code == "MISSING_SEQUENCE_FIELD"]

    assert len(matching) == 1
    assert matching[0].path.endswith(f".sequence.{missing_field}")


@pytest.mark.parametrize("typed", (False, True))
@pytest.mark.parametrize(
    "field_name, value, expected_code",
    (
        ("count", True, "INVALID_SEQUENCE_COUNT"),
        ("count", 0, "INVALID_SEQUENCE_COUNT"),
        ("count", -1, "INVALID_SEQUENCE_COUNT"),
        ("count", 1.5, "INVALID_SEQUENCE_COUNT"),
        ("start", False, "INVALID_SEQUENCE_START"),
        ("start", -1, "INVALID_SEQUENCE_START"),
        ("start", 0.0, "INVALID_SEQUENCE_START"),
        ("digits", True, "INVALID_SEQUENCE_DIGITS"),
        ("digits", 0, "INVALID_SEQUENCE_DIGITS"),
        ("digits", 13, "INVALID_SEQUENCE_DIGITS"),
        ("digits", 4.0, "INVALID_SEQUENCE_DIGITS"),
        ("setup", False, "INVALID_SEQUENCE_SETUP"),
        ("setup", -1, "INVALID_SEQUENCE_SETUP"),
        ("setup", 3, "INVALID_SEQUENCE_SETUP"),
        ("setup", 1.0, "INVALID_SEQUENCE_SETUP"),
    ),
)
def test_raw_and_typed_sequences_share_strict_scalar_contracts(
    typed,
    field_name,
    value,
    expected_code,
):
    sequence = {"count": 3, "start": 0, "digits": 4, "setup": 1}
    sequence[field_name] = value

    codes = sequence_issue_codes(sequence, typed=typed)

    assert expected_code in codes


@pytest.mark.parametrize("typed", (False, True))
def test_invalid_count_does_not_trigger_secondary_setup_range_error(typed):
    sequence = {"count": True, "start": 0, "setup": 99}

    codes = sequence_issue_codes(sequence, typed=typed)

    assert codes.count("INVALID_SEQUENCE_COUNT") == 1
    assert "INVALID_SEQUENCE_SETUP" not in codes


def test_validate_or_raise_preserves_sequence_field_path():
    document = build_document(
        build_raw_mesh({"count": 2, "start": 0, "digits": 0})
    )

    with pytest.raises(SpineValidationError) as error:
        SpineValidator().validate_or_raise(document)

    issue = next(
        item for item in error.value.issues if item.code == "INVALID_SEQUENCE_DIGITS"
    )
    assert issue.path.endswith(".sequence.digits")


def test_serializer_rejects_invalid_raw_sequence():
    document = build_document(build_raw_mesh({"count": 0, "start": 0}))

    with pytest.raises(SpineValidationError) as error:
        SpineSerializer().to_dict(document)

    assert {item.code for item in error.value.issues} == {"INVALID_SEQUENCE_COUNT"}


@pytest.mark.parametrize("typed", (False, True))
def test_serializer_preserves_valid_sequence_mapping_without_inserting_defaults(typed):
    sequence = {"count": 10, "start": 0}
    attachment = build_typed_mesh(sequence) if typed else build_raw_mesh(sequence)

    serialized = SpineSerializer().to_dict(build_document(attachment))
    serialized_sequence = serialized["skins"][0]["attachments"]["slot"]["mesh"][
        "sequence"
    ]

    assert serialized_sequence == sequence
    assert "digits" not in serialized_sequence
    assert "setup" not in serialized_sequence
