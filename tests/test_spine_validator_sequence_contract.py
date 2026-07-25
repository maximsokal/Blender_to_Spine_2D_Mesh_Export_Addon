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
_ATTACHMENT_VARIANTS = (
    "typed_mesh",
    "raw_mesh",
    "raw_region",
    "raw_linkedmesh",
)


def build_typed_mesh(sequence: dict[str, object] | None) -> MeshAttachment:
    return MeshAttachment(
        name="item",
        uvs=tuple(VALID_RAW_MESH["uvs"]),
        triangles=tuple(VALID_RAW_MESH["triangles"]),
        vertices=tuple(VALID_RAW_MESH["vertices"]),
        hull=VALID_RAW_MESH["hull"],
        edges=tuple(VALID_RAW_MESH["edges"]),
        width=VALID_RAW_MESH["width"],
        height=VALID_RAW_MESH["height"],
        sequence=sequence,
    )


def build_raw_attachment(
    sequence: object,
    *,
    variant: str,
    serialized_edges: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    if variant == "raw_mesh":
        attachment = deepcopy(VALID_RAW_MESH)
        if serialized_edges:
            attachment["edges"] = [value * 2 for value in attachment["edges"]]
        attachment["sequence"] = sequence
        return attachment, {}

    if variant == "raw_region":
        return (
            {
                "type": "region",
                "path": "item",
                "width": 64.0,
                "height": 64.0,
                "sequence": sequence,
            },
            {},
        )

    if variant == "raw_linkedmesh":
        parent = deepcopy(VALID_RAW_MESH)
        if serialized_edges:
            parent["edges"] = [value * 2 for value in parent["edges"]]
        return (
            {
                "type": "linkedmesh",
                "parent": "parent",
                "sequence": sequence,
            },
            {"parent": parent},
        )

    raise AssertionError(f"unsupported raw attachment variant: {variant}")


def build_document(
    attachment: MeshAttachment | dict[str, object],
    *,
    extra_attachments: dict[str, object] | None = None,
) -> SpineDocument:
    attachments: dict[str, object] = {}
    if extra_attachments:
        attachments.update(extra_attachments)
    attachments["item"] = attachment
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=(Slot("slot", "root", attachment="item"),),
        skins=(Skin("default", {"slot": attachments}),),
        animations={"animation": {}},
    )


def build_variant_document(
    sequence: object,
    *,
    variant: str,
    serialized_edges: bool = False,
) -> SpineDocument:
    if variant == "typed_mesh":
        if sequence is not None and not isinstance(sequence, dict):
            raise TypeError("typed MeshAttachment sequence fixture must be dict or None")
        return build_document(build_typed_mesh(sequence))

    attachment, extras = build_raw_attachment(
        sequence,
        variant=variant,
        serialized_edges=serialized_edges,
    )
    return build_document(attachment, extra_attachments=extras)


def sequence_issue_codes(
    sequence: dict[str, object],
    *,
    variant: str,
) -> tuple[str, ...]:
    return tuple(
        issue.code
        for issue in SpineValidator().validate(
            build_variant_document(sequence, variant=variant)
        )
    )


@pytest.mark.parametrize("variant", _ATTACHMENT_VARIANTS)
@pytest.mark.parametrize(
    "sequence",
    (
        {"count": 10},
        {"count": 3, "start": -7, "digits": 0, "setup": 1},
        {"count": 2, "digits": 100, "futureField": True},
    ),
)
def test_runtime_valid_setup_sequences_have_no_validation_issues(variant, sequence):
    assert sequence_issue_codes(sequence, variant=variant) == ()


@pytest.mark.parametrize("variant", ("raw_mesh", "raw_region", "raw_linkedmesh"))
def test_raw_sequence_must_be_a_mapping_for_all_supported_attachment_types(variant):
    issues = SpineValidator().validate(
        build_variant_document([1, 2, 3], variant=variant)
    )

    issue = next(item for item in issues if item.code == "INVALID_SEQUENCE_MAPPING")
    assert issue.path.endswith(".sequence")


@pytest.mark.parametrize("variant", _ATTACHMENT_VARIANTS)
def test_only_count_is_required(variant):
    codes = sequence_issue_codes({}, variant=variant)

    assert codes == ("MISSING_SEQUENCE_FIELD",)


@pytest.mark.parametrize("variant", _ATTACHMENT_VARIANTS)
@pytest.mark.parametrize(
    "field_name, value, expected_code",
    (
        ("count", True, "INVALID_SEQUENCE_COUNT"),
        ("count", 0, "INVALID_SEQUENCE_COUNT"),
        ("count", -1, "INVALID_SEQUENCE_COUNT"),
        ("count", 1.5, "INVALID_SEQUENCE_COUNT"),
        ("start", False, "INVALID_SEQUENCE_START"),
        ("start", 0.0, "INVALID_SEQUENCE_START"),
        ("start", None, "INVALID_SEQUENCE_START"),
        ("digits", True, "INVALID_SEQUENCE_DIGITS"),
        ("digits", -1, "INVALID_SEQUENCE_DIGITS"),
        ("digits", 4.0, "INVALID_SEQUENCE_DIGITS"),
        ("digits", None, "INVALID_SEQUENCE_DIGITS"),
        ("setup", False, "INVALID_SEQUENCE_SETUP"),
        ("setup", -1, "INVALID_SEQUENCE_SETUP"),
        ("setup", 3, "INVALID_SEQUENCE_SETUP"),
        ("setup", 1.0, "INVALID_SEQUENCE_SETUP"),
    ),
)
def test_all_supported_attachments_share_strict_scalar_contracts(
    variant,
    field_name,
    value,
    expected_code,
):
    sequence = {"count": 3, "start": 0, "digits": 4, "setup": 1}
    sequence[field_name] = value

    codes = sequence_issue_codes(sequence, variant=variant)

    assert expected_code in codes


@pytest.mark.parametrize("variant", _ATTACHMENT_VARIANTS)
def test_invalid_count_does_not_trigger_secondary_setup_range_error(variant):
    sequence = {"count": True, "setup": 99}

    codes = sequence_issue_codes(sequence, variant=variant)

    assert codes.count("INVALID_SEQUENCE_COUNT") == 1
    assert "INVALID_SEQUENCE_SETUP" not in codes


@pytest.mark.parametrize("variant", _ATTACHMENT_VARIANTS)
def test_omitted_runtime_defaults_are_not_materialized(variant):
    sequence = {"count": 10}
    document = build_variant_document(
        sequence,
        variant=variant,
        serialized_edges=True,
    )

    serialized = SpineSerializer().to_dict(document)
    serialized_sequence = serialized["skins"][0]["attachments"]["slot"]["item"][
        "sequence"
    ]

    assert serialized_sequence == sequence
    assert "start" not in serialized_sequence
    assert "digits" not in serialized_sequence
    assert "setup" not in serialized_sequence


def test_explicit_null_raw_sequence_is_treated_as_absent_like_runtime():
    document = build_variant_document(None, variant="raw_region")

    serialized = SpineSerializer().to_dict(document)

    assert serialized["skins"][0]["attachments"]["slot"]["item"][
        "sequence"
    ] is None


def test_unsupported_attachment_sequence_field_remains_runtime_inert():
    attachment = {
        "type": "point",
        "x": 0,
        "y": 0,
        "sequence": ["runtime", "ignores", "this"],
    }
    document = build_document(attachment)

    serialized = SpineSerializer().to_dict(document)

    assert serialized["skins"][0]["attachments"]["slot"]["item"][
        "sequence"
    ] == ["runtime", "ignores", "this"]


def test_validate_or_raise_preserves_sequence_field_path():
    document = build_variant_document(
        {"count": 2, "digits": -1},
        variant="raw_region",
    )

    with pytest.raises(SpineValidationError) as error:
        SpineValidator().validate_or_raise(document)

    issue = next(
        item for item in error.value.issues if item.code == "INVALID_SEQUENCE_DIGITS"
    )
    assert issue.path.endswith(".sequence.digits")


def test_serializer_rejects_invalid_raw_region_sequence():
    document = build_variant_document({"count": 0}, variant="raw_region")

    with pytest.raises(SpineValidationError) as error:
        SpineSerializer().to_dict(document)

    assert {item.code for item in error.value.issues} == {"INVALID_SEQUENCE_COUNT"}
