import json
from math import nan

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import (
    Bone,
    IKConstraint,
    MeshAttachment,
    Skin,
    Slot,
    SpineDocument,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.serializer import SpineSerializer
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.spine_json_contract import (
    SpineJsonContractError,
    validate_json_value,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.validator import (
    SpineValidationError,
    SpineValidator,
)


def _typed_mesh(*, edges=(0, 1)) -> MeshAttachment:
    return MeshAttachment(
        name="mesh",
        uvs=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        triangles=(0, 1, 2),
        vertices=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        hull=3,
        edges=edges,
        width=64.0,
        height=64.0,
    )


def _document(
    *,
    attachment=None,
    slot_attachment="mesh",
    skins=None,
    skeleton=None,
    ik=(),
) -> SpineDocument:
    if attachment is None:
        attachment = _typed_mesh()
    resolved_skins = skins
    if resolved_skins is None:
        resolved_skins = (Skin("default", {"slot": {"mesh": attachment}}),)
    return SpineDocument(
        skeleton={"spine": "4.2.43"} if skeleton is None else skeleton,
        bones=(Bone("root"),),
        slots=(Slot("slot", bone="root", attachment=slot_attachment),),
        skins=resolved_skins,
        ik=ik,
        animations={"animation": {}},
    )


def _codes(document: SpineDocument) -> set[str]:
    return {issue.code for issue in SpineValidator().validate(document)}


def test_json_contract_reports_exact_nested_path_and_rejects_cycles():
    with pytest.raises(SpineJsonContractError) as exc_info:
        validate_json_value({"animations": {"walk": [0.0, nan]}})
    assert exc_info.value.path == "$.animations.walk[1]"

    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(SpineJsonContractError) as cycle_info:
        validate_json_value(cyclic, path="events")
    assert cycle_info.value.path == "events[0]"


def test_json_contract_rejects_non_string_keys_and_unsupported_values():
    with pytest.raises(SpineJsonContractError, match="mapping keys must be str"):
        validate_json_value({1: "bad"}, path="skeleton")
    with pytest.raises(SpineJsonContractError, match="unsupported JSON value type set"):
        validate_json_value({"bad": {1, 2}}, path="document")


@pytest.mark.parametrize(
    ("factory", "error_type"),
    (
        (lambda: Bone("root", x=True), TypeError),
        (
            lambda: IKConstraint(
                "ik", order=True, bones=("root",), target="root"
            ),
            TypeError,
        ),
        (
            lambda: MeshAttachment(
                name="mesh",
                uvs=(0.0, 0.0),
                triangles=(),
                vertices=(0.0, 0.0),
                hull=True,
            ),
            TypeError,
        ),
    ),
)
def test_model_rejects_bool_as_number_or_integer(factory, error_type):
    with pytest.raises(error_type):
        factory()


def test_serializer_mandatorily_validates_mutated_metadata_with_exact_path():
    skeleton = {"spine": "4.2.43"}
    document = _document(skeleton=skeleton)
    skeleton["width"] = nan

    with pytest.raises(SpineValidationError) as exc_info:
        SpineSerializer().to_json(document)

    assert exc_info.value.issues[0].code == "INVALID_JSON_VALUE"
    assert exc_info.value.issues[0].path == "skeleton.width"


def test_allow_nan_false_remains_a_final_defense_after_custom_validator():
    class NoOpValidator(SpineValidator):
        def validate_or_raise(self, document):
            return None

    skeleton = {"spine": "4.2.43"}
    document = _document(skeleton=skeleton)
    skeleton["width"] = nan

    with pytest.raises(ValueError, match="Out of range float values"):
        SpineSerializer(NoOpValidator()).to_json(document)


def test_raw_mesh_mapping_is_validated_like_typed_mesh():
    raw_mesh = {
        "type": "mesh",
        "uvs": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        "triangles": [0, 1, -1],
        "vertices": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        "hull": 3,
        "edges": [0, 4],
    }
    document = _document(attachment=raw_mesh)

    assert "TRIANGLE_INDEX_OUT_OF_RANGE" in _codes(document)
    assert "EDGE_INDEX_OUT_OF_RANGE" in _codes(document)
    with pytest.raises(SpineValidationError):
        SpineSerializer().to_dict(document)


def test_raw_mesh_validation_collects_independent_array_errors():
    raw_mesh = {
        "type": "mesh",
        "uvs": "not-an-array",
        "triangles": [0, True, -1, 2],
        "vertices": [0.0, nan],
        "hull": True,
        "edges": [0, True, -1],
        "width": nan,
    }
    raw_mesh["vertices"][1] = 0.0
    raw_mesh["width"] = 1.0
    document = _document(attachment=raw_mesh)
    raw_mesh["vertices"][1] = nan
    raw_mesh["width"] = nan

    codes = _codes(document)
    assert {
        "INVALID_JSON_VALUE",
        "INVALID_UV_ARRAY",
        "INVALID_TRIANGLE_ARRAY",
        "NON_INTEGER_TRIANGLE_INDEX",
        "TRIANGLE_INDEX_OUT_OF_RANGE",
        "NON_FINITE_MESH_VALUE",
        "INVALID_HULL",
        "INVALID_EDGE_ARRAY",
        "NON_INTEGER_EDGE_INDEX",
        "EDGE_INDEX_OUT_OF_RANGE",
        "INVALID_MESH_DIMENSION",
    } <= codes


def test_non_mesh_raw_attachments_remain_supported():
    point = {"type": "point", "x": 1.0, "y": 2.0}
    document = _document(attachment=point)

    assert SpineValidator().validate(document) == ()
    assert json.loads(SpineSerializer().to_json(document))["skins"][0][
        "attachments"
    ]["slot"]["mesh"] == point


def test_duplicate_skin_and_missing_setup_attachment_are_rejected():
    skins = (
        Skin("default", {"slot": {"other": {"type": "point"}}}),
        Skin("default", {}),
    )
    document = _document(skins=skins, slot_attachment="mesh")

    codes = _codes(document)
    assert "DUPLICATE_SKIN" in codes
    assert "MISSING_SETUP_ATTACHMENT" in codes


def test_setup_attachment_may_live_in_a_non_default_skin():
    document = _document(
        skins=(Skin("variant", {"slot": {"mesh": {"type": "point"}}}),)
    )
    assert SpineValidator().validate(document) == ()


def test_typed_mesh_edge_range_is_checked_by_validator():
    document = _document(attachment=_typed_mesh(edges=(0, 3)))
    assert "EDGE_INDEX_OUT_OF_RANGE" in _codes(document)


def test_validator_does_not_crash_on_post_construction_invalid_order():
    constraint = IKConstraint("ik", order=0, bones=("root",), target="root")
    document = _document(ik=(constraint,))
    object.__setattr__(constraint, "order", [])

    issues = SpineValidator().validate(document)
    assert any(issue.code == "INVALID_CONSTRAINT_ORDER" for issue in issues)


@pytest.mark.parametrize("indent", (True, -1, 1.5))
def test_serializer_rejects_non_strict_indent(indent):
    error_type = ValueError if indent == -1 else TypeError
    with pytest.raises(error_type):
        SpineSerializer().to_json(_document(), indent=indent)
