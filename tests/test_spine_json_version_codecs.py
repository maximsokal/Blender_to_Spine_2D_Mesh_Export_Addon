"""Contracts for target-version Spine JSON codec selection and native output."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import (
    Bone,
    IKConstraint,
    MeshAttachment,
    Skin,
    Slot,
    SpineDocument,
    TransformConstraint,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.serializer import SpineSerializer
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.validator import (
    SpineValidationError,
    SpineValidator,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_codecs import (
    registered_spine_json_codecs,
    serialize_spine_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import (
    SpineJsonTarget,
    SpineJsonTargetUnavailableError,
)


def _document() -> SpineDocument:
    attachment = MeshAttachment(
        name="mesh",
        uvs=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        triangles=(0, 1, 2),
        vertices=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        hull=3,
        edges=(0, 1),
        width=64.0,
        height=64.0,
        sequence={"count": 2, "start": 0, "digits": 4, "setup": 0},
    )
    return SpineDocument(
        skeleton={"spine": "4.2.43", "images": "images/"},
        bones=(Bone("root"),),
        slots=(Slot("slot", bone="root", attachment="mesh"),),
        skins=(Skin("default", {"slot": {"mesh": attachment}}),),
        animations={"animation": {}},
    )


def _v41_document() -> SpineDocument:
    attachment = MeshAttachment(
        name="mesh",
        uvs=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        triangles=(0, 1, 2),
        vertices=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        hull=3,
        edges=(0, 1),
        width=64.0,
        height=64.0,
        sequence={"count": 3, "start": 1, "digits": 4, "setup": 1},
    )
    linked = {
        "type": "mesh",
        "parent": "mesh",
        "path": "mesh",
        "timelines": False,
    }
    return SpineDocument(
        skeleton={
            "spine": "4.1.19",
            "images": "images/",
            "referenceScale": 100,
        },
        bones=(
            Bone("root"),
            Bone(
                "control",
                parent="root",
                extras={"inherit": "onlyTranslation"},
            ),
        ),
        slots=(Slot("slot", bone="control", attachment="mesh"),),
        skins=(
            Skin(
                "default",
                {"slot": {"mesh": attachment, "linked": linked}},
                constraints=("ik", "transform"),
            ),
        ),
        ik=(IKConstraint("ik", 0, ("control",), "root"),),
        transform=(
            TransformConstraint("transform", 1, ("control",), "root"),
        ),
        animations={"animation": {}},
        extras={"physics": [{"name": "unused-4.2-only-constraint"}]},
    )


@pytest.mark.parametrize("indent", (0, 2, 4))
def test_v42_facade_is_byte_identical_to_current_serializer(indent: int) -> None:
    document = _document()

    expected = SpineSerializer().to_json(document, indent=indent)
    actual = serialize_spine_document(
        document,
        SpineJsonTarget.SPINE_4_2,
        indent=indent,
    )

    assert actual == expected
    assert json.loads(actual)["skins"][0]["attachments"]["slot"]["mesh"][
        "sequence"
    ] == {"count": 2, "start": 0, "digits": 4, "setup": 0}


def test_v42_facade_does_not_mutate_the_input_document() -> None:
    document = _document()
    before = SpineSerializer().to_json(document, indent=2)

    serialize_spine_document(document, "4.2.43", indent=2)

    assert SpineSerializer().to_json(document, indent=2) == before


def test_v41_codec_writes_native_setup_pose_fields() -> None:
    payload = json.loads(
        serialize_spine_document(
            _v41_document(),
            SpineJsonTarget.SPINE_4_1,
            indent=2,
        )
    )

    assert payload["skeleton"]["spine"] == "4.1.19"
    assert "referenceScale" not in payload["skeleton"]
    assert payload["bones"][1]["transform"] == "onlyTranslation"
    assert "inherit" not in payload["bones"][1]
    assert "physics" not in payload

    skin = payload["skins"][0]
    assert skin["ik"] == ["ik"]
    assert skin["transform"] == ["transform"]
    assert "constraints" not in skin

    attachments = skin["attachments"]["slot"]
    assert attachments["mesh"]["sequence"] == {
        "count": 3,
        "start": 1,
        "digits": 4,
        "setup": 1,
    }
    assert attachments["linked"]["parent"] == "mesh"
    assert attachments["linked"]["timelines"] is False


def test_v41_codec_does_not_mutate_the_input_document() -> None:
    document = _v41_document()
    before = SpineSerializer().to_json(document, indent=2)

    serialize_spine_document(document, SpineJsonTarget.SPINE_4_1, indent=2)

    assert SpineSerializer().to_json(document, indent=2) == before


def test_v41_facade_rejects_missing_skin_constraint_before_rewrite() -> None:
    document = _v41_document()
    broken_skin = replace(document.skins[0], constraints=("missing",))
    broken = replace(document, skins=(broken_skin,))

    with pytest.raises(SpineValidationError) as exc_info:
        serialize_spine_document(broken, SpineJsonTarget.SPINE_4_1)

    assert {issue.code for issue in exc_info.value.issues} == {
        "MISSING_SKIN_CONSTRAINT"
    }
    assert "missing" in str(exc_info.value)


@pytest.mark.parametrize(
    "target,document",
    (
        (SpineJsonTarget.SPINE_4_1, _v41_document()),
        (SpineJsonTarget.SPINE_4_2, _document()),
    ),
)
def test_facade_forwards_the_caller_selected_validator(
    target: SpineJsonTarget,
    document: SpineDocument,
) -> None:
    class SentinelValidator(SpineValidator):
        def validate_or_raise(self, document):
            raise RuntimeError("sentinel validator used")

    with pytest.raises(RuntimeError, match="sentinel validator used"):
        serialize_spine_document(
            document,
            target,
            validator=SentinelValidator(),
        )


def test_only_ready_targets_have_registered_production_codecs() -> None:
    codecs = registered_spine_json_codecs()

    assert tuple(codecs) == (
        SpineJsonTarget.SPINE_4_1,
        SpineJsonTarget.SPINE_4_2,
    )
    for target, codec in codecs.items():
        assert codec.target is target
        assert target.descriptor.serializer_ready

    for target in SpineJsonTarget:
        if target.descriptor.serializer_ready:
            continue
        with pytest.raises(SpineJsonTargetUnavailableError):
            serialize_spine_document(_document(), target)
