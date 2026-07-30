"""Contracts for production codec selection and quarantined Spine 4.1 research."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_serialization_validator import (
    ConnectedGroupSerializationValidator,
)
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
    SpineJsonCodecContext,
    registered_spine_json_codecs,
    serialize_spine_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_codecs.v41 import (
    Spine41JsonCodec,
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


def _v41_research_document() -> SpineDocument:
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
            "spine": "4.2.43",
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
                constraints=("ik", "transform-a", "transform-b"),
            ),
        ),
        ik=(IKConstraint("ik", 5, ("control",), "root"),),
        transform=(
            TransformConstraint("transform-a", 5, ("control",), "root"),
            TransformConstraint("transform-b", 11, ("control",), "root"),
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


def test_v42_facade_forwards_the_caller_selected_validator() -> None:
    class SentinelValidator(SpineValidator):
        def validate_or_raise(self, document):
            raise RuntimeError("sentinel validator used")

    with pytest.raises(RuntimeError, match="sentinel validator used"):
        serialize_spine_document(
            _document(),
            SpineJsonTarget.SPINE_4_2,
            validator=SentinelValidator(),
        )


def test_only_spine_four_two_has_a_registered_production_codec() -> None:
    codecs = registered_spine_json_codecs()

    assert tuple(codecs) == (SpineJsonTarget.SPINE_4_2,)
    assert codecs[SpineJsonTarget.SPINE_4_2].target is SpineJsonTarget.SPINE_4_2
    assert SpineJsonTarget.SPINE_4_2.descriptor.serializer_ready is True


@pytest.mark.parametrize(
    "target",
    tuple(target for target in SpineJsonTarget if target is not SpineJsonTarget.SPINE_4_2),
)
def test_unready_targets_fail_before_codec_resolution(target: SpineJsonTarget) -> None:
    with pytest.raises(SpineJsonTargetUnavailableError):
        serialize_spine_document(_document(), target)


def test_spine_four_one_facade_is_quarantined() -> None:
    assert SpineJsonTarget.SPINE_4_1.descriptor.serializer_ready is False

    with pytest.raises(SpineJsonTargetUnavailableError):
        serialize_spine_document(
            _v41_research_document(),
            SpineJsonTarget.SPINE_4_1,
        )


def test_quarantined_v41_adapter_preserves_authored_constraint_orders() -> None:
    document = _v41_research_document()
    before = SpineSerializer(
        validator=ConnectedGroupSerializationValidator()
    ).to_json(document, indent=2)

    payload = json.loads(
        Spine41JsonCodec().to_json(
            document,
            context=SpineJsonCodecContext(
                target=SpineJsonTarget.SPINE_4_1,
                validator=ConnectedGroupSerializationValidator(),
            ),
            indent=2,
        )
    )

    assert payload["skeleton"]["spine"] == "4.1.24"
    assert "referenceScale" not in payload["skeleton"]
    assert payload["bones"][1]["transform"] == "onlyTranslation"
    assert "inherit" not in payload["bones"][1]
    assert "physics" not in payload

    skin = payload["skins"][0]
    assert skin["ik"] == ["ik"]
    assert skin["transform"] == ["transform-a", "transform-b"]
    assert "constraints" not in skin

    order_by_name = {
        constraint["name"]: constraint.get("order", 0)
        for collection_name in ("ik", "transform", "path")
        for constraint in payload.get(collection_name, ())
    }
    assert order_by_name == {
        "ik": 5,
        "transform-a": 5,
        "transform-b": 11,
    }

    assert SpineSerializer(
        validator=ConnectedGroupSerializationValidator()
    ).to_json(document, indent=2) == before


def test_quarantined_v41_adapter_rejects_missing_skin_constraint_before_rewrite() -> None:
    document = _v41_research_document()
    broken_skin = replace(document.skins[0], constraints=("missing",))
    broken = replace(document, skins=(broken_skin,))

    with pytest.raises(SpineValidationError) as exc_info:
        Spine41JsonCodec().to_json(
            broken,
            context=SpineJsonCodecContext(
                target=SpineJsonTarget.SPINE_4_1,
                validator=ConnectedGroupSerializationValidator(),
            ),
        )

    assert {issue.code for issue in exc_info.value.issues} == {
        "MISSING_SKIN_CONSTRAINT"
    }
    assert "missing" in str(exc_info.value)
