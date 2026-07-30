"""Contracts for target-version Spine JSON codec selection and 4.2 parity."""

from __future__ import annotations

import json

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import (
    Bone,
    MeshAttachment,
    Skin,
    Slot,
    SpineDocument,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.serializer import SpineSerializer
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.validator import SpineValidator
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


def test_facade_forwards_the_caller_selected_validator() -> None:
    class SentinelValidator(SpineValidator):
        def validate_or_raise(self, document):
            raise RuntimeError("sentinel validator used")

    with pytest.raises(RuntimeError, match="sentinel validator used"):
        serialize_spine_document(
            _document(),
            SpineJsonTarget.SPINE_4_2,
            validator=SentinelValidator(),
        )


def test_only_ready_targets_have_registered_production_codecs() -> None:
    codecs = registered_spine_json_codecs()

    assert tuple(codecs) == (SpineJsonTarget.SPINE_4_2,)
    assert codecs[SpineJsonTarget.SPINE_4_2].target is SpineJsonTarget.SPINE_4_2

    for target in SpineJsonTarget:
        if target is SpineJsonTarget.SPINE_4_2:
            continue
        with pytest.raises(SpineJsonTargetUnavailableError):
            serialize_spine_document(_document(), target)
