"""Spine 4.0.64 codec regressions for the supported setup-pose subset."""

from __future__ import annotations

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
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_codecs import (
    Spine40JsonCodec,
    SpineJsonCodecContext,
    serialize_spine_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import SpineJsonTarget


def _document(*, sequence: bool = False) -> SpineDocument:
    attachment = MeshAttachment(
        name="mesh",
        uvs=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        triangles=(0, 1, 2),
        vertices=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        hull=3,
        width=64.0,
        height=64.0,
        sequence=(
            {"count": 2, "start": 0, "digits": 4, "setup": 0}
            if sequence
            else None
        ),
    )
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
                {"slot": {"mesh": attachment}},
                constraints=("ik", "transform"),
            ),
        ),
        ik=(
            IKConstraint(
                "ik",
                1,
                ("control",),
                "root",
                extras={"compress": True, "stretch": True},
            ),
        ),
        transform=(
            TransformConstraint(
                "transform",
                0,
                ("control",),
                "root",
                extras={
                    "local": True,
                    "relative": True,
                    "rotation": 12.5,
                    "x": -4.0,
                    "y": 8.0,
                    "scaleX": -1.0,
                    "mixRotate": 0.25,
                    "mixX": 0.5,
                    "mixY": 0.75,
                    "mixScaleX": 0.6,
                    "mixScaleY": 0.7,
                    "mixShearY": 0.0,
                },
            ),
        ),
        animations={"animation": {}},
        extras={"physics": [{"name": "unsupported"}]},
    )


def test_spine40_codec_rewrites_exact_legacy_4x_schema() -> None:
    document = _document()
    payload = json.loads(
        serialize_spine_document(
            document,
            SpineJsonTarget.SPINE_4_0,
        )
    )

    assert payload["skeleton"]["spine"] == "4.0.64"
    assert "referenceScale" not in payload["skeleton"]
    assert payload["bones"][1]["transform"] == "onlyTranslation"
    assert "inherit" not in payload["bones"][1]
    assert payload["skins"][0]["ik"] == ["ik"]
    assert payload["skins"][0]["transform"] == ["transform"]
    assert "constraints" not in payload["skins"][0]
    assert "physics" not in payload

    constraint = payload["transform"][0]
    assert constraint["order"] == 0
    assert constraint["target"] == "root"
    assert constraint["local"] is True
    assert constraint["relative"] is True
    assert constraint["mixRotate"] == 0.25
    assert constraint["mixX"] == 0.5
    assert constraint["mixY"] == 0.75
    assert constraint["mixScaleX"] == 0.6
    assert constraint["mixScaleY"] == 0.7
    assert constraint["mixShearY"] == 0.0


def test_spine40_codec_does_not_mutate_canonical_document() -> None:
    document = _document()

    serialize_spine_document(document, "4.0.64")

    assert document.skeleton["spine"] == "4.2.43"
    assert document.bones[1].extras == {"inherit": "onlyTranslation"}
    assert document.skins[0].constraints == ("ik", "transform")
    assert document.extras == {"physics": [{"name": "unsupported"}]}


def test_spine40_codec_rejects_setup_attachment_sequences() -> None:
    with pytest.raises(ValueError, match="does not support.*sequences") as exc_info:
        serialize_spine_document(
            _document(sequence=True),
            SpineJsonTarget.SPINE_4_0,
        )

    assert "document.skins[0].attachments.slot.mesh.sequence" in str(exc_info.value)


def test_spine40_codec_rejects_valid_animation_sequence_members() -> None:
    document = _document(sequence=True)
    animated = SpineDocument(
        skeleton=document.skeleton,
        bones=document.bones,
        slots=document.slots,
        skins=document.skins,
        ik=document.ik,
        transform=document.transform,
        animations={
            "animation": {
                "attachments": {
                    "default": {
                        "slot": {
                            "mesh": {
                                "sequence": [
                                    {"time": 0.0, "mode": "hold", "index": 0}
                                ]
                            }
                        }
                    }
                }
            }
        },
        extras=document.extras,
    )

    with pytest.raises(ValueError, match="does not support.*sequences") as exc_info:
        Spine40JsonCodec().to_json(
            animated,
            context=SpineJsonCodecContext(target=SpineJsonTarget.SPINE_4_0),
        )

    message = str(exc_info.value)
    assert "document.skins[0].attachments.slot.mesh.sequence" in message
    assert (
        "document.animations.animation.attachments.default.slot.mesh.sequence"
        in message
    )


def test_spine40_codec_rejects_wrong_context_target() -> None:
    with pytest.raises(ValueError, match="requires SPINE_4_0"):
        Spine40JsonCodec().to_json(
            _document(),
            context=SpineJsonCodecContext(target=SpineJsonTarget.SPINE_4_1),
        )
