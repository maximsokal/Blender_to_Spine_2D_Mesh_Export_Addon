"""Focused schema checks for the Spine 3.8 codec."""

import json

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import (
    Bone,
    Skin,
    SpineDocument,
    TransformConstraint,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_codecs import (
    serialize_spine_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import SpineJsonTarget


def _document() -> SpineDocument:
    return SpineDocument(
        skeleton={"spine": "4.2.43", "referenceScale": 100},
        bones=(
            Bone("root"),
            Bone(
                "control",
                parent="root",
                color="ff0000ff",
                icon="ik",
                extras={"inherit": "onlyTranslation"},
            ),
        ),
        slots=(),
        skins=(Skin("default", {}),),
        transform=(
            TransformConstraint(
                "copy",
                0,
                ("control",),
                "root",
                extras={
                    "mixRotate": 0.25,
                    "mixX": 0.5,
                    "mixScaleX": 0.6,
                    "mixShearY": 0.0,
                },
            ),
        ),
        animations={},
    )


def test_spine38_rewrites_legacy_fields_without_mutating_input() -> None:
    document = _document()
    payload = json.loads(
        serialize_spine_document(document, SpineJsonTarget.SPINE_3_8)
    )

    assert payload["skeleton"]["spine"] == "3.8.99"
    assert "referenceScale" not in payload["skeleton"]
    assert payload["bones"][1]["transform"] == "onlyTranslation"
    assert "inherit" not in payload["bones"][1]
    assert "color" not in payload["bones"][1]
    assert "icon" not in payload["bones"][1]
    constraint = payload["transform"][0]
    assert constraint["rotateMix"] == 0.25
    assert constraint["translateMix"] == 0.5
    assert constraint["scaleMix"] == 0.6
    assert constraint["shearMix"] == 0.0
    assert "mixX" not in constraint
    assert "mixScaleX" not in constraint
    assert document.skeleton["spine"] == "4.2.43"
    assert document.transform[0].extras["mixX"] == 0.5
