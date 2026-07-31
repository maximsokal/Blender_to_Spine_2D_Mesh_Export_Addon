"""Fail-closed axis-pair checks for Spine 3.8 combined mixes."""

import pytest

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


def _document(extras: dict[str, float]) -> SpineDocument:
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"), Bone("control", parent="root")),
        slots=(),
        skins=(Skin("default", {}),),
        transform=(
            TransformConstraint("copy", 0, ("control",), "root", extras=extras),
        ),
        animations={},
    )


@pytest.mark.parametrize(
    "extras, legacy_name, y_field",
    (
        ({"mixX": 0.25, "mixY": 0.75}, "translateMix", "mixY"),
        ({"mixScaleX": 0.2, "mixScaleY": 0.8}, "scaleMix", "mixScaleY"),
    ),
)
def test_spine38_rejects_unrepresentable_axis_pairs(
    extras: dict[str, float],
    legacy_name: str,
    y_field: str,
) -> None:
    with pytest.raises(ValueError, match=legacy_name) as exc_info:
        serialize_spine_document(_document(extras), SpineJsonTarget.SPINE_3_8)

    message = str(exc_info.value)
    assert y_field in message
    assert "document.transform[0]" in message


def test_spine38_accepts_explicit_equal_axis_pairs() -> None:
    serialize_spine_document(
        _document(
            {
                "mixX": 0.4,
                "mixY": 0.4,
                "mixScaleX": 0.7,
                "mixScaleY": 0.7,
            }
        ),
        SpineJsonTarget.SPINE_3_8,
    )
