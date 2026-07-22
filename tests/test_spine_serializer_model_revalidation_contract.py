import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import (
    Bone,
    Skin,
    Slot,
    SpineDocument,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.serializer import SpineSerializer


def build_document():
    animations = {
        "idle": {
            "events": [{"name": "step"}],
            "drawOrder": [
                {"offsets": [{"slot": "slot", "offset": 0}]}
            ],
            "slots": {
                "slot": {"attachment": [{"name": "A"}]},
            },
        }
    }
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=(Slot("slot", "root"),),
        skins=(Skin("default", {"slot": {"A": {"type": "point"}}}),),
        animations=animations,
        events={"step": {}},
    )


def test_serializer_accepts_unchanged_model_animation_payload():
    document = build_document()

    serialized = SpineSerializer().to_dict(document)

    assert serialized["animations"] == document.animations
    assert serialized["events"] == document.events


def test_serializer_rejects_mutated_event_reference():
    document = build_document()
    document.animations["idle"]["events"][0]["name"] = "missing"

    with pytest.raises(ValueError, match="undefined event 'missing'"):
        SpineSerializer().to_dict(document)


def test_serializer_rejects_mutated_draw_order_reference():
    document = build_document()
    document.animations["idle"]["drawOrder"][0]["offsets"][0][
        "slot"
    ] = "missing"

    with pytest.raises(ValueError, match="undefined slot 'missing'"):
        SpineSerializer().to_dict(document)


def test_serializer_rejects_mutated_attachment_reference():
    document = build_document()
    document.animations["idle"]["slots"]["slot"]["attachment"][0][
        "name"
    ] = "missing"

    with pytest.raises(
        ValueError,
        match="undefined attachment 'missing' for slot 'slot'",
    ):
        SpineSerializer().to_dict(document)
