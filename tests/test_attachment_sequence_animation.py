from dataclasses import replace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    AttachmentSequenceAnimationError,
    Bone,
    MeshAttachment,
    Skin,
    Slot,
    SpineDocument,
    apply_attachment_sequence_animations,
    build_attachment_sequence_timeline,
)


def _attachment(name, *, count):
    return MeshAttachment(
        name=name,
        path=f"images/{name}_",
        uvs=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        triangles=(0, 1, 2),
        vertices=(-50.0, 50.0, 50.0, 50.0, -50.0, -50.0),
        hull=3,
        edges=(0, 1, 1, 2, 2, 0),
        width=100.0,
        height=100.0,
        sequence={"count": count, "start": 0, "digits": 4, "setup": 1},
    )


def _document():
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=(
            Slot("Animated_slot", "root", attachment="Animated"),
            Slot("Hidden_slot", "root", attachment="Hidden"),
        ),
        skins=(
            Skin(
                name="default",
                attachments={
                    "Animated_slot": {"Animated": _attachment("Animated", count=3)},
                    "Hidden_slot": {"Hidden": _attachment("Hidden", count=2)},
                },
            ),
        ),
        animations={"animation": {}},
    )


def test_sequence_timeline_matches_legacy_v023_frame_schedule():
    assert build_attachment_sequence_timeline(3) == (
        {"mode": "loop", "delay": 0.0333},
        {"time": 0.0333, "mode": "loop", "index": 1},
        {"time": 0.0666, "mode": "loop", "index": 2},
    )


def test_sequence_builder_adds_typed_attachment_timelines_and_is_idempotent():
    first = apply_attachment_sequence_animations(_document())
    second = apply_attachment_sequence_animations(first)

    attachments = first.animations["animation"]["attachments"]["default"]
    assert attachments["Animated_slot"]["Animated"]["sequence"] == [
        {"mode": "loop", "delay": 0.0333},
        {"time": 0.0333, "mode": "loop", "index": 1},
        {"time": 0.0666, "mode": "loop", "index": 2},
    ]
    assert attachments["Hidden_slot"]["Hidden"]["sequence"][-1]["index"] == 1
    assert second == first


def test_sequence_builder_can_limit_generation_to_owned_slots():
    result = apply_attachment_sequence_animations(
        _document(),
        slot_names=("Animated_slot",),
    )

    skin_timelines = result.animations["animation"]["attachments"]["default"]
    assert set(skin_timelines) == {"Animated_slot"}


def test_sequence_builder_rejects_conflicting_existing_timeline():
    source = _document()
    conflicting = replace(
        source,
        animations={
            "animation": {
                "attachments": {
                    "default": {
                        "Animated_slot": {
                            "Animated": {
                                "sequence": [
                                    {"mode": "hold", "index": 0},
                                ],
                            },
                        },
                    },
                },
            },
        },
    )

    with pytest.raises(AttachmentSequenceAnimationError, match="Refusing to overwrite"):
        apply_attachment_sequence_animations(conflicting)


def test_sequence_builder_rejects_unknown_slot_filter():
    with pytest.raises(AttachmentSequenceAnimationError, match="unknown setup slots"):
        apply_attachment_sequence_animations(
            _document(),
            slot_names=("Missing_slot",),
        )
