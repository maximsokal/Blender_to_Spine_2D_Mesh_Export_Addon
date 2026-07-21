from copy import deepcopy
from math import inf, nan

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    Bone,
    Skin,
    Slot,
    SpineDocument,
    SpineSerializer,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.sequence_timeline_contract import (
    validate_animation_sequence_timelines,
)


def sequence_attachment(*, count=4, attachment_type="region"):
    return {
        "type": attachment_type,
        "sequence": {"count": count, "start": 0},
    }


def sequence_animation(
    frames,
    *,
    skin_name="default",
    slot_name="slot",
    attachment_name="item",
    extra_timelines=None,
):
    timelines = {"sequence": frames}
    if extra_timelines:
        timelines.update(extra_timelines)
    return {
        "animation": {
            "attachments": {
                skin_name: {
                    slot_name: {
                        attachment_name: timelines,
                    }
                }
            }
        }
    }


def build_document(animations, *, skins=None, slots=None):
    if slots is None:
        slots = (Slot("slot", "root"),)
    if skins is None:
        skins = (
            Skin(
                "default",
                {"slot": {"item": sequence_attachment()}},
            ),
        )
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=slots,
        skins=skins,
        animations=animations,
    )


def direct_validate(animations, *, skins=None, slot_names=("slot",)):
    if skins is None:
        skins = (
            Skin(
                "default",
                {"slot": {"item": sequence_attachment()}},
            ),
        )
    validate_animation_sequence_timelines(
        animations,
        skins=skins,
        slot_names=slot_names,
        path="document.animations",
    )


def test_all_sequence_modes_and_delay_inheritance_are_preserved():
    animations = sequence_animation(
        [
            {"mode": "hold", "index": 0, "delay": 0.1},
            {"time": 1, "mode": "once", "index": 1},
            {"time": 2, "mode": "loop", "index": 2, "delay": 0.2},
            {"time": 3, "mode": "pingpong", "index": 3},
            {"time": 4, "mode": "onceReverse", "index": 4},
            {"time": 5, "mode": "loopReverse", "index": 5},
            {
                "time": 6,
                "mode": "pingpongReverse",
                "index": 6,
                "futureField": True,
            },
        ],
        extra_timelines={
            "deform": [{"futureOnly": True}],
            "futureTimeline": {"enabled": True},
        },
    )
    source = deepcopy(animations)

    serialized = SpineSerializer().to_dict(build_document(animations))

    assert serialized["animations"] == source
    first = serialized["animations"]["animation"]["attachments"]["default"][
        "slot"
    ]["item"]["sequence"][0]
    assert "time" not in first


@pytest.mark.parametrize("attachment_type", ("region", "mesh", "linkedmesh"))
def test_supported_texture_region_attachment_types(attachment_type):
    attachment = sequence_attachment(attachment_type=attachment_type)
    if attachment_type == "mesh":
        attachment.update(
            {
                "uvs": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                "triangles": [0, 1, 2],
                "vertices": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                "hull": 3,
            }
        )
    skins = (Skin("default", {"slot": {"item": attachment}}),)

    SpineSerializer().to_dict(
        build_document(sequence_animation([{}]), skins=skins)
    )


def test_equal_sequence_times_are_allowed_and_decreasing_times_fail():
    direct_validate(sequence_animation([{"time": 0.5}, {"time": 0.5}]))

    with pytest.raises(ValueError, match="previous sequence time 1"):
        direct_validate(sequence_animation([{"time": 1}, {"time": 0.5}]))


@pytest.mark.parametrize(
    "animations, expected",
    (
        (sequence_animation([{}], skin_name="missing"), "undefined skin 'missing'"),
        (sequence_animation([{}], slot_name="missing"), "undefined slot 'missing'"),
        (
            sequence_animation([{}], attachment_name="missing"),
            "undefined attachment 'missing'",
        ),
    ),
)
def test_sequence_reference_chain_is_fail_closed(animations, expected):
    with pytest.raises(ValueError, match=expected):
        direct_validate(animations)


def test_attachment_must_exist_in_selected_skin_slot():
    skins = (
        Skin("default", {"slot": {"item": sequence_attachment()}}),
        Skin("alternate", {}),
    )
    with pytest.raises(ValueError, match="without attachments in skin 'alternate'"):
        direct_validate(
            sequence_animation([{}], skin_name="alternate"),
            skins=skins,
        )


@pytest.mark.parametrize("attachment_type", ("point", "boundingbox", "path", "clipping"))
def test_non_texture_region_attachments_are_rejected(attachment_type):
    skins = (
        Skin(
            "default",
            {
                "slot": {
                    "item": {
                        "type": attachment_type,
                        "sequence": {"count": 2, "start": 0},
                    }
                }
            },
        ),
    )
    with pytest.raises(ValueError, match="non-sequence attachment type"):
        direct_validate(sequence_animation([{}]), skins=skins)


def test_setup_sequence_is_required_and_must_have_positive_count():
    skins = (Skin("default", {"slot": {"item": {"type": "region"}}}),)
    with pytest.raises(ValueError, match="sequence is required"):
        direct_validate(sequence_animation([{}]), skins=skins)

    for sequence, error_type, expected in (
        (None, ValueError, "sequence is required"),
        ([], TypeError, "sequence must be a mapping"),
        ({"start": 0}, ValueError, "sequence.count is required"),
        ({"count": True}, TypeError, "sequence.count must be int"),
        ({"count": 0}, ValueError, "greater than or equal to 1"),
    ):
        skins = (
            Skin(
                "default",
                {"slot": {"item": {"type": "region", "sequence": sequence}}},
            ),
        )
        with pytest.raises(error_type, match=expected):
            direct_validate(sequence_animation([{}]), skins=skins)


@pytest.mark.parametrize("value", (None, True, 1, "attachments", (), []))
def test_attachments_section_must_be_mapping(value):
    with pytest.raises(TypeError, match="attachments must be a mapping"):
        direct_validate({"animation": {"attachments": value}})


@pytest.mark.parametrize("value", (None, True, 1, "timeline", {}))
def test_sequence_timeline_must_be_sequence(value):
    with pytest.raises(TypeError, match="sequence must be a list or tuple"):
        direct_validate(sequence_animation(value))


def test_sequence_timeline_cannot_be_empty():
    with pytest.raises(ValueError, match="sequence cannot be empty"):
        direct_validate(sequence_animation([]))


@pytest.mark.parametrize("value", (None, True, 1, "keyframe", (), []))
def test_sequence_keyframe_must_be_mapping(value):
    with pytest.raises(TypeError, match=r"sequence\[0\] must be a mapping"):
        direct_validate(sequence_animation([value]))


@pytest.mark.parametrize("value", (None, True, "0", (), {}))
def test_sequence_time_requires_number(value):
    with pytest.raises(TypeError, match=r"sequence\[0\]\.time"):
        direct_validate(sequence_animation([{"time": value}]))


@pytest.mark.parametrize("value", (nan, inf, -inf))
def test_sequence_time_must_be_finite(value):
    with pytest.raises(ValueError, match=r"sequence\[0\]\.time must be finite"):
        direct_validate(sequence_animation([{"time": value}]))


@pytest.mark.parametrize("value", (None, True, 1, (), {}))
def test_sequence_mode_requires_string(value):
    with pytest.raises(TypeError, match=r"sequence\[0\]\.mode must be str"):
        direct_validate(sequence_animation([{"mode": value}]))


@pytest.mark.parametrize("value", ("Hold", "LOOP", "linear", "", " pingpong "))
def test_sequence_mode_must_be_canonical(value):
    with pytest.raises(ValueError, match="mode must be one of"):
        direct_validate(sequence_animation([{"mode": value}]))


@pytest.mark.parametrize("value", (True, 1.5, "1", None, (), {}))
def test_sequence_index_requires_integer(value):
    with pytest.raises(TypeError, match=r"sequence\[0\]\.index must be int"):
        direct_validate(sequence_animation([{"index": value}]))


def test_sequence_index_must_be_non_negative_and_exactly_packable():
    with pytest.raises(ValueError, match="index must be non-negative"):
        direct_validate(sequence_animation([{"index": -1}]))

    direct_validate(sequence_animation([{"index": 1_048_575}]))
    with pytest.raises(ValueError, match="exact runtime frame packing"):
        direct_validate(sequence_animation([{"index": 1_048_576}]))


def test_sequence_index_may_exceed_setup_count_because_runtime_modes_resolve_it():
    direct_validate(sequence_animation([{"mode": "hold", "index": 100}]))
    direct_validate(
        sequence_animation([{"mode": "loop", "index": 100, "delay": 0.1}])
    )


@pytest.mark.parametrize("value", (True, "0.1", None, (), {}))
def test_sequence_delay_requires_number_when_present(value):
    with pytest.raises(TypeError, match=r"sequence\[0\]\.delay"):
        direct_validate(sequence_animation([{"delay": value}]))


@pytest.mark.parametrize("value", (nan, inf, -inf))
def test_sequence_delay_must_be_finite(value):
    with pytest.raises(ValueError, match=r"sequence\[0\]\.delay must be finite"):
        direct_validate(sequence_animation([{"delay": value}]))


def test_sequence_delay_cannot_be_negative():
    with pytest.raises(ValueError, match="delay must be non-negative"):
        direct_validate(sequence_animation([{"delay": -0.1}]))


def test_non_hold_mode_requires_positive_effective_delay():
    with pytest.raises(ValueError, match="delay must resolve to a value greater than 0"):
        direct_validate(sequence_animation([{"mode": "loop"}]))

    with pytest.raises(ValueError, match="delay must resolve to a value greater than 0"):
        direct_validate(
            sequence_animation(
                [
                    {"mode": "hold", "delay": 0},
                    {"time": 1, "mode": "once"},
                ]
            )
        )

    direct_validate(
        sequence_animation(
            [
                {"mode": "hold", "delay": 0.25},
                {"time": 1, "mode": "once"},
            ]
        )
    )


def test_serializer_revalidates_mutated_nested_sequence_payload():
    animations = sequence_animation([{"mode": "hold"}])
    document = build_document(animations)
    animations["animation"]["attachments"]["default"]["slot"]["item"][
        "sequence"
    ][0]["mode"] = "invalid"

    with pytest.raises(ValueError, match="mode must be one of"):
        SpineSerializer().to_dict(document)


def test_serializer_does_not_materialize_sequence_defaults():
    animations = sequence_animation([{"futureField": True}])
    source = deepcopy(animations)

    serialized = SpineSerializer().to_dict(build_document(animations))

    assert serialized["animations"] == source
    keyframe = serialized["animations"]["animation"]["attachments"]["default"][
        "slot"
    ]["item"]["sequence"][0]
    assert "time" not in keyframe
    assert "mode" not in keyframe
    assert "index" not in keyframe
    assert "delay" not in keyframe
