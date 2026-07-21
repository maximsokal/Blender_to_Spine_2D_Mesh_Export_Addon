from copy import deepcopy
from math import inf, nan

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    Bone,
    Skin,
    Slot,
    SpineDocument,
    SpineSerializer,
    SpineValidator,
)


def build_document(events, animations):
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=(Slot("slot", "root"),),
        skins=(Skin("default", {}),),
        animations=animations,
        events=events,
    )


@pytest.mark.parametrize(
    "events, animations",
    (
        ({}, {}),
        ({"hit": {}}, {"idle": {}}),
        (
            {"hit": {"int": 1, "float": 2.5, "string": "setup"}},
            {
                "idle": {
                    "events": [
                        {"name": "hit"},
                        {
                            "time": 0.25,
                            "name": "hit",
                            "int": 2,
                            "float": 3.5,
                            "string": "key",
                            "volume": 1.5,
                            "balance": -2.0,
                        },
                    ]
                }
            },
        ),
        (
            {"same-time": {}},
            {
                "animation with spaces": {
                    "events": (
                        {"time": 0.5, "name": "same-time"},
                        {"time": 0.5, "name": "same-time"},
                    )
                }
            },
        ),
        (
            {"future": {}},
            {
                "idle": {
                    "events": [
                        {
                            "name": "future",
                            "futureField": {"enabled": True},
                        }
                    ],
                    "futureTimeline": {"enabled": True},
                }
            },
        ),
    ),
)
def test_valid_animation_event_timelines_are_accepted(events, animations):
    document = build_document(events, animations)

    assert SpineValidator().validate(document) == ()
    assert document.events == events
    assert document.animations == animations


def test_setup_event_names_must_not_be_empty():
    with pytest.raises(ValueError, match="event name cannot be empty"):
        build_document({"": {}}, {})


@pytest.mark.parametrize("value", (None, True, 1, "animation", (), []))
def test_animation_value_must_be_a_mapping(value):
    with pytest.raises(TypeError) as error:
        build_document({}, {"animation with spaces": value})

    assert 'document.animations["animation with spaces"] must be a mapping' in str(
        error.value
    )


@pytest.mark.parametrize("value", (None, True, 1, "events", {}))
def test_event_timeline_must_be_a_sequence(value):
    with pytest.raises(TypeError, match="events must be a list or tuple"):
        build_document({"hit": {}}, {"idle": {"events": value}})


def test_event_timeline_cannot_be_empty():
    with pytest.raises(ValueError, match="events cannot be empty"):
        build_document({"hit": {}}, {"idle": {"events": []}})


@pytest.mark.parametrize("value", (None, True, 1, "key", (), []))
def test_event_keyframe_must_be_a_mapping(value):
    with pytest.raises(TypeError, match=r"events\[0\] must be a mapping"):
        build_document({"hit": {}}, {"idle": {"events": [value]}})


def test_event_keyframe_requires_name():
    with pytest.raises(ValueError, match=r"events\[0\]\.name is required"):
        build_document({"hit": {}}, {"idle": {"events": [{}]}})


@pytest.mark.parametrize("value", (None, True, 1, (), {}))
def test_event_keyframe_name_requires_a_string(value):
    with pytest.raises(TypeError, match=r"events\[0\]\.name must be str"):
        build_document({"hit": {}}, {"idle": {"events": [{"name": value}]}})


def test_event_keyframe_name_cannot_be_blank():
    with pytest.raises(ValueError, match=r"events\[0\]\.name cannot be empty"):
        build_document({"hit": {}}, {"idle": {"events": [{"name": "   "}]}})


def test_event_keyframe_must_reference_setup_event():
    with pytest.raises(ValueError, match="references undefined event 'missing'"):
        build_document({"hit": {}}, {"idle": {"events": [{"name": "missing"}]}})


@pytest.mark.parametrize("value", (None, True, "0", (), {}))
def test_event_time_requires_a_number(value):
    with pytest.raises(TypeError, match=r"events\[0\]\.time must be a finite number"):
        build_document(
            {"hit": {}},
            {"idle": {"events": [{"name": "hit", "time": value}]}},
        )


@pytest.mark.parametrize("value", (nan, inf, -inf))
def test_event_time_rejects_non_finite_values(value):
    with pytest.raises(ValueError) as error:
        build_document(
            {"hit": {}},
            {"idle": {"events": [{"name": "hit", "time": value}]}},
        )

    assert "document.animations.idle.events[0].time" in str(error.value)


def test_event_times_must_be_non_decreasing():
    with pytest.raises(ValueError, match="greater than or equal to the previous"):
        build_document(
            {"hit": {}},
            {
                "idle": {
                    "events": [
                        {"time": 1.0, "name": "hit"},
                        {"time": 0.5, "name": "hit"},
                    ]
                }
            },
        )


def test_omitted_event_time_uses_runtime_zero_for_order_validation():
    with pytest.raises(ValueError, match="previous event time 1"):
        build_document(
            {"hit": {}},
            {
                "idle": {
                    "events": [
                        {"time": 1, "name": "hit"},
                        {"name": "hit"},
                    ]
                }
            },
        )


@pytest.mark.parametrize("value", (True, 1.0, "1", None))
def test_event_keyframe_int_requires_a_strict_integer(value):
    with pytest.raises(TypeError, match=r"events\[0\]\.int must be int"):
        build_document(
            {"hit": {}},
            {"idle": {"events": [{"name": "hit", "int": value}]}},
        )


@pytest.mark.parametrize("value", (-(2**31), 2**31 - 1))
def test_event_keyframe_int_accepts_signed_32_bit_boundaries(value):
    document = build_document(
        {"hit": {}},
        {"idle": {"events": [{"name": "hit", "int": value}]}},
    )

    assert document.animations["idle"]["events"][0]["int"] == value


@pytest.mark.parametrize("value", (-(2**31) - 1, 2**31))
def test_event_keyframe_int_rejects_runtime_overflow(value):
    with pytest.raises(ValueError, match="signed 32-bit range"):
        build_document(
            {"hit": {}},
            {"idle": {"events": [{"name": "hit", "int": value}]}},
        )


@pytest.mark.parametrize("value", (None, True, 1, (), {}))
def test_event_keyframe_string_requires_a_string(value):
    with pytest.raises(TypeError, match=r"events\[0\]\.string must be str"):
        build_document(
            {"hit": {}},
            {"idle": {"events": [{"name": "hit", "string": value}]}},
        )


@pytest.mark.parametrize("field_name", ("float", "volume", "balance"))
@pytest.mark.parametrize("value", (None, True, "1", (), {}))
def test_event_keyframe_numeric_fields_require_numbers(field_name, value):
    with pytest.raises(TypeError) as error:
        build_document(
            {"hit": {}},
            {"idle": {"events": [{"name": "hit", field_name: value}]}},
        )

    assert (
        f"document.animations.idle.events[0].{field_name} must be a finite number"
        in str(error.value)
    )


@pytest.mark.parametrize("field_name", ("float", "volume", "balance"))
@pytest.mark.parametrize("value", (nan, inf, -inf))
def test_event_keyframe_numeric_fields_reject_non_finite_values(
    field_name,
    value,
):
    with pytest.raises(ValueError) as error:
        build_document(
            {"hit": {}},
            {"idle": {"events": [{"name": "hit", field_name: value}]}},
        )

    assert f"document.animations.idle.events[0].{field_name}" in str(error.value)


def test_volume_and_balance_are_not_given_unproven_ranges():
    animations = {
        "idle": {
            "events": [
                {
                    "name": "hit",
                    "volume": 2.0,
                    "balance": -3.0,
                }
            ]
        }
    }

    assert build_document({"hit": {}}, animations).animations == animations


def test_serializer_preserves_event_timeline_without_inserting_defaults():
    animations = {
        "idle": {
            "events": [
                {
                    "name": "hit",
                    "futureField": True,
                }
            ]
        }
    }
    source = deepcopy(animations)

    serialized = SpineSerializer().to_dict(build_document({"hit": {}}, animations))

    assert serialized["animations"] == source
    keyframe = serialized["animations"]["idle"]["events"][0]
    assert keyframe == {"name": "hit", "futureField": True}
    assert "time" not in keyframe
    assert "int" not in keyframe
    assert "float" not in keyframe
    assert "string" not in keyframe
