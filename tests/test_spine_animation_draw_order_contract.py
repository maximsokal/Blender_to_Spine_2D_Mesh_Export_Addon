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


def build_document(slot_names, animations):
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=tuple(Slot(slot_name, "root") for slot_name in slot_names),
        skins=(Skin("default", {}),),
        animations=animations,
    )


@pytest.mark.parametrize(
    "slot_names, animations",
    (
        ((), {}),
        (("slot",), {"idle": {}}),
        (("slot",), {"idle": {"drawOrder": [{}]}}),
        (("slot",), {"idle": {"drawOrder": [{"offsets": []}]}}),
        (
            ("slot",),
            {
                "animation with spaces": {
                    "drawOrder": (
                        {"time": 0.5},
                        {"time": 0.5, "offsets": ()},
                    )
                }
            },
        ),
        (
            ("a", "b", "c"),
            {
                "swap": {
                    "drawOrder": [
                        {
                            "time": 0.25,
                            "offsets": [
                                {"slot": "a", "offset": 2},
                                {"slot": "c", "offset": -2},
                            ],
                        }
                    ]
                }
            },
        ),
        (
            ("slot",),
            {
                "future": {
                    "drawOrder": [
                        {
                            "futureField": {"enabled": True},
                            "offsets": [],
                        }
                    ],
                    "futureTimeline": {"enabled": True},
                }
            },
        ),
    ),
)
def test_valid_draw_order_timelines_are_accepted(slot_names, animations):
    document = build_document(slot_names, animations)

    assert SpineValidator().validate(document) == ()
    assert document.animations == animations


@pytest.mark.parametrize("value", (None, True, 1, "timeline", {}))
def test_draw_order_timeline_must_be_a_sequence(value):
    with pytest.raises(TypeError, match="drawOrder must be a list or tuple"):
        build_document(("slot",), {"idle": {"drawOrder": value}})


def test_draw_order_timeline_cannot_be_empty():
    with pytest.raises(ValueError, match="drawOrder cannot be empty"):
        build_document(("slot",), {"idle": {"drawOrder": []}})


@pytest.mark.parametrize("value", (None, True, 1, "keyframe", (), []))
def test_draw_order_keyframe_must_be_a_mapping(value):
    with pytest.raises(TypeError, match=r"drawOrder\[0\] must be a mapping"):
        build_document(("slot",), {"idle": {"drawOrder": [value]}})


@pytest.mark.parametrize("value", (None, True, "0", (), {}))
def test_draw_order_time_requires_a_number(value):
    with pytest.raises(
        TypeError,
        match=r"drawOrder\[0\]\.time must be a finite number",
    ):
        build_document(
            ("slot",),
            {"idle": {"drawOrder": [{"time": value}]}},
        )


@pytest.mark.parametrize("value", (nan, inf, -inf))
def test_draw_order_time_rejects_non_finite_values(value):
    with pytest.raises(ValueError) as error:
        build_document(
            ("slot",),
            {"idle": {"drawOrder": [{"time": value}]}},
        )

    assert "document.animations.idle.drawOrder[0].time" in str(error.value)


def test_draw_order_times_must_be_non_decreasing():
    with pytest.raises(ValueError, match="previous draw order time"):
        build_document(
            ("slot",),
            {
                "idle": {
                    "drawOrder": [
                        {"time": 1.0},
                        {"time": 0.5},
                    ]
                }
            },
        )


def test_omitted_draw_order_time_uses_runtime_zero_for_order_validation():
    with pytest.raises(ValueError, match="previous draw order time 1"):
        build_document(
            ("slot",),
            {
                "idle": {
                    "drawOrder": [
                        {"time": 1},
                        {},
                    ]
                }
            },
        )


@pytest.mark.parametrize("value", (None, True, 1, "offsets", {}))
def test_draw_order_offsets_must_be_a_sequence(value):
    with pytest.raises(TypeError, match="offsets must be a list or tuple"):
        build_document(
            ("slot",),
            {"idle": {"drawOrder": [{"offsets": value}]}},
        )


@pytest.mark.parametrize("value", (None, True, 1, "offset", (), []))
def test_draw_order_offset_entry_must_be_a_mapping(value):
    with pytest.raises(TypeError, match=r"offsets\[0\] must be a mapping"):
        build_document(
            ("slot",),
            {"idle": {"drawOrder": [{"offsets": [value]}]}},
        )


def test_draw_order_offset_requires_slot():
    with pytest.raises(ValueError, match=r"offsets\[0\]\.slot is required"):
        build_document(
            ("slot",),
            {"idle": {"drawOrder": [{"offsets": [{"offset": 0}]}]}},
        )


@pytest.mark.parametrize("value", (None, True, 1, (), {}))
def test_draw_order_slot_requires_a_string(value):
    with pytest.raises(TypeError, match=r"offsets\[0\]\.slot must be str"):
        build_document(
            ("slot",),
            {
                "idle": {
                    "drawOrder": [
                        {"offsets": [{"slot": value, "offset": 0}]}
                    ]
                }
            },
        )


def test_draw_order_slot_cannot_be_blank():
    with pytest.raises(ValueError, match=r"offsets\[0\]\.slot cannot be empty"):
        build_document(
            ("slot",),
            {
                "idle": {
                    "drawOrder": [
                        {"offsets": [{"slot": "   ", "offset": 0}]}
                    ]
                }
            },
        )


def test_draw_order_slot_must_exist():
    with pytest.raises(ValueError, match="references undefined slot 'missing'"):
        build_document(
            ("slot",),
            {
                "idle": {
                    "drawOrder": [
                        {"offsets": [{"slot": "missing", "offset": 0}]}
                    ]
                }
            },
        )


def test_draw_order_cannot_reference_ambiguous_duplicate_setup_slot():
    with pytest.raises(ValueError, match="duplicated setup slot 'duplicate'"):
        build_document(
            ("duplicate", "duplicate"),
            {
                "idle": {
                    "drawOrder": [
                        {"offsets": [{"slot": "duplicate", "offset": 0}]}
                    ]
                }
            },
        )


def test_draw_order_slot_cannot_be_repeated_in_one_keyframe():
    with pytest.raises(ValueError, match="duplicates slot 'a'"):
        build_document(
            ("a", "b"),
            {
                "idle": {
                    "drawOrder": [
                        {
                            "offsets": [
                                {"slot": "a", "offset": 1},
                                {"slot": "a", "offset": 0},
                            ]
                        }
                    ]
                }
            },
        )


def test_draw_order_offset_entries_must_follow_setup_slot_order():
    with pytest.raises(ValueError, match="must follow setup slot order"):
        build_document(
            ("a", "b"),
            {
                "idle": {
                    "drawOrder": [
                        {
                            "offsets": [
                                {"slot": "b", "offset": -1},
                                {"slot": "a", "offset": 1},
                            ]
                        }
                    ]
                }
            },
        )


def test_draw_order_offset_value_is_required():
    with pytest.raises(ValueError, match=r"offsets\[0\]\.offset is required"):
        build_document(
            ("slot",),
            {"idle": {"drawOrder": [{"offsets": [{"slot": "slot"}]}]}},
        )


@pytest.mark.parametrize("value", (True, 1.0, "1", None, (), {}))
def test_draw_order_offset_requires_a_strict_integer(value):
    with pytest.raises(TypeError, match=r"offsets\[0\]\.offset must be int"):
        build_document(
            ("slot",),
            {
                "idle": {
                    "drawOrder": [
                        {"offsets": [{"slot": "slot", "offset": value}]}
                    ]
                }
            },
        )


@pytest.mark.parametrize(
    "slot_name, offset",
    (
        ("a", -1),
        ("a", 2),
        ("b", -2),
        ("b", 1),
    ),
)
def test_draw_order_offset_target_must_stay_inside_slot_range(slot_name, offset):
    with pytest.raises(ValueError, match="outside draw order range"):
        build_document(
            ("a", "b"),
            {
                "idle": {
                    "drawOrder": [
                        {"offsets": [{"slot": slot_name, "offset": offset}]}
                    ]
                }
            },
        )


def test_draw_order_moved_slots_cannot_target_the_same_index():
    with pytest.raises(ValueError, match="already used by"):
        build_document(
            ("a", "b"),
            {
                "idle": {
                    "drawOrder": [
                        {
                            "offsets": [
                                {"slot": "a", "offset": 1},
                                {"slot": "b", "offset": 0},
                            ]
                        }
                    ]
                }
            },
        )


def test_serializer_preserves_draw_order_without_inserting_defaults():
    animations = {
        "idle": {
            "drawOrder": [
                {
                    "offsets": [],
                    "futureField": True,
                }
            ]
        }
    }
    source = deepcopy(animations)

    serialized = SpineSerializer().to_dict(build_document(("slot",), animations))

    assert serialized["animations"] == source
    keyframe = serialized["animations"]["idle"]["drawOrder"][0]
    assert keyframe == {"offsets": [], "futureField": True}
    assert "time" not in keyframe
