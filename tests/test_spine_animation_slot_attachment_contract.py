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


def build_document(
    *,
    slot_names=("slot",),
    skins=None,
    animations=None,
):
    if skins is None:
        skins=(
            Skin(
                "default",
                {"slot": {"A": {"type": "point"}}},
            ),
        )
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=tuple(Slot(slot_name, "root") for slot_name in slot_names),
        skins=skins,
        animations=animations or {},
    )


@pytest.mark.parametrize(
    "animations",
    (
        {},
        {"idle": {}},
        {"idle": {"slots": {}}},
        {
            "idle": {
                "slots": {
                    "slot": {
                        "rgba": [{"time": 0, "color": "ffffffff"}],
                    }
                }
            }
        },
        {
            "idle": {
                "slots": {
                    "slot": {
                        "attachment": [
                            {},
                            {"time": 0.5, "name": None},
                            {"time": 1.0, "name": "A"},
                        ]
                    }
                }
            }
        },
        {
            "animation with spaces": {
                "slots": {
                    "slot": {
                        "attachment": (
                            {"time": 0.5, "name": "A"},
                            {"time": 0.5},
                        )
                    }
                }
            }
        },
        {
            "future": {
                "slots": {
                    "slot": {
                        "attachment": [
                            {
                                "name": "A",
                                "futureField": {"enabled": True},
                            }
                        ],
                        "futureTimeline": {"enabled": True},
                    }
                },
                "futureSection": {"enabled": True},
            }
        },
    ),
)
def test_valid_slot_attachment_timelines_are_accepted(animations):
    document = build_document(animations=animations)

    assert SpineValidator().validate(document) == ()
    assert document.animations == animations


def test_attachment_name_may_reference_any_skin_for_the_slot():
    skins = (
        Skin("default", {}),
        Skin("alternate", {"slot": {"B": {"type": "point"}}}),
    )
    animations = {
        "idle": {
            "slots": {
                "slot": {
                    "attachment": [{"name": "B"}],
                }
            }
        }
    }

    document = build_document(skins=skins, animations=animations)

    assert document.animations == animations
    assert SpineValidator().validate(document) == ()


@pytest.mark.parametrize("value", (None, True, 1, "slots", (), []))
def test_animation_slots_section_must_be_a_mapping(value):
    with pytest.raises(TypeError, match="slots must be a mapping"):
        build_document(animations={"idle": {"slots": value}})


def test_slot_timeline_must_reference_setup_slot():
    with pytest.raises(ValueError, match="references undefined slot 'missing'"):
        build_document(
            animations={"idle": {"slots": {"missing": {}}}},
        )


def test_slot_timeline_path_uses_json_key_notation_for_spaces():
    with pytest.raises(ValueError) as error:
        build_document(
            animations={"animation with spaces": {"slots": {"missing slot": {}}}},
        )

    assert (
        'document.animations["animation with spaces"].slots["missing slot"]'
        in str(error.value)
    )


def test_slot_timeline_cannot_reference_ambiguous_duplicate_setup_slot():
    with pytest.raises(ValueError, match="duplicated setup slot 'duplicate'"):
        build_document(
            slot_names=("duplicate", "duplicate"),
            skins=(
                Skin(
                    "default",
                    {"duplicate": {"A": {"type": "point"}}},
                ),
            ),
            animations={
                "idle": {
                    "slots": {
                        "duplicate": {
                            "attachment": [{"name": "A"}],
                        }
                    }
                }
            },
        )


@pytest.mark.parametrize("value", (None, True, 1, "slot", (), []))
def test_slot_timeline_payload_must_be_a_mapping(value):
    with pytest.raises(TypeError, match="document.animations.idle.slots.slot"):
        build_document(
            animations={"idle": {"slots": {"slot": value}}},
        )


@pytest.mark.parametrize("value", (None, True, 1, "attachment", {}))
def test_attachment_timeline_must_be_a_sequence(value):
    with pytest.raises(TypeError, match="attachment must be a list or tuple"):
        build_document(
            animations={
                "idle": {
                    "slots": {
                        "slot": {"attachment": value},
                    }
                }
            },
        )


def test_attachment_timeline_cannot_be_empty():
    with pytest.raises(ValueError, match="attachment cannot be empty"):
        build_document(
            animations={
                "idle": {
                    "slots": {
                        "slot": {"attachment": []},
                    }
                }
            },
        )


@pytest.mark.parametrize("value", (None, True, 1, "keyframe", (), []))
def test_attachment_keyframe_must_be_a_mapping(value):
    with pytest.raises(TypeError, match=r"attachment\[0\] must be a mapping"):
        build_document(
            animations={
                "idle": {
                    "slots": {
                        "slot": {"attachment": [value]},
                    }
                }
            },
        )


@pytest.mark.parametrize("value", (None, True, "0", (), {}))
def test_attachment_time_requires_a_number(value):
    with pytest.raises(
        TypeError,
        match=r"attachment\[0\]\.time must be a finite number",
    ):
        build_document(
            animations={
                "idle": {
                    "slots": {
                        "slot": {
                            "attachment": [{"time": value}],
                        }
                    }
                }
            },
        )


@pytest.mark.parametrize("value", (nan, inf, -inf))
def test_attachment_time_rejects_non_finite_values(value):
    with pytest.raises(ValueError) as error:
        build_document(
            animations={
                "idle": {
                    "slots": {
                        "slot": {
                            "attachment": [{"time": value}],
                        }
                    }
                }
            },
        )

    assert "document.animations.idle.slots.slot.attachment[0].time" in str(
        error.value
    )


def test_attachment_times_must_be_non_decreasing():
    with pytest.raises(ValueError, match="previous attachment time"):
        build_document(
            animations={
                "idle": {
                    "slots": {
                        "slot": {
                            "attachment": [
                                {"time": 1, "name": "A"},
                                {"time": 0.5, "name": "A"},
                            ],
                        }
                    }
                }
            },
        )


def test_omitted_attachment_time_uses_runtime_zero_for_order_validation():
    with pytest.raises(ValueError, match="previous attachment time 1"):
        build_document(
            animations={
                "idle": {
                    "slots": {
                        "slot": {
                            "attachment": [
                                {"time": 1, "name": "A"},
                                {"name": "A"},
                            ],
                        }
                    }
                }
            },
        )


@pytest.mark.parametrize("value", (True, 1, (), {}))
def test_attachment_name_requires_a_string_or_null(value):
    with pytest.raises(TypeError, match=r"attachment\[0\]\.name must be str"):
        build_document(
            animations={
                "idle": {
                    "slots": {
                        "slot": {
                            "attachment": [{"name": value}],
                        }
                    }
                }
            },
        )


def test_attachment_name_cannot_be_blank():
    with pytest.raises(ValueError, match=r"attachment\[0\]\.name cannot be empty"):
        build_document(
            animations={
                "idle": {
                    "slots": {
                        "slot": {
                            "attachment": [{"name": "   "}],
                        }
                    }
                }
            },
        )


def test_attachment_name_must_exist_for_the_same_slot():
    skins = (
        Skin(
            "default",
            {
                "slot": {"A": {"type": "point"}},
                "other": {"B": {"type": "point"}},
            },
        ),
    )

    with pytest.raises(
        ValueError,
        match="undefined attachment 'B' for slot 'slot'",
    ):
        build_document(
            slot_names=("slot", "other"),
            skins=skins,
            animations={
                "idle": {
                    "slots": {
                        "slot": {
                            "attachment": [{"name": "B"}],
                        }
                    }
                }
            },
        )


def test_serializer_preserves_attachment_timeline_without_defaults():
    animations = {
        "idle": {
            "slots": {
                "slot": {
                    "attachment": [
                        {
                            "name": None,
                            "futureField": True,
                        },
                        {
                            "time": 1,
                            "name": "A",
                        },
                    ]
                }
            }
        }
    }
    source = deepcopy(animations)

    serialized = SpineSerializer().to_dict(
        build_document(animations=animations),
    )

    assert serialized["animations"] == source
    first_keyframe = serialized["animations"]["idle"]["slots"]["slot"][
        "attachment"
    ][0]
    assert first_keyframe == {"name": None, "futureField": True}
    assert "time" not in first_keyframe
