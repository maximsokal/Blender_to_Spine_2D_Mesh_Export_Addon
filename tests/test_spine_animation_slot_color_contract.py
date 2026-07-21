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
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.slot_color_timeline_contract import (
    validate_animation_slot_color_timelines,
)


def build_document(animations, *, slot_names=("slot",)):
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=tuple(Slot(slot_name, "root") for slot_name in slot_names),
        skins=(Skin("default", {}),),
        animations=animations,
    )


def test_all_known_slot_color_timelines_are_accepted_and_preserved():
    animations = {
        "idle": {
            "slots": {
                "slot": {
                    "rgba": [
                        {"color": "FFFFFFFF"},
                        {
                            "time": 1,
                            "color": "aabbccdd",
                            "curve": "stepped",
                        },
                    ],
                    "rgb": [
                        {"color": "AABBCC"},
                        {"time": 1, "color": "001122"},
                    ],
                    "alpha": [
                        {},
                        {"time": 1, "value": 1.5},
                    ],
                    "rgba2": [
                        {"light": "FFFFFFFF", "dark": "000000"},
                    ],
                    "rgb2": [
                        {"light": "abcdef", "dark": "123456"},
                    ],
                    "futureTimeline": {"enabled": True},
                }
            }
        }
    }
    source = deepcopy(animations)
    document = build_document(animations)

    assert SpineValidator().validate(document) == ()
    assert SpineSerializer().to_dict(document)["animations"] == source
    assert document.animations == source


def test_equal_slot_color_keyframe_times_are_allowed():
    document = build_document(
        {
            "idle": {
                "slots": {
                    "slot": {
                        "rgba": [
                            {"time": 0.5, "color": "FFFFFFFF"},
                            {"time": 0.5, "color": "00000000"},
                        ]
                    }
                }
            }
        }
    )

    SpineSerializer().to_dict(document)


@pytest.mark.parametrize("timeline_name", ("rgba", "rgb", "alpha", "rgba2", "rgb2"))
@pytest.mark.parametrize("value", (None, True, 1, "timeline", {}))
def test_known_slot_color_timeline_must_be_a_sequence(timeline_name, value):
    document = build_document(
        {"idle": {"slots": {"slot": {timeline_name: value}}}}
    )

    with pytest.raises(TypeError, match="must be a list or tuple"):
        SpineSerializer().to_dict(document)


@pytest.mark.parametrize("timeline_name", ("rgba", "rgb", "alpha", "rgba2", "rgb2"))
def test_known_slot_color_timeline_cannot_be_empty(timeline_name):
    document = build_document(
        {"idle": {"slots": {"slot": {timeline_name: []}}}}
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        SpineSerializer().to_dict(document)


@pytest.mark.parametrize("value", (None, True, 1, "keyframe", (), []))
def test_slot_color_keyframe_must_be_a_mapping(value):
    document = build_document(
        {"idle": {"slots": {"slot": {"rgba": [value]}}}}
    )

    with pytest.raises(TypeError, match=r"rgba\[0\] must be a mapping"):
        SpineSerializer().to_dict(document)


@pytest.mark.parametrize("value", (None, True, "0", (), {}))
def test_slot_color_time_requires_a_number(value):
    document = build_document(
        {
            "idle": {
                "slots": {
                    "slot": {
                        "rgba": [{"time": value, "color": "FFFFFFFF"}],
                    }
                }
            }
        }
    )

    with pytest.raises(TypeError, match=r"rgba\[0\]\.time"):
        SpineSerializer().to_dict(document)


@pytest.mark.parametrize("value", (nan, inf, -inf))
def test_slot_color_time_rejects_non_finite_values(value):
    document = build_document(
        {
            "idle": {
                "slots": {
                    "slot": {
                        "rgba": [{"time": value, "color": "FFFFFFFF"}],
                    }
                }
            }
        }
    )

    with pytest.raises((ValueError, TypeError)) as error:
        SpineSerializer().to_dict(document)

    assert "document.animations.idle.slots.slot.rgba[0].time" in str(error.value)


def test_slot_color_times_must_be_non_decreasing():
    document = build_document(
        {
            "idle": {
                "slots": {
                    "slot": {
                        "rgb": [
                            {"time": 1, "color": "FFFFFF"},
                            {"time": 0.5, "color": "000000"},
                        ]
                    }
                }
            }
        }
    )

    with pytest.raises(ValueError, match="previous rgb time 1"):
        SpineSerializer().to_dict(document)


@pytest.mark.parametrize(
    "timeline_name, keyframe, missing_field",
    (
        ("rgba", {}, "color"),
        ("rgb", {}, "color"),
        ("rgba2", {"dark": "000000"}, "light"),
        ("rgba2", {"light": "FFFFFFFF"}, "dark"),
        ("rgb2", {"dark": "000000"}, "light"),
        ("rgb2", {"light": "FFFFFF"}, "dark"),
    ),
)
def test_slot_color_fields_are_required(timeline_name, keyframe, missing_field):
    document = build_document(
        {"idle": {"slots": {"slot": {timeline_name: [keyframe]}}}}
    )

    with pytest.raises(ValueError, match=rf"\.{missing_field} is required"):
        SpineSerializer().to_dict(document)


@pytest.mark.parametrize(
    "timeline_name, keyframe, expected_text",
    (
        ("rgba", {"color": "FFFFFF"}, "8 hexadecimal RGBA"),
        ("rgba", {"color": "#FFFFFFFF"}, "8 hexadecimal RGBA"),
        ("rgba", {"color": "GGGGGGGG"}, "8 hexadecimal RGBA"),
        ("rgb", {"color": "FFFFFFFF"}, "6 hexadecimal RGB"),
        ("rgb", {"color": "#FFFFFF"}, "6 hexadecimal RGB"),
        (
            "rgba2",
            {"light": "FFFFFF", "dark": "000000"},
            "8 hexadecimal RGBA",
        ),
        (
            "rgba2",
            {"light": "FFFFFFFF", "dark": "00000000"},
            "6 hexadecimal RGB",
        ),
        (
            "rgb2",
            {"light": "FFFFFFFF", "dark": "000000"},
            "6 hexadecimal RGB",
        ),
    ),
)
def test_slot_color_hex_formats_are_exact(
    timeline_name,
    keyframe,
    expected_text,
):
    document = build_document(
        {"idle": {"slots": {"slot": {timeline_name: [keyframe]}}}}
    )

    with pytest.raises(ValueError, match=expected_text):
        SpineSerializer().to_dict(document)


@pytest.mark.parametrize(
    "timeline_name, keyframe",
    (
        ("rgba", {"color": None}),
        ("rgb", {"color": True}),
        ("rgba2", {"light": 1, "dark": "000000"}),
        ("rgb2", {"light": "FFFFFF", "dark": {}}),
    ),
)
def test_slot_color_fields_require_strings(timeline_name, keyframe):
    document = build_document(
        {"idle": {"slots": {"slot": {timeline_name: [keyframe]}}}}
    )

    with pytest.raises(TypeError, match="must be str"):
        SpineSerializer().to_dict(document)


@pytest.mark.parametrize("value", (True, "1", None, (), {}))
def test_alpha_value_requires_a_number_when_present(value):
    document = build_document(
        {"idle": {"slots": {"slot": {"alpha": [{"value": value}]}}}}
    )

    with pytest.raises(TypeError, match=r"alpha\[0\]\.value"):
        SpineSerializer().to_dict(document)


@pytest.mark.parametrize("value", (nan, inf, -inf))
def test_alpha_value_rejects_non_finite_numbers(value):
    document = build_document(
        {"idle": {"slots": {"slot": {"alpha": [{"value": value}]}}}}
    )

    with pytest.raises(ValueError, match=r"alpha\[0\]\.value must be finite"):
        SpineSerializer().to_dict(document)


def test_unknown_slot_timeline_and_curve_are_preserved_without_validation():
    animations = {
        "idle": {
            "slots": {
                "slot": {
                    "futureTimeline": None,
                    "rgba": [
                        {
                            "color": "FFFFFFFF",
                            "curve": {"futureCurve": True},
                        }
                    ],
                }
            }
        }
    }

    assert SpineSerializer().to_dict(build_document(animations))["animations"] == (
        animations
    )


def test_serializer_revalidates_mutated_nested_animation_payload():
    animations = {
        "idle": {
            "slots": {
                "slot": {
                    "rgba": [{"color": "FFFFFFFF"}],
                }
            }
        }
    }
    document = build_document(animations)
    animations["idle"]["slots"]["slot"]["rgba"][0]["color"] = "invalid"

    with pytest.raises(ValueError, match="8 hexadecimal RGBA"):
        SpineSerializer().to_dict(document)


def test_direct_contract_rejects_undefined_and_ambiguous_slots():
    with pytest.raises(ValueError, match="undefined slot 'missing'"):
        validate_animation_slot_color_timelines(
            {
                "animation with spaces": {
                    "slots": {
                        "missing": {
                            "rgba": [{"color": "FFFFFFFF"}],
                        }
                    }
                }
            },
            slot_names=("slot",),
            path="document.animations",
        )

    with pytest.raises(ValueError, match="duplicated setup slot 'duplicate'"):
        validate_animation_slot_color_timelines(
            {
                "idle": {
                    "slots": {
                        "duplicate": {
                            "rgb": [{"color": "FFFFFF"}],
                        }
                    }
                }
            },
            slot_names=("duplicate", "duplicate"),
            path="document.animations",
        )
