from copy import deepcopy
from math import inf, nan

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    Bone,
    Skin,
    Slot,
    SpineDocument,
    SpineSerializer,
    build_legacy_preview_animation,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.curve_timeline_contract import (
    validate_animation_curves,
)


def build_document(animations, *, bones=None, slots=None):
    if bones is None:
        bones=(Bone("root"),)
    if slots is None:
        slots=(Slot("slot", "root"),)
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=bones,
        slots=slots,
        skins=(Skin("default", {}),),
        animations=animations,
    )


def curve_values(channel_count):
    return [0.25, -10.0, 0.75, 10.0] * channel_count


def nested_animation(section, owner, timeline_name, curve):
    return {
        "animation": {
            section: {
                owner: {
                    timeline_name: [
                        {"curve": curve},
                        {"time": 1},
                    ]
                }
            }
        }
    }


def direct_animation(section, owner, curve):
    return {
        "animation": {
            section: {
                owner: [
                    {"curve": curve},
                    {"time": 1},
                ]
            }
        }
    }


@pytest.mark.parametrize(
    "section, owner, timeline_name, channel_count",
    (
        ("slots", "slot", "alpha", 1),
        ("slots", "slot", "rgb", 3),
        ("slots", "slot", "rgba", 4),
        ("slots", "slot", "rgb2", 6),
        ("slots", "slot", "rgba2", 7),
        ("bones", "root", "rotate", 1),
        ("bones", "root", "translate", 2),
        ("bones", "root", "scale", 2),
        ("bones", "root", "shear", 2),
        ("path", "path constraint", "position", 1),
        ("path", "path constraint", "spacing", 1),
        ("path", "path constraint", "mix", 3),
        ("physics", "", "inertia", 1),
        ("physics", "physics constraint", "mix", 1),
    ),
)
def test_nested_bezier_channel_counts_are_accepted(
    section,
    owner,
    timeline_name,
    channel_count,
):
    validate_animation_curves(
        nested_animation(
            section,
            owner,
            timeline_name,
            curve_values(channel_count),
        ),
        path="document.animations",
    )


@pytest.mark.parametrize(
    "section, owner, channel_count",
    (
        ("ik", "arm IK", 2),
        ("transform", "root transform", 6),
    ),
)
def test_direct_constraint_bezier_channel_counts_are_accepted(
    section,
    owner,
    channel_count,
):
    validate_animation_curves(
        direct_animation(section, owner, tuple(curve_values(channel_count))),
        path="document.animations",
    )


@pytest.mark.parametrize(
    "animations",
    (
        nested_animation("bones", "root", "rotate", "stepped"),
        nested_animation("slots", "slot", "rgba2", "stepped"),
        direct_animation("ik", "arm", "stepped"),
        direct_animation("transform", "root", "stepped"),
    ),
)
def test_exact_stepped_curve_is_accepted(animations):
    validate_animation_curves(animations, path="document.animations")


@pytest.mark.parametrize("curve", ("Stepped", "STEPPED", "linear", "", " stepped "))
def test_unknown_curve_strings_are_rejected(curve):
    with pytest.raises(ValueError, match='exactly "stepped"'):
        validate_animation_curves(
            nested_animation("bones", "root", "rotate", curve),
            path="document.animations",
        )


@pytest.mark.parametrize("curve", (None, True, 1, 1.5, {}, set()))
def test_curve_requires_stepped_or_numeric_sequence(curve):
    with pytest.raises(TypeError, match='exactly "stepped"'):
        validate_animation_curves(
            nested_animation("bones", "root", "rotate", curve),
            path="document.animations",
        )


@pytest.mark.parametrize(
    "channel_count, supplied_length",
    (
        (1, 0),
        (1, 3),
        (1, 5),
        (2, 4),
        (2, 7),
        (2, 9),
        (3, 8),
        (3, 13),
        (4, 12),
        (6, 20),
        (7, 24),
        (7, 29),
    ),
)
def test_bezier_length_must_equal_four_numbers_per_channel(
    channel_count,
    supplied_length,
):
    curve = [0.0] * supplied_length
    with pytest.raises(
        ValueError,
        match=rf"exactly {channel_count * 4} Bezier numbers",
    ):
        validate_animation_curves(
            nested_animation("slots", "slot", {
                1: "alpha",
                2: "translate",
                3: "rgb",
                4: "rgba",
                6: "rgb2",
                7: "rgba2",
            }[channel_count], curve)
            if channel_count != 2
            else nested_animation("bones", "root", "translate", curve),
            path="document.animations",
        )


@pytest.mark.parametrize("value", (True, "0", None, (), {}))
def test_bezier_components_must_be_numbers(value):
    curve = [0.0, value, 1.0, 1.0]
    with pytest.raises(TypeError, match=r"curve\[1\] must be a finite number"):
        validate_animation_curves(
            nested_animation("bones", "root", "rotate", curve),
            path="document.animations",
        )


@pytest.mark.parametrize("value", (nan, inf, -inf))
def test_bezier_components_must_be_finite(value):
    curve = [0.0, value, 1.0, 1.0]
    with pytest.raises(ValueError, match=r"curve\[1\] must be finite"):
        validate_animation_curves(
            nested_animation("bones", "root", "rotate", curve),
            path="document.animations",
        )


@pytest.mark.parametrize("value", (None, True, "0", (), {}))
def test_curve_timeline_time_requires_a_number(value):
    animations = {
        "animation": {
            "bones": {
                "root": {
                    "rotate": [
                        {"time": value},
                    ]
                }
            }
        }
    }
    with pytest.raises(TypeError, match=r"rotate\[0\]\.time"):
        validate_animation_curves(animations, path="document.animations")


@pytest.mark.parametrize("value", (nan, inf, -inf))
def test_curve_timeline_time_must_be_finite(value):
    animations = {
        "animation": {
            "bones": {
                "root": {
                    "rotate": [
                        {"time": value},
                    ]
                }
            }
        }
    }
    with pytest.raises(ValueError, match=r"rotate\[0\]\.time must be finite"):
        validate_animation_curves(animations, path="document.animations")


def test_curve_timeline_times_must_be_non_decreasing():
    animations = {
        "animation": {
            "bones": {
                "root": {
                    "rotate": [
                        {"time": 1},
                        {"time": 0.5},
                    ]
                }
            }
        }
    }
    with pytest.raises(ValueError, match="previous timeline time 1"):
        validate_animation_curves(animations, path="document.animations")


def test_equal_curve_timeline_times_are_allowed():
    animations = {
        "animation": {
            "bones": {
                "root": {
                    "rotate": [
                        {"time": 0.5, "curve": "stepped"},
                        {"time": 0.5},
                    ]
                }
            }
        }
    }
    validate_animation_curves(animations, path="document.animations")


@pytest.mark.parametrize("value", (None, True, 1, "section", (), []))
def test_known_curve_section_must_be_a_mapping(value):
    with pytest.raises(TypeError, match="document.animations.animation.bones"):
        validate_animation_curves(
            {"animation": {"bones": value}},
            path="document.animations",
        )


@pytest.mark.parametrize("value", (None, True, 1, "owner", (), []))
def test_nested_curve_owner_payload_must_be_a_mapping(value):
    with pytest.raises(TypeError, match="document.animations.animation.bones.root"):
        validate_animation_curves(
            {"animation": {"bones": {"root": value}}},
            path="document.animations",
        )


@pytest.mark.parametrize("value", (None, True, 1, "timeline", {}))
def test_known_curve_timeline_must_be_a_sequence(value):
    with pytest.raises(TypeError, match="rotate must be a list or tuple"):
        validate_animation_curves(
            {"animation": {"bones": {"root": {"rotate": value}}}},
            path="document.animations",
        )


def test_known_curve_timeline_cannot_be_empty():
    with pytest.raises(ValueError, match="rotate cannot be empty"):
        validate_animation_curves(
            {"animation": {"bones": {"root": {"rotate": []}}}},
            path="document.animations",
        )


@pytest.mark.parametrize("value", (None, True, 1, "keyframe", (), []))
def test_curve_keyframe_must_be_a_mapping(value):
    with pytest.raises(TypeError, match=r"rotate\[0\] must be a mapping"):
        validate_animation_curves(
            {"animation": {"bones": {"root": {"rotate": [value]}}}},
            path="document.animations",
        )


def test_unknown_and_discrete_timelines_are_preserved_without_curve_parsing():
    animations = {
        "animation": {
            "futureSection": {"curve": {"future": True}},
            "bones": {
                "root": {
                    "futureTimeline": {"curve": {"future": True}},
                    "inherit": [{"curve": {"future": True}}],
                }
            },
            "slots": {
                "slot": {
                    "attachment": [{"name": None, "curve": {"future": True}}],
                }
            },
            "physics": {
                "": {
                    "reset": [{"curve": {"future": True}}],
                }
            },
        }
    }
    source = deepcopy(animations)

    serialized = SpineSerializer().to_dict(build_document(animations))

    assert serialized["animations"] == source


def test_legacy_preview_curves_remain_valid_and_byte_preserved():
    preview = build_legacy_preview_animation("Hero")
    animations = {"preview": preview}
    source = deepcopy(animations)
    bones = (
        Bone("root"),
        Bone("Hero_rotation_X", parent="root"),
        Bone("Hero_rotation_Z", parent="root"),
        Bone("Hero_rotation_Y", parent="root"),
    )

    serialized = SpineSerializer().to_dict(
        build_document(animations, bones=bones),
    )

    assert serialized["animations"] == source
    assert serialized["animations"]["preview"]["bones"][
        "Hero_rotation_X"
    ]["rotate"][1]["curve"] == "stepped"
    assert serialized["animations"]["preview"]["bones"][
        "Hero_rotation_Y"
    ]["rotate"][1]["curve"] == [2.667, -360, 3.333, -360]


def test_serializer_revalidates_mutated_nested_curve_payload():
    animations = nested_animation(
        "bones",
        "root",
        "rotate",
        curve_values(1),
    )
    document = build_document(animations)
    animations["animation"]["bones"]["root"]["rotate"][0]["curve"] = [0, 1]

    with pytest.raises(ValueError, match="exactly 4 Bezier numbers"):
        SpineSerializer().to_dict(document)


def test_serializer_preserves_valid_curves_without_defaults_or_normalization():
    animations = {
        "animation": {
            "bones": {
                "root": {
                    "rotate": [
                        {
                            "curve": [0.25, -720, 0.75, 720],
                            "futureField": True,
                        },
                        {"time": 1},
                    ]
                }
            }
        }
    }
    source = deepcopy(animations)

    serialized = SpineSerializer().to_dict(build_document(animations))

    assert serialized["animations"] == source
    first = serialized["animations"]["animation"]["bones"]["root"][
        "rotate"
    ][0]
    assert "time" not in first
    assert first["curve"] == [0.25, -720, 0.75, 720]
