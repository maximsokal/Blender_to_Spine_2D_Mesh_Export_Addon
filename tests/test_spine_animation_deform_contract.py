from copy import deepcopy
from math import inf, nan

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    Bone,
    MeshAttachment,
    Skin,
    Slot,
    SpineDocument,
    SpineSerializer,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.deform_timeline_contract import (
    validate_animation_deform_timelines,
)


def unweighted_mesh(name="mesh"):
    return MeshAttachment(
        name=name,
        uvs=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        triangles=(0, 1, 2),
        vertices=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        hull=3,
    )


def weighted_mesh(name="weighted"):
    vertices = []
    for x, y in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)):
        vertices.extend(
            (
                2,
                0,
                x,
                y,
                0.5,
                1,
                x,
                y,
                0.5,
            )
        )
    return MeshAttachment(
        name=name,
        uvs=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        triangles=(0, 1, 2),
        vertices=tuple(vertices),
        hull=3,
    )


def deform_animation(
    frames,
    *,
    skin_name="default",
    slot_name="slot",
    attachment_name="mesh",
    extra_attachment_timelines=None,
):
    attachment_timelines = {"deform": frames}
    if extra_attachment_timelines:
        attachment_timelines.update(extra_attachment_timelines)
    return {
        "animation": {
            "attachments": {
                skin_name: {
                    slot_name: {
                        attachment_name: attachment_timelines,
                    }
                }
            }
        }
    }


def build_document(
    animations,
    *,
    skins=None,
    slots=None,
    bones=None,
):
    if bones is None:
        bones = (Bone("root"), Bone("child", parent="root"))
    if slots is None:
        slots = (Slot("slot", "root"),)
    if skins is None:
        skins = (
            Skin(
                "default",
                {"slot": {"mesh": unweighted_mesh()}},
            ),
        )
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=bones,
        slots=slots,
        skins=skins,
        animations=animations,
    )


def direct_validate(animations, *, skins=None, slot_names=("slot",)):
    if skins is None:
        skins = (Skin("default", {"slot": {"mesh": unweighted_mesh()}}),)
    validate_animation_deform_timelines(
        animations,
        skins=skins,
        slot_names=slot_names,
        path="document.animations",
    )


def test_unweighted_deform_is_accepted_and_preserved_without_defaults():
    animations = deform_animation(
        [
            {"curve": [0.25, 0.0, 0.75, 1.0]},
            {
                "time": 1.0,
                "offset": 2,
                "vertices": [0.25, -0.5],
                "curve": "stepped",
                "futureField": True,
            },
            {"time": 2.0, "vertices": []},
        ],
        extra_attachment_timelines={
            "sequence": [{"time": 0.0, "mode": "hold"}],
            "futureTimeline": {"enabled": True},
        },
    )
    source = deepcopy(animations)

    serialized = SpineSerializer().to_dict(build_document(animations))

    assert serialized["animations"] == source
    first = serialized["animations"]["animation"]["attachments"]["default"][
        "slot"
    ]["mesh"]["deform"][0]
    assert "time" not in first
    assert "vertices" not in first


def test_weighted_capacity_uses_two_coordinates_per_influence():
    mesh = weighted_mesh()
    skins = (Skin("default", {"slot": {"weighted": mesh}}),)
    animations = deform_animation(
        [{"vertices": [0.1] * 12}],
        attachment_name="weighted",
    )

    SpineSerializer().to_dict(build_document(animations, skins=skins))

    overflow = deform_animation(
        [{"vertices": [0.1] * 14}],
        attachment_name="weighted",
    )
    with pytest.raises(ValueError, match="exceeds deform capacity 12"):
        SpineSerializer().to_dict(build_document(overflow, skins=skins))


@pytest.mark.parametrize("attachment_type", ("boundingbox", "path", "clipping"))
def test_raw_vertex_attachment_types_are_supported(attachment_type):
    attachment = {
        "type": attachment_type,
        "vertexCount": 3,
        "vertices": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    }
    if attachment_type == "path":
        attachment["lengths"] = [1.0]
    skins = (Skin("default", {"slot": {attachment_type: attachment}}),)
    animations = deform_animation(
        [{"offset": 2, "vertices": [0.25, -0.25]}],
        attachment_name=attachment_type,
    )

    SpineSerializer().to_dict(build_document(animations, skins=skins))


def test_linked_mesh_inherits_parent_capacity_from_default_skin():
    skins = (
        Skin("default", {"slot": {"parent": weighted_mesh("parent")}}),
        Skin(
            "alternate",
            {
                "slot": {
                    "linked": {
                        "type": "linkedmesh",
                        "parent": "parent",
                        "skin": "default",
                        "timelines": False,
                    }
                }
            },
        ),
    )
    animations = deform_animation(
        [{"offset": 10, "vertices": [0.25, -0.25]}],
        skin_name="alternate",
        attachment_name="linked",
    )

    SpineSerializer().to_dict(build_document(animations, skins=skins))


def test_linked_mesh_without_skin_uses_default_parent_skin():
    skins = (
        Skin("default", {"slot": {"parent": unweighted_mesh("parent")}}),
        Skin(
            "alternate",
            {"slot": {"linked": {"type": "mesh", "parent": "parent"}}},
        ),
    )
    animations = deform_animation(
        [{"vertices": [0.0] * 6}],
        skin_name="alternate",
        attachment_name="linked",
    )

    SpineSerializer().to_dict(build_document(animations, skins=skins))


@pytest.mark.parametrize(
    "animations, expected",
    (
        (
            deform_animation([], skin_name="missing"),
            "undefined skin 'missing'",
        ),
        (
            deform_animation([], slot_name="missing"),
            "undefined slot 'missing'",
        ),
        (
            deform_animation([], attachment_name="missing"),
            "undefined attachment 'missing'",
        ),
    ),
)
def test_deform_references_must_exist(animations, expected):
    with pytest.raises(ValueError, match=expected):
        direct_validate(animations)


def test_attachment_must_exist_in_the_selected_skin_slot():
    skins = (
        Skin("default", {"slot": {"mesh": unweighted_mesh()}}),
        Skin("alternate", {}),
    )
    animations = deform_animation(
        [{}],
        skin_name="alternate",
    )

    with pytest.raises(ValueError, match="without attachments in skin 'alternate'"):
        direct_validate(animations, skins=skins)


@pytest.mark.parametrize("attachment_type", ("region", "point"))
def test_non_vertex_attachments_are_rejected(attachment_type):
    skins = (
        Skin(
            "default",
            {"slot": {"item": {"type": attachment_type}}},
        ),
    )
    animations = deform_animation([{}], attachment_name="item")

    with pytest.raises(ValueError, match="non-deformable attachment type"):
        direct_validate(animations, skins=skins)


def test_missing_linked_parent_is_rejected():
    skins = (
        Skin(
            "default",
            {"slot": {"linked": {"type": "linkedmesh", "parent": "missing"}}},
        ),
    )
    animations = deform_animation([{}], attachment_name="linked")

    with pytest.raises(ValueError, match="undefined attachment 'missing'"):
        direct_validate(animations, skins=skins)


def test_linked_parent_cycle_is_rejected():
    skins = (
        Skin(
            "default",
            {
                "slot": {
                    "a": {"type": "linkedmesh", "parent": "b"},
                    "b": {"type": "linkedmesh", "parent": "a"},
                }
            },
        ),
    )
    animations = deform_animation([{}], attachment_name="a")

    with pytest.raises(ValueError, match="linked mesh parent cycle"):
        direct_validate(animations, skins=skins)


@pytest.mark.parametrize("value", (None, True, 1, "attachments", (), []))
def test_attachments_section_must_be_a_mapping(value):
    with pytest.raises(TypeError, match="attachments must be a mapping"):
        direct_validate({"animation": {"attachments": value}})


@pytest.mark.parametrize("value", (None, True, 1, "skin", (), []))
def test_skin_timeline_payload_must_be_a_mapping(value):
    with pytest.raises(TypeError, match="attachments.default must be a mapping"):
        direct_validate({"animation": {"attachments": {"default": value}}})


@pytest.mark.parametrize("value", (None, True, 1, "slot", (), []))
def test_slot_timeline_payload_must_be_a_mapping(value):
    with pytest.raises(TypeError, match="attachments.default.slot must be a mapping"):
        direct_validate(
            {"animation": {"attachments": {"default": {"slot": value}}}}
        )


@pytest.mark.parametrize("value", (None, True, 1, "attachment", (), []))
def test_attachment_timeline_payload_must_be_a_mapping(value):
    with pytest.raises(TypeError, match="attachments.default.slot.mesh must be a mapping"):
        direct_validate(
            {
                "animation": {
                    "attachments": {"default": {"slot": {"mesh": value}}}
                }
            }
        )


@pytest.mark.parametrize("value", (None, True, 1, "timeline", {}))
def test_deform_timeline_must_be_a_sequence(value):
    with pytest.raises(TypeError, match="deform must be a list or tuple"):
        direct_validate(deform_animation(value))


def test_deform_timeline_cannot_be_empty():
    with pytest.raises(ValueError, match="deform cannot be empty"):
        direct_validate(deform_animation([]))


@pytest.mark.parametrize("value", (None, True, 1, "keyframe", (), []))
def test_deform_keyframe_must_be_a_mapping(value):
    with pytest.raises(TypeError, match=r"deform\[0\] must be a mapping"):
        direct_validate(deform_animation([value]))


@pytest.mark.parametrize("value", (None, True, "0", (), {}))
def test_deform_time_requires_a_number(value):
    with pytest.raises(TypeError, match=r"deform\[0\]\.time"):
        direct_validate(deform_animation([{"time": value}]))


@pytest.mark.parametrize("value", (nan, inf, -inf))
def test_deform_time_must_be_finite(value):
    with pytest.raises(ValueError, match=r"deform\[0\]\.time must be finite"):
        direct_validate(deform_animation([{"time": value}]))


def test_deform_times_are_non_decreasing_and_equal_times_are_allowed():
    direct_validate(deform_animation([{"time": 0.5}, {"time": 0.5}]))

    with pytest.raises(ValueError, match="previous deform time 1"):
        direct_validate(deform_animation([{"time": 1}, {"time": 0.5}]))


@pytest.mark.parametrize("value", (None, True, 1, "vertices", {}))
def test_vertices_must_be_a_sequence_when_present(value):
    with pytest.raises(TypeError, match=r"vertices must be a list or tuple"):
        direct_validate(deform_animation([{"vertices": value}]))


@pytest.mark.parametrize("value", (True, "0", None, (), {}))
def test_vertex_components_must_be_numbers(value):
    with pytest.raises(TypeError, match=r"vertices\[1\] must be a finite number"):
        direct_validate(deform_animation([{"vertices": [0.0, value]}]))


@pytest.mark.parametrize("value", (nan, inf, -inf))
def test_vertex_components_must_be_finite(value):
    with pytest.raises(ValueError, match=r"vertices\[1\] must be finite"):
        direct_validate(deform_animation([{"vertices": [0.0, value]}]))


def test_vertices_must_preserve_xy_pairs():
    with pytest.raises(ValueError, match="vertices must contain X/Y pairs"):
        direct_validate(deform_animation([{"vertices": [0.0]}]))


@pytest.mark.parametrize("value", (True, 1.5, "2", None, (), {}))
def test_consumed_offset_requires_an_integer(value):
    with pytest.raises(TypeError, match=r"offset must be int"):
        direct_validate(deform_animation([{"vertices": [], "offset": value}]))


def test_consumed_offset_must_be_non_negative_and_even():
    with pytest.raises(ValueError, match="offset must be non-negative"):
        direct_validate(deform_animation([{"vertices": [], "offset": -2}]))

    with pytest.raises(ValueError, match="preserve X/Y pair alignment"):
        direct_validate(deform_animation([{"vertices": [], "offset": 1}]))


def test_deform_range_cannot_exceed_attachment_capacity():
    with pytest.raises(ValueError, match=r"range \[4, 8\) exceeds deform capacity 6"):
        direct_validate(
            deform_animation([{"offset": 4, "vertices": [0.0] * 4}])
        )


def test_offset_without_vertices_is_runtime_inert_and_preserved():
    animations = deform_animation(
        [
            {"offset": "ignored-without-vertices"},
            {"time": 1},
        ]
    )
    source = deepcopy(animations)

    serialized = SpineSerializer().to_dict(build_document(animations))

    assert serialized["animations"] == source


@pytest.mark.parametrize(
    "curve",
    (
        "linear",
        [0.0, 1.0],
        [0.0, True, 1.0, 1.0],
    ),
)
def test_consumed_deform_curve_uses_shared_single_channel_contract(curve):
    with pytest.raises((TypeError, ValueError)):
        direct_validate(
            deform_animation(
                [
                    {"curve": curve},
                    {"time": 1},
                ]
            )
        )


def test_valid_deform_curves_and_inert_terminal_curve_are_preserved():
    animations = deform_animation(
        [
            {"curve": [0.25, 0.0, 0.75, 1.0]},
            {"time": 1, "curve": "stepped"},
            {"time": 2, "curve": {"terminalFutureMetadata": True}},
        ]
    )
    source = deepcopy(animations)

    serialized = SpineSerializer().to_dict(build_document(animations))

    assert serialized["animations"] == source


def test_serializer_revalidates_mutated_nested_deform_payload():
    animations = deform_animation([{"vertices": [0.0, 0.0]}])
    document = build_document(animations)
    animations["animation"]["attachments"]["default"]["slot"]["mesh"][
        "deform"
    ][0]["vertices"] = [0.0] * 8

    with pytest.raises(ValueError, match="exceeds deform capacity 6"):
        SpineSerializer().to_dict(document)
