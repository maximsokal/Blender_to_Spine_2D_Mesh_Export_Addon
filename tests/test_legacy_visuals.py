from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    Bone,
    Skin,
    Slot,
    SpineDocument,
    SpineSerializer,
    apply_legacy_visual_options,
    build_legacy_control_slots_and_attachments,
    build_legacy_preview_animation,
)


def _base_document() -> SpineDocument:
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(
            Bone(name="root"),
            Bone(name="Hero_rotation_X", parent="root"),
            Bone(name="Hero_rotation_Z", parent="root"),
            Bone(name="Hero_rotation_Y", parent="root"),
            Bone(name="Hero_main", parent="root"),
        ),
        slots=(Slot(name="Hero_Segment_0", bone="root", attachment="mesh"),),
        skins=(
            Skin(
                name="default",
                attachments={
                    "Hero_Segment_0": {
                        "mesh": {"type": "point", "x": 1, "y": 2}
                    }
                },
            ),
        ),
        animations={"animation": {}},
    )


def test_control_shapes_preserve_legacy_order_counts_and_colors():
    slots, attachments = build_legacy_control_slots_and_attachments("Hero")

    assert tuple(slot.name for slot in slots) == (
        "Hero_rotation_X",
        "Hero_rotation_Z",
        "Hero_rotation_Y",
        "Hero_main",
    )
    expected = {
        "Hero_rotation_X": (64, "ff0000ff", 128),
        "Hero_rotation_Z": (64, "002cffff", 128),
        "Hero_rotation_Y": (64, "00ff18ff", 128),
        "Hero_main": (24, "df00ffff", 48),
    }
    for name, (vertex_count, color, coordinate_count) in expected.items():
        payload = attachments[name][name]
        assert payload["type"] == "boundingbox"
        assert payload["vertexCount"] == vertex_count
        assert payload["color"] == color
        assert len(payload["vertices"]) == coordinate_count

    assert attachments["Hero_main"]["Hero_main"]["vertices"][:4] == [
        -21.11,
        20.72,
        -20.96,
        68.4,
    ]
    assert attachments["Hero_rotation_X"]["Hero_rotation_X"]["vertices"][-4:] == [
        0.46,
        -96.63,
        24.45,
        -90.78,
    ]


def test_preview_animation_matches_legacy_timelines():
    preview = build_legacy_preview_animation("Hero")
    bones = preview["bones"]

    assert tuple(bones) == (
        "Hero_rotation_Y",
        "Hero_rotation_Z",
        "Hero_rotation_X",
    )
    assert bones["Hero_rotation_Y"]["rotate"][1] == {
        "time": 2,
        "value": -360,
        "curve": [2.667, -360, 3.333, -360],
    }
    assert bones["Hero_rotation_Z"]["rotate"][-1] == {"time": 6}
    assert bones["Hero_rotation_X"]["rotate"][-1] == {
        "time": 8,
        "value": -360,
    }


def test_visual_options_are_applied_before_serialization():
    result = apply_legacy_visual_options(
        _base_document(),
        prefix="Hero",
        include_control_icons=True,
        include_preview_animation=True,
    )
    data = SpineSerializer().to_dict(result)

    assert tuple(slot["name"] for slot in data["slots"]) == (
        "Hero_rotation_X",
        "Hero_rotation_Z",
        "Hero_rotation_Y",
        "Hero_main",
        "Hero_Segment_0",
    )
    attachments = data["skins"][0]["attachments"]
    assert tuple(attachments) == (
        "Hero_rotation_X",
        "Hero_rotation_Z",
        "Hero_rotation_Y",
        "Hero_main",
        "Hero_Segment_0",
    )
    assert "preview" in data["animations"]
    assert data["animations"]["animation"] == {}


def test_visual_options_can_be_disabled_independently():
    base = _base_document()
    none = apply_legacy_visual_options(
        base,
        prefix="Hero",
        include_control_icons=False,
        include_preview_animation=False,
    )
    preview_only = apply_legacy_visual_options(
        base,
        prefix="Hero",
        include_control_icons=False,
        include_preview_animation=True,
    )
    icons_only = apply_legacy_visual_options(
        base,
        prefix="Hero",
        include_control_icons=True,
        include_preview_animation=False,
    )

    assert none == base
    assert tuple(slot.name for slot in preview_only.slots) == ("Hero_Segment_0",)
    assert "preview" in preview_only.animations
    assert tuple(slot.name for slot in icons_only.slots[:4]) == (
        "Hero_rotation_X",
        "Hero_rotation_Z",
        "Hero_rotation_Y",
        "Hero_main",
    )
    assert "preview" not in icons_only.animations
