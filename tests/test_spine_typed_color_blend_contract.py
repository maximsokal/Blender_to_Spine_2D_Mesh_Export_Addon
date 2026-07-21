import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    Bone,
    Skin,
    Slot,
    SpineDocument,
    SpineSerializer,
    SpineValidator,
)


def build_document(*, bone: Bone, slot: Slot) -> SpineDocument:
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(bone,),
        slots=(slot,),
        skins=(Skin("default", {}),),
        animations={"animation": {}},
    )


@pytest.mark.parametrize(
    "color",
    (None, "FFFFFFFF", "AABBCCDD", "aabbccdd", "00000000"),
)
def test_bone_accepts_optional_official_rgba_color(color):
    bone = Bone("root", color=color)
    document = build_document(bone=bone, slot=Slot("slot", "root"))

    assert SpineValidator().validate(document) == ()


@pytest.mark.parametrize(
    "color",
    (None, "FFFFFFFF", "AABBCCDD", "aabbccdd", "00000000"),
)
def test_slot_accepts_optional_official_rgba_color(color):
    slot = Slot("slot", "root", color=color)
    document = build_document(bone=Bone("root"), slot=slot)

    assert SpineValidator().validate(document) == ()


@pytest.mark.parametrize(
    "value, exception_type",
    (
        (True, TypeError),
        (1, TypeError),
        ("", ValueError),
        ("FFFFFF", ValueError),
        ("#FFFFFFFF", ValueError),
        ("GGGGGGGG", ValueError),
        ("123456789", ValueError),
    ),
)
@pytest.mark.parametrize("owner", ("bone", "slot"))
def test_bone_and_slot_reject_noncanonical_color(owner, value, exception_type):
    constructor = (
        (lambda: Bone("root", color=value))
        if owner == "bone"
        else (lambda: Slot("slot", "root", color=value))
    )

    with pytest.raises(exception_type, match="color"):
        constructor()


@pytest.mark.parametrize(
    "blend",
    (None, "normal", "additive", "multiply", "screen"),
)
def test_slot_accepts_only_official_blend_tokens(blend):
    slot = Slot("slot", "root", blend=blend)
    document = build_document(bone=Bone("root"), slot=slot)

    assert SpineValidator().validate(document) == ()


@pytest.mark.parametrize(
    "blend, exception_type",
    (
        (True, TypeError),
        (1, TypeError),
        ("", ValueError),
        ("NORMAL", ValueError),
        ("Normal", ValueError),
        (" normal", ValueError),
        ("normal ", ValueError),
        ("overlay", ValueError),
        ("erase", ValueError),
    ),
)
def test_slot_rejects_noncanonical_blend_tokens(blend, exception_type):
    with pytest.raises(exception_type, match="blend"):
        Slot("slot", "root", blend=blend)


def test_serializer_preserves_typed_color_and_blend_without_normalization():
    document = build_document(
        bone=Bone("root", color="aabbccdd"),
        slot=Slot("slot", "root", color="11223344", blend="multiply"),
    )

    serialized = SpineSerializer().to_dict(document)

    assert serialized["bones"][0]["color"] == "aabbccdd"
    assert serialized["slots"][0]["color"] == "11223344"
    assert serialized["slots"][0]["blend"] == "multiply"


def test_serializer_omits_optional_color_and_blend_fields_when_none():
    serialized = SpineSerializer().to_dict(
        build_document(bone=Bone("root"), slot=Slot("slot", "root"))
    )

    assert "color" not in serialized["bones"][0]
    assert "color" not in serialized["slots"][0]
    assert "blend" not in serialized["slots"][0]
