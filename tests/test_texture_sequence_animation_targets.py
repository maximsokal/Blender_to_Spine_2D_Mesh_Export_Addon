import json

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.baking import TextureSequenceTiming
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    Bone,
    MeshAttachment,
    Skin,
    Slot,
    SpineDocument,
    SpineJsonTarget,
    SpineTextureAnimationEncoding,
    finalize_texture_sequence_animation,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_codecs import (
    serialize_spine_document,
)


def _sequence_attachment() -> MeshAttachment:
    return MeshAttachment(
        name="Animated",
        path="images/Animated_Baked_",
        uvs=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        triangles=(0, 1, 2),
        vertices=(-50.0, 50.0, 50.0, 50.0, -50.0, -50.0),
        hull=3,
        edges=(0, 1, 1, 2, 2, 0),
        width=100.0,
        height=100.0,
        sequence={"count": 3, "start": 5, "digits": 4, "setup": 0},
    )


def _document() -> SpineDocument:
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=(Slot("Animated_slot", "root", attachment="Animated"),),
        skins=(
            Skin(
                name="default",
                attachments={
                    "Animated_slot": {"Animated": _sequence_attachment()},
                },
            ),
        ),
        animations={"animation": {}},
    )


def _timing() -> TextureSequenceTiming:
    return TextureSequenceTiming(
        scene_fps=30000,
        scene_fps_base=1001.0,
    )


def _contains_sequence(value: object) -> bool:
    if isinstance(value, dict):
        return "sequence" in value or any(
            _contains_sequence(child) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_sequence(child) for child in value)
    return False


@pytest.mark.parametrize(
    "target",
    (
        SpineJsonTarget.SPINE_4_1,
        SpineJsonTarget.SPINE_4_2,
        SpineJsonTarget.SPINE_4_3,
    ),
)
def test_native_targets_keep_one_attachment_and_compact_loop_timeline(
    target: SpineJsonTarget,
) -> None:
    result = finalize_texture_sequence_animation(
        _document(),
        target=target,
        timing=_timing(),
    )

    attachments = result.skins[0].attachments["Animated_slot"]
    assert tuple(attachments) == ("Animated",)
    assert attachments["Animated"].sequence == {
        "count": 3,
        "start": 5,
        "digits": 4,
        "setup": 0,
    }
    timeline = result.animations["animation"]["attachments"]["default"][
        "Animated_slot"
    ]["Animated"]["sequence"]
    assert timeline == [
        {"time": 0.0, "mode": "loop", "index": 0, "delay": 0.033367},
        {"time": 0.1001, "mode": "loop", "index": 0, "delay": 0.033367},
    ]
    assert result.skeleton["fps"] == pytest.approx(29.97003)
    assert target.texture_animation_encoding is SpineTextureAnimationEncoding.NATIVE_SEQUENCE


@pytest.mark.parametrize(
    "target",
    (
        SpineJsonTarget.SPINE_3_8,
        SpineJsonTarget.SPINE_4_0,
    ),
)
def test_legacy_targets_expand_frames_and_key_every_attachment(
    target: SpineJsonTarget,
) -> None:
    result = finalize_texture_sequence_animation(
        _document(),
        target=target,
        timing=_timing(),
    )

    attachments = result.skins[0].attachments["Animated_slot"]
    assert tuple(attachments) == (
        "Animated_0005",
        "Animated_0006",
        "Animated_0007",
    )
    assert result.slots[0].attachment == "Animated_0005"
    assert tuple(attachment.path for attachment in attachments.values()) == (
        "images/Animated_Baked_0005",
        "images/Animated_Baked_0006",
        "images/Animated_Baked_0007",
    )
    assert all(attachment.sequence is None for attachment in attachments.values())
    assert result.animations["animation"]["slots"]["Animated_slot"][
        "attachment"
    ] == [
        {"time": 0.0, "name": "Animated_0005"},
        {"time": 0.033367, "name": "Animated_0006"},
        {"time": 0.066733, "name": "Animated_0007"},
        {"time": 0.1001, "name": "Animated_0005"},
    ]
    assert target.texture_animation_encoding is SpineTextureAnimationEncoding.ATTACHMENT_SWAP

    encoded = json.loads(serialize_spine_document(result, target, indent=2))
    assert not _contains_sequence(encoded)
    assert encoded["slots"][0]["attachment"] == "Animated_0005"
    assert len(encoded["animations"]["animation"]["slots"]["Animated_slot"]["attachment"]) == 4


def test_static_document_is_returned_without_fps_or_animation_changes() -> None:
    source = SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=(Slot("Static_slot", "root", attachment="Static"),),
        skins=(
            Skin(
                name="default",
                attachments={
                    "Static_slot": {
                        "Static": MeshAttachment(
                            name="Static",
                            path="images/Static",
                            uvs=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
                            triangles=(0, 1, 2),
                            vertices=(-1.0, 1.0, 1.0, 1.0, -1.0, -1.0),
                            hull=3,
                        )
                    }
                },
            ),
        ),
        animations={},
    )

    assert finalize_texture_sequence_animation(
        source,
        target=SpineJsonTarget.SPINE_3_8,
        timing=_timing(),
    ) is source
