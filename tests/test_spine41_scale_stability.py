"""Regression contracts for Spine 4.1 world-constraint scale stability."""

from __future__ import annotations

import json

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import (
    Bone,
    MeshAttachment,
    Skin,
    Slot,
    SpineDocument,
    TransformConstraint,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_codecs import (
    serialize_spine_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import (
    SpineJsonTarget,
)


def _attachment() -> MeshAttachment:
    return MeshAttachment(
        name="mesh",
        uvs=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        triangles=(0, 1, 2),
        vertices=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        hull=3,
        width=64.0,
        height=64.0,
    )


def _document(
    bones: tuple[Bone, ...],
    transform: tuple[TransformConstraint, ...],
) -> SpineDocument:
    return SpineDocument(
        skeleton={"spine": "4.2.43", "images": "images/"},
        bones=bones,
        slots=(Slot("slot", bone="driven", attachment="mesh"),),
        skins=(Skin("default", {"slot": {"mesh": _attachment()}}),),
        transform=transform,
        animations={},
    )


def _bone_by_name(payload: dict[str, object], name: str) -> dict[str, object]:
    for bone in payload["bones"]:
        if bone["name"] == name:
            return bone
    raise AssertionError(f"Serialized bone is missing: {name}")


def test_v41_stabilizes_only_zero_scale_ancestry_used_by_world_constraints() -> None:
    document = _document(
        bones=(
            Bone("root"),
            Bone("axis-collapse", parent="root", scale_x=0.0),
            Bone("driven", parent="axis-collapse"),
            Bone("safe-zero", parent="root", scale_x=0.0),
            Bone("target", parent="root"),
        ),
        transform=(
            TransformConstraint(
                "world-transform",
                0,
                ("driven",),
                "target",
                extras={
                    "relative": True,
                    "mixRotate": 0.0,
                    "mixX": 0.0,
                    "mixY": 0.0,
                    "mixScaleX": 0.0,
                    "mixScaleY": 0.0,
                    "mixShearY": 0.0,
                },
            ),
        ),
    )

    payload = json.loads(
        serialize_spine_document(
            document,
            SpineJsonTarget.SPINE_4_1,
            indent=2,
        )
    )

    assert _bone_by_name(payload, "axis-collapse")["scaleX"] == pytest.approx(
        0.001
    )
    assert _bone_by_name(payload, "safe-zero")["scaleX"] == 0.0

    # Target conversion operates on a detached mapping and cannot mutate the canonical
    # typed document used by the 4.2 path.
    assert document.bones[1].scale_x == 0.0
    assert document.bones[3].scale_x == 0.0


def test_v41_keeps_zero_scale_outside_local_constraint_inverse_path() -> None:
    document = _document(
        bones=(
            Bone("root"),
            Bone("axis-collapse", parent="root", scale_x=0.0),
            Bone("driven", parent="axis-collapse"),
            Bone("target", parent="root"),
        ),
        transform=(
            TransformConstraint(
                "local-transform",
                0,
                ("driven",),
                "target",
                extras={
                    "local": True,
                    "relative": True,
                    "mixRotate": 0.0,
                    "mixX": 0.0,
                    "mixY": 0.0,
                    "mixScaleX": 0.0,
                    "mixScaleY": 0.0,
                    "mixShearY": 0.0,
                },
            ),
        ),
    )

    payload = json.loads(
        serialize_spine_document(document, SpineJsonTarget.SPINE_4_1)
    )

    assert _bone_by_name(payload, "axis-collapse")["scaleX"] == 0.0


def test_v41_fails_closed_when_nested_collapsed_axes_remain_near_singular() -> None:
    document = _document(
        bones=(
            Bone("root"),
            Bone("collapse-a", parent="root", scale_x=0.0),
            Bone("collapse-b", parent="collapse-a", scale_x=0.0),
            Bone("driven", parent="collapse-b"),
            Bone("target", parent="root"),
        ),
        transform=(
            TransformConstraint(
                "world-transform",
                0,
                ("driven",),
                "target",
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="retain singular parent matrices",
    ):
        serialize_spine_document(document, SpineJsonTarget.SPINE_4_1)
