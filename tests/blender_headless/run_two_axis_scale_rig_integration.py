"""Validate the generalized two-axis plus scale rig inside Blender 5.2.

This integration is intentionally render-free. It exercises the exact domain builder,
attachment weighting, profile-aware visuals, Spine validation, and deterministic JSON
serialization while running in Blender's bundled Python runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import traceback

import bpy


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (  # noqa: E402
    A1RigProfile,
    A1RigSetupPoseMode,
    LegacyAttachmentVertex,
    LegacyMeshAttachmentRequest,
    LegacyRigBuildRequest,
    LegacyZGroup,
    SpineSerializer,
    SpineValidator,
    apply_rig_visual_options,
    build_legacy_mesh_document,
    build_two_axis_scale_rig,
    decode_weighted_vertices,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _build_document():
    rig = build_two_axis_scale_rig(
        LegacyRigBuildRequest(
            prefix="TwoAxisFixture",
            texture_width=500,
            texture_height=500,
            z_groups=(
                LegacyZGroup(-1.0, height_real_pixels=-200.0),
                LegacyZGroup(1.0, height_real_pixels=300.0),
            ),
            main_position_pixels=(125.0, -50.0),
            setup_pose_mode=A1RigSetupPoseMode.NORMALIZED_SINGLE,
        )
    )
    request = LegacyMeshAttachmentRequest(
        slot_name="TwoAxisFixture_Segment_0",
        attachment_name="TwoAxisFixture_Segment_0",
        vertex_prefix="TwoAxisFixture_Segment_0",
        image_path="images/TwoAxisFixture_Baked.png",
        width=500.0,
        height=500.0,
        vertices=(
            LegacyAttachmentVertex(
                index=0,
                uv=(0.0, 0.0),
                bone_position_pixels=(-250.0, -250.0),
                z_group_index=1,
            ),
            LegacyAttachmentVertex(
                index=1,
                uv=(1.0, 0.0),
                bone_position_pixels=(250.0, -250.0),
                z_group_index=1,
            ),
            LegacyAttachmentVertex(
                index=2,
                uv=(1.0, 1.0),
                bone_position_pixels=(250.0, 250.0),
                z_group_index=2,
            ),
            LegacyAttachmentVertex(
                index=3,
                uv=(0.0, 1.0),
                bone_position_pixels=(-250.0, 250.0),
                z_group_index=2,
            ),
        ),
        triangles=(0, 1, 2, 0, 2, 3),
        hull=4,
        edges=(0, 1, 1, 2, 2, 3, 3, 0),
    )
    built = build_legacy_mesh_document(
        rig,
        (request,),
        skeleton_metadata={
            "spine": "4.2.43",
            "width": 500.0,
            "height": 500.0,
            "images": "./images/",
        },
    )
    document = apply_rig_visual_options(
        built.document,
        prefix="TwoAxisFixture",
        rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        include_control_icons=True,
        include_preview_animation=True,
    )
    return rig, built, document


def test_two_axis_scale_rig_serializes_in_blender_runtime() -> None:
    rig, built, document = _build_document()
    SpineValidator().validate_or_raise(document)

    bone_names = tuple(bone.name for bone in document.bones)
    _assert(
        "TwoAxisFixture_rotation_Z" not in bone_names,
        "two-axis profile unexpectedly generated a Z rotation control",
    )
    _assert(
        "TwoAxisFixture_scale" in bone_names,
        "two-axis profile did not generate the independent scale control",
    )
    bone_by_name = {bone.name: bone for bone in document.bones}
    main = bone_by_name["TwoAxisFixture_main"]
    internal_base = bone_by_name["TwoAxisFixture"]
    rotation_x = bone_by_name["TwoAxisFixture_rotation_X"]
    rotation_y = bone_by_name["TwoAxisFixture_rotation_Y"]
    scale = bone_by_name["TwoAxisFixture_scale"]
    _assert((main.x, main.y) == (0.0, 0.0), "single main must be neutral")
    _assert(
        (internal_base.x, internal_base.y) == (125.0, -50.0),
        "single placement was not transferred to the internal base",
    )
    _assert(rotation_x.rotation == 0.0, "X control setup rotation must be zero")
    _assert(rotation_y.rotation == 0.0, "Y control setup rotation must be zero")
    _assert(
        rotation_x.x == rotation_y.x == scale.x,
        "X, Y, and Scale controls must share one X column",
    )
    _assert(
        rotation_x.y - rotation_y.y == 200.0,
        "X and Y controls are not spaced by one control length",
    )
    _assert(
        rotation_y.y - scale.y == 200.0,
        "Y and Scale controls are not spaced by one control length",
    )

    rotate_x, rotate_y, _scale_constraint, _depth = rig.transform
    _assert(
        rotate_x.extras.get("rotation") == -134.67,
        "X reference angle was not transferred to the constraint offset",
    )
    _assert(
        rotate_y.extras.get("rotation") == -17.43,
        "Y reference angle was not transferred to the constraint offset",
    )

    combined_orders = tuple(item.order for item in (*rig.ik, *rig.transform))
    _assert(
        set(combined_orders) == {0, 1, 2, 3, 4},
        f"constraint orders must cover 0..4 exactly: {combined_orders}",
    )
    scale_constraint = next(
        item for item in rig.transform if item.name == "TwoAxisFixture_scale"
    )
    _assert(
        scale_constraint.bones
        == (
            "TwoAxisFixture_rotate_X",
            "TwoAxisFixture_2",
            "TwoAxisFixture_1",
        ),
        f"unexpected scale targets: {scale_constraint.bones}",
    )

    component = built.components[0]
    decoded = decode_weighted_vertices(
        component.attachment.vertices,
        expected_vertex_count=4,
    )
    expected_indices = tuple(
        component.vertex_bone_start_index + index for index in range(4)
    )
    actual_indices = tuple(vertex.influences[0].bone_index for vertex in decoded)
    _assert(
        actual_indices == expected_indices,
        f"weighted bone indices changed: expected={expected_indices}, actual={actual_indices}",
    )
    _assert(
        all(len(vertex.influences) == 1 for vertex in decoded),
        "every reference vertex must keep one full-weight influence",
    )

    serialized = SpineSerializer().to_dict(document)
    serialized_control_names = {
        bone["name"] for bone in serialized["bones"]
    }
    _assert(
        "TwoAxisFixture_scale" in serialized_control_names,
        "serialized JSON lost the scale control",
    )
    _assert(
        "TwoAxisFixture_rotation_Z" not in serialized_control_names,
        "serialized JSON contains a forbidden Z control",
    )
    preview_bones = set(serialized["animations"]["preview"]["bones"])
    _assert(
        preview_bones
        == {
            "TwoAxisFixture_rotation_X",
            "TwoAxisFixture_rotation_Y",
            "TwoAxisFixture_scale",
        },
        f"preview references unexpected controls: {preview_bones}",
    )

    with tempfile.TemporaryDirectory(prefix="spine2d-two-axis-rig-") as directory:
        output_path = Path(directory) / "TwoAxisFixture.json"
        SpineSerializer().write_json(document, output_path)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        _assert(payload["skeleton"]["spine"] == "4.2.43", "wrong Spine version")
        _assert(len(payload["ik"]) == 1, "reference rig must serialize one IK")
        _assert(
            len(payload["transform"]) == 4,
            "reference rig must serialize four Transform constraints",
        )


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    name = test_two_axis_scale_rig_serializes_in_blender_runtime.__name__
    print(f"[TWO_AXIS_SCALE_RIG] RUN {name}")
    test_two_axis_scale_rig_serializes_in_blender_runtime()
    print(f"[TWO_AXIS_SCALE_RIG] PASS {name}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
