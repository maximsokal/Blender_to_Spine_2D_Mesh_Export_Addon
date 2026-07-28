"""Protect multi-object placement from single-object setup normalization."""

from __future__ import annotations

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    A1RigSetupPoseMode,
    LegacyRigBuildRequest,
    LegacyZGroup,
    SpineCompositionSettings,
    SpineDocument,
    SpineDocumentComponent,
    SpineValidator,
    build_two_axis_scale_rig,
    compose_spine_documents,
)


def _document(prefix: str, position: tuple[float, float]) -> SpineDocument:
    rig = build_two_axis_scale_rig(
        LegacyRigBuildRequest(
            prefix=prefix,
            texture_width=500,
            texture_height=500,
            z_groups=(
                LegacyZGroup(-1.0, height_real_pixels=-200.0),
                LegacyZGroup(1.0, height_real_pixels=300.0),
            ),
            main_position_pixels=position,
            setup_pose_mode=A1RigSetupPoseMode.PRESERVE_COMPOSITION,
        )
    )
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=rig.bones,
        slots=(),
        skins=(),
        ik=rig.ik,
        transform=rig.transform,
    )


def test_standalone_multi_composition_preserves_each_main_position_and_distance():
    first_position = (125.0, -50.0)
    second_position = (-275.0, 160.0)
    result = compose_spine_documents(
        (
            SpineDocumentComponent("first", _document("First", first_position)),
            SpineDocumentComponent("second", _document("Second", second_position)),
        ),
        SpineCompositionSettings(shared_bone_names=("root",)),
    )

    SpineValidator().validate_or_raise(result.document)
    bones = {bone.name: bone for bone in result.document.bones}
    first_main = bones["First_main"]
    second_main = bones["Second_main"]

    assert (first_main.x, first_main.y) == first_position
    assert (second_main.x, second_main.y) == second_position
    assert (
        second_main.x - first_main.x,
        second_main.y - first_main.y,
    ) == (
        second_position[0] - first_position[0],
        second_position[1] - first_position[1],
    )

    assert bones["First_rotation_X"].rotation == -134.67
    assert bones["First_rotation_Y"].rotation == -17.43
    assert bones["Second_rotation_X"].rotation == -134.67
    assert bones["Second_rotation_Y"].rotation == -17.43

    # Multi mode must not transfer placement into the internal base layer.
    assert bones["First"].x is None and bones["First"].y is None
    assert bones["Second"].x is None and bones["Second"].y is None
