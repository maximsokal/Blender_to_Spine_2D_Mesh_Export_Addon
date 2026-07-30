"""Spine 4.1 setup-matrix safety contracts for generated control rigs."""

from __future__ import annotations

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_rig_contracts import (
    LegacyRigBuildRequest,
    LegacyZGroup,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import (
    Bone,
    SpineDocument,
    TransformConstraint,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import (
    A1RigSetupPoseMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.spine41_setup_safety import (
    Spine41RigSafetyError,
    calculate_spine41_setup_matrices,
    find_spine41_unsafe_world_constraints,
    validate_spine41_setup_safety,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.two_axis_scale_rig_assembly import (
    build_two_axis_scale_rig,
)


def _document(
    *,
    local: bool = False,
    bridge_only_translation: bool = False,
) -> SpineDocument:
    bones = [
        Bone("root"),
        Bone("collapsed", parent="root", scale_x=0.0),
    ]
    parent_name = "collapsed"
    if bridge_only_translation:
        bones.append(
            Bone(
                "bridge",
                parent="collapsed",
                extras={"inherit": "onlyTranslation"},
            )
        )
        parent_name = "bridge"
    bones.extend(
        (
            Bone("constrained", parent=parent_name),
            Bone("target", parent="root"),
        )
    )
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=tuple(bones),
        slots=(),
        skins=(),
        transform=(
            TransformConstraint(
                name="world-transform",
                order=0,
                bones=("constrained",),
                target="target",
                extras={
                    "local": local,
                    "mixRotate": 1,
                    "mixX": 0,
                    "mixY": 0,
                    "mixScaleX": 0,
                    "mixScaleY": 0,
                    "mixShearY": 0,
                },
            ),
        ),
    )


def test_world_constraint_under_zero_scale_parent_is_rejected() -> None:
    document = _document()

    unsafe = find_spine41_unsafe_world_constraints(document)

    assert len(unsafe) == 1
    assert unsafe[0].constraint_name == "world-transform"
    assert unsafe[0].bone_name == "constrained"
    assert unsafe[0].parent_name == "collapsed"
    assert unsafe[0].parent_determinant == pytest.approx(0.0)
    with pytest.raises(Spine41RigSafetyError, match="singular parent"):
        validate_spine41_setup_safety(document)


def test_local_constraint_does_not_invert_the_parent_world_matrix() -> None:
    document = _document(local=True)

    assert find_spine41_unsafe_world_constraints(document) == ()
    validate_spine41_setup_safety(document)


def test_only_translation_bridge_breaks_singular_matrix_inheritance() -> None:
    document = _document(bridge_only_translation=True)

    matrices = calculate_spine41_setup_matrices(document.bones)

    assert matrices["collapsed"].determinant == pytest.approx(0.0)
    assert matrices["bridge"].determinant == pytest.approx(1.0)
    assert matrices["constrained"].determinant == pytest.approx(1.0)
    assert find_spine41_unsafe_world_constraints(document) == ()
    validate_spine41_setup_safety(document)


def test_current_two_axis_rig_is_explicitly_quarantined_for_spine_four_one() -> None:
    rig = build_two_axis_scale_rig(
        LegacyRigBuildRequest(
            prefix="Cone",
            texture_width=256,
            texture_height=256,
            z_groups=(
                LegacyZGroup(z_value=0.0, height_real_pixels=0.0),
                LegacyZGroup(z_value=1.0, height_real_pixels=128.0),
            ),
            main_position_pixels=(0.0, 0.0),
            setup_pose_mode=A1RigSetupPoseMode.NORMALIZED_SINGLE,
        )
    )
    document = SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=rig.bones,
        slots=(),
        skins=(),
        ik=rig.ik,
        transform=rig.transform,
    )

    unsafe = find_spine41_unsafe_world_constraints(document)
    unsafe_names = {item.constraint_name for item in unsafe}

    assert rig.profile.scale_constraint("Cone") in unsafe_names
    assert rig.profile.scale_depth_constraint("Cone") in unsafe_names
    with pytest.raises(Spine41RigSafetyError):
        validate_spine41_setup_safety(document)
