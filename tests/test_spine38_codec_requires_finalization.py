"""Fail-closed contract for direct Spine 3.8 two-axis serialization."""

from __future__ import annotations

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_rig_contracts import (
    LegacyRigBuildRequest,
    LegacyZGroup,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import SpineDocument
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_builder import build_rig
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import (
    A1RigProfile,
    A1RigSetupPoseMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_codecs import (
    serialize_spine_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import SpineJsonTarget


def _canonical_two_axis_document() -> SpineDocument:
    rig = build_rig(
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
        ),
        A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        spine_target=SpineJsonTarget.SPINE_3_8,
    )
    return SpineDocument(
        skeleton={"spine": SpineJsonTarget.SPINE_3_8.exact_version},
        bones=rig.bones,
        slots=(),
        skins=(),
        ik=rig.ik,
        transform=rig.transform,
    )


def test_spine38_codec_rejects_unfinalized_canonical_two_axis_document() -> None:
    document = _canonical_two_axis_document()

    with pytest.raises(
        ValueError,
        match="two-axis constraint inventory is incomplete",
    ):
        serialize_spine_document(
            document,
            SpineJsonTarget.SPINE_3_8,
        )
