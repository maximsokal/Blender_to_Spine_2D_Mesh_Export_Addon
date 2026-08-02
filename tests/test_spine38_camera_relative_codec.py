"""Spine 3.8 serialization regressions for rigid camera-relative two-axis rigs."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application.a1_document_assembly import (
    A1DocumentAssemblyResult,
    A1DocumentAssemblySettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_document_preparation import (
    finalize_a1_document_assembly_for_target,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_attachment_builder import (
    LegacyAttachmentVertex,
    LegacyMeshAttachmentRequest,
    build_legacy_mesh_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_rig_contracts import (
    LegacyRigBuildRequest,
    LegacyZGroup,
    LegacyZGroupOriginMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_builder import build_rig
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import (
    A1CameraLayerProjectionKind,
    A1RigProfile,
    A1RigSetupPoseMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_codecs.registry import (
    resolve_spine_json_codec,
    serialize_spine_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_codecs.v38 import (
    Spine38JsonCodec,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_codecs.v38_camera_relative import (
    Spine38CameraRelativeJsonCodec,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import (
    SpineJsonTarget,
)


_PREFIX = "CameraCodec"


def _assembly() -> A1DocumentAssemblyResult:
    target = SpineJsonTarget.SPINE_3_8
    rig = build_rig(
        LegacyRigBuildRequest(
            prefix=_PREFIX,
            texture_width=128,
            texture_height=128,
            z_groups=(LegacyZGroup(-4.5),),
            main_position_pixels=(19.0, -11.0),
            setup_pose_mode=A1RigSetupPoseMode.PREPROJECTED_SCREEN,
            z_group_origin_mode=LegacyZGroupOriginMode.OBJECT_ORIGIN,
            camera_layer_projection_kind=(
                A1CameraLayerProjectionKind.PERSPECTIVE
            ),
        ),
        A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        spine_target=target,
    )
    z_group_index = rig.info.z_groups[0].index
    request = LegacyMeshAttachmentRequest(
        slot_name=f"{_PREFIX}_Segment_0",
        attachment_name=f"{_PREFIX}_Segment_0",
        vertex_prefix=f"{_PREFIX}_Segment_0",
        image_path=f"images/{_PREFIX}_Baked",
        width=128,
        height=128,
        vertices=(
            LegacyAttachmentVertex(
                index=0,
                uv=(0.0, 0.0),
                bone_position_pixels=(-24.0, -20.0),
                z_group_index=z_group_index,
            ),
            LegacyAttachmentVertex(
                index=1,
                uv=(1.0, 0.0),
                bone_position_pixels=(24.0, -20.0),
                z_group_index=z_group_index,
            ),
            LegacyAttachmentVertex(
                index=2,
                uv=(0.5, 1.0),
                bone_position_pixels=(0.0, 20.0),
                z_group_index=z_group_index,
            ),
        ),
        triangles=(0, 1, 2),
        hull=3,
    )
    document_build = build_legacy_mesh_document(
        rig,
        (request,),
        skeleton_metadata={
            "spine": target.exact_version,
            "width": 128,
            "height": 128,
        },
    )
    return A1DocumentAssemblyResult(
        settings=A1DocumentAssemblySettings(
            prefix=_PREFIX,
            uv_layer_name="SpineBakeUV",
            image_path=f"images/{_PREFIX}_Baked",
            attachment_width=128,
            attachment_height=128,
            center_x=0.0,
            center_y=0.0,
        ),
        rig=rig,
        z_groups=object(),
        projections=(),
        document_build=document_build,
    )


def _finalized() -> A1DocumentAssemblyResult:
    return finalize_a1_document_assembly_for_target(
        _assembly(),
        spine_target=SpineJsonTarget.SPINE_3_8,
        prefix=_PREFIX,
    )


def _records(payload: dict[str, object], name: str) -> dict[str, dict[str, object]]:
    values = payload.get(name)
    assert isinstance(values, list)
    return {
        value["name"]: value
        for value in values
        if isinstance(value, dict) and isinstance(value.get("name"), str)
    }


def test_registry_uses_camera_relative_aware_spine38_codec() -> None:
    codec = resolve_spine_json_codec(SpineJsonTarget.SPINE_3_8)

    assert isinstance(codec, Spine38CameraRelativeJsonCodec)
    assert isinstance(codec, Spine38JsonCodec)


def test_camera_relative_spine38_document_serializes_without_position_scale() -> None:
    finalized = _finalized()

    payload = json.loads(
        serialize_spine_document(
            finalized.document,
            SpineJsonTarget.SPINE_3_8,
        )
    )
    transforms = _records(payload, "transform")
    ik = _records(payload, "ik")

    position_name = f"{_PREFIX}_scale_spine38_position"
    assert position_name not in transforms

    rotation_x = transforms[f"{_PREFIX}_rotation_X_constraint"]
    depth = transforms[f"{_PREFIX}_scale_rotate_X_constraint"]
    rotation_y = transforms[f"{_PREFIX}_rotation_Y"]
    scale = transforms[f"{_PREFIX}_scale"]
    scale_ik = ik[f"{_PREFIX}_IK"]

    assert (
        int(rotation_x.get("order", 0)),
        int(scale_ik.get("order", 0)),
        int(depth.get("order", 0)),
        int(rotation_y.get("order", 0)),
        int(scale.get("order", 0)),
    ) == (0, 1, 2, 3, 4)
    assert scale["bones"] == [_PREFIX]
    assert scale["target"] == f"{_PREFIX}_scale"
    assert float(scale["rotateMix"]) == 0.0
    assert float(scale["translateMix"]) == 0.0
    assert float(scale["scaleMix"]) == 1.0
    assert float(scale["shearMix"]) == 0.0


def test_codec_rejects_camera_relative_scale_retargeted_to_orbital_layer() -> None:
    finalized = _finalized()
    profile = finalized.rig.profile
    scale_name = profile.scale_constraint(_PREFIX)
    layer_name = finalized.rig.info.z_groups[0].bone_name
    malformed_transform = tuple(
        replace(constraint, bones=(layer_name,))
        if constraint.name == scale_name
        else constraint
        for constraint in finalized.document.transform
    )
    malformed = replace(finalized.document, transform=malformed_transform)

    with pytest.raises(ValueError, match="must constrain only object base"):
        serialize_spine_document(malformed, SpineJsonTarget.SPINE_3_8)
