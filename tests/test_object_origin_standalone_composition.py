"""Final standalone JSON contracts for independent per-object Blender pivots."""

from __future__ import annotations

import json

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.composition import (
    ConstraintOrderPolicy,
    SpineCompositionSettings,
    SpineDocumentComponent,
    compose_spine_documents,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_rig_contracts import (
    LegacyRigBuildRequest,
    LegacyZGroup,
    LegacyZGroupOriginMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import SpineDocument
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import (
    A1RigSetupPoseMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.two_axis_scale_rig import (
    build_two_axis_scale_rig,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_codecs import (
    serialize_spine_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import (
    SpineJsonTarget,
)


def _document(
    prefix: str,
    *,
    main_position_pixels: tuple[float, float],
    z_values: tuple[float, ...],
) -> SpineDocument:
    rig = build_two_axis_scale_rig(
        LegacyRigBuildRequest(
            prefix=prefix,
            texture_width=100,
            texture_height=100,
            z_groups=tuple(LegacyZGroup(value) for value in z_values),
            main_position_pixels=main_position_pixels,
            setup_pose_mode=A1RigSetupPoseMode.PRESERVE_COMPOSITION,
            z_group_origin_mode=LegacyZGroupOriginMode.OBJECT_ORIGIN,
        )
    )
    return SpineDocument(
        skeleton={"spine": SpineJsonTarget.SPINE_4_2.exact_version},
        bones=rig.bones,
        slots=(),
        skins=(),
        ik=rig.ik,
        transform=rig.transform,
    )


def _bone_by_name(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    bones = payload["bones"]
    assert isinstance(bones, list)
    return {
        str(bone["name"]): bone
        for bone in bones
        if isinstance(bone, dict) and "name" in bone
    }


def test_standalone_json_preserves_one_root_and_each_object_origin() -> None:
    composition = compose_spine_documents(
        (
            SpineDocumentComponent(
                "first",
                _document(
                    "First",
                    main_position_pixels=(125.0, -75.0),
                    z_values=(-1.0, 0.0, 2.0),
                ),
            ),
            SpineDocumentComponent(
                "second",
                _document(
                    "Second",
                    main_position_pixels=(-40.0, 210.0),
                    z_values=(1.0, 3.0),
                ),
            ),
        ),
        SpineCompositionSettings(
            shared_bone_names=("root",),
            constraint_order_policy=ConstraintOrderPolicy.REBASE_CONTIGUOUS,
        ),
    )
    payload = json.loads(
        serialize_spine_document(
            composition.document,
            SpineJsonTarget.SPINE_4_2,
        )
    )
    bones = payload["bones"]
    assert isinstance(bones, list)
    assert sum(
        1
        for bone in bones
        if isinstance(bone, dict) and bone.get("name") == "root"
    ) == 1

    by_name = _bone_by_name(payload)
    assert by_name["First_main"]["parent"] == "root"
    assert by_name["First_main"]["x"] == 125.0
    assert by_name["First_main"]["y"] == -75.0
    assert by_name["Second_main"]["parent"] == "root"
    assert by_name["Second_main"]["x"] == -40.0
    assert by_name["Second_main"]["y"] == 210.0

    assert by_name["First_1_scale"]["y"] == -100.0
    assert by_name["First_2_scale"].get("y", 0.0) == 0.0
    assert by_name["First_3_scale"]["y"] == 200.0
    assert by_name["Second_1_scale"]["y"] == 100.0
    assert by_name["Second_2_scale"]["y"] == 300.0


def test_standalone_json_does_not_create_missing_zero_depth_group() -> None:
    composition = compose_spine_documents(
        (
            SpineDocumentComponent(
                "positive",
                _document(
                    "Positive",
                    main_position_pixels=(0.0, 0.0),
                    z_values=(1.0, 2.0),
                ),
            ),
            SpineDocumentComponent(
                "negative",
                _document(
                    "Negative",
                    main_position_pixels=(10.0, 20.0),
                    z_values=(-3.0, -1.0),
                ),
            ),
        ),
        SpineCompositionSettings(
            shared_bone_names=("root",),
            constraint_order_policy=ConstraintOrderPolicy.REBASE_CONTIGUOUS,
        ),
    )
    by_name = _bone_by_name(
        json.loads(
            serialize_spine_document(
                composition.document,
                SpineJsonTarget.SPINE_4_2,
            )
        )
    )

    assert "Positive_3_scale" not in by_name
    assert "Negative_3_scale" not in by_name
    assert tuple(by_name[name]["y"] for name in ("Positive_1_scale", "Positive_2_scale")) == (
        100.0,
        200.0,
    )
    assert tuple(by_name[name]["y"] for name in ("Negative_1_scale", "Negative_2_scale")) == (
        -300.0,
        -100.0,
    )
