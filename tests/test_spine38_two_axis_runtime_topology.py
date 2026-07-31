"""Regression contracts for Spine 3.8 two-axis runtime-cache safety."""

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
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.composition import (
    ConstraintOrderPolicy,
    SpineCompositionSettings,
    SpineDocumentComponent,
    compose_spine_documents,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_attachment_builder import (
    LegacyMeshDocumentBuildResult,
)
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
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.spine41_setup_safety import (
    calculate_spine41_setup_matrices,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.two_axis_scale_spine38 import (
    adapt_two_axis_document_for_spine38_with_report,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_codecs import (
    serialize_spine_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import SpineJsonTarget


def _assembly(
    target: SpineJsonTarget,
    *,
    prefix: str = "Cone",
) -> A1DocumentAssemblyResult:
    rig = build_rig(
        LegacyRigBuildRequest(
            prefix=prefix,
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
        spine_target=target,
    )
    document = SpineDocument(
        skeleton={"spine": target.exact_version},
        bones=rig.bones,
        slots=(),
        skins=(),
        ik=rig.ik,
        transform=rig.transform,
    )
    document_build = LegacyMeshDocumentBuildResult(
        rig=rig,
        requests=(),
        components=(),
        skins=(),
        document=document,
    )
    return A1DocumentAssemblyResult(
        settings=A1DocumentAssemblySettings(
            prefix=prefix,
            uv_layer_name="SpineBakeUV",
            image_path=f"images/{prefix}_Baked",
            attachment_width=256,
            attachment_height=256,
            center_x=0.0,
            center_y=0.0,
        ),
        rig=rig,
        z_groups=object(),
        projections=(),
        document_build=document_build,
    )


def _finalized(
    target: SpineJsonTarget,
    *,
    prefix: str = "Cone",
) -> A1DocumentAssemblyResult:
    return finalize_a1_document_assembly_for_target(
        _assembly(target, prefix=prefix),
        spine_target=target,
        prefix=prefix,
    )


def _transform_by_name(document: SpineDocument):
    return {constraint.name: constraint for constraint in document.transform}


def _ik_by_name(document: SpineDocument):
    return {constraint.name: constraint for constraint in document.ik}


def _orders_for_prefix(document: SpineDocument, prefix: str) -> tuple[int, ...]:
    transforms = _transform_by_name(document)
    ik = _ik_by_name(document)
    return (
        transforms[f"{prefix}_rotation_X_constraint"].order,
        ik[f"{prefix}_IK"].order,
        transforms[f"{prefix}_scale_rotate_X_constraint"].order,
        transforms[f"{prefix}_scale"].order,
        transforms[f"{prefix}_rotation_Y"].order,
    )


def test_spine38_depth_precedes_uniform_scale_and_scale_owns_wrappers() -> None:
    assembly = _assembly(SpineJsonTarget.SPINE_3_8)
    source_document = assembly.document
    profile = assembly.rig.profile

    finalized = finalize_a1_document_assembly_for_target(
        assembly,
        spine_target=SpineJsonTarget.SPINE_3_8,
        prefix="Cone",
    )
    document = finalized.document
    transforms = _transform_by_name(document)
    uniform_scale = transforms[profile.scale_constraint("Cone")]
    depth_scale = transforms[profile.scale_depth_constraint("Cone")]
    rotation_y = transforms[profile.rotation_y_constraint("Cone")]

    assert _orders_for_prefix(document, "Cone") == (0, 1, 2, 3, 4)

    wrappers = assembly.rig.info.sub_bone_scale_names
    layers = assembly.rig.info.sub_bone_names
    source_scale = _transform_by_name(source_document)[
        profile.scale_constraint("Cone")
    ]
    wrapper_by_layer = dict(zip(layers, wrappers, strict=True))
    expected_scale_bones = tuple(
        wrapper_by_layer.get(name, name) for name in source_scale.bones
    )

    assert depth_scale.bones == wrappers
    assert uniform_scale.bones == expected_scale_bones
    assert uniform_scale.bones.count(profile.scale_rotate_x_bone("Cone")) == 1
    assert set(wrappers).issubset(uniform_scale.bones)
    assert set(layers).isdisjoint(uniform_scale.bones)
    assert set(rotation_y.bones) == set(layers)

    final_bones = {bone.name: bone for bone in document.bones}
    for wrapper_name in wrappers:
        bridge_name = f"{wrapper_name}_spine41_bridge"
        wrapper = final_bones[wrapper_name]
        bridge = final_bones[bridge_name]
        assert wrapper.parent == bridge_name
        assert bridge.parent == profile.rotate_x_bone("Cone")
        assert bridge.extras == {"inherit": "onlyTranslation"}

    source_matrices = calculate_spine41_setup_matrices(source_document.bones)
    final_matrices = calculate_spine41_setup_matrices(document.bones)
    for name in (*wrappers, *layers):
        assert final_matrices[name] == source_matrices[name]


def test_spine38_standalone_composition_preserves_each_cache_safe_phase_block() -> None:
    first = _finalized(SpineJsonTarget.SPINE_3_8, prefix="Cone")
    second = _finalized(SpineJsonTarget.SPINE_3_8, prefix="Pyramid")

    composition = compose_spine_documents(
        (
            SpineDocumentComponent("first", first.document),
            SpineDocumentComponent("second", second.document),
        ),
        SpineCompositionSettings(
            shared_bone_names=("root",),
            constraint_order_policy=ConstraintOrderPolicy.REBASE_CONTIGUOUS,
        ),
    )

    assert _orders_for_prefix(composition.document, "Cone") == (0, 1, 2, 3, 4)
    assert _orders_for_prefix(composition.document, "Pyramid") == (5, 6, 7, 8, 9)


def test_spine38_codec_serializes_only_the_cache_safe_two_axis_topology() -> None:
    finalized = _finalized(SpineJsonTarget.SPINE_3_8)
    profile = finalized.rig.profile
    payload = json.loads(
        serialize_spine_document(
            finalized.document,
            SpineJsonTarget.SPINE_3_8,
        )
    )

    transform_by_name = {
        constraint["name"]: constraint for constraint in payload["transform"]
    }
    ik_by_name = {constraint["name"]: constraint for constraint in payload["ik"]}
    assert (
        transform_by_name[profile.rotation_x_constraint("Cone")]["order"],
        ik_by_name[profile.scale_ik_constraint("Cone")]["order"],
        transform_by_name[profile.scale_depth_constraint("Cone")]["order"],
        transform_by_name[profile.scale_constraint("Cone")]["order"],
        transform_by_name[profile.rotation_y_constraint("Cone")]["order"],
    ) == (0, 1, 2, 3, 4)
    assert set(
        transform_by_name[profile.scale_constraint("Cone")]["bones"]
    ) == {
        profile.scale_rotate_x_bone("Cone"),
        *finalized.rig.info.sub_bone_scale_names,
    }
    assert set(finalized.rig.info.sub_bone_names).isdisjoint(
        transform_by_name[profile.scale_constraint("Cone")]["bones"]
    )


def test_spine38_codec_rejects_the_previous_stale_child_topology() -> None:
    assembly = _assembly(SpineJsonTarget.SPINE_3_8)
    finalized = finalize_a1_document_assembly_for_target(
        assembly,
        spine_target=SpineJsonTarget.SPINE_3_8,
        prefix="Cone",
    )
    profile = finalized.rig.profile
    source_scale = _transform_by_name(assembly.document)[
        profile.scale_constraint("Cone")
    ]
    unsafe_scale_bones = tuple(
        profile.scale_rotate_x_bone("Cone")
        if name == profile.rotate_x_bone("Cone")
        else name
        for name in source_scale.bones
    )
    scale_name = profile.scale_constraint("Cone")
    depth_name = profile.scale_depth_constraint("Cone")
    unsafe_transform = tuple(
        replace(constraint, order=2, bones=unsafe_scale_bones)
        if constraint.name == scale_name
        else replace(constraint, order=3)
        if constraint.name == depth_name
        else constraint
        for constraint in finalized.document.transform
    )
    unsafe_document = replace(
        finalized.document,
        transform=unsafe_transform,
    )

    with pytest.raises(ValueError, match="X/IK/Depth/Scale/Y"):
        serialize_spine_document(
            unsafe_document,
            SpineJsonTarget.SPINE_3_8,
        )


def test_spine38_finalization_is_idempotent() -> None:
    first = _finalized(SpineJsonTarget.SPINE_3_8)
    second = finalize_a1_document_assembly_for_target(
        first,
        spine_target=SpineJsonTarget.SPINE_3_8,
        prefix="Cone",
    )

    assert second.document == first.document
    assert second.document_build.components == first.document_build.components
    assert second.document_build.skins == first.document_build.skins
    assert second.rig is first.rig


@pytest.mark.parametrize(
    "target",
    (SpineJsonTarget.SPINE_4_0, SpineJsonTarget.SPINE_4_1),
)
def test_spine40_and_spine41_keep_their_working_scale_before_depth_policy(
    target: SpineJsonTarget,
) -> None:
    assembly = _assembly(target)
    profile = assembly.rig.profile

    finalized = finalize_a1_document_assembly_for_target(
        assembly,
        spine_target=target,
        prefix="Cone",
    )
    document = finalized.document
    transforms = _transform_by_name(document)
    ik = _ik_by_name(document)

    rotation_x = transforms[profile.rotation_x_constraint("Cone")]
    scale_ik = ik[profile.scale_ik_constraint("Cone")]
    uniform_scale = transforms[profile.scale_constraint("Cone")]
    depth_scale = transforms[profile.scale_depth_constraint("Cone")]
    rotation_y = transforms[profile.rotation_y_constraint("Cone")]

    assert (
        rotation_x.order,
        scale_ik.order,
        uniform_scale.order,
        depth_scale.order,
        rotation_y.order,
    ) == (0, 1, 2, 3, 4)
    assert set(assembly.rig.info.sub_bone_names).issubset(uniform_scale.bones)
    assert set(assembly.rig.info.sub_bone_scale_names).isdisjoint(
        uniform_scale.bones
    )


def test_spine38_adapter_rejects_non_dense_phase_drift() -> None:
    assembly = _assembly(SpineJsonTarget.SPINE_3_8)
    profile = assembly.rig.profile
    scale_name = profile.scale_constraint("Cone")
    drifted_transform = tuple(
        replace(constraint, order=9)
        if constraint.name == scale_name
        else constraint
        for constraint in assembly.document.transform
    )
    drifted = replace(assembly.document, transform=drifted_transform)

    with pytest.raises(ValueError, match="must form either the canonical"):
        adapt_two_axis_document_for_spine38_with_report(
            drifted,
            profile=profile,
            prefix="Cone",
        )
