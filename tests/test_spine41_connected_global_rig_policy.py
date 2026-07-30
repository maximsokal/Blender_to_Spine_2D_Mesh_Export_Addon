"""Spine 4.1-safe constraint policy for the connected global wrapper."""

from __future__ import annotations

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_contracts import (
    ConnectedZLayer,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import (
    IKConstraint,
    TransformConstraint,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.two_axis_scale_profile import (
    TwoAxisScaleRigProfile,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.two_axis_scale_spine41 import (
    adapt_connected_two_axis_constraints_for_spine41,
)


def _layers() -> tuple[ConnectedZLayer, ...]:
    return (
        ConnectedZLayer(
            layer_index=0,
            representative_relative_z=0.0,
            component_ids=("cone-a", "cone-b"),
            scale_bone_name="all_objects_0_scale",
            layer_bone_name="all_objects_layer_0",
        ),
        ConnectedZLayer(
            layer_index=1,
            representative_relative_z=1.0,
            component_ids=("cone-c",),
            scale_bone_name="all_objects_1_scale",
            layer_bone_name="all_objects_layer_1",
        ),
    )


def test_connected_global_policy_changes_only_scale_evaluation_ownership() -> None:
    profile = TwoAxisScaleRigProfile()
    prefix = "all_objects"
    ik = (
        IKConstraint(
            name=profile.scale_ik_constraint(prefix),
            order=4,
            bones=(profile.rotate_x_constraint_bone(prefix),),
            target=profile.rotate_x_constraint_ik_bone(prefix),
        ),
    )
    transform = (
        TransformConstraint(
            name=profile.scale_constraint(prefix),
            order=8,
            bones=(
                profile.rotate_x_bone(prefix),
                "all_objects_layer_0",
                "all_objects_layer_1",
            ),
            target=profile.scale_control_bone(prefix),
            extras={
                "relative": True,
                "mixRotate": 0,
                "mixX": 0,
                "mixY": 0,
                "mixShearY": 0,
            },
        ),
        TransformConstraint(
            name=profile.scale_depth_constraint(prefix),
            order=12,
            bones=("all_objects_0_scale", "all_objects_1_scale"),
            target=profile.rotate_x_constraint_bone(prefix),
            extras={
                "rotation": -90,
                "scaleX": -1,
                "mixRotate": 0,
                "mixX": 0,
                "mixShearY": 0,
            },
        ),
    )

    adapted_ik, adapted_transform = (
        adapt_connected_two_axis_constraints_for_spine41(
            ik,
            transform,
            profile=profile,
            group_prefix=prefix,
            layers=_layers(),
        )
    )

    assert adapted_ik is ik
    assert adapted_transform[0].name == transform[0].name
    assert adapted_transform[0].order == transform[0].order
    assert adapted_transform[0].bones == transform[0].bones
    assert adapted_transform[0].target == transform[0].target
    assert adapted_transform[0].extras == {**transform[0].extras, "local": True}

    assert adapted_transform[1].name == transform[1].name
    assert adapted_transform[1].order == transform[1].order
    assert adapted_transform[1].target == transform[1].target
    assert adapted_transform[1].extras == transform[1].extras
    assert adapted_transform[1].bones == (
        "all_objects_layer_0",
        "all_objects_layer_1",
    )


def test_connected_global_policy_is_idempotent() -> None:
    profile = TwoAxisScaleRigProfile()
    prefix = "all_objects"
    ik = ()
    transform = (
        TransformConstraint(
            name=profile.scale_constraint(prefix),
            order=0,
            bones=(profile.rotate_x_bone(prefix),),
            target=profile.scale_control_bone(prefix),
            extras={
                "local": True,
                "relative": True,
                "mixRotate": 0,
                "mixX": 0,
                "mixShearY": 0,
            },
        ),
        TransformConstraint(
            name=profile.scale_depth_constraint(prefix),
            order=1,
            bones=("all_objects_layer_0", "all_objects_layer_1"),
            target=profile.rotate_x_constraint_bone(prefix),
            extras={
                "rotation": -90,
                "scaleX": -1,
                "mixRotate": 0,
                "mixX": 0,
                "mixShearY": 0,
            },
        ),
    )

    first = adapt_connected_two_axis_constraints_for_spine41(
        ik,
        transform,
        profile=profile,
        group_prefix=prefix,
        layers=_layers(),
    )
    second = adapt_connected_two_axis_constraints_for_spine41(
        first[0],
        first[1],
        profile=profile,
        group_prefix=prefix,
        layers=_layers(),
    )

    assert second == first
