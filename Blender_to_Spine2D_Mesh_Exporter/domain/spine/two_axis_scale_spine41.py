"""Build the Spine 4.1-safe constraint variant of the two-axis scale rig.

The canonical two-axis hierarchy intentionally contains axis-collapse bones with
``scaleX == 0``. Spine 4.1 world-space transform constraints call
``Bone.updateAppliedTransform`` and must invert the constrained bone's parent matrix.
Two canonical constraints therefore require a target-aware builder representation:

- the uniform scale constraint becomes relative-local, so it updates applied scale
  without inverting the singular parent of ``<prefix>_rotate_X``;
- the depth scale constraint targets the final layer bones rather than their
  ``onlyTranslation`` wrappers, so every constrained bone has an invertible parent.

This module changes no setup bone scale, uses no epsilon replacement, and performs no
post-serialization JSON repair.
"""

from __future__ import annotations

from dataclasses import replace

from .legacy_rig_contracts import LegacyRigBuildResult
from .model import SpineDocument, TransformConstraint
from .rig_profiles import A1RigProfile, resolve_a1_rig_profile
from .spine41_setup_safety import validate_spine41_setup_safety
from .two_axis_scale_profile import TwoAxisScaleRigProfile


def _constraint_by_name(
    constraints: tuple[TransformConstraint, ...],
    name: str,
) -> TransformConstraint:
    matches = tuple(item for item in constraints if item.name == name)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one two-axis constraint named {name!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _adapt_uniform_scale_constraint(
    constraint: TransformConstraint,
) -> TransformConstraint:
    """Move relative scale evaluation from world space to applied local space."""

    extras = dict(constraint.extras)
    if extras.get("relative") is not True:
        raise ValueError(
            f"Constraint {constraint.name!r} must be relative before Spine 4.1 adaptation"
        )
    if extras.get("local", False) not in {False, None}:
        raise ValueError(
            f"Constraint {constraint.name!r} is already local and cannot be adapted"
        )
    for field_name in ("mixRotate", "mixX", "mixShearY"):
        if extras.get(field_name) != 0:
            raise ValueError(
                f"Constraint {constraint.name!r} requires {field_name}=0 before "
                "Spine 4.1 adaptation"
            )

    extras["local"] = True
    return replace(constraint, extras=extras)


def _adapt_depth_scale_constraint(
    constraint: TransformConstraint,
    *,
    source_wrapper_bones: tuple[str, ...],
    target_layer_bones: tuple[str, ...],
) -> TransformConstraint:
    """Retarget depth scaling to bones with invertible setup parents."""

    if constraint.bones != source_wrapper_bones:
        raise ValueError(
            f"Constraint {constraint.name!r} bone schema changed: "
            f"expected={source_wrapper_bones}, actual={constraint.bones}"
        )
    if len(target_layer_bones) != len(source_wrapper_bones):
        raise ValueError("Spine 4.1 depth target mapping must be one-to-one")
    return replace(constraint, bones=target_layer_bones)


def adapt_two_axis_scale_rig_for_spine41(
    rig: LegacyRigBuildResult,
) -> LegacyRigBuildResult:
    """Return a detached Spine 4.1-safe two-axis constraint variant.

    The input result is not mutated. Constraint names, targets, orders, bone hierarchy,
    and attachment-facing layer names remain stable.
    """

    if not isinstance(rig, LegacyRigBuildResult):
        raise TypeError("rig must be LegacyRigBuildResult")
    profile_id = resolve_a1_rig_profile(rig.profile.profile_id)
    if profile_id is not A1RigProfile.TWO_AXIS_ROTATION_SCALE:
        raise TypeError("Spine 4.1 two-axis adaptation requires TWO_AXIS_ROTATION_SCALE")
    if not isinstance(rig.profile, TwoAxisScaleRigProfile):
        raise TypeError("rig.profile must be TwoAxisScaleRigProfile")

    prefix = rig.request.prefix
    scale_name = rig.profile.scale_constraint(prefix)
    depth_name = rig.profile.scale_depth_constraint(prefix)
    source_scale = _constraint_by_name(rig.transform, scale_name)
    source_depth = _constraint_by_name(rig.transform, depth_name)

    adapted_by_name = {item.name: item for item in rig.transform}
    adapted_by_name[scale_name] = _adapt_uniform_scale_constraint(source_scale)
    adapted_by_name[depth_name] = _adapt_depth_scale_constraint(
        source_depth,
        source_wrapper_bones=rig.info.sub_bone_scale_names,
        target_layer_bones=rig.info.sub_bone_names,
    )
    adapted = replace(
        rig,
        transform=tuple(adapted_by_name[item.name] for item in rig.transform),
    )

    # The exact runtime remains the final acceptance oracle. This pure domain guard
    # catches the known singular-parent failure before any JSON is emitted.
    validate_spine41_setup_safety(
        SpineDocument(
            skeleton={"spine": "4.1.24"},
            bones=adapted.bones,
            slots=(),
            skins=(),
            ik=adapted.ik,
            transform=adapted.transform,
        )
    )
    return adapted


__all__ = ["adapt_two_axis_scale_rig_for_spine41"]
