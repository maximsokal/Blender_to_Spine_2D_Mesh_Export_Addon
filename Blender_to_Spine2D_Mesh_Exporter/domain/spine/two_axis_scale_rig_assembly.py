"""Orchestrate complete two-axis rotation plus scale rig construction."""

from __future__ import annotations

import logging

from .legacy_rig_contracts import LegacyRigBuildRequest, LegacyRigBuildResult
from .legacy_rig_error import LegacyRigBuildError
from .legacy_rig_plan import build_legacy_rig_plan
from .legacy_rig_validation import validate_legacy_rig_plan
from .two_axis_scale_profile import TwoAxisScaleRigProfile
from .two_axis_scale_rig_bones import build_two_axis_scale_bones
from .two_axis_scale_rig_constraints import build_two_axis_scale_constraints
from .two_axis_scale_rig_plan import build_two_axis_scale_layout
from .two_axis_scale_rig_validation import validate_two_axis_scale_rig_result


logger = logging.getLogger(__name__)


def build_two_axis_scale_rig(
    request: LegacyRigBuildRequest,
    profile: TwoAxisScaleRigProfile | None = None,
) -> LegacyRigBuildResult:
    """Build the complete generalized X/Y rotation plus scale rig."""

    if not isinstance(request, LegacyRigBuildRequest):
        raise TypeError("request must be LegacyRigBuildRequest")
    resolved_profile = TwoAxisScaleRigProfile() if profile is None else profile
    if not isinstance(resolved_profile, TwoAxisScaleRigProfile):
        raise TypeError("profile must be TwoAxisScaleRigProfile")

    stage = "PLAN"
    try:
        plan = build_legacy_rig_plan(request, resolved_profile)
        validate_legacy_rig_plan(plan)

        stage = "LAYOUT"
        layout = build_two_axis_scale_layout(plan)

        stage = "BONES"
        bones = build_two_axis_scale_bones(plan, layout)

        stage = "CONSTRAINTS"
        ik, transform = build_two_axis_scale_constraints(plan, layout)

        stage = "RESULT"
        result = LegacyRigBuildResult(
            request=request,
            profile=resolved_profile,
            bones=bones,
            ik=ik,
            transform=transform,
            info=plan.info,
        )
        validate_two_axis_scale_rig_result(result)
        logger.debug(
            "Built two-axis scale rig for %s: bones=%d z_groups=%d",
            request.prefix,
            len(result.bones),
            len(result.info.z_groups),
        )
        return result
    except LegacyRigBuildError as exc:
        raise LegacyRigBuildError(
            f"Unable to build two-axis scale rig for '{request.prefix}' at {stage}: {exc}"
        ) from exc
    except Exception as exc:
        raise LegacyRigBuildError(
            f"Unable to build two-axis scale rig for '{request.prefix}' at {stage}: {exc}"
        ) from exc


__all__ = ["build_two_axis_scale_rig"]
