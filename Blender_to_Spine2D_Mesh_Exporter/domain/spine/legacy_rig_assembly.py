"""Orchestrate exact legacy A1 rig construction from physical owners."""

from __future__ import annotations

from .legacy_profile import LegacyRigProfile
from .legacy_rig_bones import build_legacy_rig_bones
from .legacy_rig_constraints import build_legacy_rig_constraints
from .legacy_rig_contracts import LegacyRigBuildRequest, LegacyRigBuildResult
from .legacy_rig_error import LegacyRigBuildError
from .legacy_rig_plan import build_legacy_rig_plan
from .legacy_rig_validation import (
    validate_legacy_rig_plan,
    validate_legacy_rig_result,
)


def build_legacy_rig(
    request: LegacyRigBuildRequest,
    profile: LegacyRigProfile | None = None,
) -> LegacyRigBuildResult:
    """Build and validate the complete ordered A1 control hierarchy."""

    if not isinstance(request, LegacyRigBuildRequest):
        raise TypeError("request must be LegacyRigBuildRequest")
    resolved_profile = LegacyRigProfile() if profile is None else profile
    if not isinstance(resolved_profile, LegacyRigProfile):
        raise TypeError("profile must be LegacyRigProfile")

    stage = "PLAN"
    try:
        plan = build_legacy_rig_plan(request, resolved_profile)
        validate_legacy_rig_plan(plan)

        stage = "BONES"
        bones = build_legacy_rig_bones(plan)

        stage = "CONSTRAINTS"
        ik, transform = build_legacy_rig_constraints(plan)

        stage = "RESULT"
        result = LegacyRigBuildResult(
            request=request,
            profile=resolved_profile,
            bones=bones,
            ik=ik,
            transform=transform,
            info=plan.info,
        )
        validate_legacy_rig_result(result)
        return result
    except LegacyRigBuildError as exc:
        raise LegacyRigBuildError(
            f"Unable to build A1 rig for '{request.prefix}' at {stage}: {exc}"
        ) from exc
    except Exception as exc:
        raise LegacyRigBuildError(
            f"Unable to build A1 rig for '{request.prefix}' at {stage}: {exc}"
        ) from exc


__all__ = ["build_legacy_rig"]
