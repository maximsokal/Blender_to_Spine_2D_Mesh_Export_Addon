"""Compatibility facade for the decomposed legacy A1 rig builder."""

from .legacy_rig_assembly import build_legacy_rig
from .legacy_rig_bones import build_z_group_bones_for_request as _build_z_group_bones
from .legacy_rig_constraints import build_legacy_constraints as _build_constraints
from .legacy_rig_contracts import (
    LegacyRigBuildRequest,
    LegacyRigBuildResult,
    LegacyRigInfo,
    LegacyZGroup,
    LegacyZGroupBuildInfo,
    UniformScaleMode,
)
from .legacy_rig_error import LegacyRigBuildError
from .legacy_rig_scale import (
    calculate_uniform_scale,
    resolve_main_position as _main_position,
)


__all__ = [
    "LegacyRigBuildError",
    "LegacyRigBuildRequest",
    "LegacyRigBuildResult",
    "LegacyRigInfo",
    "LegacyZGroup",
    "LegacyZGroupBuildInfo",
    "UniformScaleMode",
    "_build_constraints",
    "_build_z_group_bones",
    "_main_position",
    "build_legacy_rig",
    "calculate_uniform_scale",
]
