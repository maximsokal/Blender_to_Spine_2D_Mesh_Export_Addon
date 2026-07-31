"""Fail-closed target, rig-profile, and composition-scope export capabilities.

A JSON codec being registered does not prove that every generated rig topology is safe
for that target runtime. This module owns the narrower production contract used before
Blender geometry preparation starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import FrozenSet, Mapping, Tuple

from .rig_profiles import A1RigProfile, resolve_a1_rig_profile
from .version_target import (
    SpineJsonTarget,
    SpineJsonTargetUnavailableError,
    require_spine_json_target_serializable,
)


class SpineJsonExportCapabilityError(SpineJsonTargetUnavailableError):
    """Raised when a codec exists but the requested rig/scope is not accepted."""


class SpineJsonExportScope(str, Enum):
    """Stable Blender-independent identifiers for generated document topology."""

    SINGLE_OBJECT = "SINGLE_OBJECT"
    STANDALONE_MULTI_OBJECT = "STANDALONE_MULTI_OBJECT"
    CONNECTED_MULTI_OBJECT = "CONNECTED_MULTI_OBJECT"
    MIXED_MULTI_OBJECT = "MIXED_MULTI_OBJECT"


@dataclass(frozen=True, slots=True)
class SpineJsonExportCapability:
    """One accepted target/profile pair and its explicitly enabled scopes."""

    target: SpineJsonTarget
    rig_profile: A1RigProfile
    scopes: FrozenSet[SpineJsonExportScope]
    limitations: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.target, SpineJsonTarget):
            raise TypeError("target must be SpineJsonTarget")
        if not isinstance(self.rig_profile, A1RigProfile):
            raise TypeError("rig_profile must be A1RigProfile")
        if not isinstance(self.scopes, frozenset) or not self.scopes:
            raise ValueError("scopes must be a non-empty frozenset")
        if not all(isinstance(scope, SpineJsonExportScope) for scope in self.scopes):
            raise TypeError("scopes must contain SpineJsonExportScope values")
        if not isinstance(self.limitations, tuple) or not all(
            isinstance(item, str) and item.strip() for item in self.limitations
        ):
            raise TypeError("limitations must contain non-empty strings")


_ALL_SCOPES = frozenset(SpineJsonExportScope)
_LIMITED_STANDALONE_SCOPES = frozenset(
    {
        SpineJsonExportScope.SINGLE_OBJECT,
        SpineJsonExportScope.STANDALONE_MULTI_OBJECT,
    }
)

_CAPABILITIES: Mapping[
    tuple[SpineJsonTarget, A1RigProfile],
    SpineJsonExportCapability,
] = MappingProxyType(
    {
        (
            SpineJsonTarget.SPINE_4_0,
            A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        ): SpineJsonExportCapability(
            target=SpineJsonTarget.SPINE_4_0,
            rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE,
            scopes=_LIMITED_STANDALONE_SCOPES,
            limitations=(
                "Attachment and animation sequences are not supported by Spine 4.0.64.",
            ),
        ),
        (
            SpineJsonTarget.SPINE_4_1,
            A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        ): SpineJsonExportCapability(
            target=SpineJsonTarget.SPINE_4_1,
            rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE,
            scopes=_LIMITED_STANDALONE_SCOPES,
        ),
        (
            SpineJsonTarget.SPINE_4_2,
            A1RigProfile.THREE_AXIS_ROTATION,
        ): SpineJsonExportCapability(
            target=SpineJsonTarget.SPINE_4_2,
            rig_profile=A1RigProfile.THREE_AXIS_ROTATION,
            scopes=_ALL_SCOPES,
        ),
        (
            SpineJsonTarget.SPINE_4_2,
            A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        ): SpineJsonExportCapability(
            target=SpineJsonTarget.SPINE_4_2,
            rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE,
            scopes=_ALL_SCOPES,
        ),
        (
            SpineJsonTarget.SPINE_4_3,
            A1RigProfile.THREE_AXIS_ROTATION,
        ): SpineJsonExportCapability(
            target=SpineJsonTarget.SPINE_4_3,
            rig_profile=A1RigProfile.THREE_AXIS_ROTATION,
            scopes=_LIMITED_STANDALONE_SCOPES,
            limitations=(
                "Connected and mixed 4.3 compositions await runtime acceptance.",
            ),
        ),
        (
            SpineJsonTarget.SPINE_4_3,
            A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        ): SpineJsonExportCapability(
            target=SpineJsonTarget.SPINE_4_3,
            rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE,
            scopes=_LIMITED_STANDALONE_SCOPES,
            limitations=(
                "Connected and mixed 4.3 compositions await runtime acceptance.",
            ),
        ),
    }
)


def resolve_spine_json_export_scope(value: object) -> SpineJsonExportScope:
    """Resolve one scope enum or exact persisted identifier without fallback."""

    if isinstance(value, SpineJsonExportScope):
        return value
    if not isinstance(value, str):
        raise TypeError("export scope must be SpineJsonExportScope or str")
    normalized = value.strip().upper()
    if not normalized:
        raise ValueError("export scope cannot be empty")
    try:
        return SpineJsonExportScope(normalized)
    except ValueError as exc:
        supported = tuple(scope.value for scope in SpineJsonExportScope)
        raise ValueError(
            f"Unsupported Spine JSON export scope {value!r}; supported={supported}"
        ) from exc


def registered_spine_json_export_capabilities() -> Mapping[
    tuple[SpineJsonTarget, A1RigProfile],
    SpineJsonExportCapability,
]:
    """Return the immutable accepted target/profile capability registry."""

    return _CAPABILITIES


def require_spine_json_export_capability(
    target: object,
    rig_profile: object,
    scope: object,
) -> SpineJsonExportCapability:
    """Return the accepted capability or fail before any Blender geometry work."""

    resolved_target = require_spine_json_target_serializable(target)
    resolved_profile = resolve_a1_rig_profile(rig_profile)
    resolved_scope = resolve_spine_json_export_scope(scope)
    capability = _CAPABILITIES.get((resolved_target, resolved_profile))

    if capability is None:
        supported_profiles = tuple(
            profile.label
            for candidate_target, profile in _CAPABILITIES
            if candidate_target is resolved_target
        )
        raise SpineJsonExportCapabilityError(
            f"{resolved_target.label} ({resolved_target.exact_version}) is not enabled "
            f"for rig profile {resolved_profile.label}; "
            f"supported profiles={supported_profiles}"
        )

    if resolved_scope not in capability.scopes:
        supported_scopes = tuple(
            candidate.value
            for candidate in SpineJsonExportScope
            if candidate in capability.scopes
        )
        raise SpineJsonExportCapabilityError(
            f"{resolved_target.label} ({resolved_target.exact_version}) with rig profile "
            f"{resolved_profile.label} does not support {resolved_scope.value}; "
            f"supported scopes={supported_scopes}"
        )

    return capability


__all__ = [
    "SpineJsonExportCapability",
    "SpineJsonExportCapabilityError",
    "SpineJsonExportScope",
    "registered_spine_json_export_capabilities",
    "require_spine_json_export_capability",
    "resolve_spine_json_export_scope",
]
