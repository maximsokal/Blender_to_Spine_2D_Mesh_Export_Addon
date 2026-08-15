"""Resolve persistent per-family Spine project versions from AddonPreferences."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Tuple

import bpy

from ..domain.spine.version_target import (
    SpineJsonTarget,
    resolve_spine_json_target,
    validate_spine_json_exact_version_for_target,
)


@dataclass(frozen=True, slots=True)
class SpineExactVersionPreferenceSpec:
    """One stable AddonPreferences field owned by a Spine schema family."""

    target: SpineJsonTarget
    property_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.target, SpineJsonTarget):
            raise TypeError("target must be SpineJsonTarget")
        if not isinstance(self.property_name, str) or not self.property_name.strip():
            raise ValueError("property_name must be a non-empty string")
        if not self.property_name.startswith("spine2d_exact_version_"):
            raise ValueError("Invalid Spine exact-version preference property name")

    @property
    def default_version(self) -> str:
        return self.target.exact_version

    @property
    def label(self) -> str:
        return f"{self.target.label} project version"


SPINE_EXACT_VERSION_PREFERENCE_SPECS: Tuple[SpineExactVersionPreferenceSpec, ...] = (
    SpineExactVersionPreferenceSpec(SpineJsonTarget.SPINE_3_8, "spine2d_exact_version_3_8"),
    SpineExactVersionPreferenceSpec(SpineJsonTarget.SPINE_4_0, "spine2d_exact_version_4_0"),
    SpineExactVersionPreferenceSpec(SpineJsonTarget.SPINE_4_1, "spine2d_exact_version_4_1"),
    SpineExactVersionPreferenceSpec(SpineJsonTarget.SPINE_4_2, "spine2d_exact_version_4_2"),
    SpineExactVersionPreferenceSpec(SpineJsonTarget.SPINE_4_3, "spine2d_exact_version_4_3"),
)

_SPEC_BY_TARGET: Mapping[SpineJsonTarget, SpineExactVersionPreferenceSpec] = MappingProxyType(
    {spec.target: spec for spec in SPINE_EXACT_VERSION_PREFERENCE_SPECS}
)

if len(_SPEC_BY_TARGET) != len(SpineJsonTarget):
    raise RuntimeError("Every Spine JSON target must own one exact-version preference")
if len({spec.property_name for spec in SPINE_EXACT_VERSION_PREFERENCE_SPECS}) != len(
    SPINE_EXACT_VERSION_PREFERENCE_SPECS
):
    raise RuntimeError("Spine exact-version preference property names must be unique")


def spine_exact_version_preference_spec(target: object) -> SpineExactVersionPreferenceSpec:
    resolved = resolve_spine_json_target(target)
    try:
        return _SPEC_BY_TARGET[resolved]
    except KeyError as exc:
        raise RuntimeError(f"No exact-version preference for {resolved.value}") from exc


def addon_root_package_name() -> str:
    package = str(__package__ or "").strip()
    marker = ".blender_adapter"
    if marker not in package:
        raise RuntimeError(f"Unable to resolve add-on root package from {package!r}")
    root = package.split(marker, 1)[0].strip()
    if not root:
        raise RuntimeError("Resolved add-on root package is empty")
    return root


def get_spine_addon_preferences(
    context: Any | None = None,
    *,
    required: bool = False,
) -> Any | None:
    """Return the installed extension's AddonPreferences for the active Blender profile.

    Source-registered development tests intentionally have no installed add-on entry;
    callers may therefore request a non-required lookup and fall back to descriptor
    defaults without manufacturing a fake preferences object.
    """

    runtime_context = bpy.context if context is None else context
    preferences = getattr(runtime_context, "preferences", None)
    addons = getattr(preferences, "addons", None)
    addon = None if addons is None else addons.get(addon_root_package_name())
    result = None if addon is None else getattr(addon, "preferences", None)
    if result is None and required:
        raise RuntimeError("Spine2D AddonPreferences are unavailable")
    return result


def read_spine_project_exact_version_raw(
    target: object,
    *,
    preferences: Any | None = None,
    context: Any | None = None,
) -> object:
    """Read one preference without validating it.

    Readiness/cache code needs the raw persisted value in its signature so a scripted
    preference mutation invalidates cached diagnostics even when Blender's RNA update
    callback is bypassed. Validation remains centralized in
    :func:`resolve_spine_project_exact_version`.
    """

    resolved_target = resolve_spine_json_target(target)
    spec = spine_exact_version_preference_spec(resolved_target)
    prefs = preferences
    if prefs is None:
        prefs = get_spine_addon_preferences(context, required=False)
    if prefs is None:
        return spec.default_version
    return getattr(prefs, spec.property_name, spec.default_version)


def resolve_spine_project_exact_version(
    target: object,
    *,
    preferences: Any | None = None,
    context: Any | None = None,
) -> str:
    """Resolve the user's exact project version for one schema family."""

    resolved_target = resolve_spine_json_target(target)
    raw = read_spine_project_exact_version_raw(
        resolved_target,
        preferences=preferences,
        context=context,
    )
    return validate_spine_json_exact_version_for_target(resolved_target, raw)


def assign_spine_project_exact_version(
    preferences: Any,
    target: object,
    value: object,
) -> str:
    """Validate and assign one exact project version to AddonPreferences."""

    if preferences is None:
        raise ValueError("preferences cannot be None")
    resolved_target = resolve_spine_json_target(target)
    spec = spine_exact_version_preference_spec(resolved_target)
    normalized = validate_spine_json_exact_version_for_target(resolved_target, value)
    if not hasattr(preferences, spec.property_name):
        raise AttributeError(
            f"AddonPreferences is missing {spec.property_name!r} for {resolved_target.value}"
        )
    setattr(preferences, spec.property_name, normalized)
    return normalized


__all__ = [
    "SPINE_EXACT_VERSION_PREFERENCE_SPECS",
    "SpineExactVersionPreferenceSpec",
    "addon_root_package_name",
    "assign_spine_project_exact_version",
    "get_spine_addon_preferences",
    "read_spine_project_exact_version_raw",
    "resolve_spine_project_exact_version",
    "spine_exact_version_preference_spec",
]
