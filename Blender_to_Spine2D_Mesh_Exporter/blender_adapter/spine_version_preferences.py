"""Resolve persistent per-family Spine project versions from AddonPreferences."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Tuple

import bpy

from .. import __package__ as _ADDON_BASE_PACKAGE
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
    """Return Blender's authoritative root add-on package identifier.

    Blender Extensions add the repository to the runtime module namespace, for example
    ``bl_ext.user_default.blender_to_spine2d_mesh_exporter``. Subpackages must therefore
    reuse the root package's own ``__package__`` value instead of reconstructing it from
    their local package string. This is also the identifier used by AddonPreferences.
    """

    root = str(_ADDON_BASE_PACKAGE or "").strip()
    if not root:
        raise RuntimeError("Resolved add-on root package is empty")
    return root


def _addon_entry_from_key(addons: Any, key: str) -> Any | None:
    """Read one Blender Addon collection entry without assuming a concrete collection."""

    getter = getattr(addons, "get", None)
    if callable(getter):
        try:
            result = getter(key)
        except Exception:
            result = None
        if result is not None:
            return result
    try:
        return addons[key]
    except (KeyError, IndexError, TypeError, AttributeError):
        return None


def _iter_addon_entries(addons: Any) -> tuple[tuple[str, Any], ...]:
    """Return deterministic ``(module-key, Addon)`` entries from Blender or test doubles."""

    entries: list[tuple[str, Any]] = []
    seen: set[int] = set()

    keys_method = getattr(addons, "keys", None)
    if callable(keys_method):
        try:
            keys = tuple(str(key) for key in keys_method())
        except Exception:
            keys = ()
        for key in keys:
            addon = _addon_entry_from_key(addons, key)
            if addon is None or id(addon) in seen:
                continue
            seen.add(id(addon))
            entries.append((key, addon))

    try:
        iterable = tuple(addons)
    except (TypeError, RuntimeError):
        iterable = ()
    for raw_entry in iterable:
        addon = raw_entry
        key = str(getattr(addon, "module", "") or "").strip()
        if isinstance(raw_entry, str):
            key = raw_entry
            addon = _addon_entry_from_key(addons, key)
        if addon is None or id(addon) in seen:
            continue
        seen.add(id(addon))
        entries.append((key, addon))

    return tuple(entries)


def _spine_preferences_from_addon(addon: Any) -> Any | None:
    preferences = getattr(addon, "preferences", None)
    if preferences is None:
        return None
    if not all(
        hasattr(preferences, spec.property_name)
        for spec in SPINE_EXACT_VERSION_PREFERENCE_SPECS
    ):
        return None
    return preferences


def _installed_extension_preferences_fallback(
    addons: Any,
    root_package: str,
) -> Any | None:
    """Find our installed Preferences when Blender exposes an unexpected collection key.

    The semantic fallback is intentionally restricted to installed ``bl_ext.*`` runtimes.
    It identifies this add-on by the complete stable set of exact-version RNA fields, then
    uses module identity only to disambiguate multiple installed copies.
    """

    candidates: list[tuple[str, Any, Any]] = []
    for key, addon in _iter_addon_entries(addons):
        preferences = _spine_preferences_from_addon(addon)
        if preferences is None:
            continue
        candidates.append((key, addon, preferences))

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0][2]

    root_casefold = root_package.casefold()
    root_leaf = root_package.rsplit(".", 1)[-1].casefold()
    ranked: list[tuple[int, Any]] = []
    for key, addon, preferences in candidates:
        module = str(getattr(addon, "module", "") or "").strip()
        identifiers = tuple(value for value in (key, module) if value)
        score = 0
        if any(value.casefold() == root_casefold for value in identifiers):
            score = 2
        elif any(
            value.rsplit(".", 1)[-1].casefold() == root_leaf
            for value in identifiers
        ):
            score = 1
        ranked.append((score, preferences))

    best_score = max(score for score, _preferences in ranked)
    best = tuple(
        preferences
        for score, preferences in ranked
        if score == best_score
    )
    if best_score > 0 and len(best) == 1:
        return best[0]

    raise RuntimeError(
        "Multiple Spine2D AddonPreferences entries are enabled; "
        f"unable to select runtime package {root_package!r}"
    )


def _preferences_from_runtime_context(
    runtime_context: Any,
    root_package: str,
) -> Any | None:
    preferences = getattr(runtime_context, "preferences", None)
    addons = getattr(preferences, "addons", None)
    if addons is None:
        return None

    exact_addon = _addon_entry_from_key(addons, root_package)
    exact_preferences = (
        None if exact_addon is None else getattr(exact_addon, "preferences", None)
    )
    if exact_preferences is not None:
        if not root_package.startswith("bl_ext."):
            return exact_preferences
        resolved = _spine_preferences_from_addon(exact_addon)
        if resolved is not None:
            return resolved

    if root_package.startswith("bl_ext."):
        return _installed_extension_preferences_fallback(addons, root_package)
    return None


def get_spine_addon_preferences(
    context: Any | None = None,
    *,
    required: bool = False,
) -> Any | None:
    """Return the installed extension's AddonPreferences for the active Blender profile.

    UI draw callbacks may provide an area-local context whose ``preferences.addons`` view
    does not expose the enabled extension entry even though Blender's global context does.
    Explicit context lookup is therefore attempted first, then the same global context used
    by production export/settings resolution is used as a deterministic fallback.
    """

    root_package = addon_root_package_name()
    runtime_context = bpy.context if context is None else context
    result = _preferences_from_runtime_context(runtime_context, root_package)

    global_context = getattr(bpy, "context", None)
    if result is None and context is not None and global_context is not runtime_context:
        result = _preferences_from_runtime_context(global_context, root_package)

    if result is None and required:
        raise RuntimeError(
            "Spine2D AddonPreferences are unavailable for "
            f"runtime package {root_package!r}"
        )
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
        root_package = addon_root_package_name()
        if root_package.startswith("bl_ext."):
            raise RuntimeError(
                "Installed Spine2D extension cannot resolve its AddonPreferences entry: "
                f"{root_package!r}"
            )
        return spec.default_version
    if not hasattr(prefs, spec.property_name):
        raise AttributeError(
            f"AddonPreferences is missing {spec.property_name!r} for {resolved_target.value}"
        )
    return getattr(prefs, spec.property_name)


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
