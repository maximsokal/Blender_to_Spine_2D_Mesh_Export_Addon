"""One-time migration owner for persisted Rewrite Scene settings."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import bpy

from ..domain.geometry import DepthProjectionBaseMode
from ..domain.spine.rig_profiles import A1RigProfile, resolve_a1_rig_profile
from ..domain.spine.version_target import (
    DEFAULT_SPINE_JSON_TARGET,
    SpineJsonTarget,
    resolve_spine_json_target,
)

try:
    from bpy.app.handlers import persistent
except Exception:  # pragma: no cover - real Blender always provides this decorator.
    def persistent(function):
        return function


logger = logging.getLogger(__name__)
CURRENT_SETTINGS_SCHEMA_VERSION = 8
_FILE_LOADING = False
_SCHEMA_PROPERTY = "spine2d_settings_schema_version"
_RIG_PROPERTY = "spine2d_rig_profile"
_SEAM_PROPERTY = "spine2d_seam_maker_mode"
_TARGET_PROPERTY = "spine2d_target_spine_version"
_DEPTH_DEFAULTS: tuple[tuple[str, object], ...] = (
    ("spine2d_depth_smoothing", 0.35),
    ("spine2d_depth_edge_threshold", 0.08),
    ("spine2d_depth_mesh_error_pixels", 4.0),
    ("spine2d_depth_max_points", 128),
    ("spine2d_depth_parallax_horizon_angle", 0.0),
    (
        "spine2d_depth_base_mode",
        DepthProjectionBaseMode.FARTHEST_VISIBLE.value,
    ),
)
_MISSING = object()


@dataclass(frozen=True, slots=True)
class _PreRegistrationSceneState:
    """Raw persisted values captured before Blender binds Rewrite RNA defaults."""

    schema_version: int
    persisted_keys: frozenset[str]
    seam_mode: str
    rig_profile_raw: object
    rig_profile_persisted: bool
    spine_target_raw: object
    spine_target_persisted: bool


_PRE_REGISTRATION_SCENE_STATES: dict[
    tuple[str, int],
    _PreRegistrationSceneState,
] = {}


def migration_file_loading() -> bool:
    """Return whether Blender is currently restoring persisted Scene RNA values."""

    return _FILE_LOADING


def _scene_identity(scene: Any) -> tuple[str, int]:
    """Return one process-local identity stable across RNA registration."""

    as_pointer = getattr(scene, "as_pointer", None)
    if callable(as_pointer):
        try:
            pointer = int(as_pointer())
        except (TypeError, ValueError, OverflowError, RuntimeError):
            pointer = 0
        if pointer > 0:
            return ("BLENDER_POINTER", pointer)
    return ("PYTHON_OBJECT", id(scene))


def migration_registration_pending(scene: Any) -> bool:
    """Return whether ``scene`` has a pre-RNA snapshot awaiting migration.

    This is the narrow lifecycle signal needed by Scene RNA update callbacks while
    properties are first bound. It deliberately replaces the old dependency on the
    extension-wide registration state machine.
    """

    if scene is None:
        return False
    return _scene_identity(scene) in _PRE_REGISTRATION_SCENE_STATES


def _coerce_schema_version(raw: object) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, value)


def _normalize_seam_mode(raw: object) -> str:
    value = str(raw or "AUTO").strip().upper()
    return value or "AUTO"


def _stored_schema_version(scene: Any) -> int:
    return _coerce_schema_version(getattr(scene, _SCHEMA_PROPERTY, 0))


def _stored_seam_mode(scene: Any) -> str:
    """Return one normalized persisted seam mode for diagnostics."""

    return _normalize_seam_mode(getattr(scene, _SEAM_PROPERTY, "AUTO"))


def _persisted_scene_keys(scene: Any) -> frozenset[str]:
    """Read actual ID-property keys without treating RNA defaults as persisted data."""

    keys = getattr(scene, "keys", None)
    if not callable(keys):
        return frozenset()
    try:
        return frozenset(str(key) for key in keys())
    except Exception:
        logger.debug("Unable to inspect Scene ID-property keys", exc_info=True)
        return frozenset()


def _persisted_id_value(
    scene: Any,
    property_name: str,
    persisted_keys: frozenset[str],
    default: object,
) -> object:
    """Read one raw Blender ID-property before an RNA descriptor can shadow it."""

    if property_name not in persisted_keys:
        return default

    try:
        return scene[property_name]
    except Exception:
        logger.debug(
            "Unable to read persisted Scene property %s through item access",
            property_name,
            exc_info=True,
        )
        return default


def _capture_scene_state(scene: Any) -> _PreRegistrationSceneState:
    """Capture only raw persisted values needed by the schema migration."""

    if scene is None:
        raise ValueError("scene cannot be None")

    persisted_keys = _persisted_scene_keys(scene)
    raw_schema = _persisted_id_value(
        scene,
        _SCHEMA_PROPERTY,
        persisted_keys,
        0,
    )
    raw_seam = _persisted_id_value(
        scene,
        _SEAM_PROPERTY,
        persisted_keys,
        "AUTO",
    )
    raw_rig = _persisted_id_value(
        scene,
        _RIG_PROPERTY,
        persisted_keys,
        _MISSING,
    )
    raw_target = _persisted_id_value(
        scene,
        _TARGET_PROPERTY,
        persisted_keys,
        _MISSING,
    )
    return _PreRegistrationSceneState(
        schema_version=_coerce_schema_version(raw_schema),
        persisted_keys=persisted_keys,
        seam_mode=_normalize_seam_mode(raw_seam),
        rig_profile_raw=raw_rig,
        rig_profile_persisted=raw_rig is not _MISSING,
        spine_target_raw=raw_target,
        spine_target_persisted=raw_target is not _MISSING,
    )


def _capture_pre_registration_scene_state_for_scenes(
    scenes: tuple[Any, ...],
) -> int:
    """Replace the pending snapshot set with the supplied deterministic Scene tuple."""

    if not isinstance(scenes, tuple):
        raise TypeError("scenes must be a tuple")
    if any(scene is None for scene in scenes):
        raise ValueError("scenes cannot contain None")

    _PRE_REGISTRATION_SCENE_STATES.clear()
    for scene in scenes:
        identity = _scene_identity(scene)
        if identity in _PRE_REGISTRATION_SCENE_STATES:
            raise ValueError("scenes cannot contain duplicate Scene identities")
        _PRE_REGISTRATION_SCENE_STATES[identity] = _capture_scene_state(scene)
    return len(_PRE_REGISTRATION_SCENE_STATES)


def capture_pre_registration_scene_state() -> int:
    """Capture current Scene ID-properties immediately before RNA registration."""

    scenes = tuple(getattr(bpy.data, "scenes", ()))
    captured = _capture_pre_registration_scene_state_for_scenes(scenes)
    logger.debug("Captured pre-registration settings for %d Scene(s)", captured)
    return captured


def clear_pre_registration_scene_state() -> None:
    """Discard pending pre-registration snapshots after rollback or unregistration."""

    _PRE_REGISTRATION_SCENE_STATES.clear()


def _is_fresh_scene(
    scene: Any,
    current_schema: int,
    snapshot: _PreRegistrationSceneState | None,
) -> bool:
    """Return True only when no previous Rewrite setting was stored in the Scene."""

    if current_schema != 0:
        return False
    persisted = (
        snapshot.persisted_keys
        if snapshot is not None
        else _persisted_scene_keys(scene)
    )
    return not any(
        key.startswith("spine2d_") and key != _SCHEMA_PROPERTY
        for key in persisted
    )


def _resolve_stored_rig_profile(raw: object) -> A1RigProfile:
    try:
        return resolve_a1_rig_profile(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid persisted rig profile %r; using two-axis default", raw)
        return A1RigProfile.TWO_AXIS_ROTATION_SCALE


def _stored_rig_profile(
    scene: Any,
    snapshot: _PreRegistrationSceneState | None,
) -> A1RigProfile:
    if snapshot is not None and snapshot.rig_profile_persisted:
        return _resolve_stored_rig_profile(snapshot.rig_profile_raw)
    raw = getattr(
        scene,
        _RIG_PROPERTY,
        A1RigProfile.TWO_AXIS_ROTATION_SCALE.value,
    )
    return _resolve_stored_rig_profile(raw)


def _resolve_stored_spine_target(raw: object) -> SpineJsonTarget:
    try:
        return resolve_spine_json_target(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid persisted Spine JSON target %r; using %s",
            raw,
            DEFAULT_SPINE_JSON_TARGET.value,
        )
        return DEFAULT_SPINE_JSON_TARGET


def _stored_spine_target(
    scene: Any,
    snapshot: _PreRegistrationSceneState | None,
) -> SpineJsonTarget:
    """Preserve one valid saved target and default missing/invalid values to 4.2."""

    if snapshot is not None:
        if not snapshot.spine_target_persisted:
            return DEFAULT_SPINE_JSON_TARGET
        return _resolve_stored_spine_target(snapshot.spine_target_raw)

    persisted_keys = _persisted_scene_keys(scene)
    if _TARGET_PROPERTY not in persisted_keys:
        return DEFAULT_SPINE_JSON_TARGET
    raw = _persisted_id_value(
        scene,
        _TARGET_PROPERTY,
        persisted_keys,
        DEFAULT_SPINE_JSON_TARGET.value,
    )
    return _resolve_stored_spine_target(raw)


def _initialize_depth_defaults(
    scene: Any,
    persisted_keys: frozenset[str],
) -> tuple[str, ...]:
    """Initialize only depth fields that did not already exist in the saved Scene."""

    initialized: list[str] = []
    for property_name, default in _DEPTH_DEFAULTS:
        if property_name in persisted_keys:
            continue
        setattr(scene, property_name, default)
        initialized.append(property_name)
    return tuple(initialized)


def migrate_scene_settings(scene: Any) -> bool:
    """Migrate one Scene once without overwriting established user choices.

    Schema 4 introduced selectable rig profiles. Schema 5 changed the fresh-Scene rig
    default. Schema 6 added the Spine JSON target. Schema 7 added Depth Camera Projection
    quality controls. Schema 8 adds the Parallax Horizon Angle with a zero-degree default,
    preserving every 0.81.0 front-only export until the user opts into reserve coverage.
    """

    if scene is None:
        raise ValueError("scene cannot be None")

    identity = _scene_identity(scene)
    snapshot = _PRE_REGISTRATION_SCENE_STATES.get(identity)
    current = (
        snapshot.schema_version
        if snapshot is not None
        else _stored_schema_version(scene)
    )
    if current >= CURRENT_SETTINGS_SCHEMA_VERSION:
        _PRE_REGISTRATION_SCENE_STATES.pop(identity, None)
        return False

    persisted_keys = (
        snapshot.persisted_keys
        if snapshot is not None
        else _persisted_scene_keys(scene)
    )
    fresh_scene = _is_fresh_scene(scene, current, snapshot)
    previous_mode = (
        snapshot.seam_mode
        if snapshot is not None
        else _stored_seam_mode(scene)
    )
    seam_changed = current < 3 and not fresh_scene

    if current >= 4:
        rig_profile = _stored_rig_profile(scene, snapshot)
    elif fresh_scene:
        rig_profile = A1RigProfile.TWO_AXIS_ROTATION_SCALE
    else:
        rig_profile = A1RigProfile.THREE_AXIS_ROTATION

    spine_target = _stored_spine_target(scene, snapshot)

    try:
        if seam_changed:
            scene.spine2d_seam_maker_mode = "AUTO"
        scene.spine2d_rig_profile = rig_profile.value
        scene.spine2d_target_spine_version = spine_target.value
        initialized_depth = (
            _initialize_depth_defaults(scene, persisted_keys)
            if current < 8
            else ()
        )
        scene.spine2d_settings_schema_version = CURRENT_SETTINGS_SCHEMA_VERSION
    except Exception:
        logger.exception(
            "Unable to apply migrated settings to Scene '%s'",
            str(getattr(scene, "name", "<unnamed>")),
        )
        raise
    else:
        _PRE_REGISTRATION_SCENE_STATES.pop(identity, None)

    logger.info(
        "Migrated Spine2D Rewrite Scene '%s' settings schema %d -> %d; "
        "fresh=%s; Seam Maker %s -> %s; Rig -> %s; Spine target -> %s (%s); "
        "depth_defaults=%s",
        str(getattr(scene, "name", "<unnamed>")),
        current,
        CURRENT_SETTINGS_SCHEMA_VERSION,
        fresh_scene,
        previous_mode,
        "AUTO" if seam_changed else previous_mode,
        rig_profile.value,
        spine_target.value,
        spine_target.exact_version,
        initialized_depth,
    )
    return True


def migrate_all_scenes() -> int:
    """Migrate every current Blender Scene and return the changed count."""

    changed = 0
    for scene in tuple(getattr(bpy.data, "scenes", ())):
        try:
            changed += int(migrate_scene_settings(scene))
        except Exception:
            logger.exception(
                "Unable to migrate Spine2D Rewrite settings for Scene '%s'",
                str(getattr(scene, "name", "<unnamed>")),
            )
    return changed


@persistent
def spine2d_scene_settings_load_pre(_dummy: Any) -> None:
    """Prevent RNA update callbacks from marking old values during file loading."""

    global _FILE_LOADING
    clear_pre_registration_scene_state()
    _FILE_LOADING = True


@persistent
def spine2d_scene_settings_load_post(_dummy: Any) -> None:
    """Apply migrations after a .blend file has restored persisted Scene values."""

    global _FILE_LOADING
    try:
        migrate_all_scenes()
    finally:
        _FILE_LOADING = False


def register() -> None:
    """Register persistent load handlers and migrate already-open Scenes."""

    load_pre_handlers = bpy.app.handlers.load_pre
    load_post_handlers = bpy.app.handlers.load_post
    added_pre = False
    added_post = False

    if spine2d_scene_settings_load_pre not in load_pre_handlers:
        load_pre_handlers.append(spine2d_scene_settings_load_pre)
        added_pre = True
    if spine2d_scene_settings_load_post not in load_post_handlers:
        load_post_handlers.append(spine2d_scene_settings_load_post)
        added_post = True

    try:
        migrate_all_scenes()
    except Exception:
        if added_post:
            while spine2d_scene_settings_load_post in load_post_handlers:
                load_post_handlers.remove(spine2d_scene_settings_load_post)
        if added_pre:
            while spine2d_scene_settings_load_pre in load_pre_handlers:
                load_pre_handlers.remove(spine2d_scene_settings_load_pre)
        raise


def unregister() -> None:
    """Remove every owned persistent migration handler and transient migration state."""

    global _FILE_LOADING
    load_pre_handlers = bpy.app.handlers.load_pre
    load_post_handlers = bpy.app.handlers.load_post
    while spine2d_scene_settings_load_pre in load_pre_handlers:
        load_pre_handlers.remove(spine2d_scene_settings_load_pre)
    while spine2d_scene_settings_load_post in load_post_handlers:
        load_post_handlers.remove(spine2d_scene_settings_load_post)
    clear_pre_registration_scene_state()
    _FILE_LOADING = False


__all__ = [
    "CURRENT_SETTINGS_SCHEMA_VERSION",
    "capture_pre_registration_scene_state",
    "clear_pre_registration_scene_state",
    "migrate_all_scenes",
    "migrate_scene_settings",
    "migration_file_loading",
    "migration_registration_pending",
    "register",
    "spine2d_scene_settings_load_post",
    "spine2d_scene_settings_load_pre",
    "unregister",
]
