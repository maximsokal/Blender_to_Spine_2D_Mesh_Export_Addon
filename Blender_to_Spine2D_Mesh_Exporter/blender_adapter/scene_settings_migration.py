"""One-time migration owner for persisted Rewrite Scene settings."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import bpy

from ..domain.spine.rig_profiles import A1RigProfile, resolve_a1_rig_profile

try:
    from bpy.app.handlers import persistent
except Exception:  # pragma: no cover - real Blender always provides this decorator.
    def persistent(function):
        return function


logger = logging.getLogger(__name__)
CURRENT_SETTINGS_SCHEMA_VERSION = 5
_REGISTERED = False
_FILE_LOADING = False
_SCHEMA_PROPERTY = "spine2d_settings_schema_version"
_RIG_PROPERTY = "spine2d_rig_profile"
_SEAM_PROPERTY = "spine2d_seam_maker_mode"
_MISSING = object()


@dataclass(frozen=True, slots=True)
class _PreRegistrationSceneState:
    """Raw persisted values captured before Blender binds Rewrite RNA defaults."""

    schema_version: int
    persisted_keys: frozenset[str]
    seam_mode: str
    rig_profile_raw: object
    rig_profile_persisted: bool


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
    """Read one raw Blender ID-property before an RNA descriptor can shadow it.

    The key set is captured first through ``Scene.keys()``. Once membership is proven,
    item access is the single authoritative Blender API for the raw persisted value.
    This deliberately avoids the retired dynamic ``Scene.get`` compatibility bridge.
    """

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
    return _PreRegistrationSceneState(
        schema_version=_coerce_schema_version(raw_schema),
        persisted_keys=persisted_keys,
        seam_mode=_normalize_seam_mode(raw_seam),
        rig_profile_raw=raw_rig,
        rig_profile_persisted=raw_rig is not _MISSING,
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
    """Capture current Scene ID-properties immediately before RNA registration.

    Registering an EnumProperty over an older saved ID-property can make Blender expose
    the new RNA default before the migration owner runs. This snapshot preserves the
    actual pre-registration schema, seam mode, and rig choice for that one lifecycle.
    """

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


def migrate_scene_settings(scene: Any) -> bool:
    """Migrate one Scene once without overwriting established rig choices.

    Schema 4 introduced selectable rig profiles and assigned older saved projects to the
    byte-compatible three-axis rig. Schema 5 changes only the default for genuinely fresh
    Scenes. Existing schema-4 Scenes keep the profile already chosen by the user.
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
        # Never silently change established pre-profile projects.
        rig_profile = A1RigProfile.THREE_AXIS_ROTATION

    try:
        if seam_changed:
            scene.spine2d_seam_maker_mode = "AUTO"
        scene.spine2d_rig_profile = rig_profile.value
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
        "fresh=%s; Seam Maker %s -> %s; Rig -> %s",
        str(getattr(scene, "name", "<unnamed>")),
        current,
        CURRENT_SETTINGS_SCHEMA_VERSION,
        fresh_scene,
        previous_mode,
        "AUTO" if seam_changed else previous_mode,
        rig_profile.value,
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

    global _REGISTERED
    if _REGISTERED:
        return
    load_pre_handlers = bpy.app.handlers.load_pre
    load_post_handlers = bpy.app.handlers.load_post
    if spine2d_scene_settings_load_pre not in load_pre_handlers:
        load_pre_handlers.append(spine2d_scene_settings_load_pre)
    if spine2d_scene_settings_load_post not in load_post_handlers:
        load_post_handlers.append(spine2d_scene_settings_load_post)
    try:
        migrate_all_scenes()
    except Exception:
        while spine2d_scene_settings_load_pre in load_pre_handlers:
            load_pre_handlers.remove(spine2d_scene_settings_load_pre)
        while spine2d_scene_settings_load_post in load_post_handlers:
            load_post_handlers.remove(spine2d_scene_settings_load_post)
        raise
    _REGISTERED = True


def unregister() -> None:
    """Remove every copy of the persistent migration handlers."""

    global _REGISTERED, _FILE_LOADING
    load_pre_handlers = bpy.app.handlers.load_pre
    load_post_handlers = bpy.app.handlers.load_post
    while spine2d_scene_settings_load_pre in load_pre_handlers:
        load_pre_handlers.remove(spine2d_scene_settings_load_pre)
    while spine2d_scene_settings_load_post in load_post_handlers:
        load_post_handlers.remove(spine2d_scene_settings_load_post)
    clear_pre_registration_scene_state()
    _FILE_LOADING = False
    _REGISTERED = False


__all__ = [
    "CURRENT_SETTINGS_SCHEMA_VERSION",
    "capture_pre_registration_scene_state",
    "clear_pre_registration_scene_state",
    "migrate_all_scenes",
    "migrate_scene_settings",
    "migration_file_loading",
    "register",
    "spine2d_scene_settings_load_post",
    "spine2d_scene_settings_load_pre",
    "unregister",
]
