"""One-time migration owner for persisted Rewrite Scene settings."""

from __future__ import annotations

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


def migration_file_loading() -> bool:
    """Return whether Blender is currently restoring persisted Scene RNA values."""

    return _FILE_LOADING


def _stored_schema_version(scene: Any) -> int:
    raw = getattr(scene, _SCHEMA_PROPERTY, 0)
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, value)


def _stored_seam_mode(scene: Any) -> str:
    """Return one normalized persisted seam mode for diagnostics."""

    raw = getattr(scene, "spine2d_seam_maker_mode", "AUTO")
    value = str(raw or "AUTO").strip().upper()
    return value or "AUTO"


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


def _is_fresh_scene(scene: Any, current_schema: int) -> bool:
    """Return True only when no previous Rewrite setting was stored in the Scene."""

    if current_schema != 0:
        return False
    persisted = _persisted_scene_keys(scene)
    return not any(
        key.startswith("spine2d_") and key != _SCHEMA_PROPERTY
        for key in persisted
    )


def _stored_rig_profile(scene: Any) -> A1RigProfile:
    raw = getattr(scene, _RIG_PROPERTY, A1RigProfile.TWO_AXIS_ROTATION_SCALE.value)
    try:
        return resolve_a1_rig_profile(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid persisted rig profile %r; using two-axis default", raw)
        return A1RigProfile.TWO_AXIS_ROTATION_SCALE


def migrate_scene_settings(scene: Any) -> bool:
    """Migrate one Scene once without overwriting established rig choices.

    Schema 4 introduced selectable rig profiles and assigned older saved projects to the
    byte-compatible three-axis rig. Schema 5 changes only the default for genuinely fresh
    Scenes. Existing schema-4 Scenes keep the profile already chosen by the user.
    """

    if scene is None:
        raise ValueError("scene cannot be None")

    current = _stored_schema_version(scene)
    if current >= CURRENT_SETTINGS_SCHEMA_VERSION:
        return False

    fresh_scene = _is_fresh_scene(scene, current)
    previous_mode = _stored_seam_mode(scene)
    seam_changed = current < 3 and not fresh_scene
    if seam_changed:
        scene.spine2d_seam_maker_mode = "AUTO"

    if current >= 4:
        rig_profile = _stored_rig_profile(scene)
    elif fresh_scene:
        rig_profile = A1RigProfile.TWO_AXIS_ROTATION_SCALE
    else:
        # Never silently change established pre-profile projects.
        rig_profile = A1RigProfile.THREE_AXIS_ROTATION

    scene.spine2d_rig_profile = rig_profile.value
    scene.spine2d_settings_schema_version = CURRENT_SETTINGS_SCHEMA_VERSION
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
    _FILE_LOADING = False
    _REGISTERED = False


__all__ = [
    "CURRENT_SETTINGS_SCHEMA_VERSION",
    "migrate_all_scenes",
    "migrate_scene_settings",
    "migration_file_loading",
    "register",
    "spine2d_scene_settings_load_post",
    "spine2d_scene_settings_load_pre",
    "unregister",
]
