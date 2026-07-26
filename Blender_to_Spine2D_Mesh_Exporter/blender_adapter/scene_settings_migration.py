"""One-time migration owner for persisted Rewrite Scene settings."""

from __future__ import annotations

import logging
from typing import Any

import bpy
try:
    from bpy.app.handlers import persistent
except Exception:  # pragma: no cover - real Blender always provides this decorator.
    def persistent(function):
        return function


logger = logging.getLogger(__name__)
CURRENT_SETTINGS_SCHEMA_VERSION = 2
_REGISTERED = False
_FILE_LOADING = False


def migration_file_loading() -> bool:
    """Return whether Blender is currently restoring persisted Scene RNA values."""

    return _FILE_LOADING


def _stored_schema_version(scene: Any) -> int:
    raw = getattr(scene, "spine2d_settings_schema_version", 0)
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


def migrate_scene_settings(scene: Any) -> bool:
    """Migrate one Scene exactly once without overwriting later user choices.

    Schema 2 deliberately repeats the historical seam-mode reset for Scenes that were
    already marked as schema 1 by the first migration implementation. Those Scenes could
    still contain the persisted CUSTOM value because the old marker was written before
    the load lifecycle was fully covered. Once schema 2 is stored, a later deliberate
    CUSTOM choice remains stable.
    """

    if scene is None:
        raise ValueError("scene cannot be None")

    current = _stored_schema_version(scene)
    if current >= CURRENT_SETTINGS_SCHEMA_VERSION:
        return False

    previous_mode = _stored_seam_mode(scene)
    scene.spine2d_seam_maker_mode = "AUTO"
    scene.spine2d_settings_schema_version = CURRENT_SETTINGS_SCHEMA_VERSION
    logger.info(
        "Migrated Spine2D Rewrite Scene '%s' settings schema %d -> %d; "
        "Seam Maker %s -> AUTO",
        str(getattr(scene, "name", "<unnamed>")),
        current,
        CURRENT_SETTINGS_SCHEMA_VERSION,
        previous_mode,
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
