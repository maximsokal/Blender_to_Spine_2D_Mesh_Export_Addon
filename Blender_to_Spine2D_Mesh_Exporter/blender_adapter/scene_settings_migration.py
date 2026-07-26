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
CURRENT_SETTINGS_SCHEMA_VERSION = 1
_REGISTERED = False


def _stored_schema_version(scene: Any) -> int:
    raw = getattr(scene, "spine2d_settings_schema_version", 0)
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, value)


def migrate_scene_settings(scene: Any) -> bool:
    """Migrate one Scene exactly once without overwriting later user choices."""

    if scene is None:
        raise ValueError("scene cannot be None")

    current = _stored_schema_version(scene)
    if current >= CURRENT_SETTINGS_SCHEMA_VERSION:
        return False

    # Versions before schema 1 persisted the historical CUSTOM value in old .blend
    # files even after the Rewrite default changed. Reset it once, then mark the Scene
    # so a deliberate CUSTOM choice made afterwards remains stable.
    scene.spine2d_seam_maker_mode = "AUTO"
    scene.spine2d_settings_schema_version = CURRENT_SETTINGS_SCHEMA_VERSION
    logger.info(
        "Migrated Spine2D Rewrite Scene '%s' settings schema %d -> %d; "
        "Seam Maker reset to AUTO",
        str(getattr(scene, "name", "<unnamed>")),
        current,
        CURRENT_SETTINGS_SCHEMA_VERSION,
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
def spine2d_scene_settings_load_post(_dummy: Any) -> None:
    """Apply migrations after a .blend file has restored persisted Scene values."""

    migrate_all_scenes()


def register() -> None:
    """Register one persistent load handler and migrate already-open Scenes."""

    global _REGISTERED
    if _REGISTERED:
        return
    handlers = bpy.app.handlers.load_post
    if spine2d_scene_settings_load_post not in handlers:
        handlers.append(spine2d_scene_settings_load_post)
    try:
        migrate_all_scenes()
    except Exception:
        while spine2d_scene_settings_load_post in handlers:
            handlers.remove(spine2d_scene_settings_load_post)
        raise
    _REGISTERED = True


def unregister() -> None:
    """Remove every copy of the persistent migration handler."""

    global _REGISTERED
    handlers = bpy.app.handlers.load_post
    while spine2d_scene_settings_load_post in handlers:
        handlers.remove(spine2d_scene_settings_load_post)
    _REGISTERED = False


__all__ = [
    "CURRENT_SETTINGS_SCHEMA_VERSION",
    "migrate_all_scenes",
    "migrate_scene_settings",
    "register",
    "spine2d_scene_settings_load_post",
    "unregister",
]
