"""Precise depsgraph invalidation for cached A1 Rewrite readiness reports.

Readiness analysis creates temporary Blender datablocks. Blender may publish queued
updates for those datablocks after the new report is cached; the former broad handler
then changed READY to STALE immediately. This owner records the real scene
dependencies represented by a report and ignores exporter-owned temporary updates and
selection-only Object updates while preserving invalidation for real export inputs.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from bpy.app.handlers import persistent as _persistent
except Exception:  # pragma: no cover - non-Blender unit-test mocks.
    def _persistent(function):
        return function

from . import a1_export_readiness as _readiness
from .a1_ui_selection import _rna_identity


logger = logging.getLogger(__name__)

DependencyIdentity = tuple[str, tuple[str, object]]
_UPDATE_FLAGS = (
    "is_updated_geometry",
    "is_updated_transform",
    "is_updated_shading",
)
_TEMPORARY_PREFIXES = ("__Spine2D_",)
_TEMPORARY_MARKERS = (".spine2d-stage-v2~",)

_BASE_STORE = _readiness.store_a1_export_readiness
_BASE_CLEAR = _readiness.clear_a1_export_readiness
_BASE_HANDLER = _readiness.a1_readiness_depsgraph_update_post
_DEPENDENCIES_BY_SCENE: dict[int, frozenset[DependencyIdentity]] = {}
_REGISTERED = False


def _safe_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    try:
        return tuple(value)
    except Exception:
        logger.debug("Unable to iterate Blender RNA collection", exc_info=True)
        return ()


def _dependency_identity(value: Any) -> DependencyIdentity | None:
    if value is None:
        return None
    id_type = str(getattr(value, "id_type", "") or "").strip().upper()
    if id_type not in _readiness._RELEVANT_ID_TYPES:
        return None
    return id_type, _rna_identity(value)


def _add_dependency(
    dependencies: set[DependencyIdentity],
    value: Any,
) -> DependencyIdentity | None:
    identity = _dependency_identity(value)
    if identity is not None:
        dependencies.add(identity)
    return identity


def _collect_node_tree(
    node_tree: Any,
    dependencies: set[DependencyIdentity],
    visited: set[DependencyIdentity],
) -> None:
    identity = _add_dependency(dependencies, node_tree)
    if identity is None or identity in visited:
        return
    visited.add(identity)
    for node in _safe_tuple(getattr(node_tree, "nodes", ())):
        _add_dependency(dependencies, getattr(node, "image", None))
        _add_dependency(dependencies, getattr(node, "texture", None))
        nested = getattr(node, "node_tree", None)
        if nested is not None:
            _collect_node_tree(nested, dependencies, visited)


def _collect_material(
    material: Any,
    dependencies: set[DependencyIdentity],
    visited_node_trees: set[DependencyIdentity],
) -> None:
    if material is None:
        return
    _add_dependency(dependencies, material)
    node_tree = getattr(material, "node_tree", None)
    if node_tree is not None:
        _collect_node_tree(node_tree, dependencies, visited_node_trees)


def _collect_object(
    obj: Any,
    dependencies: set[DependencyIdentity],
    visited_node_trees: set[DependencyIdentity],
) -> None:
    if obj is None:
        return
    _add_dependency(dependencies, obj)
    _add_dependency(dependencies, getattr(obj, "data", None))
    for slot in _safe_tuple(getattr(obj, "material_slots", ())):
        _collect_material(
            getattr(slot, "material", None),
            dependencies,
            visited_node_trees,
        )


def _capture_dependencies(context: Any) -> frozenset[DependencyIdentity]:
    scene = getattr(context, "scene", None)
    dependencies: set[DependencyIdentity] = set()
    visited_node_trees: set[DependencyIdentity] = set()

    _add_dependency(dependencies, scene)
    world = getattr(scene, "world", None)
    _add_dependency(dependencies, world)
    _collect_node_tree(
        getattr(world, "node_tree", None),
        dependencies,
        visited_node_trees,
    )
    _collect_node_tree(
        getattr(scene, "node_tree", None),
        dependencies,
        visited_node_trees,
    )

    objects: dict[tuple[str, object], Any] = {}
    for obj in _safe_tuple(getattr(scene, "objects", ())):
        objects.setdefault(_rna_identity(obj), obj)
    for obj in _readiness._request_mesh_objects(context):
        objects.setdefault(_rna_identity(obj), obj)
    camera = getattr(scene, "camera", None)
    if camera is not None:
        objects.setdefault(_rna_identity(camera), camera)

    for obj in objects.values():
        _collect_object(obj, dependencies, visited_node_trees)
    return frozenset(dependencies)


def _datablock_name(value: Any) -> str:
    return str(
        getattr(value, "name_full", None)
        or getattr(value, "name", None)
        or ""
    ).strip()


def _is_temporary_datablock(value: Any) -> bool:
    name = _datablock_name(value)
    if not name:
        return False
    if any(name.startswith(prefix) for prefix in _TEMPORARY_PREFIXES):
        return True
    folded = name.casefold()
    return any(marker in folded for marker in _TEMPORARY_MARKERS)


def _update_flags(update: Any) -> tuple[bool, ...]:
    flags: list[bool] = []
    for name in _UPDATE_FLAGS:
        try:
            value = getattr(update, name)
        except Exception:
            continue
        if isinstance(value, bool):
            flags.append(value)
    return tuple(flags)


def store_a1_export_readiness(context: Any, report: Any) -> None:
    """Store the base report and the exact live IDs represented by it."""

    _BASE_STORE(context, report)
    key = _readiness._scene_key(getattr(context, "scene", None))
    try:
        _DEPENDENCIES_BY_SCENE[key] = _capture_dependencies(context)
    except Exception:
        logger.exception("Unable to capture A1 readiness dependencies")
        # Empty dependencies keep all non-temporary updates conservative.
        _DEPENDENCIES_BY_SCENE[key] = frozenset()


def clear_a1_export_readiness(scene: Any | None = None) -> None:
    """Clear the base report and matching dependency snapshot."""

    if scene is None:
        _DEPENDENCIES_BY_SCENE.clear()
        _BASE_CLEAR()
        return
    _DEPENDENCIES_BY_SCENE.pop(_readiness._scene_key(scene), None)
    _BASE_CLEAR(scene)


@_persistent
def a1_readiness_depsgraph_update_post(scene: Any, depsgraph: Any) -> None:
    """Invalidate only reports touched by real export-relevant datablocks."""

    if scene is None or not _readiness._READINESS_CACHE:
        return
    try:
        key = _readiness._scene_key(scene)
    except Exception:
        logger.debug("Unable to resolve readiness Scene key", exc_info=True)
        return
    entry = _readiness._READINESS_CACHE.get(key)
    if entry is None or entry.stale:
        return

    try:
        updates = tuple(getattr(depsgraph, "updates", ()))
    except Exception:
        logger.debug("Unable to read depsgraph updates", exc_info=True)
        return
    if not updates:
        return

    dependencies = _DEPENDENCIES_BY_SCENE.get(key, frozenset())
    for update in updates:
        updated_id = getattr(update, "id", None)
        identity = _dependency_identity(updated_id)
        if identity is None or _is_temporary_datablock(updated_id):
            continue

        id_type = identity[0]
        known_dependency = identity in dependencies
        flags = _update_flags(update)
        if id_type == "OBJECT" and known_dependency and flags and not any(flags):
            # Temporary operator activation can restore selection/active state without
            # changing geometry, transform, or shading.
            continue
        if id_type == "SCENE":
            # Rewrite settings are already part of build_a1_readiness_signature().
            # Scene link/unlink noise from temporary objects must not defeat a fresh
            # report; World, camera, light, material and object IDs are tracked directly.
            continue

        # Known inputs changed, or a new non-temporary relevant ID entered the graph.
        entry.stale = True
        return


def _remove_all(handlers: Any, callback: Any) -> None:
    while callback in handlers:
        handlers.remove(callback)


def _load_bpy() -> Any:
    try:
        import bpy
    except Exception as exc:
        raise RuntimeError(
            "Blender bpy module is required for readiness invalidation"
        ) from exc
    return bpy


def _install_ui_bindings(ui_module: Any) -> None:
    ui_module.store_a1_export_readiness = store_a1_export_readiness
    ui_module.clear_a1_export_readiness = clear_a1_export_readiness
    ui_module.a1_readiness_depsgraph_update_post = (
        a1_readiness_depsgraph_update_post
    )


def _restore_ui_bindings(ui_module: Any) -> None:
    ui_module.store_a1_export_readiness = _BASE_STORE
    ui_module.clear_a1_export_readiness = _BASE_CLEAR
    ui_module.a1_readiness_depsgraph_update_post = _BASE_HANDLER


def register() -> None:
    """Replace the broad handler after the Rewrite UI has registered it."""

    global _REGISTERED
    if _REGISTERED:
        return
    bpy = _load_bpy()
    from .. import ui

    handlers = bpy.app.handlers.depsgraph_update_post
    try:
        _install_ui_bindings(ui)
        _remove_all(handlers, _BASE_HANDLER)
        _remove_all(handlers, a1_readiness_depsgraph_update_post)
        handlers.append(a1_readiness_depsgraph_update_post)
    except Exception:
        logger.exception("Unable to install precise A1 readiness invalidation")
        _remove_all(handlers, a1_readiness_depsgraph_update_post)
        if _BASE_HANDLER not in handlers:
            handlers.append(_BASE_HANDLER)
        _restore_ui_bindings(ui)
        raise
    _REGISTERED = True


def unregister() -> None:
    """Restore the original UI bindings and handler symmetrically."""

    global _REGISTERED
    if not _REGISTERED:
        _DEPENDENCIES_BY_SCENE.clear()
        return
    bpy = _load_bpy()
    from .. import ui

    handlers = bpy.app.handlers.depsgraph_update_post
    _remove_all(handlers, a1_readiness_depsgraph_update_post)
    _restore_ui_bindings(ui)
    if _BASE_HANDLER not in handlers:
        handlers.append(_BASE_HANDLER)
    _DEPENDENCIES_BY_SCENE.clear()
    _REGISTERED = False


__all__ = [
    "a1_readiness_depsgraph_update_post",
    "clear_a1_export_readiness",
    "register",
    "store_a1_export_readiness",
    "unregister",
]
