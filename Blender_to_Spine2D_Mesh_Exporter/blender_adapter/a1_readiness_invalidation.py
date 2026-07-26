"""Own precise depsgraph invalidation for cached A1 readiness reports.

The deep readiness analysis creates and removes temporary Blender datablocks. Blender
may publish those queued dependency-graph updates after the fresh report has already
been cached. The original broad handler treated every relevant ID update (and even an
empty update batch) as a user edit, which immediately changed READY back to STALE.

This owner replaces that broad runtime handler, records the real scene dependencies at
cache time, and ignores exporter-owned temporary datablocks plus selection-only Object
updates. Actual geometry, transform, shading, material, image, camera, light, world, and
new scene dependency changes still invalidate the report.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from bpy.app.handlers import persistent as _persistent
except Exception:  # pragma: no cover - exercised by non-Blender unit-test mocks.
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
_REWRITE_TEMPORARY_PREFIXES = ("__Spine2D_",)
_REWRITE_TEMPORARY_MARKERS = (".spine2d-stage-v2~",)

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


def _id_type(value: Any) -> str:
    return str(getattr(value, "id_type", "") or "").strip().upper()


def _dependency_identity(value: Any) -> DependencyIdentity | None:
    if value is None:
        return None
    resolved_type = _id_type(value)
    if resolved_type not in _readiness._RELEVANT_ID_TYPES:
        return None
    return resolved_type, _rna_identity(value)


def _add_dependency(
    dependencies: set[DependencyIdentity],
    value: Any,
) -> DependencyIdentity | None:
    identity = _dependency_identity(value)
    if identity is not None:
        dependencies.add(identity)
    return identity


def _collect_node_tree_dependencies(
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
        nested_tree = getattr(node, "node_tree", None)
        if nested_tree is not None:
            _collect_node_tree_dependencies(nested_tree, dependencies, visited)


def _collect_material_dependencies(
    material: Any,
    dependencies: set[DependencyIdentity],
    visited_node_trees: set[DependencyIdentity],
) -> None:
    if material is None:
        return
    _add_dependency(dependencies, material)
    node_tree = getattr(material, "node_tree", None)
    if node_tree is not None:
        _collect_node_tree_dependencies(
            node_tree,
            dependencies,
            visited_node_trees,
        )


def _collect_object_dependencies(
    obj: Any,
    dependencies: set[DependencyIdentity],
    visited_node_trees: set[DependencyIdentity],
) -> None:
    if obj is None:
        return
    _add_dependency(dependencies, obj)
    _add_dependency(dependencies, getattr(obj, "data", None))
    for slot in _safe_tuple(getattr(obj, "material_slots", ())):
        _collect_material_dependencies(
            getattr(slot, "material", None),
            dependencies,
            visited_node_trees,
        )


def _capture_readiness_dependencies(context: Any) -> frozenset[DependencyIdentity]:
    dependencies: set[DependencyIdentity] = set()
    visited_node_trees: set[DependencyIdentity] = set()
    scene = getattr(context, "scene", None)

    # The Scene itself is retained so a real Scene update can be recognized. A
    # signature comparison still owns ordinary Rewrite setting changes.
    _add_dependency(dependencies, scene)

    world = getattr(scene, "world", None)
    _add_dependency(dependencies, world)
    if world is not None:
        world_tree = getattr(world, "node_tree", None)
        if world_tree is not None:
            _collect_node_tree_dependencies(
                world_tree,
                dependencies,
                visited_node_trees,
            )

    scene_tree = getattr(scene, "node_tree", None)
    if scene_tree is not None:
        _collect_node_tree_dependencies(
            scene_tree,
            dependencies,
            visited_node_trees,
        )

    unique_objects: dict[tuple[str, object], Any] = {}
    for obj in _safe_tuple(getattr(scene, "objects", ())):
        unique_objects.setdefault(_rna_identity(obj), obj)
    for obj in _readiness._request_mesh_objects(context):
        unique_objects.setdefault(_rna_identity(obj), obj)

    camera = getattr(scene, "camera", None)
    if camera is not None:
        unique_objects.setdefault(_rna_identity(camera), camera)

    for obj in unique_objects.values():
        _collect_object_dependencies(
            obj,
            dependencies,
            visited_node_trees,
        )

    return frozenset(dependencies)


def _datablock_name(value: Any) -> str:
    return str(
        getattr(value, "name_full", None)
        or getattr(value, "name", None)
        or ""
    ).strip()


def _is_rewrite_temporary_datablock(value: Any) -> bool:
    name = _datablock_name(value)
    if not name:
        return False
    if any(name.startswith(prefix) for prefix in _REWRITE_TEMPORARY_PREFIXES):
        return True
    folded = name.casefold()
    return any(marker in folded for marker in _REWRITE_TEMPORARY_MARKERS)


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


def _is_selection_only_known_object_update(
    update: Any,
    *,
    id_type: str,
    known_dependency: bool,
) -> bool:
    if id_type != "OBJECT" or not known_dependency:
        return False
    flags = _update_flags(update)
    return bool(flags) and not any(flags)


def store_a1_export_readiness(context: Any, report: Any) -> None:
    """Cache the base report and snapshot the live dependencies it represents."""

    _BASE_STORE(context, report)
    scene = getattr(context, "scene", None)
    key = _readiness._scene_key(scene)
    try:
        _DEPENDENCIES_BY_SCENE[key] = _capture_readiness_dependencies(context)
    except Exception:
        # A missing dependency snapshot must never make a READY report unsafely
        # permanent. An empty set keeps the handler conservative for non-temp IDs.
        logger.exception("Unable to capture A1 readiness dependency identities")
        _DEPENDENCIES_BY_SCENE[key] = frozenset()


def clear_a1_export_readiness(scene: Any | None = None) -> None:
    """Clear the base report and its matching dependency snapshot."""

    if scene is None:
        _DEPENDENCIES_BY_SCENE.clear()
        _BASE_CLEAR()
        return
    key = _readiness._scene_key(scene)
    _DEPENDENCIES_BY_SCENE.pop(key, None)
    _BASE_CLEAR(scene)


@_persistent
def a1_readiness_depsgraph_update_post(scene: Any, depsgraph: Any) -> None:
    """Invalidate only reports touched by real, export-relevant datablocks."""

    if not _readiness._READINESS_CACHE or scene is None:
        return
    try:
        key = _readiness._scene_key(scene)
    except Exception:
        logger.debug("Unable to resolve Scene key for readiness invalidation", exc_info=True)
        return

    entry = _readiness._READINESS_CACHE.get(key)
    if entry is None or entry.stale:
        return
    try:
        updates = tuple(getattr(depsgraph, "updates", ()))
    except Exception:
        logger.debug("Unable to read dependency-graph updates", exc_info=True)
        return

    # An empty batch carries no evidence that the cached request changed. The old
    # handler treated it as globally relevant and disabled Export immediately.
    if not updates:
        return

    dependencies = _DEPENDENCIES_BY_SCENE.get(key, frozenset())
    for update in updates:
        updated_id = getattr(update, "id", None)
        id_type = _id_type(updated_id)
        if id_type not in _readiness._RELEVANT_ID_TYPES:
            continue
        if _is_rewrite_temporary_datablock(updated_id):
            continue

        identity = _dependency_identity(updated_id)
        known_dependency = identity is not None and identity in dependencies

        # Activating the isolated temporary unwrap object temporarily changes
        # selection/active Object state. Blender can report that restoration as an
        # Object update with all semantic update flags false.
        if _is_selection_only_known_object_update(
            update,
            id_type=id_type,
            known_dependency=known_dependency,
        ):
            continue

        if id_type == "SCENE":
            # Scene-level property changes are already represented by the cheap request
            # signature. Link/unlink noise from temporary objects must not defeat a
            # freshly stored report.
            continue

        # A known dependency changed, or a new non-temporary relevant datablock entered
        # the dependency graph. Both require a fresh production-backed analysis.
        entry.stale = True
        return


def _remove_all(handlers: Any, callback: Any) -> None:
    while callback in handlers:
        handlers.remove(callback)


def _load_bpy() -> Any:
    try:
        import bpy
    except Exception as exc:
        raise RuntimeError("Blender bpy module is required for readiness invalidation") from exc
    return bpy


def _install_bindings(ui_module: Any) -> None:
    _readiness.store_a1_export_readiness = store_a1_export_readiness
    _readiness.clear_a1_export_readiness = clear_a1_export_readiness
    _readiness.a1_readiness_depsgraph_update_post = a1_readiness_depsgraph_update_post
    ui_module.store_a1_export_readiness = store_a1_export_readiness
    ui_module.clear_a1_export_readiness = clear_a1_export_readiness
    ui_module.a1_readiness_depsgraph_update_post = a1_readiness_depsgraph_update_post


def _restore_bindings(ui_module: Any) -> None:
    _readiness.store_a1_export_readiness = _BASE_STORE
    _readiness.clear_a1_export_readiness = _BASE_CLEAR
    _readiness.a1_readiness_depsgraph_update_post = _BASE_HANDLER
    ui_module.store_a1_export_readiness = _BASE_STORE
    ui_module.clear_a1_export_readiness = _BASE_CLEAR
    ui_module.a1_readiness_depsgraph_update_post = _BASE_HANDLER


def register() -> None:
    """Replace the broad UI handler after the Rewrite UI has registered it."""

    global _REGISTERED
    if _REGISTERED:
        return
    bpy = _load_bpy()
    from .. import ui

    handlers = bpy.app.handlers.depsgraph_update_post
    try:
        _install_bindings(ui)
        _remove_all(handlers, _BASE_HANDLER)
        _remove_all(handlers, a1_readiness_depsgraph_update_post)
        handlers.append(a1_readiness_depsgraph_update_post)
    except Exception:
        logger.exception("Unable to install precise A1 readiness invalidation")
        _remove_all(handlers, a1_readiness_depsgraph_update_post)
        if _BASE_HANDLER not in handlers:
            handlers.append(_BASE_HANDLER)
        _restore_bindings(ui)
        raise
    _REGISTERED = True


def unregister() -> None:
    """Restore the base bindings so standalone UI teardown remains symmetrical."""

    global _REGISTERED
    if not _REGISTERED:
        _DEPENDENCIES_BY_SCENE.clear()
        return
    bpy = _load_bpy()
    from .. import ui

    handlers = bpy.app.handlers.depsgraph_update_post
    _remove_all(handlers, a1_readiness_depsgraph_update_post)
    _restore_bindings(ui)
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
