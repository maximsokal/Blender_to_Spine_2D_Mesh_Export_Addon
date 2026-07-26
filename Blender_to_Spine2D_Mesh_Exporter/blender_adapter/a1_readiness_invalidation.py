"""Stable, dependency-scoped invalidation for cached A1 Rewrite readiness.

The production readiness pipeline creates temporary Blender datablocks and performs
context-sensitive UV operations. Blender may publish those dependency-graph updates
while Analyze is running or immediately after temporary data has been removed. This
module owns the UI-facing readiness lifecycle so those internal updates cannot turn a
fresh report into STALE, while real changes to the current export request still do.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
import logging
from math import isfinite
from typing import Any, Mapping

try:
    from bpy.app.handlers import persistent as _persistent
except Exception:  # pragma: no cover - non-Blender unit-test mocks.
    def _persistent(function):
        return function

from ..application import A1ExportReadinessReport, A1ReadinessState
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
_RENDER_PIPELINE_MARKERS = (
    "B4",
    "CAMERA",
    "CAMERA_PROJECTION",
    "GROUPED_CAMERA",
    "PROJECTION",
)

_BASE_ANALYSE = _readiness.analyse_a1_export_readiness
_BASE_BUILD_SIGNATURE = _readiness.build_a1_readiness_signature
_BASE_STORE = _readiness.store_a1_export_readiness
_BASE_CLEAR = _readiness.clear_a1_export_readiness
_BASE_HANDLER = _readiness.a1_readiness_depsgraph_update_post
_BASE_REQUIRE = _readiness.require_current_a1_export_readiness


@dataclass(frozen=True, slots=True)
class _DependencySnapshot:
    identities: frozenset[DependencyIdentity]
    labels: Mapping[DependencyIdentity, str]
    values: Mapping[DependencyIdentity, Any]
    states: Mapping[DependencyIdentity, Mapping[str, object]]
    uses_scene_rendering: bool


_DEPENDENCIES_BY_SCENE: dict[int, _DependencySnapshot] = {}
_STALE_REASONS_BY_SCENE: dict[int, str] = {}
_ANALYSIS_SCENES: set[int] = set()
_REGISTERED = False
_BASE_PANEL_DRAW_READINESS: Any | None = None


def _safe_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    try:
        return tuple(value)
    except Exception:
        logger.debug("Unable to iterate Blender RNA collection", exc_info=True)
        return ()


def _safe_float_tuple(value: Any, size: int) -> tuple[float, ...] | str:
    try:
        result = tuple(float(value[index]) for index in range(size))
    except Exception:
        return str(value)
    return result if all(isfinite(item) for item in result) else str(result)


def _matrix_signature(matrix: Any) -> tuple[float, ...] | str:
    try:
        values = tuple(
            float(matrix[row][column])
            for row in range(4)
            for column in range(4)
        )
    except Exception:
        return str(matrix)
    return values if all(isfinite(value) for value in values) else str(values)


def _dependency_identity(value: Any) -> DependencyIdentity | None:
    if value is None:
        return None
    id_type = str(getattr(value, "id_type", "") or "").strip().upper()
    if id_type not in _readiness._RELEVANT_ID_TYPES:
        return None
    return id_type, _rna_identity(value)


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


def _statistics_items(report: Any) -> tuple[tuple[str, object], ...]:
    if report is None:
        return ()
    items: list[tuple[str, object]] = []
    for key, value in getattr(report, "statistics", {}).items():
        items.append((str(key), value))
    for object_report in tuple(getattr(report, "objects", ())):
        for key, value in getattr(object_report, "statistics", {}).items():
            items.append((str(key), value))
    return tuple(items)


def _report_uses_scene_rendering(report: Any) -> bool:
    for key, value in _statistics_items(report):
        normalized_key = key.upper()
        if not any(
            token in normalized_key
            for token in ("PIPELINE", "STRATEGY", "PROJECTION", "BAKE_MODE")
        ):
            continue
        normalized_value = str(value).upper()
        if any(marker in normalized_value for marker in _RENDER_PIPELINE_MARKERS):
            return True
    return False


def _add_dependency(
    identities: set[DependencyIdentity],
    labels: dict[DependencyIdentity, str],
    values: dict[DependencyIdentity, Any],
    value: Any,
    *,
    label: str,
) -> DependencyIdentity | None:
    identity = _dependency_identity(value)
    if identity is None:
        return None
    identities.add(identity)
    labels.setdefault(identity, label)
    values.setdefault(identity, value)
    return identity


def _collect_node_tree(
    node_tree: Any,
    identities: set[DependencyIdentity],
    labels: dict[DependencyIdentity, str],
    values: dict[DependencyIdentity, Any],
    visited: set[DependencyIdentity],
    *,
    owner_label: str,
) -> None:
    identity = _add_dependency(
        identities,
        labels,
        values,
        node_tree,
        label=f"{owner_label} node tree",
    )
    if identity is None or identity in visited:
        return
    visited.add(identity)
    for node in _safe_tuple(getattr(node_tree, "nodes", ())):
        image = getattr(node, "image", None)
        if image is not None:
            _add_dependency(
                identities,
                labels,
                values,
                image,
                label=f"Image '{_datablock_name(image) or 'unnamed'}'",
            )
        texture = getattr(node, "texture", None)
        if texture is not None:
            _add_dependency(
                identities,
                labels,
                values,
                texture,
                label=f"Texture '{_datablock_name(texture) or 'unnamed'}'",
            )
        nested = getattr(node, "node_tree", None)
        if nested is not None:
            _collect_node_tree(
                nested,
                identities,
                labels,
                values,
                visited,
                owner_label=owner_label,
            )


def _collect_material(
    material: Any,
    identities: set[DependencyIdentity],
    labels: dict[DependencyIdentity, str],
    values: dict[DependencyIdentity, Any],
    visited_node_trees: set[DependencyIdentity],
) -> None:
    if material is None:
        return
    material_name = _datablock_name(material) or "unnamed"
    _add_dependency(
        identities,
        labels,
        values,
        material,
        label=f"Material '{material_name}'",
    )
    node_tree = getattr(material, "node_tree", None)
    if node_tree is not None:
        _collect_node_tree(
            node_tree,
            identities,
            labels,
            values,
            visited_node_trees,
            owner_label=f"Material '{material_name}'",
        )


def _collect_object(
    obj: Any,
    identities: set[DependencyIdentity],
    labels: dict[DependencyIdentity, str],
    values: dict[DependencyIdentity, Any],
    visited_node_trees: set[DependencyIdentity],
    *,
    role: str,
) -> None:
    if obj is None:
        return
    object_name = _datablock_name(obj) or "unnamed"
    _add_dependency(
        identities,
        labels,
        values,
        obj,
        label=f"{role} '{object_name}'",
    )
    data = getattr(obj, "data", None)
    if data is not None:
        _add_dependency(
            identities,
            labels,
            values,
            data,
            label=f"{role} '{object_name}' data",
        )
    for slot in _safe_tuple(getattr(obj, "material_slots", ())):
        _collect_material(
            getattr(slot, "material", None),
            identities,
            labels,
            values,
            visited_node_trees,
        )


def _hash_value(digest: Any, value: object) -> None:
    digest.update(repr(value).encode("utf-8", errors="backslashreplace"))
    digest.update(b"\0")


def _mesh_geometry_digest(mesh: Any) -> str:
    """Hash exact source geometry/UV values without allocating a BMesh."""

    digest = sha256()
    for vertex in _safe_tuple(getattr(mesh, "vertices", ())):
        _hash_value(
            digest,
            (
                int(getattr(vertex, "index", 0)),
                _safe_float_tuple(getattr(vertex, "co", ()), 3),
            ),
        )
    for edge in _safe_tuple(getattr(mesh, "edges", ())):
        _hash_value(
            digest,
            (
                int(getattr(edge, "index", 0)),
                tuple(int(value) for value in getattr(edge, "vertices", ())),
            ),
        )
    for polygon in _safe_tuple(getattr(mesh, "polygons", ())):
        _hash_value(
            digest,
            (
                int(getattr(polygon, "index", 0)),
                tuple(int(value) for value in getattr(polygon, "vertices", ())),
                int(getattr(polygon, "material_index", 0)),
                bool(getattr(polygon, "use_smooth", False)),
            ),
        )
    for layer in _safe_tuple(getattr(mesh, "uv_layers", ())):
        _hash_value(digest, _datablock_name(layer))
        for loop_uv in _safe_tuple(getattr(layer, "data", ())):
            _hash_value(
                digest,
                _safe_float_tuple(getattr(loop_uv, "uv", ()), 2),
            )
    return digest.hexdigest()


def _dependency_state(
    identity: DependencyIdentity,
    value: Any,
    *,
    include_geometry_digest: bool,
) -> Mapping[str, object]:
    id_type = identity[0]
    state: dict[str, object] = {
        "id_type": id_type,
        "identity": identity[1],
        "name": _datablock_name(value),
    }
    if id_type == "OBJECT":
        state.update(
            {
                "object_type": str(getattr(value, "type", "")),
                "matrix_world": _matrix_signature(getattr(value, "matrix_world", None)),
                "location": _safe_float_tuple(getattr(value, "location", ()), 3),
                "rotation": _safe_float_tuple(getattr(value, "rotation_euler", ()), 3),
                "scale": _safe_float_tuple(getattr(value, "scale", ()), 3),
                "hide_render": bool(getattr(value, "hide_render", False)),
                "data": _dependency_identity(getattr(value, "data", None)),
            }
        )
    elif id_type == "MESH":
        state.update(
            {
                "vertices": len(_safe_tuple(getattr(value, "vertices", ()))),
                "edges": len(_safe_tuple(getattr(value, "edges", ()))),
                "loops": len(_safe_tuple(getattr(value, "loops", ()))),
                "polygons": len(_safe_tuple(getattr(value, "polygons", ()))),
            }
        )
        if include_geometry_digest:
            state["geometry_digest"] = _mesh_geometry_digest(value)
    elif id_type == "NODETREE":
        state.update(
            {
                "nodes": len(_safe_tuple(getattr(value, "nodes", ()))),
                "links": len(_safe_tuple(getattr(value, "links", ()))),
            }
        )
    elif id_type in {"MATERIAL", "WORLD"}:
        state.update(
            {
                "use_nodes": bool(getattr(value, "use_nodes", False)),
                "node_tree": _dependency_identity(getattr(value, "node_tree", None)),
            }
        )
    elif id_type == "IMAGE":
        state.update(
            {
                "filepath": str(getattr(value, "filepath", "")),
                "source": str(getattr(value, "source", "")),
                "size": _safe_float_tuple(getattr(value, "size", ()), 2),
            }
        )
    return state


def _capture_dependencies(
    context: Any,
    report: Any,
    *,
    include_exact_states: bool,
) -> _DependencySnapshot:
    scene = getattr(context, "scene", None)
    uses_scene_rendering = _report_uses_scene_rendering(report)
    identities: set[DependencyIdentity] = set()
    labels: dict[DependencyIdentity, str] = {}
    values: dict[DependencyIdentity, Any] = {}
    visited_node_trees: set[DependencyIdentity] = set()

    for obj in _readiness._request_mesh_objects(context):
        _collect_object(
            obj,
            identities,
            labels,
            values,
            visited_node_trees,
            role="Object",
        )

    if uses_scene_rendering:
        world = getattr(scene, "world", None)
        if world is not None:
            world_name = _datablock_name(world) or "World"
            _add_dependency(
                identities,
                labels,
                values,
                world,
                label=f"World '{world_name}'",
            )
            _collect_node_tree(
                getattr(world, "node_tree", None),
                identities,
                labels,
                values,
                visited_node_trees,
                owner_label=f"World '{world_name}'",
            )

        _collect_node_tree(
            getattr(scene, "node_tree", None),
            identities,
            labels,
            values,
            visited_node_trees,
            owner_label="Scene compositor",
        )

        camera = getattr(scene, "camera", None)
        if camera is not None:
            _collect_object(
                camera,
                identities,
                labels,
                values,
                visited_node_trees,
                role="Camera",
            )

        for scene_object in _safe_tuple(getattr(scene, "objects", ())):
            if str(getattr(scene_object, "type", "")).upper() != "LIGHT":
                continue
            _collect_object(
                scene_object,
                identities,
                labels,
                values,
                visited_node_trees,
                role="Light",
            )

    states = {
        identity: _dependency_state(
            identity,
            values[identity],
            include_geometry_digest=include_exact_states,
        )
        for identity in identities
    }
    return _DependencySnapshot(
        identities=frozenset(identities),
        labels=dict(labels),
        values=dict(values),
        states=states,
        uses_scene_rendering=uses_scene_rendering,
    )


def _cached_report(context: Any) -> A1ExportReadinessReport | None:
    scene = getattr(context, "scene", None)
    if scene is None:
        return None
    try:
        entry = _readiness._READINESS_CACHE.get(_readiness._scene_key(scene))
    except Exception:
        return None
    return None if entry is None else entry.report


def build_a1_readiness_signature(
    context: Any,
    report: A1ExportReadinessReport | None = None,
) -> str:
    """Build a cheap request/dependency signature suitable for every UI redraw."""

    base_signature = _BASE_BUILD_SIGNATURE(context)
    scene = getattr(context, "scene", None)
    try:
        key = _readiness._scene_key(scene)
    except Exception:
        key = None

    resolved_report = report
    if resolved_report is None:
        if key is not None and key in _ANALYSIS_SCENES:
            return base_signature
        resolved_report = _cached_report(context)
    if resolved_report is None:
        return base_signature

    snapshot = _capture_dependencies(
        context,
        resolved_report,
        include_exact_states=False,
    )
    dependency_state = tuple(
        snapshot.states[identity]
        for identity in sorted(snapshot.identities, key=repr)
    )
    payload = {
        "base": base_signature,
        "uses_scene_rendering": snapshot.uses_scene_rendering,
        "dependencies": dependency_state,
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def _flush_context_updates(context: Any) -> None:
    view_layer = getattr(context, "view_layer", None)
    update = getattr(view_layer, "update", None)
    if callable(update):
        update()


def analyse_a1_export_readiness(context: Any) -> A1ExportReadinessReport:
    """Run Analyze under a guard and sign the fully restored post-analysis state."""

    scene = getattr(context, "scene", None)
    key = _readiness._scene_key(scene)
    _ANALYSIS_SCENES.add(key)
    _STALE_REASONS_BY_SCENE.pop(key, None)
    _DEPENDENCIES_BY_SCENE.pop(key, None)
    _BASE_CLEAR(scene)
    try:
        report = _BASE_ANALYSE(context)
        _flush_context_updates(context)
        final_signature = build_a1_readiness_signature(context, report)
        if report.signature != final_signature:
            report = replace(report, signature=final_signature)
        return report
    finally:
        _ANALYSIS_SCENES.discard(key)


def store_a1_export_readiness(context: Any, report: Any) -> None:
    """Cache a report only after the live Blender state has been stabilized."""

    if not isinstance(report, A1ExportReadinessReport):
        raise TypeError("report must be A1ExportReadinessReport")
    scene = getattr(context, "scene", None)
    key = _readiness._scene_key(scene)
    added_guard = key not in _ANALYSIS_SCENES
    if added_guard:
        _ANALYSIS_SCENES.add(key)
    try:
        _flush_context_updates(context)
        final_signature = build_a1_readiness_signature(context, report)
        stabilized_report = (
            report
            if report.signature == final_signature
            else replace(report, signature=final_signature)
        )
        _BASE_STORE(context, stabilized_report)
        _DEPENDENCIES_BY_SCENE[key] = _capture_dependencies(
            context,
            stabilized_report,
            include_exact_states=True,
        )
        _STALE_REASONS_BY_SCENE.pop(key, None)
    finally:
        if added_guard:
            _ANALYSIS_SCENES.discard(key)


def clear_a1_export_readiness(scene: Any | None = None) -> None:
    """Clear the base report and every v2 dependency/reason sidecar."""

    if scene is None:
        _DEPENDENCIES_BY_SCENE.clear()
        _STALE_REASONS_BY_SCENE.clear()
        _ANALYSIS_SCENES.clear()
        _BASE_CLEAR()
        return
    key = _readiness._scene_key(scene)
    _DEPENDENCIES_BY_SCENE.pop(key, None)
    _STALE_REASONS_BY_SCENE.pop(key, None)
    _ANALYSIS_SCENES.discard(key)
    _BASE_CLEAR(scene)


def _update_flags(update: Any) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    for name in _UPDATE_FLAGS:
        try:
            value = getattr(update, name)
        except Exception:
            continue
        if isinstance(value, bool):
            flags[name] = value
    return flags


def _reason_for_update(
    identity: DependencyIdentity,
    label: str,
    flags: Mapping[str, bool],
) -> str:
    id_type = identity[0]
    if flags.get("is_updated_geometry", False):
        return f"{label} geometry changed"
    if flags.get("is_updated_transform", False):
        return f"{label} transform changed"
    if flags.get("is_updated_shading", False):
        return f"{label} shading changed"
    if id_type in {"MATERIAL", "NODETREE", "TEXTURE", "IMAGE"}:
        return f"{label} changed"
    if id_type in {"CAMERA", "LIGHT", "WORLD"}:
        return f"{label} rendering state changed"
    return f"{label} changed"


def _semantic_state_unchanged(
    identity: DependencyIdentity,
    updated_id: Any,
    flags: Mapping[str, bool],
    snapshot: _DependencySnapshot,
) -> bool:
    id_type = identity[0]
    compare_state = False
    if id_type == "MESH" and flags.get("is_updated_geometry", False):
        compare_state = True
    elif (
        id_type == "OBJECT"
        and flags.get("is_updated_transform", False)
        and not flags.get("is_updated_geometry", False)
        and not flags.get("is_updated_shading", False)
    ):
        compare_state = True
    if not compare_state:
        return False
    previous = snapshot.states.get(identity)
    if previous is None:
        return False
    try:
        current = _dependency_state(
            identity,
            updated_id,
            include_geometry_digest=True,
        )
    except Exception:
        logger.debug("Unable to compare delayed depsgraph state", exc_info=True)
        return False
    return current == previous


def _mark_stale(key: int, entry: Any, reason: str) -> None:
    entry.stale = True
    _STALE_REASONS_BY_SCENE[key] = reason
    logger.debug("A1 readiness marked stale: %s", reason)


@_persistent
def a1_readiness_depsgraph_update_post(scene: Any, depsgraph: Any) -> None:
    """Invalidate only when an exact dependency of the cached request changed."""

    if scene is None or not _readiness._READINESS_CACHE:
        return
    try:
        key = _readiness._scene_key(scene)
    except Exception:
        logger.debug("Unable to resolve readiness Scene key", exc_info=True)
        return
    if key in _ANALYSIS_SCENES:
        return

    entry = _readiness._READINESS_CACHE.get(key)
    if entry is None or entry.stale:
        return
    snapshot = _DEPENDENCIES_BY_SCENE.get(key)
    if snapshot is None:
        return

    try:
        updates = tuple(getattr(depsgraph, "updates", ()))
    except Exception:
        logger.debug("Unable to read depsgraph updates", exc_info=True)
        return
    if not updates:
        return

    for update in updates:
        updated_id = getattr(update, "id", None)
        identity = _dependency_identity(updated_id)
        if identity is None or _is_temporary_datablock(updated_id):
            continue
        if identity not in snapshot.identities:
            # Unrelated Blender IDs must not disable the current export request.
            continue

        flags = _update_flags(update)
        if flags and not any(flags.values()):
            # Selection/active-object restoration publishes zero-semantic updates.
            continue
        if _semantic_state_unchanged(identity, updated_id, flags, snapshot):
            # Blender can deliver a queued source update after Analyze even though the
            # restored source data is byte-for-byte unchanged.
            continue

        label = snapshot.labels.get(
            identity,
            _datablock_name(updated_id) or identity[0].title(),
        )
        _mark_stale(key, entry, _reason_for_update(identity, label, flags))
        return


def current_a1_readiness_reason(context: Any) -> str:
    """Return a user-facing reason for the current STALE state."""

    scene = getattr(context, "scene", None)
    if scene is None:
        return "The active Scene is unavailable"
    try:
        key = _readiness._scene_key(scene)
    except Exception:
        return "The active Scene identity changed"

    explicit = _STALE_REASONS_BY_SCENE.get(key)
    if explicit:
        return explicit

    entry = _readiness._READINESS_CACHE.get(key)
    if entry is None:
        return "No cached analysis is available"
    try:
        current_signature = build_a1_readiness_signature(context)
    except Exception:
        logger.debug("Unable to explain readiness signature change", exc_info=True)
        return "The export request can no longer be validated"
    if current_signature != entry.signature:
        return "Export selection, frame, Scene, or settings changed"
    if entry.stale:
        return "An export dependency changed"
    return ""


def require_current_a1_export_readiness(context: Any) -> tuple[bool, str]:
    allowed, message = _BASE_REQUIRE(context)
    if allowed:
        return True, ""
    state, _report = _readiness.current_a1_export_readiness(context)
    if state is A1ReadinessState.STALE:
        reason = current_a1_readiness_reason(context)
        return False, f"Export analysis is outdated: {reason}. Run Analyze again"
    return False, message


def _draw_readiness_v2(self: Any, layout: Any, context: Any) -> bool:
    """Draw the existing readiness UI with a precise STALE explanation."""

    state, report = _readiness.current_a1_export_readiness(context)
    box = layout.box()
    row = box.row(align=True)
    row.label(text="Export readiness:")
    row.operator(
        "object.spine2d_refresh_info",
        text="Analyze",
        icon="VIEWZOOM",
    )

    if state is A1ReadinessState.NOT_ANALYSED:
        box.label(text="Not analyzed", icon="QUESTION")
        box.label(text="Run Analyze before export")
        return False
    if state is A1ReadinessState.STALE:
        box.label(text="Analysis outdated", icon="FILE_REFRESH")
        box.label(text=current_a1_readiness_reason(context))
        return False
    if report is None:
        box.label(text="Analysis cache unavailable", icon="CANCEL")
        return False

    box.label(
        text=(
            f"{state.value}: {report.blocker_count} blocker(s), "
            f"{report.warning_count} warning(s)"
        ),
        icon=self._state_icon(state),
    )
    for issue in report.issues[:6]:
        box.label(
            text=f"{issue.code}: {issue.message}",
            icon=self._issue_icon(issue.severity),
        )
    for item in report.objects:
        self._draw_object_readiness(box, item)
    return report.can_export


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


def _install_bindings(ui_module: Any) -> None:
    global _BASE_PANEL_DRAW_READINESS

    _readiness.analyse_a1_export_readiness = analyse_a1_export_readiness
    _readiness.build_a1_readiness_signature = build_a1_readiness_signature
    _readiness.store_a1_export_readiness = store_a1_export_readiness
    _readiness.clear_a1_export_readiness = clear_a1_export_readiness
    _readiness.require_current_a1_export_readiness = require_current_a1_export_readiness
    _readiness.a1_readiness_depsgraph_update_post = a1_readiness_depsgraph_update_post

    ui_module.analyse_a1_export_readiness = analyse_a1_export_readiness
    ui_module.store_a1_export_readiness = store_a1_export_readiness
    ui_module.clear_a1_export_readiness = clear_a1_export_readiness
    ui_module.require_current_a1_export_readiness = require_current_a1_export_readiness
    ui_module.a1_readiness_depsgraph_update_post = a1_readiness_depsgraph_update_post

    panel_class = ui_module.OBJECT_PT_Spine2DMeshPanel
    if _BASE_PANEL_DRAW_READINESS is None:
        _BASE_PANEL_DRAW_READINESS = panel_class._draw_readiness
    panel_class._draw_readiness = _draw_readiness_v2


def _restore_bindings(ui_module: Any) -> None:
    global _BASE_PANEL_DRAW_READINESS

    _readiness.analyse_a1_export_readiness = _BASE_ANALYSE
    _readiness.build_a1_readiness_signature = _BASE_BUILD_SIGNATURE
    _readiness.store_a1_export_readiness = _BASE_STORE
    _readiness.clear_a1_export_readiness = _BASE_CLEAR
    _readiness.require_current_a1_export_readiness = _BASE_REQUIRE
    _readiness.a1_readiness_depsgraph_update_post = _BASE_HANDLER

    ui_module.analyse_a1_export_readiness = _BASE_ANALYSE
    ui_module.store_a1_export_readiness = _BASE_STORE
    ui_module.clear_a1_export_readiness = _BASE_CLEAR
    ui_module.require_current_a1_export_readiness = _BASE_REQUIRE
    ui_module.a1_readiness_depsgraph_update_post = _BASE_HANDLER

    if _BASE_PANEL_DRAW_READINESS is not None:
        ui_module.OBJECT_PT_Spine2DMeshPanel._draw_readiness = (
            _BASE_PANEL_DRAW_READINESS
        )
        _BASE_PANEL_DRAW_READINESS = None


def register() -> None:
    """Replace the broad readiness lifecycle after the Rewrite UI is registered."""

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
        logger.exception("Unable to install A1 readiness v2 lifecycle")
        _remove_all(handlers, a1_readiness_depsgraph_update_post)
        if _BASE_HANDLER not in handlers:
            handlers.append(_BASE_HANDLER)
        _restore_bindings(ui)
        raise
    _REGISTERED = True


def unregister() -> None:
    """Restore the original UI bindings, method, and handler symmetrically."""

    global _REGISTERED
    if not _REGISTERED:
        _DEPENDENCIES_BY_SCENE.clear()
        _STALE_REASONS_BY_SCENE.clear()
        _ANALYSIS_SCENES.clear()
        return
    bpy = _load_bpy()
    from .. import ui

    handlers = bpy.app.handlers.depsgraph_update_post
    _remove_all(handlers, a1_readiness_depsgraph_update_post)
    _restore_bindings(ui)
    if _BASE_HANDLER not in handlers:
        handlers.append(_BASE_HANDLER)
    _DEPENDENCIES_BY_SCENE.clear()
    _STALE_REASONS_BY_SCENE.clear()
    _ANALYSIS_SCENES.clear()
    _REGISTERED = False


__all__ = [
    "a1_readiness_depsgraph_update_post",
    "analyse_a1_export_readiness",
    "build_a1_readiness_signature",
    "clear_a1_export_readiness",
    "current_a1_readiness_reason",
    "register",
    "require_current_a1_export_readiness",
    "store_a1_export_readiness",
    "unregister",
]
