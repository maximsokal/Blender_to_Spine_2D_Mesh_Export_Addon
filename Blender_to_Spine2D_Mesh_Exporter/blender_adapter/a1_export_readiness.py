"""Run and cache production-backed A1 export-readiness analysis for Blender UI."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import logging
from math import isfinite
from typing import Any, Mapping, Tuple

try:
    from bpy.app.handlers import persistent as _persistent
except Exception:  # pragma: no cover - exercised by non-Blender unit-test mocks.
    def _persistent(function):
        return function

from ..application import (
    A1ExportReadinessReport,
    A1MultiObjectMode,
    A1ObjectReadiness,
    A1ReadinessState,
    ExportIssue,
    IssueSeverity,
    ReadinessStatistic,
)
from ..domain.geometry import build_edge_to_faces
from .a1_mixed_composition import (
    compose_a1_mixed_document,
    partition_mixed_prepared_objects,
)
from .a1_mixed_object_export import prepare_a1_mixed_object
from .a1_multi_object_composition import compose_a1_multi_object_document
from .a1_multi_object_contracts import A1MultiObjectPreparationError
from .a1_multi_object_export import prepare_a1_multi_object
from .a1_object_preparation import (
    A1ObjectPreparationError,
    PreparedA1Object,
    prepare_a1_object,
)
from .a1_ui_export_plan import (
    A1UiMultiExportPlan,
    build_active_ui_export_plan,
    build_selected_ui_export_plan,
)
from .a1_ui_scene_capture import _resolve_spine_target
from .a1_ui_selection import _ordered_selected_meshes, _rna_identity
from .spine_version_preferences import read_spine_project_exact_version_raw


logger = logging.getLogger(__name__)
_RELEVANT_ID_TYPES = frozenset(
    {
        "CAMERA",
        "IMAGE",
        "LIGHT",
        "MATERIAL",
        "MESH",
        "NODETREE",
        "OBJECT",
        "SCENE",
        "TEXTURE",
        "WORLD",
    }
)


@dataclass(slots=True)
class _ReadinessCacheEntry:
    signature: str
    report: A1ExportReadinessReport
    stale: bool = False


_READINESS_CACHE: dict[int, _ReadinessCacheEntry] = {}


def _scene_key(scene: Any) -> int:
    if scene is None:
        raise ValueError("scene cannot be None")
    pointer = getattr(scene, "as_pointer", None)
    if callable(pointer):
        try:
            resolved = int(pointer())
            if resolved:
                return resolved
        except Exception:
            logger.debug("Unable to read Scene RNA pointer", exc_info=True)
    return id(scene)


def _safe_int(value: Any, default: int = 0) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return str(value) if value is not None else default


def _safe_float(value: Any, default: float = 0.0) -> float | str:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return str(value) if value is not None else default
    return result if isfinite(result) else str(result)


def _safe_float_tuple(value: Any, size: int) -> tuple[float, ...] | str:
    try:
        result = tuple(float(value[index]) for index in range(size))
    except Exception:
        return str(value)
    return result if all(isfinite(item) for item in result) else str(result)


def _matrix_signature(matrix: Any) -> tuple[float, ...] | str:
    try:
        values = tuple(
            float(matrix[row][column]) for row in range(4) for column in range(4)
        )
    except Exception:
        return str(matrix)
    return values if all(isfinite(item) for item in values) else str(values)


def _blend_file_path() -> str:
    try:
        import bpy

        return str(getattr(bpy.data, "filepath", ""))
    except Exception:
        return ""


def _material_signature(obj: Any) -> tuple[tuple[object, ...], ...]:
    result: list[tuple[object, ...]] = []
    for slot in tuple(getattr(obj, "material_slots", ())):
        material = getattr(slot, "material", None)
        if material is None:
            result.append(("EMPTY",))
            continue
        node_tree = getattr(material, "node_tree", None)
        result.append(
            (
                _rna_identity(material),
                str(
                    getattr(material, "name_full", None)
                    or getattr(material, "name", "")
                ),
                bool(getattr(material, "use_nodes", False)),
                None if node_tree is None else _rna_identity(node_tree),
                len(tuple(getattr(node_tree, "nodes", ()))) if node_tree else 0,
                len(tuple(getattr(node_tree, "links", ()))) if node_tree else 0,
            )
        )
    return tuple(result)


def _modifier_signature(obj: Any) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            str(getattr(modifier, "name", "")),
            str(getattr(modifier, "type", "")),
            bool(getattr(modifier, "show_viewport", True)),
            bool(getattr(modifier, "show_render", True)),
        )
        for modifier in tuple(getattr(obj, "modifiers", ()))
    )


def _object_signature(obj: Any) -> Mapping[str, object]:
    mesh = getattr(obj, "data", None)
    bake = getattr(obj, "spine2d_bake_settings", None)
    connect = getattr(obj, "spine2d_connect_settings", None)
    return {
        "identity": _rna_identity(obj),
        "name": str(getattr(obj, "name_full", None) or getattr(obj, "name", "")),
        "type": str(getattr(obj, "type", "")),
        "mesh_identity": None if mesh is None else _rna_identity(mesh),
        "vertices": len(tuple(getattr(mesh, "vertices", ()))) if mesh is not None else 0,
        "edges": len(tuple(getattr(mesh, "edges", ()))) if mesh is not None else 0,
        "loops": len(tuple(getattr(mesh, "loops", ()))) if mesh is not None else 0,
        "faces": len(tuple(getattr(mesh, "polygons", ()))) if mesh is not None else 0,
        "matrix_world": _matrix_signature(getattr(obj, "matrix_world", None)),
        "location": _safe_float_tuple(getattr(obj, "location", ()), 3),
        "rotation": _safe_float_tuple(getattr(obj, "rotation_euler", ()), 3),
        "scale": _safe_float_tuple(getattr(obj, "scale", ()), 3),
        "hide_render": bool(getattr(obj, "hide_render", False)),
        "modifiers": _modifier_signature(obj),
        "materials": _material_signature(obj),
        "connect": bool(getattr(connect, "enabled", False)),
        "bake_start": _safe_int(getattr(bake, "bake_frame_start", 0)),
        "bake_frames": _safe_int(getattr(bake, "frames_for_render", 0)),
    }


def _selected_meshes(context: Any) -> Tuple[Any, ...]:
    if context is None:
        return ()
    unique_by_identity: dict[tuple[str, object], Any] = {}
    for obj in getattr(context, "selected_objects", ()):
        if getattr(obj, "type", None) == "MESH":
            unique_by_identity.setdefault(_rna_identity(obj), obj)
    return tuple(unique_by_identity.values())


def _request_mesh_objects(context: Any) -> Tuple[Any, ...]:
    """Mirror the UI router: multi uses its exact ordered selection helper."""

    selected = _selected_meshes(context)
    if len(selected) > 1:
        return _ordered_selected_meshes(context)

    active = getattr(context, "active_object", None)
    if active is not None and getattr(active, "type", None) == "MESH":
        return (active,)
    return selected


def _ordered_signature_objects(context: Any) -> Tuple[Any, ...]:
    return _request_mesh_objects(context)


def build_a1_readiness_signature(context: Any) -> str:
    """Return a cheap signature for the exact UI export request.

    Geometry/material edits are invalidated by the persistent depsgraph handler. This
    signature additionally catches active/selection and every export-setting change.
    """

    if context is None:
        raise ValueError("context cannot be None")
    scene = getattr(context, "scene", None)
    if scene is None:
        raise ValueError("context.scene is missing")

    active = getattr(context, "active_object", None)
    ordered = _ordered_signature_objects(context)
    active_identity = None if active is None else _rna_identity(active)
    request_active_identity = (
        active_identity
        if any(_rna_identity(obj) == active_identity for obj in ordered)
        else None
    )
    render = getattr(scene, "render", None)
    camera = getattr(scene, "camera", None)
    spine_target_raw = getattr(scene, "spine2d_target_spine_version", "SPINE_4_2")
    spine_target = _resolve_spine_target(scene)
    spine_exact_version_raw = read_spine_project_exact_version_raw(
        spine_target,
        context=context,
    )
    payload = {
        "blend_file": _blend_file_path(),
        "scene": _rna_identity(scene),
        "active": request_active_identity,
        "objects": tuple(_object_signature(obj) for obj in ordered),
        "frame_current": _safe_int(getattr(scene, "frame_current", 0)),
        "camera": None if camera is None else _rna_identity(camera),
        "camera_matrix": (
            None if camera is None else _matrix_signature(getattr(camera, "matrix_world", None))
        ),
        "render_engine": str(getattr(render, "engine", "")),
        "settings": {
            "texture_mode": str(
                getattr(scene, "spine2d_texture_export_mode", "NORMAL_UV_SEGMENTS")
            ),
            "projection_direction": str(
                getattr(scene, "spine2d_projection_direction", "POSITIVE_Z")
            ),
            "spine_target": str(spine_target_raw),
            "spine_exact_version_raw": spine_exact_version_raw,
            "texture_size": _safe_int(
                getattr(scene, "spine2d_texture_size", 1024)
            ),
            "json_path": str(getattr(scene, "spine2d_json_path", "")),
            "images_path": str(getattr(scene, "spine2d_images_path", "")),
            "control_icons": bool(getattr(scene, "spine2d_control_icons", False)),
            "preview": bool(
                getattr(scene, "spine2d_export_preview_animation", False)
            ),
            "seam_mode": str(getattr(scene, "spine2d_seam_maker_mode", "AUTO")),
            "angle_limit": _safe_float(
                getattr(scene, "spine2d_angle_limit", 30.0)
            ),
            "angular_mode": str(
                getattr(scene, "spine2d_angular_mode", "SEED_CONE")
            ),
            "local_angle_limit": _safe_float(
                getattr(scene, "spine2d_local_angle_limit", 30.0)
            ),
            "frames": _safe_int(
                getattr(scene, "spine2d_frames_for_render", 0)
            ),
            "frame_start": _safe_int(
                getattr(scene, "spine2d_bake_frame_start", 0)
            ),
            "sequence_fps_override": _safe_float(
                getattr(scene, "spine2d_sequence_fps_override", 0.0)
            ),
            "material_policy": str(
                getattr(scene, "spine2d_material_source_policy", "REQUIRE_SOURCE")
            ),
            "generated_pattern": str(
                getattr(scene, "spine2d_generated_material_pattern", "SOLID_GRAY")
            ),
            "projection_alpha": _safe_float(
                getattr(scene, "spine2d_projection_alpha_threshold", 1.0 / 255.0)
            ),
            "include_scene_shadows": bool(
                getattr(scene, "spine2d_include_scene_shadows", True)
            ),
            "include_scene_reflection_transmission": bool(
                getattr(
                    scene,
                    "spine2d_include_scene_reflection_transmission",
                    True,
                )
            ),
            "world_affects_lighting_reflections": bool(
                getattr(scene, "spine2d_world_affects_lighting_reflections", True)
            ),
            "depth_smoothing": _safe_float(
                getattr(scene, "spine2d_depth_smoothing", 0.35)
            ),
            "depth_edge_threshold": _safe_float(
                getattr(scene, "spine2d_depth_edge_threshold", 0.08)
            ),
            "depth_mesh_error_pixels": _safe_float(
                getattr(scene, "spine2d_depth_mesh_error_pixels", 4.0)
            ),
            "depth_max_points": _safe_int(
                getattr(scene, "spine2d_depth_max_points", 128)
            ),
            "depth_base_mode": str(
                getattr(scene, "spine2d_depth_base_mode", "FARTHEST_VISIBLE")
            ),
        },
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(serialized.encode("utf-8")).hexdigest()


def _readiness_statistics(
    values: Mapping[str, object],
) -> dict[str, ReadinessStatistic]:
    result: dict[str, ReadinessStatistic] = {}
    for key, value in values.items():
        name = str(key).strip()
        if not name:
            continue
        if isinstance(value, bool):
            result[name] = int(value)
        elif isinstance(value, int):
            result[name] = value
        elif isinstance(value, float):
            result[name] = value if isfinite(value) else str(value)
        elif isinstance(value, str):
            result[name] = value
        else:
            result[name] = str(value)
    return result


def _prepared_statistics(prepared: PreparedA1Object) -> dict[str, ReadinessStatistic]:
    if not isinstance(prepared, PreparedA1Object):
        raise TypeError("prepared must be PreparedA1Object")
    snapshot = prepared.source_snapshot
    edge_to_faces = build_edge_to_faces(snapshot)
    used_vertex_ids = {loop.vertex_id for loop in snapshot.loops}
    used_edge_ids = {loop.edge_id for loop in snapshot.loops}
    attachment_vertices = sum(
        len(projection.request.vertices)
        for projection in prepared.document_assembly.projections
    )
    triangulated_faces = sum(
        len(region.snapshot.faces) for region in prepared.geometry.regions
    )
    statistics = _readiness_statistics(prepared.statistics)
    statistics.update(
        {
            "used_vertices": len(used_vertex_ids),
            "loose_vertices": len(snapshot.vertices) - len(used_vertex_ids),
            "loose_edges": len(snapshot.edges) - len(used_edge_ids),
            "boundary_edges": sum(len(faces) == 1 for faces in edge_to_faces.values()),
            "non_manifold_edges": sum(
                len(faces) > 2 for faces in edge_to_faces.values()
            ),
            "n_gon_count": sum(len(face.loop_ids) > 4 for face in snapshot.faces),
            "triangles_after_triangulation": triangulated_faces,
            "exported_attachment_vertices": attachment_vertices,
            "uv_duplicated_vertices": max(
                0,
                attachment_vertices - len(used_vertex_ids),
            ),
        }
    )
    return statistics


def _prepared_object_readiness(prepared: PreparedA1Object) -> A1ObjectReadiness:
    return A1ObjectReadiness(
        object_id=prepared.object_id,
        issues=prepared.warnings,
        statistics=_prepared_statistics(prepared),
    )


def _error_issue(
    *,
    stage: str,
    exc: Exception,
    object_id: str | None,
) -> ExportIssue:
    normalized_stage = str(stage or "VALIDATE_REQUEST").strip() or "VALIDATE_REQUEST"
    code_stage = "".join(
        character if character.isalnum() else "_"
        for character in normalized_stage.upper()
    ).strip("_")
    return ExportIssue(
        severity=IssueSeverity.ERROR,
        stage=normalized_stage,
        code=f"A1_READINESS_{code_stage or 'VALIDATE_REQUEST'}_FAILED",
        message=str(exc).strip() or type(exc).__name__,
        object_id=object_id,
        technical_details=repr(exc),
    )


def _requested_object_ids(context: Any) -> Tuple[str, ...]:
    names = tuple(
        str(getattr(obj, "name_full", None) or getattr(obj, "name", "")).strip()
        for obj in _request_mesh_objects(context)
    )
    return tuple(dict.fromkeys(value for value in names if value))


def _failure_report(
    *,
    signature: str,
    requested_object_ids: Tuple[str, ...],
    error: ExportIssue,
    warnings: Tuple[ExportIssue, ...] = (),
    statistics: Mapping[str, object] | None = None,
) -> A1ExportReadinessReport:
    recognized = set(requested_object_ids)
    all_issues = warnings + (error,)
    per_object = tuple(
        A1ObjectReadiness(
            object_id=object_id,
            issues=tuple(
                issue for issue in all_issues if issue.object_id == object_id
            ),
            statistics={},
        )
        for object_id in requested_object_ids
    )
    global_issues = tuple(
        issue
        for issue in all_issues
        if issue.object_id is None or issue.object_id not in recognized
    )
    return A1ExportReadinessReport(
        signature=signature,
        objects=per_object,
        issues=global_issues,
        statistics=_readiness_statistics(statistics or {}),
    )


def _composition_statistics(document: Any) -> dict[str, ReadinessStatistic]:
    skins = tuple(getattr(document, "skins", ()))
    return {
        "composed_bone_count": len(tuple(getattr(document, "bones", ()))),
        "composed_slot_count": len(tuple(getattr(document, "slots", ()))),
        "composed_attachment_count": sum(
            len(attachments)
            for skin in skins
            for attachments in getattr(skin, "attachments", {}).values()
        ),
    }


def _analyse_multi_plan(
    plan: A1UiMultiExportPlan,
    *,
    context: Any,
    scene: Any,
) -> tuple[Tuple[PreparedA1Object, ...], Mapping[str, object]]:
    if plan.settings.mode is A1MultiObjectMode.MIXED:
        prepared = prepare_a1_mixed_object(
            plan.connected_sources,
            plan.standalone_sources,
            plan.settings,
            context=context,
            scene=scene,
        )
        partition = partition_mixed_prepared_objects(
            prepared.objects,
            plan.connected_sources,
            plan.standalone_sources,
        )
        composition = compose_a1_mixed_document(
            plan.connected_sources,
            plan.standalone_sources,
            partition,
            plan.settings,
        )
        document = composition.document
    else:
        prepared = prepare_a1_multi_object(
            plan.all_sources,
            plan.settings,
            context=context,
            scene=scene,
        )
        composition = compose_a1_multi_object_document(
            plan.all_sources,
            prepared.objects,
            plan.settings,
        )
        document = composition.document
    statistics = _readiness_statistics(prepared.statistics)
    statistics.update(_composition_statistics(document))
    statistics["output_write_probe"] = "NOT_RUN"
    return prepared.objects, statistics


def _fallback_signature(context: Any, exc: Exception) -> str:
    payload = {
        "scene": id(getattr(context, "scene", None)),
        "objects": _requested_object_ids(context),
        "signature_error": repr(exc),
    }
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return sha256(serialized.encode("utf-8")).hexdigest()


def analyse_a1_export_readiness(context: Any) -> A1ExportReadinessReport:
    """Run the production preparation/composition pipeline without staging files."""

    requested_ids = _requested_object_ids(context)
    try:
        signature = build_a1_readiness_signature(context)
    except Exception as exc:
        logger.exception("Unable to build A1 readiness signature")
        return _failure_report(
            signature=_fallback_signature(context, exc),
            requested_object_ids=requested_ids,
            error=_error_issue(
                stage="VALIDATE_REQUEST",
                exc=exc,
                object_id=None,
            ),
        )

    scene = getattr(context, "scene", None)
    try:
        selected_meshes = _selected_meshes(context)
        if len(selected_meshes) > 1:
            plan = build_selected_ui_export_plan(context)
            prepared_objects, statistics = _analyse_multi_plan(
                plan,
                context=context,
                scene=scene,
            )
            return A1ExportReadinessReport(
                signature=signature,
                objects=tuple(
                    _prepared_object_readiness(prepared)
                    for prepared in prepared_objects
                ),
                issues=plan.issues,
                statistics=statistics,
            )

        plan = build_active_ui_export_plan(context)
        prepared = prepare_a1_object(
            plan.source_object,
            plan.settings,
            context=context,
            scene=scene,
        )
        return A1ExportReadinessReport(
            signature=signature,
            objects=(_prepared_object_readiness(prepared),),
            statistics={
                "object_count": 1,
                "mode": "SINGLE",
                "output_write_probe": "NOT_RUN",
                **_composition_statistics(prepared.document),
            },
        )
    except A1ObjectPreparationError as exc:
        return _failure_report(
            signature=signature,
            requested_object_ids=requested_ids,
            error=_error_issue(
                stage=exc.stage.value,
                exc=exc.cause,
                object_id=exc.object_id,
            ),
            warnings=exc.warnings,
            statistics=exc.statistics,
        )
    except A1MultiObjectPreparationError as exc:
        return _failure_report(
            signature=signature,
            requested_object_ids=requested_ids,
            error=_error_issue(
                stage=exc.stage.value,
                exc=exc.cause,
                object_id=exc.object_id,
            ),
            warnings=exc.warnings,
            statistics=exc.statistics,
        )
    except Exception as exc:
        logger.exception("A1 export readiness analysis failed")
        return _failure_report(
            signature=signature,
            requested_object_ids=requested_ids,
            error=_error_issue(
                stage="VALIDATE_REQUEST",
                exc=exc,
                object_id=None,
            ),
        )


def store_a1_export_readiness(context: Any, report: A1ExportReadinessReport) -> None:
    if context is None:
        raise ValueError("context cannot be None")
    if not isinstance(report, A1ExportReadinessReport):
        raise TypeError("report must be A1ExportReadinessReport")
    scene = getattr(context, "scene", None)
    key = _scene_key(scene)
    _READINESS_CACHE[key] = _ReadinessCacheEntry(
        signature=report.signature,
        report=report,
    )


def clear_a1_export_readiness(scene: Any | None = None) -> None:
    if scene is None:
        _READINESS_CACHE.clear()
        return
    _READINESS_CACHE.pop(_scene_key(scene), None)


def current_a1_export_readiness(
    context: Any,
) -> tuple[A1ReadinessState, A1ExportReadinessReport | None]:
    if context is None:
        return A1ReadinessState.NOT_ANALYSED, None
    scene = getattr(context, "scene", None)
    if scene is None:
        return A1ReadinessState.NOT_ANALYSED, None
    entry = _READINESS_CACHE.get(_scene_key(scene))
    if entry is None:
        return A1ReadinessState.NOT_ANALYSED, None
    if entry.stale:
        return A1ReadinessState.STALE, entry.report
    try:
        signature = build_a1_readiness_signature(context)
    except Exception:
        logger.debug("Unable to refresh A1 readiness signature", exc_info=True)
        return A1ReadinessState.STALE, entry.report
    if signature != entry.signature:
        return A1ReadinessState.STALE, entry.report
    return entry.report.state, entry.report


def require_current_a1_export_readiness(context: Any) -> tuple[bool, str]:
    state, report = current_a1_export_readiness(context)
    if state is A1ReadinessState.NOT_ANALYSED:
        return False, "Run Analyze before export"
    if state is A1ReadinessState.STALE:
        return False, "Export analysis is outdated; run Analyze again"
    if report is None:
        return False, "Export analysis cache is unavailable"
    if state is A1ReadinessState.BLOCKED:
        return False, f"Export is blocked by {report.blocker_count} issue(s)"
    return True, ""


@_persistent
def a1_readiness_depsgraph_update_post(_scene: Any, depsgraph: Any) -> None:
    """Mark cached reports stale when export-relevant Blender IDs change."""

    if not _READINESS_CACHE:
        return
    try:
        updates = tuple(getattr(depsgraph, "updates", ()))
    except Exception:
        updates = ()
    relevant = not updates
    for update in updates:
        updated_id = getattr(update, "id", None)
        id_type = str(getattr(updated_id, "id_type", "")).upper()
        if id_type in _RELEVANT_ID_TYPES:
            relevant = True
            break
    if relevant:
        for entry in _READINESS_CACHE.values():
            entry.stale = True


__all__ = [
    "a1_readiness_depsgraph_update_post",
    "analyse_a1_export_readiness",
    "build_a1_readiness_signature",
    "clear_a1_export_readiness",
    "current_a1_export_readiness",
    "require_current_a1_export_readiness",
    "store_a1_export_readiness",
]
