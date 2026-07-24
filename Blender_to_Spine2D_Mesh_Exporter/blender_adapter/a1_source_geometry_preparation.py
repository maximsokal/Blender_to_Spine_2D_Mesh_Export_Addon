"""Read and prepare source geometry for one A1 object export."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Mapping, Tuple

from ..application import (
    A1GeometryPreparationResult,
    A1ResolvedOutputPaths,
    A1SingleObjectExportSettings,
    A1SingleObjectStage,
    A1SourceGeometryMode,
    A1SourceUvBoundaryMode,
    A1ZGroupAssignmentPlan,
    ExportIssue,
    build_a1_z_group_assignment,
    prepare_a1_geometry_regions,
    resolve_a1_names,
    resolve_a1_output_paths,
)
from ..domain.geometry import (
    LineageSeverity,
    MeshSnapshot,
    normalize_mesh_snapshot_world_transform,
)
from .a1_preparation_contracts import (
    A1ObjectPreparationError,
    StatisticsValue,
    freeze_statistics,
    warning_issue,
)
from .evaluated_mesh_reader import read_evaluated_mesh_snapshot
from .mesh_reader import read_source_mesh_snapshot
from .render_engine_contract import (
    RenderEngineContract,
    render_engine_contract_from_execution,
)
from .scene_context_contract import (
    BlenderSceneContextError,
    require_depsgraph_scene_consistency,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class A1SourceGeometryPreparationResult:
    """Validated source metadata and geometry products used by later stages."""

    source_object: Any
    object_id: str
    prefix: str
    settings: A1SingleObjectExportSettings
    output_paths: A1ResolvedOutputPaths
    renderer: RenderEngineContract
    source_snapshot: MeshSnapshot
    z_groups: A1ZGroupAssignmentPlan
    geometry: A1GeometryPreparationResult
    warnings: Tuple[ExportIssue, ...]
    statistics: Mapping[str, StatisticsValue]

    def __post_init__(self) -> None:
        if self.source_object is None:
            raise ValueError("source_object cannot be None")
        for field_name in ("object_id", "prefix"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        expected = (
            ("settings", A1SingleObjectExportSettings),
            ("output_paths", A1ResolvedOutputPaths),
            ("renderer", RenderEngineContract),
            ("source_snapshot", MeshSnapshot),
            ("z_groups", A1ZGroupAssignmentPlan),
            ("geometry", A1GeometryPreparationResult),
        )
        for field_name, expected_type in expected:
            if not isinstance(getattr(self, field_name), expected_type):
                raise TypeError(f"{field_name} must be {expected_type.__name__}")
        if self.source_snapshot.source_object_id != self.object_id:
            raise ValueError("source_snapshot.source_object_id must match object_id")
        if not isinstance(self.warnings, tuple) or not all(
            isinstance(issue, ExportIssue) for issue in self.warnings
        ):
            raise TypeError("warnings must be a tuple of ExportIssue values")
        if not isinstance(self.statistics, Mapping):
            raise TypeError("statistics must be a mapping")


def object_name(source_obj: Any) -> str:
    """Resolve and validate the stable Blender object name used by A1 contracts."""

    if source_obj is None or getattr(source_obj, "type", None) != "MESH":
        raise ValueError("source_obj must be a Blender MESH object")
    value = str(
        getattr(source_obj, "name_full", None)
        or getattr(source_obj, "name", None)
        or ""
    ).strip()
    if not value:
        raise ValueError("source_obj name is empty")
    if getattr(source_obj, "data", None) is None:
        raise ValueError("source_obj.data is missing")
    return value


def _resolved_evaluation_owners(
    scene: Any | None,
) -> tuple[Any, Any]:
    """Return one Blender Scene and its current evaluated dependency graph."""

    try:
        import bpy
    except Exception as exc:
        raise ValueError(
            "Blender bpy module is unavailable for evaluated geometry"
        ) from exc

    resolved_scene = scene or getattr(bpy.context, "scene", None)
    if resolved_scene is None:
        raise ValueError("A Blender Scene is required for evaluated geometry")
    try:
        resolved_depsgraph = bpy.context.evaluated_depsgraph_get()
    except Exception as exc:
        raise ValueError(
            "Unable to acquire Blender evaluated dependency graph"
        ) from exc
    if resolved_depsgraph is None:
        raise ValueError("Blender returned no evaluated dependency graph")
    try:
        require_depsgraph_scene_consistency(
            resolved_depsgraph,
            resolved_scene,
        )
    except BlenderSceneContextError as exc:
        raise ValueError(
            f"Evaluated geometry scene and dependency graph disagree: {exc}"
        ) from exc
    return resolved_scene, resolved_depsgraph


def _read_source_snapshot(
    source_obj: Any,
    object_id: str,
    settings: A1SingleObjectExportSettings,
    *,
    scene: Any | None,
) -> tuple[MeshSnapshot, int, Tuple[ExportIssue, ...]]:
    stage = A1SingleObjectStage.READ_GEOMETRY
    if settings.source_geometry_mode is A1SourceGeometryMode.EVALUATED:
        resolved_scene, resolved_depsgraph = _resolved_evaluation_owners(scene)
        evaluated = read_evaluated_mesh_snapshot(
            source_obj,
            scene=resolved_scene,
            depsgraph=resolved_depsgraph,
            source_object_id=object_id,
            snapshot_id=f"{object_id}:a1-evaluated",
            lineage_policy=settings.modifier_lineage_policy,
        )
        warnings = tuple(
            warning_issue(
                stage=stage,
                code=f"MODIFIER_{issue.code}",
                message=issue.message,
                object_id=object_id,
                context={"channel": issue.channel},
            )
            for issue in evaluated.lineage_report.issues
            if issue.severity is LineageSeverity.WARNING
        )
        return evaluated.snapshot, len(evaluated.modifier_stack), warnings
    snapshot = read_source_mesh_snapshot(
        source_obj,
        source_object_id=object_id,
        snapshot_id=f"{object_id}:a1-source",
    )
    return snapshot, 0, ()


def _resolve_source_uv_boundary_layer(
    snapshot: MeshSnapshot,
    settings: A1SingleObjectExportSettings,
) -> str | None:
    """Resolve the exact pre-unwrap UV layer allowed to affect segmentation."""

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    if not isinstance(settings, A1SingleObjectExportSettings):
        raise TypeError("settings must be A1SingleObjectExportSettings")

    mode = settings.source_uv_boundary_mode
    if mode is A1SourceUvBoundaryMode.DISABLED:
        return None
    if mode is A1SourceUvBoundaryMode.EXPLICIT_LAYER:
        layer_name = settings.source_uv_boundary_layer_name
        if layer_name is None:
            raise ValueError(
                "EXPLICIT_LAYER requires source_uv_boundary_layer_name"
            )
        if layer_name not in snapshot.uv_layer_names:
            raise ValueError(
                f"Source UV boundary layer '{layer_name}' is absent from "
                f"snapshot '{snapshot.snapshot_id}'"
            )
        return layer_name
    if mode is A1SourceUvBoundaryMode.ACTIVE_LAYER_LEGACY:
        layer_name = snapshot.active_uv_layer
        if layer_name is None:
            raise ValueError(
                "ACTIVE_LAYER_LEGACY requires the source mesh to have an active UV layer"
            )
        return layer_name
    raise TypeError(f"Unsupported source UV boundary mode: {mode!r}")


def prepare_a1_source_geometry(
    source_obj: Any,
    settings: A1SingleObjectExportSettings,
    *,
    scene: Any | None = None,
) -> A1SourceGeometryPreparationResult:
    """Validate one request and prepare immutable source geometry state."""

    stage = A1SingleObjectStage.VALIDATE_REQUEST
    object_id: str | None = None
    warnings: Tuple[ExportIssue, ...] = ()
    statistics: Mapping[str, StatisticsValue] = {}
    try:
        if not isinstance(settings, A1SingleObjectExportSettings):
            raise TypeError("settings must be A1SingleObjectExportSettings")
        object_id = object_name(source_obj)
        prefix, _ = resolve_a1_names(object_id, settings)
        output_paths = resolve_a1_output_paths(object_id, settings)
        renderer = render_engine_contract_from_execution(settings.bake_execution)
        statistics = freeze_statistics(
            {
                "source_object": object_id,
                "rig_prefix": prefix,
                "source_geometry_mode": settings.source_geometry_mode.value,
                "source_uv_boundary_mode": settings.source_uv_boundary_mode.value,
                "source_uv_boundary_configured_layer": (
                    settings.source_uv_boundary_layer_name or ""
                ),
                "include_control_icons": int(settings.include_control_icons),
                "include_preview_animation": int(settings.include_preview_animation),
                "render_engine": renderer.blender_engine,
                "shader_render_target": renderer.shader_target,
            }
        )

        stage = A1SingleObjectStage.READ_GEOMETRY
        source_snapshot, modifier_count, warnings = _read_source_snapshot(
            source_obj,
            object_id,
            settings,
            scene=scene,
        )

        stage = A1SingleObjectStage.PREPARE_GEOMETRY
        world_transform = normalize_mesh_snapshot_world_transform(source_snapshot)
        source_snapshot = world_transform.snapshot
        if world_transform.mirrored:
            warnings = warnings + (
                warning_issue(
                    stage=stage,
                    code="MIRRORED_OBJECT_TRANSFORM",
                    message=(
                        "Object matrix_world has a negative determinant. Rewrite "
                        "preserved the mirrored geometry and oriented normals while "
                        "normalizing rotation/scale into the mesh snapshot."
                    ),
                    object_id=object_id,
                    context={
                        "determinant": world_transform.determinant,
                    },
                ),
            )

        resolved_source_uv_boundary_layer = _resolve_source_uv_boundary_layer(
            source_snapshot,
            settings,
        )
        statistics = freeze_statistics(
            statistics,
            {
                "modifier_count": modifier_count,
                "source_vertices": len(source_snapshot.vertices),
                "source_edges": len(source_snapshot.edges),
                "source_faces": len(source_snapshot.faces),
                "object_linear_transform_baked": int(world_transform.changed),
                "object_world_determinant": world_transform.determinant,
                "object_world_mirrored": int(world_transform.mirrored),
                "object_world_translation_x": world_transform.translation[0],
                "object_world_translation_y": world_transform.translation[1],
                "object_world_translation_z": world_transform.translation[2],
                "source_uv_boundary_resolved_layer": (
                    resolved_source_uv_boundary_layer or ""
                ),
            },
        )

        stage = A1SingleObjectStage.ASSIGN_Z_GROUPS
        z_groups = build_a1_z_group_assignment(source_snapshot)
        statistics = freeze_statistics(
            statistics,
            {"z_group_count": len(z_groups.groups)},
        )

        stage = A1SingleObjectStage.PREPARE_GEOMETRY
        geometry = prepare_a1_geometry_regions(
            source_snapshot,
            settings.resolved_geometry_settings(),
        )
        statistics = freeze_statistics(
            statistics,
            {
                "segment_count": len(geometry.segmentation.segments),
                "region_count": len(geometry.regions),
                "decomposition_cut_count": len(geometry.decomposition.cuts),
            },
        )
        logger.debug(
            "Prepared source geometry for %s: vertices=%d faces=%d regions=%d "
            "world_transform_baked=%s determinant=%s mirrored=%s "
            "source_uv_boundary_mode=%s source_uv_boundary_layer=%s",
            object_id,
            len(source_snapshot.vertices),
            len(source_snapshot.faces),
            len(geometry.regions),
            world_transform.changed,
            world_transform.determinant,
            world_transform.mirrored,
            settings.source_uv_boundary_mode.value,
            resolved_source_uv_boundary_layer,
        )
        return A1SourceGeometryPreparationResult(
            source_object=source_obj,
            object_id=object_id,
            prefix=prefix,
            settings=settings,
            output_paths=output_paths,
            renderer=renderer,
            source_snapshot=source_snapshot,
            z_groups=z_groups,
            geometry=geometry,
            warnings=warnings,
            statistics=statistics,
        )
    except A1ObjectPreparationError:
        raise
    except Exception as exc:
        raise A1ObjectPreparationError(
            stage=stage,
            object_id=object_id,
            cause=exc,
            statistics=statistics,
            warnings=warnings,
        ) from exc


__all__ = [
    "A1SourceGeometryPreparationResult",
    "object_name",
    "prepare_a1_source_geometry",
]
