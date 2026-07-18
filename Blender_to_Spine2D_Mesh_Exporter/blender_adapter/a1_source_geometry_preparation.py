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
    A1ZGroupAssignmentPlan,
    ExportIssue,
    build_a1_z_group_assignment,
    prepare_a1_geometry_regions,
    resolve_a1_names,
    resolve_a1_output_paths,
)
from ..domain.geometry import LineageSeverity, MeshSnapshot
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


def _read_source_snapshot(
    source_obj: Any,
    object_id: str,
    settings: A1SingleObjectExportSettings,
    *,
    scene: Any | None,
) -> tuple[MeshSnapshot, int, Tuple[ExportIssue, ...]]:
    stage = A1SingleObjectStage.READ_GEOMETRY
    if settings.source_geometry_mode is A1SourceGeometryMode.EVALUATED:
        evaluated = read_evaluated_mesh_snapshot(
            source_obj,
            scene=scene,
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
        statistics = freeze_statistics(
            statistics,
            {
                "modifier_count": modifier_count,
                "source_vertices": len(source_snapshot.vertices),
                "source_edges": len(source_snapshot.edges),
                "source_faces": len(source_snapshot.faces),
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
            "Prepared source geometry for %s: vertices=%d faces=%d regions=%d",
            object_id,
            len(source_snapshot.vertices),
            len(source_snapshot.faces),
            len(geometry.regions),
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
