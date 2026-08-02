"""Typed contracts shared by every staged A1 object-preparation service."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Tuple

from ..application import (
    A1DocumentAssemblyResult,
    A1GeometryPreparationResult,
    A1ResolvedOutputPaths,
    A1SingleObjectExportSettings,
    A1SingleObjectStage,
    A1TexturingTopology,
    A1UvPropagationResult,
    A1ZGroupAssignmentPlan,
    ExportIssue,
    IssueSeverity,
)
from ..domain.baking import BakePlan, ObjectMaterialAnalysis, TextureSequenceTiming
from ..domain.baking.generated_materials import GeneratedBakePlan
from ..domain.geometry import MeshSnapshot
from ..domain.spine import LegacyRigBuildResult, SpineDocument
from ..domain.uv import UvUnwrapResult


StatisticsValue = int | float | str


def freeze_statistics(
    *values: Mapping[str, StatisticsValue],
) -> Mapping[str, StatisticsValue]:
    """Merge statistics into one immutable mapping, with later stages taking precedence."""

    merged: dict[str, StatisticsValue] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise TypeError("statistics values must be mappings")
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"statistics keys must be non-empty strings")
            if isinstance(item, bool) or not isinstance(item, (int, float, str)):
                raise TypeError(
                    "statistics values must be int, float, or str; "
                    f"got {type(item).__name__} for {key!r}"
                )
            merged[key] = item
    return MappingProxyType(merged)


def build_skeleton_metadata(
    settings: A1SingleObjectExportSettings,
) -> dict[str, object]:
    """Build shared Spine skeleton metadata before and after render finalization."""

    if not isinstance(settings, A1SingleObjectExportSettings):
        raise TypeError("settings must be A1SingleObjectExportSettings")
    metadata: dict[str, object] = {
        "hash": "hash_value_placeholder",
        "spine": settings.export.spine_version,
        "x": 0,
        "y": 0,
        "width": settings.export.texture_width,
        "height": settings.export.texture_height,
        "images": "",
        "audio": "./audio",
    }
    sequence_frame_count = getattr(settings.export, "sequence_frame_count", 0)
    if sequence_frame_count > 0:
        sequence_timing = getattr(
            settings.export,
            "sequence_timing",
            TextureSequenceTiming(),
        )
        if not isinstance(sequence_timing, TextureSequenceTiming):
            raise TypeError("settings.export.sequence_timing must be TextureSequenceTiming")
        metadata["fps"] = round(sequence_timing.resolved_fps, 6)
    return metadata


class A1ObjectPreparationError(RuntimeError):
    """Wrap one failed preparation stage without hiding the original exception."""

    def __init__(
        self,
        *,
        stage: A1SingleObjectStage,
        object_id: str | None,
        cause: Exception,
        statistics: Mapping[str, StatisticsValue],
        warnings: Tuple[ExportIssue, ...],
    ) -> None:
        if not isinstance(stage, A1SingleObjectStage):
            raise TypeError("stage must be A1SingleObjectStage")
        if object_id is not None and (
            not isinstance(object_id, str) or not object_id.strip()
        ):
            raise ValueError("object_id must be a non-empty string or None")
        if not isinstance(cause, Exception):
            raise TypeError("cause must be Exception")
        if not isinstance(statistics, Mapping):
            raise TypeError("statistics must be a mapping")
        if not isinstance(warnings, tuple) or not all(
            isinstance(issue, ExportIssue) for issue in warnings
        ):
            raise TypeError("warnings must be a tuple of ExportIssue values")

        self.stage = stage
        self.object_id = object_id
        self.cause = cause
        self.statistics = freeze_statistics(statistics)
        self.warnings = warnings
        message = str(cause) or type(cause).__name__
        super().__init__(
            f"A1 object preparation failed at {stage.value}"
            + ("" if object_id is None else f" for '{object_id}'")
            + f": {message}"
        )


@dataclass(frozen=True, slots=True)
class A1BlenderFinalizationContext:
    """Immutable Blender runtime references required only after texture staging.

    Camera Projection derives its final pivot after render/crop analysis. Carrying the
    caller-selected context and Scene with the prepared object preserves the historical
    two-argument finalizer API while still using the exact Blender runtime selected by
    the public export entry point. ``None`` values deliberately fall back to
    ``bpy.context`` during real Blender execution.
    """

    context: Any | None = None
    scene: Any | None = None


@dataclass(frozen=True, slots=True)
class PreparedA1Object:
    """Complete immutable in-memory product of one A1 object preparation pipeline."""

    source_object: Any
    object_id: str
    prefix: str
    settings: A1SingleObjectExportSettings
    output_paths: A1ResolvedOutputPaths
    source_snapshot: MeshSnapshot
    z_groups: A1ZGroupAssignmentPlan
    geometry: A1GeometryPreparationResult
    texturing_topology: A1TexturingTopology
    unwrap_result: UvUnwrapResult
    uv_regions: A1UvPropagationResult
    material_analysis: ObjectMaterialAnalysis
    bake_plan: BakePlan
    rig: LegacyRigBuildResult
    document_assembly: A1DocumentAssemblyResult
    warnings: Tuple[ExportIssue, ...]
    statistics: Mapping[str, StatisticsValue]
    finalization_context: A1BlenderFinalizationContext = field(
        default_factory=A1BlenderFinalizationContext
    )

    def __post_init__(self) -> None:
        if self.source_object is None:
            raise ValueError("source_object cannot be None")
        for field_name in ("object_id", "prefix"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        expected_types = (
            ("settings", A1SingleObjectExportSettings),
            ("output_paths", A1ResolvedOutputPaths),
            ("source_snapshot", MeshSnapshot),
            ("z_groups", A1ZGroupAssignmentPlan),
            ("geometry", A1GeometryPreparationResult),
            ("texturing_topology", A1TexturingTopology),
            ("unwrap_result", UvUnwrapResult),
            ("uv_regions", A1UvPropagationResult),
            ("material_analysis", ObjectMaterialAnalysis),
            ("bake_plan", BakePlan),
            ("rig", LegacyRigBuildResult),
            ("document_assembly", A1DocumentAssemblyResult),
            ("finalization_context", A1BlenderFinalizationContext),
        )
        for field_name, expected_type in expected_types:
            if not isinstance(getattr(self, field_name), expected_type):
                raise TypeError(f"{field_name} must be {expected_type.__name__}")
        if not isinstance(self.warnings, tuple) or not all(
            isinstance(issue, ExportIssue) for issue in self.warnings
        ):
            raise TypeError("warnings must be a tuple of ExportIssue values")
        if not isinstance(self.statistics, Mapping):
            raise TypeError("statistics must be a mapping")
        if self.source_snapshot.source_object_id != self.object_id:
            raise ValueError("source_snapshot.source_object_id must match object_id")
        if self.bake_plan.source_object_id != self.object_id:
            raise ValueError("bake_plan.source_object_id must match object_id")
        if self.rig.request.prefix != self.prefix:
            raise ValueError("rig prefix must match prepared prefix")

        assembly_rig = getattr(self.document_assembly, "rig", None)
        if assembly_rig is not None:
            if not isinstance(assembly_rig, LegacyRigBuildResult):
                raise TypeError(
                    "document_assembly.rig must be LegacyRigBuildResult when present"
                )
            if assembly_rig.request.prefix != self.prefix:
                raise ValueError(
                    "document_assembly rig prefix must match prepared prefix"
                )

        object.__setattr__(self, "statistics", freeze_statistics(self.statistics))

    @property
    def document(self) -> SpineDocument:
        return self.document_assembly.document

    @property
    def bake_target_snapshot(self) -> MeshSnapshot:
        if isinstance(self.bake_plan, GeneratedBakePlan):
            return self.bake_plan.generated_material.target_snapshot
        return self.unwrap_result.snapshot

    @property
    def world_position(self) -> Tuple[float, float, float]:
        matrix = self.source_snapshot.world_matrix
        if len(matrix) != 16:
            raise ValueError("source_snapshot.world_matrix must contain 16 values")
        return float(matrix[3]), float(matrix[7]), float(matrix[11])


def warning_issue(
    *,
    stage: A1SingleObjectStage,
    code: str,
    message: str,
    object_id: str,
    context: Mapping[str, object] | None = None,
) -> ExportIssue:
    """Build one normalized preparation warning."""

    if not isinstance(stage, A1SingleObjectStage):
        raise TypeError("stage must be A1SingleObjectStage")
    for field_name, value in (
        ("code", code),
        ("message", message),
        ("object_id", object_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
    if context is not None and not isinstance(context, Mapping):
        raise TypeError("context must be a mapping or None")
    return ExportIssue(
        severity=IssueSeverity.WARNING,
        stage=stage.value,
        code=code.strip(),
        message=message,
        object_id=object_id,
        context={} if context is None else dict(context),
    )


__all__ = [
    "A1BlenderFinalizationContext",
    "A1ObjectPreparationError",
    "PreparedA1Object",
    "StatisticsValue",
    "build_skeleton_metadata",
    "freeze_statistics",
    "warning_issue",
]
