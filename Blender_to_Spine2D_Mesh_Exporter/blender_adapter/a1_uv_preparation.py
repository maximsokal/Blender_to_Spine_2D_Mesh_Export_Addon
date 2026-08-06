"""Build texturing topology and UV state for one prepared A1 source mesh."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Mapping, Tuple

from ..application import (
    A1SingleObjectStage,
    A1TexturingTopology,
    A1UvPropagationResult,
    ExportIssue,
    build_a1_texturing_topology,
    propagate_texturing_uv_to_regions,
)
from ..domain.baking import A1TextureExportMode
from ..domain.geometry import (
    MeshSnapshot,
    UvTransferReport,
    transfer_uv_by_source_loop,
)
from ..domain.projection import A1ProjectionDirection
from ..domain.uv import (
    UvRangePolicy,
    UvUnwrapResult,
    calculate_uv_statistics,
    inspect_uv_range,
)
from .a1_preparation_contracts import (
    A1ObjectPreparationError,
    StatisticsValue,
    freeze_statistics,
    warning_issue,
)
from .a1_source_geometry_preparation import A1SourceGeometryPreparationResult
from .scene_context_contract import require_context_scene_consistency
from .uv_unwrap import unwrap_snapshot_uv


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class A1UvPreparationResult:
    """UV products plus separate export and source-material geometry.

    ``unwrap_result.snapshot`` owns the generated UV layout on projected export
    geometry. ``material_bake_snapshot`` owns the same generated destination UV on the
    original Blender-local geometry. Object baking uses the latter so selecting +X, -Y,
    +Z, or Active Camera cannot change source-material normals or positions.
    """

    source: A1SourceGeometryPreparationResult
    texturing_topology: A1TexturingTopology
    unwrap_result: UvUnwrapResult
    uv_regions: A1UvPropagationResult
    warnings: Tuple[ExportIssue, ...]
    statistics: Mapping[str, StatisticsValue]
    # Appended to preserve positional compatibility with existing test doubles.
    material_bake_snapshot: MeshSnapshot | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, A1SourceGeometryPreparationResult):
            raise TypeError("source must be A1SourceGeometryPreparationResult")
        expected = (
            ("texturing_topology", A1TexturingTopology),
            ("unwrap_result", UvUnwrapResult),
            ("uv_regions", A1UvPropagationResult),
        )
        for field_name, expected_type in expected:
            if not isinstance(getattr(self, field_name), expected_type):
                raise TypeError(f"{field_name} must be {expected_type.__name__}")

        material_bake_snapshot = self.material_bake_snapshot
        if material_bake_snapshot is None:
            # Compatibility fallback for unit-test doubles created before the separate
            # material-evaluation geometry contract.
            material_bake_snapshot = self.unwrap_result.snapshot
            object.__setattr__(
                self,
                "material_bake_snapshot",
                material_bake_snapshot,
            )
        if not isinstance(material_bake_snapshot, MeshSnapshot):
            raise TypeError("material_bake_snapshot must be MeshSnapshot or None")
        if material_bake_snapshot.source_object_id != self.source.object_id:
            raise ValueError(
                "material_bake_snapshot.source_object_id must match source.object_id"
            )
        layer_name = self.source.settings.uv.layer_name
        if layer_name not in material_bake_snapshot.uv_layer_names:
            raise ValueError(
                f"material_bake_snapshot is missing generated UV layer {layer_name!r}"
            )
        if material_bake_snapshot.active_uv_layer != layer_name:
            raise ValueError(
                "material_bake_snapshot.active_uv_layer must be the generated "
                f"destination UV layer {layer_name!r}"
            )

        if not isinstance(self.warnings, tuple) or not all(
            isinstance(issue, ExportIssue) for issue in self.warnings
        ):
            raise TypeError("warnings must be a tuple of ExportIssue values")
        if not isinstance(self.statistics, Mapping):
            raise TypeError("statistics must be a mapping")


def _depth_camera_uv_result(
    source: A1SourceGeometryPreparationResult,
    texturing_topology: A1TexturingTopology,
) -> UvUnwrapResult:
    """Validate the camera UV already authored by the depth geometry source."""

    layer_name = source.settings.uv.layer_name
    snapshot = texturing_topology.snapshot
    if snapshot.active_uv_layer != layer_name:
        raise ValueError(
            "Depth Camera Projection topology lost its active camera UV layer; "
            f"expected={layer_name!r}, actual={snapshot.active_uv_layer!r}"
        )
    if snapshot.render_uv_layer != layer_name:
        raise ValueError(
            "Depth Camera Projection topology lost its render camera UV layer; "
            f"expected={layer_name!r}, actual={snapshot.render_uv_layer!r}"
        )
    statistics = calculate_uv_statistics(snapshot, layer_name)
    return UvUnwrapResult(
        snapshot=snapshot,
        settings=source.settings.uv,
        statistics=statistics,
    )


def transfer_normal_uv_to_material_bake_snapshot(
    projected_uv_snapshot: MeshSnapshot,
    material_snapshot: MeshSnapshot,
    *,
    layer_name: str,
) -> tuple[MeshSnapshot, UvTransferReport]:
    """Copy only generated destination UVs to source-material geometry.

    The two snapshots must share exact ``SourceLoopId`` lineage. Export projection may
    change positions, normals and ``matrix_world`` on ``projected_uv_snapshot``; none of
    those values are allowed to leak into the returned material snapshot.
    """

    if not isinstance(projected_uv_snapshot, MeshSnapshot):
        raise TypeError("projected_uv_snapshot must be MeshSnapshot")
    if not isinstance(material_snapshot, MeshSnapshot):
        raise TypeError("material_snapshot must be MeshSnapshot")
    if not isinstance(layer_name, str) or not layer_name.strip():
        raise ValueError("layer_name must be a non-empty string")
    resolved_layer_name = layer_name.strip()

    updated, report = transfer_uv_by_source_loop(
        projected_uv_snapshot,
        material_snapshot,
        source_layer_name=resolved_layer_name,
        target_layer_name=resolved_layer_name,
        require_complete=True,
        duplicate_tolerance=0.0,
    )
    if report.updated_loop_count != len(material_snapshot.loops):
        raise ValueError(
            "Generated material-bake UV transfer did not update every target loop; "
            f"updated={report.updated_loop_count}, "
            f"target_loops={len(material_snapshot.loops)}"
        )
    if report.missing_source_loop_ids:
        raise ValueError(
            "Generated material-bake UV transfer contains missing SourceLoopId values: "
            f"{report.missing_source_loop_ids}"
        )
    if report.unused_source_loop_ids:
        raise ValueError(
            "Projected unwrap contains SourceLoopId values absent from material "
            f"geometry: {report.unused_source_loop_ids}"
        )

    # Transfer must never alter source-material evaluation geometry.
    if updated.vertices != material_snapshot.vertices:
        raise ValueError("Material-bake UV transfer changed vertex geometry")
    if updated.edges != material_snapshot.edges:
        raise ValueError("Material-bake UV transfer changed edge topology")
    if updated.faces != material_snapshot.faces:
        raise ValueError("Material-bake UV transfer changed face topology")
    if updated.world_matrix != material_snapshot.world_matrix:
        raise ValueError("Material-bake UV transfer changed matrix_world")
    if updated.render_uv_layer != material_snapshot.render_uv_layer:
        raise ValueError("Material-bake UV transfer changed source render UV role")

    return updated, report


def _normal_material_bake_snapshot(
    source: A1SourceGeometryPreparationResult,
    unwrap_result: UvUnwrapResult,
) -> tuple[MeshSnapshot, int]:
    """Transfer generated output UVs onto unprojected material-evaluation geometry."""

    if not isinstance(source, A1SourceGeometryPreparationResult):
        raise TypeError("source must be A1SourceGeometryPreparationResult")
    if not isinstance(unwrap_result, UvUnwrapResult):
        raise TypeError("unwrap_result must be UvUnwrapResult")

    target = source.material_bake_snapshot
    if not isinstance(target, MeshSnapshot):
        raise TypeError(
            "source.material_bake_snapshot must be MeshSnapshot after validation"
        )
    updated, report = transfer_normal_uv_to_material_bake_snapshot(
        unwrap_result.snapshot,
        target,
        layer_name=source.settings.uv.layer_name,
    )
    return updated, report.updated_loop_count


def _resolve_material_bake_snapshot(
    source: A1SourceGeometryPreparationResult,
    unwrap_result: UvUnwrapResult,
) -> tuple[MeshSnapshot, int, bool]:
    """Return the correct texture execution mesh for the selected public mode."""

    mode = source.settings.bake_execution.texture_export_mode
    if mode is A1TextureExportMode.NORMAL_UV_SEGMENTS:
        snapshot, transfer_count = _normal_material_bake_snapshot(
            source,
            unwrap_result,
        )
        return snapshot, transfer_count, True

    # Rendered Camera Projection modes do not execute object UV baking. Retain the
    # already prepared layout as a validated compatibility target for dispatch.
    return unwrap_result.snapshot, 0, False


def prepare_a1_uv(
    source: A1SourceGeometryPreparationResult,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> A1UvPreparationResult:
    """Build seam topology, resolve UV layout, and prepare texture geometry.

    Normal / UV Segments unwraps projected export geometry, propagates that layout to
    attachments, then transfers the exact generated destination UV back to unprojected
    Blender-local geometry for material evaluation. Depth Camera Projection owns direct
    full-frame camera UV and does not invoke another unwrap operation.
    """

    if not isinstance(source, A1SourceGeometryPreparationResult):
        raise TypeError("source must be A1SourceGeometryPreparationResult")
    stage = A1SingleObjectStage.BUILD_TEXTURING_TOPOLOGY
    warnings = source.warnings
    depth_camera_projection = (
        source.settings.bake_execution.texture_export_mode
        is A1TextureExportMode.DEPTH_CAMERA_PROJECTION
    )
    statistics = freeze_statistics(
        source.statistics,
        {
            "attachment_invert_y": 1,
            "camera_attachment_y_compensated": int(
                source.settings.projection_direction
                is A1ProjectionDirection.ACTIVE_CAMERA
            ),
            "depth_camera_direct_uv": int(depth_camera_projection),
        },
    )
    try:
        require_context_scene_consistency(context, scene)

        texturing_topology = build_a1_texturing_topology(
            source.source_snapshot,
            source.geometry,
        )
        statistics = freeze_statistics(
            statistics,
            {"texturing_seam_count": len(texturing_topology.all_seam_edge_ids)},
        )

        stage = A1SingleObjectStage.UNWRAP_TEXTURE_UV
        unwrap_result = (
            _depth_camera_uv_result(source, texturing_topology)
            if depth_camera_projection
            else unwrap_snapshot_uv(
                texturing_topology.snapshot,
                source.settings.uv,
                context=context,
                scene=scene,
            )
        )
        raw_outside_count = unwrap_result.statistics.outside_unit_square_count
        range_report = inspect_uv_range(
            unwrap_result.snapshot,
            source.settings.uv.layer_name,
            epsilon=source.settings.uv.range_epsilon,
        )
        material_bake_snapshot, material_uv_transfer_count, projection_independent = (
            _resolve_material_bake_snapshot(source, unwrap_result)
        )
        statistics = freeze_statistics(
            statistics,
            {
                "uv_loop_count": unwrap_result.statistics.loop_count,
                "uv_outside_unit_square": raw_outside_count,
                "uv_outside_range_tolerance": range_report.outside_loop_count,
                "uv_range_policy": source.settings.uv.range_policy.value,
                "uv_range_epsilon": source.settings.uv.range_epsilon,
                "material_bake_uv_transfer_count": material_uv_transfer_count,
                "material_bake_projection_independent": int(
                    projection_independent
                ),
                "material_bake_render_uv_layer": (
                    material_bake_snapshot.render_uv_layer or ""
                ),
            },
        )
        if (
            range_report.violations
            and source.settings.uv.range_policy is UvRangePolicy.WARN_ONLY
        ):
            warnings = warnings + (
                warning_issue(
                    stage=stage,
                    code="UV_OUTSIDE_UNIT_SQUARE",
                    message=(
                        f"{range_report.outside_loop_count} UV loops are outside "
                        "the unit square beyond epsilon "
                        f"{range_report.epsilon}; export may continue because "
                        "uv.range_policy is WARN_ONLY"
                    ),
                    object_id=source.object_id,
                    context={
                        "layer_name": range_report.layer_name,
                        "epsilon": range_report.epsilon,
                        "outside_loop_count": range_report.outside_loop_count,
                    },
                ),
            )

        stage = A1SingleObjectStage.PROPAGATE_REGION_UV
        uv_regions = propagate_texturing_uv_to_regions(
            unwrap_result.snapshot,
            source.geometry,
            source_layer_name=source.settings.uv.layer_name,
            target_layer_name=source.settings.uv.layer_name,
        )
        logger.debug(
            "Prepared UVs for %s: mode=%s loops=%d regions=%d raw_outside=%d "
            "outside_tolerance=%d policy=%s epsilon=%s "
            "material_projection_independent=%s material_uv_loops=%d",
            source.object_id,
            source.settings.bake_execution.texture_export_mode.value,
            unwrap_result.statistics.loop_count,
            len(uv_regions.snapshots),
            raw_outside_count,
            range_report.outside_loop_count,
            source.settings.uv.range_policy.value,
            source.settings.uv.range_epsilon,
            projection_independent,
            material_uv_transfer_count,
        )
        return A1UvPreparationResult(
            source=source,
            texturing_topology=texturing_topology,
            unwrap_result=unwrap_result,
            uv_regions=uv_regions,
            warnings=warnings,
            statistics=statistics,
            material_bake_snapshot=material_bake_snapshot,
        )
    except A1ObjectPreparationError:
        raise
    except Exception as exc:
        raise A1ObjectPreparationError(
            stage=stage,
            object_id=source.object_id,
            cause=exc,
            statistics=statistics,
            warnings=warnings,
        ) from exc


__all__ = [
    "A1UvPreparationResult",
    "prepare_a1_uv",
    "transfer_normal_uv_to_material_bake_snapshot",
]
