"""Prepare one evaluated visible depth-relief geometry source for A1 export."""

from __future__ import annotations

from dataclasses import replace
import logging
from typing import Any, Mapping, Tuple

from ..application import (
    A1SingleObjectExportSettings,
    A1SingleObjectStage,
    A1SourceGeometryMode,
    ExportIssue,
    build_a1_z_group_assignment,
    prepare_a1_geometry_regions,
)
from ..domain.baking import A1TextureExportMode
from ..domain.geometry import (
    DepthCameraProjectionResult,
    build_depth_camera_projection_surface,
    calculate_a1_projected_snapshot_depth_range,
)
from ..domain.geometry.depth_camera_distance import (
    convert_depth_result_to_camera_distance,
)
from ..domain.projection import A1ProjectionDirection
from ..domain.spine import calculate_uniform_scale
from .a1_preparation_contracts import (
    A1ObjectPreparationError,
    StatisticsValue,
    freeze_statistics,
)
from .a1_source_geometry_preparation import (
    A1SourceGeometryPreparationResult,
    _ProjectionPreparation,
    _build_prepared_statistics,
    _log_prepared_source,
    _normalize_source_geometry,
    _read_source_snapshot,
    _resolve_source_request,
)
from .active_camera_projection import resolve_a1_active_camera_projection_frame


logger = logging.getLogger(__name__)


def _normal_camera_request_settings(
    settings: A1SingleObjectExportSettings,
) -> A1SingleObjectExportSettings:
    """Return settings accepted by shared geometry validation without losing output."""

    if not isinstance(settings, A1SingleObjectExportSettings):
        raise TypeError("settings must be A1SingleObjectExportSettings")
    if (
        settings.bake_execution.texture_export_mode
        is not A1TextureExportMode.DEPTH_CAMERA_PROJECTION
    ):
        raise ValueError(
            "Depth source preparation requires DEPTH_CAMERA_PROJECTION mode"
        )
    return replace(
        settings,
        source_geometry_mode=A1SourceGeometryMode.EVALUATED,
        projection_direction=A1ProjectionDirection.ACTIVE_CAMERA,
        bake_execution=replace(
            settings.bake_execution,
            texture_export_mode=A1TextureExportMode.NORMAL_UV_SEGMENTS,
        ),
    )


def _depth_statistics(
    base: Mapping[str, StatisticsValue],
    result: DepthCameraProjectionResult,
) -> Mapping[str, StatisticsValue]:
    if not isinstance(result, DepthCameraProjectionResult):
        raise TypeError("result must be DepthCameraProjectionResult")
    camera_distances = tuple(
        float(vertex.position[2]) for vertex in result.snapshot.vertices
    )
    if not camera_distances or any(distance <= 0.0 for distance in camera_distances):
        raise ValueError(
            "Depth rig snapshot must contain positive distances from shared camera zero"
        )
    return freeze_statistics(
        base,
        {
            "projection_kind": "DEPTH_ACTIVE_CAMERA",
            "depth_projection_active": 1,
            "depth_projection_base_mode": result.base_mode.value,
            "depth_projection_base_depth": result.base_depth,
            "depth_projection_farthest_visible_depth": result.farthest_visible_depth,
            "depth_projection_nearest_visible_depth": result.nearest_visible_depth,
            "depth_projection_nearest_camera_distance": min(camera_distances),
            "depth_projection_farthest_camera_distance": max(camera_distances),
            "depth_projection_camera_zero_shared": 1,
            "depth_projection_maximum_relief": result.maximum_relief,
            "depth_projection_requested_spacing_pixels": (
                result.requested_spacing_pixels
            ),
            "depth_projection_resolved_spacing_x_pixels": (
                result.resolved_spacing_x_pixels
            ),
            "depth_projection_resolved_spacing_y_pixels": (
                result.resolved_spacing_y_pixels
            ),
            "depth_projection_source_triangle_count": result.source_triangle_count,
            "depth_projection_point_count": result.sampled_point_count,
            "active_camera_projection_applied": 1,
            "active_camera_preprojection_triangulation": 1,
            "active_camera_name": result.frame.camera_id,
            "active_camera_type": result.frame.kind.value,
            "active_camera_clip_start": result.frame.clip_start,
            "active_camera_clip_end": result.frame.clip_end,
            "projection_canvas_width": result.frame.texture_width,
            "projection_canvas_height": result.frame.texture_height,
            "attachment_invert_y": 1,
        },
    )


def prepare_a1_depth_source_geometry(
    source_obj: Any,
    settings: A1SingleObjectExportSettings,
    *,
    scene: Any | None = None,
) -> A1SourceGeometryPreparationResult:
    """Build one optimized camera-depth surface without mutating source geometry."""

    stage = A1SingleObjectStage.VALIDATE_REQUEST
    object_id: str | None = None
    warnings: Tuple[ExportIssue, ...] = ()
    statistics: Mapping[str, StatisticsValue] = {}
    try:
        validated_settings = _normal_camera_request_settings(settings)
        request = _resolve_source_request(source_obj, validated_settings, scene)
        object_id = request.object_id
        statistics = freeze_statistics(
            request.statistics,
            {
                "texture_export_mode": A1TextureExportMode.DEPTH_CAMERA_PROJECTION.value,
                "source_geometry_mode": A1SourceGeometryMode.EVALUATED.value,
                "projection_direction": A1ProjectionDirection.ACTIVE_CAMERA.value,
            },
        )

        stage = A1SingleObjectStage.READ_GEOMETRY
        source_snapshot, modifier_count, warnings, uv_report = _read_source_snapshot(
            source_obj,
            request.object_id,
            validated_settings,
            scene=request.scene,
            depsgraph=request.depsgraph,
        )

        stage = A1SingleObjectStage.PREPARE_GEOMETRY
        normalized = _normalize_source_geometry(
            source_snapshot,
            validated_settings,
            warnings,
            object_id=request.object_id,
        )
        warnings = normalized.warnings
        if request.scene is None or request.depsgraph is None:
            raise ValueError(
                "Depth Camera Projection lost its evaluated Scene or dependency graph"
            )
        frame = resolve_a1_active_camera_projection_frame(
            request.scene,
            texture_width=settings.export.texture_width,
            texture_height=settings.export.texture_height,
            depsgraph=request.depsgraph,
        )
        uniform_scale = calculate_uniform_scale(
            settings.export.texture_width,
            settings.export.texture_height,
            settings.rig_scale_mode,
        )
        projected_depth = build_depth_camera_projection_surface(
            normalized.snapshot,
            frame,
            uniform_scale=uniform_scale,
            uv_layer_name=settings.uv.layer_name,
            settings=settings.bake_execution.depth_projection,
        )
        depth = convert_depth_result_to_camera_distance(projected_depth)

        stage = A1SingleObjectStage.ASSIGN_Z_GROUPS
        z_groups = build_a1_z_group_assignment(depth.snapshot)

        stage = A1SingleObjectStage.PREPARE_GEOMETRY
        # Region preparation remains available for diagnostics and the established UV
        # lineage pipeline. Depth document assembly consumes the complete UV snapshot as
        # one attachment and deliberately does not serialize these decomposed regions.
        geometry = prepare_a1_geometry_regions(
            depth.snapshot,
            request.geometry_settings,
        )
        projection = _ProjectionPreparation(
            snapshot=depth.snapshot,
            geometry=geometry,
            projected_origin=depth.projected_origin,
            depth_range=calculate_a1_projected_snapshot_depth_range(depth.snapshot),
            camera_projection_kind=frame.kind,
            statistics=_depth_statistics({}, depth),
        )
        statistics = _build_prepared_statistics(
            statistics,
            modifier_count=modifier_count,
            uv_report=uv_report,
            normalized=normalized,
            projection=projection,
            z_groups=z_groups,
            geometry=geometry,
        )
        statistics = _depth_statistics(statistics, depth)
        _log_prepared_source(
            request,
            settings,
            normalized,
            projection,
            geometry,
            uv_report,
        )
        camera_distances = tuple(
            float(vertex.position[2]) for vertex in depth.snapshot.vertices
        )
        logger.info(
            "Prepared Depth Camera Projection source '%s': triangles=%d points=%d "
            "camera_z=[%s, %s] camera_distance=[%s, %s] relief=%s",
            request.object_id,
            depth.source_triangle_count,
            depth.sampled_point_count,
            depth.farthest_visible_depth,
            depth.nearest_visible_depth,
            min(camera_distances),
            max(camera_distances),
            depth.maximum_relief,
        )
        return A1SourceGeometryPreparationResult(
            source_object=source_obj,
            object_id=request.object_id,
            prefix=request.prefix,
            settings=settings,
            output_paths=request.output_paths,
            renderer=request.renderer,
            source_snapshot=depth.snapshot,
            z_groups=z_groups,
            geometry=geometry,
            warnings=warnings,
            statistics=statistics,
            camera_projection_kind=frame.kind,
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
    "_depth_statistics",
    "prepare_a1_depth_source_geometry",
]
