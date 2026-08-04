"""Prepare one evaluated Depth Camera Projection source with parallax reserve."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import degrees
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
    DepthParallaxGeometryPackage,
    MeshSnapshot,
    MeshSnapshotValidator,
    ModifierLineagePolicy,
    build_depth_camera_projection_surface,
    build_depth_parallax_geometry_package,
    calculate_a1_projected_snapshot_depth_range,
)
from ..domain.geometry.depth_camera_distance import (
    convert_depth_snapshot_to_camera_distance,
)
from ..domain.geometry.depth_parallax_identity import (
    canonicalize_depth_parallax_package_identity,
)
from ..domain.geometry.evaluated_identity import (
    EvaluatedIdentityRebaseResult,
    rebase_mesh_snapshot_to_evaluated_identity,
)
from ..domain.projection import A1ProjectionDirection
from ..domain.spine import calculate_uniform_scale
from .a1_preparation_contracts import (
    A1ObjectPreparationError,
    StatisticsValue,
    freeze_statistics,
    warning_issue,
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
from .depth_parallax_camera_views import resolve_depth_parallax_camera_views


logger = logging.getLogger(__name__)
_LINEAGE_POSITION_TOLERANCE = 1.0e-9


@dataclass(frozen=True, slots=True, kw_only=True)
class A1DepthSourceGeometryPreparationResult(A1SourceGeometryPreparationResult):
    """Depth source plus front/reserve topology required by later render stages."""

    parallax_package: DepthParallaxGeometryPackage

    def __post_init__(self) -> None:
        A1SourceGeometryPreparationResult.__post_init__(self)
        if not isinstance(self.parallax_package, DepthParallaxGeometryPackage):
            raise TypeError("parallax_package must be DepthParallaxGeometryPackage")
        if self.source_snapshot != self.parallax_package.union_snapshot:
            raise ValueError(
                "Depth source_snapshot must be the parallax union snapshot"
            )


def _normal_camera_request_settings(
    settings: A1SingleObjectExportSettings,
) -> A1SingleObjectExportSettings:
    """Return the evaluated-geometry contract consumed by shared preparation.

    Depth Camera Projection renders and projects the evaluated modifier result. Array,
    Mirror, and equivalent duplication may legitimately produce several evaluated
    elements from one stamped source element. The permissive lineage policy validates
    that every required evaluated element still has safe provenance; evaluated-local
    identities are canonicalized immediately after the read so unrelated copies remain
    distinct throughout depth adjacency, rig generation, and render-face ownership.
    """

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
        modifier_lineage_policy=ModifierLineagePolicy.ALLOW_SOURCE_DUPLICATION,
        projection_direction=A1ProjectionDirection.ACTIVE_CAMERA,
        bake_execution=replace(
            settings.bake_execution,
            texture_export_mode=A1TextureExportMode.NORMAL_UV_SEGMENTS,
        ),
    )


def _canonicalize_depth_evaluated_identity(
    snapshot: MeshSnapshot,
    warnings: Tuple[ExportIssue, ...],
    statistics: Mapping[str, StatisticsValue],
    *,
    object_id: str,
) -> tuple[
    MeshSnapshot,
    Tuple[ExportIssue, ...],
    Mapping[str, StatisticsValue],
    EvaluatedIdentityRebaseResult,
]:
    """Replace duplicate modifier lineage with unique evaluated-local identities."""

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    if not isinstance(warnings, tuple) or not all(
        isinstance(issue, ExportIssue) for issue in warnings
    ):
        raise TypeError("warnings must be a tuple of ExportIssue values")
    if not isinstance(statistics, Mapping):
        raise TypeError("statistics must be a mapping")
    if not isinstance(object_id, str) or not object_id.strip():
        raise ValueError("object_id must be a non-empty string")

    result = rebase_mesh_snapshot_to_evaluated_identity(snapshot)
    resolved_statistics = freeze_statistics(
        statistics,
        {
            "evaluated_identity_rebased": int(result.changed),
            "evaluated_identity_duplicate_vertex_source_ids": (
                result.duplicate_vertex_source_id_count
            ),
            "evaluated_identity_duplicate_edge_source_ids": (
                result.duplicate_edge_source_id_count
            ),
            "evaluated_identity_duplicate_face_source_ids": (
                result.duplicate_face_source_id_count
            ),
            "evaluated_identity_duplicate_loop_source_ids": (
                result.duplicate_loop_source_id_count
            ),
            "evaluated_identity_generated_edge_count": (
                result.missing_edge_source_id_count
            ),
        },
    )
    if not result.changed:
        return result.snapshot, warnings, resolved_statistics, result

    resolved_warnings = warnings + (
        warning_issue(
            stage=A1SingleObjectStage.READ_GEOMETRY,
            code="EVALUATED_IDENTITY_REBASED",
            message=(
                "Validated modifier topology was canonicalized to unique "
                "evaluated-local identities. Duplicated or merged modifier copies "
                "remain independent in depth adjacency, rig generation, and "
                "virtual-view face ownership."
            ),
            object_id=object_id,
            context={
                "duplicate_vertex_source_ids": (
                    result.duplicate_vertex_source_id_count
                ),
                "duplicate_edge_source_ids": result.duplicate_edge_source_id_count,
                "duplicate_face_source_ids": result.duplicate_face_source_id_count,
                "duplicate_loop_source_ids": result.duplicate_loop_source_id_count,
                "generated_edges": result.missing_edge_source_id_count,
            },
        ),
    )
    logger.info(
        "Depth evaluated identity rebased for '%s': vertex_collisions=%d "
        "edge_collisions=%d loop_collisions=%d face_collisions=%d "
        "generated_edges=%d",
        object_id,
        result.duplicate_vertex_source_id_count,
        result.duplicate_edge_source_id_count,
        result.duplicate_loop_source_id_count,
        result.duplicate_face_source_id_count,
        result.missing_edge_source_id_count,
    )
    return result.snapshot, resolved_warnings, resolved_statistics, result


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
            "depth_projection_requested_spacing_pixels": result.requested_spacing_pixels,
            "depth_projection_resolved_spacing_x_pixels": result.resolved_spacing_x_pixels,
            "depth_projection_resolved_spacing_y_pixels": result.resolved_spacing_y_pixels,
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


def _require_subset_union_lineage(
    subset: MeshSnapshot,
    union: MeshSnapshot,
    *,
    label: str,
) -> None:
    """Require subset vertices to retain exact union IDs and projected positions."""

    if not isinstance(subset, MeshSnapshot):
        raise TypeError("subset must be MeshSnapshot")
    if not isinstance(union, MeshSnapshot):
        raise TypeError("union must be MeshSnapshot")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a non-empty string")
    MeshSnapshotValidator().validate_or_raise(subset)
    MeshSnapshotValidator().validate_or_raise(union)

    union_by_source = {vertex.source_id: vertex for vertex in union.vertices}
    if len(union_by_source) != len(union.vertices):
        raise ValueError("Parallax union contains duplicate SourceVertexId values")

    for vertex in subset.vertices:
        union_vertex = union_by_source.get(vertex.source_id)
        if union_vertex is None:
            raise ValueError(
                f"{label} contains vertex lineage absent from parallax union: "
                f"source_id={vertex.source_id}"
            )
        if any(
            abs(float(vertex.position[index]) - float(union_vertex.position[index]))
            > _LINEAGE_POSITION_TOLERANCE
            for index in range(3)
        ):
            raise ValueError(
                f"{label} vertex position differs from parallax union for "
                f"source_id={vertex.source_id}; subset={vertex.position}, "
                f"union={union_vertex.position}"
            )


def _package_to_camera_distance(
    package: DepthParallaxGeometryPackage,
) -> DepthParallaxGeometryPackage:
    """Convert the complete package without reconstructing lineage by coordinates."""

    if not isinstance(package, DepthParallaxGeometryPackage):
        raise TypeError("package must be DepthParallaxGeometryPackage")

    _require_subset_union_lineage(
        package.front_snapshot,
        package.union_snapshot,
        label="front parallax subset",
    )
    for surface in package.reserve_surfaces:
        _require_subset_union_lineage(
            surface.snapshot,
            package.union_snapshot,
            label=f"reserve parallax subset {surface.view.view_id.value}",
        )

    union_distance = convert_depth_snapshot_to_camera_distance(
        package.union_snapshot,
        snapshot_suffix="parallax-union-camera-distance",
    )
    front_distance = convert_depth_snapshot_to_camera_distance(
        package.front_snapshot,
        snapshot_suffix="parallax-front-camera-distance",
    )
    reserve_distance = tuple(
        replace(
            surface,
            snapshot=convert_depth_snapshot_to_camera_distance(
                surface.snapshot,
                snapshot_suffix=(
                    f"parallax-{surface.view.view_id.value.lower()}-camera-distance"
                ),
            ),
        )
        for surface in package.reserve_surfaces
    )
    front_result = replace(package.front_result, snapshot=front_distance)
    return replace(
        package,
        front_result=front_result,
        union_snapshot=union_distance,
        front_snapshot=front_distance,
        reserve_surfaces=reserve_distance,
    )


def _parallax_statistics(
    base: Mapping[str, StatisticsValue],
    package: DepthParallaxGeometryPackage,
) -> Mapping[str, StatisticsValue]:
    if not isinstance(package, DepthParallaxGeometryPackage):
        raise TypeError("package must be DepthParallaxGeometryPackage")
    view_ids = ",".join(
        surface.view.view_id.value for surface in package.reserve_surfaces
    )
    maximum_angle = max(
        (
            surface.maximum_accumulated_angle_radians
            for surface in package.reserve_surfaces
        ),
        default=0.0,
    )
    return freeze_statistics(
        base,
        {
            "depth_parallax_horizon_angle_radians": package.horizon_angle_radians,
            "depth_parallax_horizon_angle_degrees": degrees(package.horizon_angle_radians),
            "depth_parallax_enabled": int(package.reserve_enabled),
            "depth_parallax_front_source_face_count": len(package.front_face_indices),
            "depth_parallax_reserve_source_face_count": len(package.reserve_face_indices),
            "depth_parallax_reserve_attachment_count": len(package.reserve_surfaces),
            "depth_parallax_attachment_count": package.attachment_count,
            "depth_parallax_union_point_count": len(package.union_snapshot.vertices),
            "depth_parallax_view_ids": view_ids,
            "depth_parallax_maximum_accumulated_angle_radians": maximum_angle,
            "depth_parallax_maximum_accumulated_angle_degrees": degrees(maximum_angle),
        },
    )


def prepare_a1_depth_source_geometry(
    source_obj: Any,
    settings: A1SingleObjectExportSettings,
    *,
    scene: Any | None = None,
) -> A1DepthSourceGeometryPreparationResult:
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
                "modifier_lineage_policy": (
                    validated_settings.modifier_lineage_policy.value
                ),
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
        (
            source_snapshot,
            warnings,
            statistics,
            _identity_rebase,
        ) = _canonicalize_depth_evaluated_identity(
            source_snapshot,
            warnings,
            statistics,
            object_id=request.object_id,
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
        projected_front = build_depth_camera_projection_surface(
            normalized.snapshot,
            frame,
            uniform_scale=uniform_scale,
            uv_layer_name=settings.uv.layer_name,
            settings=settings.bake_execution.depth_projection,
        )
        horizon_angle = settings.bake_execution.depth_parallax.horizon_angle_radians
        reserve_views = resolve_depth_parallax_camera_views(
            request.scene,
            normalized.snapshot,
            frame,
            horizon_angle_radians=horizon_angle,
            depsgraph=request.depsgraph,
        )
        camera_z_package = build_depth_parallax_geometry_package(
            normalized.snapshot,
            projected_front,
            reserve_views,
            uniform_scale=uniform_scale,
            uv_layer_name=settings.uv.layer_name,
            horizon_angle_radians=horizon_angle,
            max_points=settings.bake_execution.depth_projection.max_points,
        )
        camera_z_package = canonicalize_depth_parallax_package_identity(
            camera_z_package,
            uv_layer_name=settings.uv.layer_name,
        )
        depth_package = _package_to_camera_distance(camera_z_package)
        depth = depth_package.front_result
        stage = A1SingleObjectStage.ASSIGN_Z_GROUPS
        z_groups = build_a1_z_group_assignment(depth_package.union_snapshot)
        stage = A1SingleObjectStage.PREPARE_GEOMETRY
        geometry = prepare_a1_geometry_regions(
            depth_package.union_snapshot,
            request.geometry_settings,
        )
        projection = _ProjectionPreparation(
            snapshot=depth_package.union_snapshot,
            geometry=geometry,
            projected_origin=depth.projected_origin,
            depth_range=calculate_a1_projected_snapshot_depth_range(
                camera_z_package.union_snapshot
            ),
            camera_projection_kind=frame.kind,
            statistics=_parallax_statistics(_depth_statistics({}, depth), depth_package),
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
        statistics = _parallax_statistics(
            _depth_statistics(statistics, depth),
            depth_package,
        )
        _log_prepared_source(
            request,
            settings,
            normalized,
            projection,
            geometry,
            uv_report,
        )
        camera_distances = tuple(
            float(vertex.position[2])
            for vertex in depth_package.union_snapshot.vertices
        )
        logger.info(
            "Prepared Depth Camera Projection source '%s': front_triangles=%d "
            "front_points=%d reserve_faces=%d reserve_attachments=%d "
            "union_points=%d horizon=%sdeg camera_distance=[%s, %s]",
            request.object_id,
            depth.source_triangle_count,
            depth.sampled_point_count,
            len(depth_package.reserve_face_indices),
            len(depth_package.reserve_surfaces),
            len(depth_package.union_snapshot.vertices),
            degrees(depth_package.horizon_angle_radians),
            min(camera_distances),
            max(camera_distances),
        )
        return A1DepthSourceGeometryPreparationResult(
            source_object=source_obj,
            object_id=request.object_id,
            prefix=request.prefix,
            settings=settings,
            output_paths=request.output_paths,
            renderer=request.renderer,
            source_snapshot=depth_package.union_snapshot,
            z_groups=z_groups,
            geometry=geometry,
            warnings=warnings,
            statistics=statistics,
            camera_projection_kind=frame.kind,
            parallax_package=depth_package,
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
    "A1DepthSourceGeometryPreparationResult",
    "_canonicalize_depth_evaluated_identity",
    "_depth_statistics",
    "prepare_a1_depth_source_geometry",
]
