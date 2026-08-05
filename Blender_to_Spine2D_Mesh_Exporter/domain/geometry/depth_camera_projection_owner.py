"""Public routing owner for source-bounded Depth Camera Projection geometry."""

from __future__ import annotations

import logging
from math import isfinite

from ..camera_projection import A1CameraProjectionFrame
from .depth_camera_projection import (
    DepthCameraProjectionError,
    DepthCameraProjectionResult,
    DepthCameraProjectionSettings,
    _translation_only_origin,
)
from .depth_camera_projection_bounded import (
    _projected_triangles,
    build_depth_camera_projection_surface as _build_bounded_surface,
)
from .depth_camera_projection_component_envelope import (
    _ComponentEnvelopeUnavailable,
    _build_component_envelope_surface,
    is_sparse_lattice_failure,
)
from .depth_camera_projection_visible_topology import (
    _LocalTopologyUnavailable,
    _build_visible_topology_surface,
)
from .model import MeshSnapshot
from .validator import MeshSnapshotValidator


logger = logging.getLogger(__name__)


def _crosses_camera_frame(
    snapshot: MeshSnapshot,
    frame: A1CameraProjectionFrame,
) -> bool:
    origin = _translation_only_origin(snapshot.world_matrix)
    _triangulated, projected_by_vertex, _triangles = _projected_triangles(
        snapshot,
        frame,
        origin,
    )
    half_width = float(frame.texture_width) / 2.0
    half_height = float(frame.texture_height) / 2.0
    tolerance = max(
        float(frame.texture_width),
        float(frame.texture_height),
        1.0,
    ) * 1.0e-9
    return any(
        point.u < -half_width - tolerance
        or point.u > half_width + tolerance
        or point.v < -half_height - tolerance
        or point.v > half_height + tolerance
        for point in projected_by_vertex.values()
    )


def _build_full_frame_surface(
    snapshot: MeshSnapshot,
    frame: A1CameraProjectionFrame,
    *,
    uniform_scale: float,
    uv_layer_name: str,
    settings: DepthCameraProjectionSettings,
) -> DepthCameraProjectionResult:
    """Build full-frame geometry and repair only sparse-lattice failures."""

    try:
        return _build_bounded_surface(
            snapshot,
            frame,
            uniform_scale=uniform_scale,
            uv_layer_name=uv_layer_name,
            settings=settings,
        )
    except DepthCameraProjectionError as lattice_error:
        if not is_sparse_lattice_failure(lattice_error):
            raise

        logger.warning(
            "Depth lattice for '%s' could not retain a connected sparse surface: "
            "%s; using budgeted component envelopes",
            snapshot.source_object_id,
            lattice_error,
        )
        try:
            return _build_component_envelope_surface(
                snapshot,
                frame,
                uniform_scale=uniform_scale,
                uv_layer_name=uv_layer_name,
                settings=settings,
            )
        except _ComponentEnvelopeUnavailable as envelope_error:
            raise DepthCameraProjectionError(
                "depth lattice and component-envelope fallback both failed; "
                f"lattice={lattice_error}; envelope={envelope_error}"
            ) from lattice_error


def build_depth_camera_projection_surface(
    snapshot: MeshSnapshot,
    frame: A1CameraProjectionFrame,
    *,
    uniform_scale: float,
    uv_layer_name: str,
    settings: DepthCameraProjectionSettings = DepthCameraProjectionSettings(),
) -> DepthCameraProjectionResult:
    """Keep full-frame behavior stable and locally repair camera-clipped polygons."""

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    if not isinstance(frame, A1CameraProjectionFrame):
        raise TypeError("frame must be A1CameraProjectionFrame")
    if isinstance(uniform_scale, bool) or not isinstance(uniform_scale, (int, float)):
        raise TypeError("uniform_scale must be a finite positive number")
    resolved_scale = float(uniform_scale)
    if not isfinite(resolved_scale) or resolved_scale <= 0.0:
        raise ValueError("uniform_scale must be a finite positive number")
    if not isinstance(uv_layer_name, str) or not uv_layer_name.strip():
        raise ValueError("uv_layer_name must be a non-empty string")
    if not isinstance(settings, DepthCameraProjectionSettings):
        raise TypeError("settings must be DepthCameraProjectionSettings")

    MeshSnapshotValidator().validate_or_raise(snapshot)
    if len(snapshot.vertices) < 3:
        raise DepthCameraProjectionError(
            "depth projection requires at least three evaluated source vertices"
        )

    if not _crosses_camera_frame(snapshot, frame):
        return _build_full_frame_surface(
            snapshot,
            frame,
            uniform_scale=resolved_scale,
            uv_layer_name=uv_layer_name,
            settings=settings,
        )

    try:
        return _build_visible_topology_surface(
            snapshot,
            frame,
            uniform_scale=resolved_scale,
            uv_layer_name=uv_layer_name,
            settings=settings,
        )
    except _LocalTopologyUnavailable as exc:
        logger.error(
            "Camera-clipped source '%s' cannot use local topology repair: %s",
            snapshot.source_object_id,
            exc,
        )
        raise DepthCameraProjectionError(
            "camera-clipped source cannot be repaired within the local topology "
            f"contract: {exc}"
        ) from exc


__all__ = ["build_depth_camera_projection_surface"]
