"""Public routing owner for source-bounded Depth Camera Projection geometry."""

from __future__ import annotations

import logging
from math import isfinite
from time import perf_counter

from ..camera_projection import A1CameraProjectionFrame
from .depth_camera_projection import (
    DepthCameraProjectionError,
    DepthCameraProjectionResult,
    DepthCameraProjectionSettings,
    _translation_only_origin,
    _world_point,
)
from .depth_camera_projection_bounded import (
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

_DIRECT_VERTEX_LIMIT = 512
_DIRECT_TRIANGLE_LIMIT = 1024
_MINIMUM_SPARSE_COMPONENT_THRESHOLD = 8
_MAXIMUM_SPARSE_COMPONENT_THRESHOLD = 32


def _crosses_camera_frame(
    snapshot: MeshSnapshot,
    frame: A1CameraProjectionFrame,
) -> bool:
    """Check source vertices directly; triangulation is unnecessary for frame bounds."""

    origin = _translation_only_origin(snapshot.world_matrix)
    half_width = float(frame.texture_width) / 2.0
    half_height = float(frame.texture_height) / 2.0
    tolerance = max(
        float(frame.texture_width),
        float(frame.texture_height),
        1.0,
    ) * 1.0e-9

    for vertex in snapshot.vertices:
        point = frame.project_world_point(
            _world_point(origin, vertex.position),
            field_name=f"frame-preflight.vertex[{vertex.id.index}]",
        )
        if (
            point.u < -half_width - tolerance
            or point.u > half_width + tolerance
            or point.v < -half_height - tolerance
            or point.v > half_height + tolerance
        ):
            return True
    return False


def _face_component_count_at_least(
    snapshot: MeshSnapshot,
    threshold: int,
) -> bool:
    """Return once local shared-edge topology reaches ``threshold`` components."""

    if isinstance(threshold, bool) or not isinstance(threshold, int):
        raise TypeError("threshold must be int")
    if threshold < 1:
        raise ValueError("threshold must be positive")
    if not snapshot.faces:
        return False

    loops = snapshot.loop_by_id()
    owners_by_edge: dict[int, list[int]] = {}
    for face in snapshot.faces:
        for loop_id in face.loop_ids:
            edge_index = loops[loop_id].edge_id.index
            owners_by_edge.setdefault(edge_index, []).append(face.id.index)

    adjacency: dict[int, set[int]] = {
        face.id.index: set() for face in snapshot.faces
    }
    for owners in owners_by_edge.values():
        unique = tuple(sorted(set(owners)))
        for owner_index, first in enumerate(unique):
            for second in unique[owner_index + 1 :]:
                adjacency[first].add(second)
                adjacency[second].add(first)

    remaining = set(adjacency)
    component_count = 0
    while remaining:
        component_count += 1
        if component_count >= threshold:
            return True
        seed = min(remaining)
        remaining.remove(seed)
        pending = [seed]
        while pending:
            current = pending.pop()
            for neighbour in adjacency[current]:
                if neighbour not in remaining:
                    continue
                remaining.remove(neighbour)
                pending.append(neighbour)
    return False


def _prefers_component_envelope(
    snapshot: MeshSnapshot,
    settings: DepthCameraProjectionSettings,
) -> bool:
    """Detect dense disconnected sources before paying for a failed lattice pass."""

    if (
        len(snapshot.vertices) <= _DIRECT_VERTEX_LIMIT
        and len(snapshot.faces) <= _DIRECT_TRIANGLE_LIMIT
    ):
        return False
    threshold = max(
        _MINIMUM_SPARSE_COMPONENT_THRESHOLD,
        min(
            _MAXIMUM_SPARSE_COMPONENT_THRESHOLD,
            max(1, settings.max_points // 4),
        ),
    )
    return _face_component_count_at_least(snapshot, threshold)


def _component_envelope_or_raise(
    snapshot: MeshSnapshot,
    frame: A1CameraProjectionFrame,
    *,
    uniform_scale: float,
    uv_layer_name: str,
    settings: DepthCameraProjectionSettings,
    reason: str,
    lattice_error: DepthCameraProjectionError | None = None,
) -> DepthCameraProjectionResult:
    started = perf_counter()
    try:
        result = _build_component_envelope_surface(
            snapshot,
            frame,
            uniform_scale=uniform_scale,
            uv_layer_name=uv_layer_name,
            settings=settings,
        )
    except _ComponentEnvelopeUnavailable as envelope_error:
        if lattice_error is not None:
            raise DepthCameraProjectionError(
                "depth lattice and component-envelope fallback both failed; "
                f"lattice={lattice_error}; envelope={envelope_error}"
            ) from lattice_error
        raise DepthCameraProjectionError(
            "component-envelope preflight route failed; "
            f"reason={reason}; envelope={envelope_error}"
        ) from envelope_error

    logger.info(
        "Depth component-envelope for '%s': reason=%s points=%d elapsed=%.3fs",
        snapshot.source_object_id,
        reason,
        len(result.snapshot.vertices),
        perf_counter() - started,
    )
    return result


def _build_full_frame_surface(
    snapshot: MeshSnapshot,
    frame: A1CameraProjectionFrame,
    *,
    uniform_scale: float,
    uv_layer_name: str,
    settings: DepthCameraProjectionSettings,
) -> DepthCameraProjectionResult:
    """Build full-frame geometry without repeating predictable sparse failures."""

    if _prefers_component_envelope(snapshot, settings):
        logger.warning(
            "Dense disconnected Depth source '%s' bypasses regular lattice and uses "
            "budgeted component envelopes",
            snapshot.source_object_id,
        )
        return _component_envelope_or_raise(
            snapshot,
            frame,
            uniform_scale=uniform_scale,
            uv_layer_name=uv_layer_name,
            settings=settings,
            reason="dense-disconnected-preflight",
        )

    lattice_started = perf_counter()
    try:
        result = _build_bounded_surface(
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
        return _component_envelope_or_raise(
            snapshot,
            frame,
            uniform_scale=uniform_scale,
            uv_layer_name=uv_layer_name,
            settings=settings,
            reason="recoverable-lattice-failure",
            lattice_error=lattice_error,
        )

    logger.info(
        "Depth bounded surface for '%s': points=%d elapsed=%.3fs",
        snapshot.source_object_id,
        len(result.snapshot.vertices),
        perf_counter() - lattice_started,
    )
    return result


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


__all__ = [
    "_crosses_camera_frame",
    "_face_component_count_at_least",
    "_prefers_component_envelope",
    "build_depth_camera_projection_surface",
]
