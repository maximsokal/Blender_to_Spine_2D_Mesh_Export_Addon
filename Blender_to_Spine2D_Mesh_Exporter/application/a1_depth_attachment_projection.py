"""Project one complete generated depth surface into one compensated Spine attachment.

Unlike Normal / UV Segments, a camera-depth surface is already an optimized screen-space
mesh. Splitting it into disk regions changes its visible topology and creates unrelated
``Segment_N`` attachments. This projector keeps the full generated surface in one mesh,
orders one deterministic screen-space convex hull, and compensates every vertex-bone Y
for its parent depth-group setup offset.
"""

from __future__ import annotations

from math import isfinite
from typing import Mapping, Tuple

from ..domain.geometry import LoopId, MeshSnapshot, MeshSnapshotValidator, VertexId
from ..domain.spine import (
    LegacyAttachmentVertex,
    LegacyMeshAttachmentRequest,
    LegacyRigBuildResult,
)
from .a1_attachment_projection_service import (
    A1AttachmentProjectionError,
    A1AttachmentProjectionResult,
    A1AttachmentProjectionSettings,
    A1AttachmentVertexKey,
)


_Position2D = tuple[float, float]
_AREA_EPSILON = 1.0e-9


def _cross(first: _Position2D, second: _Position2D, third: _Position2D) -> float:
    return (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _loop_keys(
    snapshot: MeshSnapshot,
    layer_name: str,
) -> dict[LoopId, A1AttachmentVertexKey]:
    if layer_name not in snapshot.uv_layer_names:
        raise A1AttachmentProjectionError(
            f"Depth surface is missing UV layer {layer_name!r}"
        )
    resolved: dict[LoopId, A1AttachmentVertexKey] = {}
    for loop in sorted(snapshot.loops, key=lambda item: item.id.index):
        uv = loop.uv(layer_name)
        if uv is None:
            raise A1AttachmentProjectionError(
                f"Depth surface loop {loop.id.index} has no UV in {layer_name!r}"
            )
        key = A1AttachmentVertexKey(
            vertex_id=loop.vertex_id,
            uv=(float(uv[0]), float(uv[1])),
        )
        if loop.id in resolved:
            raise A1AttachmentProjectionError(
                f"Depth surface contains duplicate LoopId {loop.id.index}"
            )
        resolved[loop.id] = key
    return resolved


def _ordered_unique_keys(
    snapshot: MeshSnapshot,
    loop_keys: Mapping[LoopId, A1AttachmentVertexKey],
) -> tuple[A1AttachmentVertexKey, ...]:
    ordered: list[A1AttachmentVertexKey] = []
    seen: set[A1AttachmentVertexKey] = set()
    for face in sorted(snapshot.faces, key=lambda item: item.id.index):
        if len(face.loop_ids) != 3:
            raise A1AttachmentProjectionError(
                f"Depth surface face {face.id.index} is not triangulated"
            )
        for loop_id in face.loop_ids:
            key = loop_keys[loop_id]
            if key in seen:
                continue
            seen.add(key)
            ordered.append(key)
    if len(ordered) < 3:
        raise A1AttachmentProjectionError(
            "Depth surface must contain at least three attachment vertices"
        )
    return tuple(ordered)


def _z_group_by_vertex(
    snapshot: MeshSnapshot,
    settings: A1AttachmentProjectionSettings,
    rig: LegacyRigBuildResult,
) -> dict[VertexId, int]:
    mapping = {
        binding.vertex_id: binding.z_group_index for binding in settings.z_bindings
    }
    expected = {vertex.id for vertex in snapshot.vertices}
    missing = expected - set(mapping)
    unknown = set(mapping) - expected
    if missing or unknown:
        raise A1AttachmentProjectionError(
            "Depth z-bindings must cover generated vertices exactly; "
            f"missing={tuple(sorted(item.index for item in missing))}, "
            f"unknown={tuple(sorted(item.index for item in unknown))}"
        )
    valid_groups = {group.index for group in rig.info.z_groups}
    invalid = tuple(sorted(set(mapping.values()) - valid_groups))
    if invalid:
        raise A1AttachmentProjectionError(
            f"Depth z-bindings reference unknown rig groups {invalid}; "
            f"available={tuple(sorted(valid_groups))}"
        )
    return mapping


def _setup_position_by_key(
    snapshot: MeshSnapshot,
    keys: tuple[A1AttachmentVertexKey, ...],
    settings: A1AttachmentProjectionSettings,
    rig: LegacyRigBuildResult,
) -> dict[A1AttachmentVertexKey, _Position2D]:
    vertex_map = snapshot.vertex_by_id()
    resolved: dict[A1AttachmentVertexKey, _Position2D] = {}
    for key in keys:
        vertex = vertex_map[key.vertex_id]
        position = (
            (
                float(vertex.position[0]) - float(settings.center_x)
            ) * float(rig.info.uniform_scale),
            -(
                float(vertex.position[1]) - float(settings.center_y)
            ) * float(rig.info.uniform_scale),
        )
        if not all(isfinite(component) for component in position):
            raise A1AttachmentProjectionError(
                f"Depth projected setup position became non-finite for {key}"
            )
        resolved[key] = position
    return resolved


def _convex_hull_keys(
    keys: tuple[A1AttachmentVertexKey, ...],
    setup_positions: Mapping[A1AttachmentVertexKey, _Position2D],
) -> tuple[A1AttachmentVertexKey, ...]:
    representatives: dict[_Position2D, A1AttachmentVertexKey] = {}
    order = {key: index for index, key in enumerate(keys)}
    for key in keys:
        position = setup_positions[key]
        previous = representatives.get(position)
        if previous is None or order[key] < order[previous]:
            representatives[position] = key

    points = tuple(sorted(representatives))
    if len(points) < 3:
        raise A1AttachmentProjectionError(
            "Depth attachment has fewer than three unique screen positions"
        )

    x_values = tuple(point[0] for point in points)
    y_values = tuple(point[1] for point in points)
    extent = max(
        max(x_values) - min(x_values),
        max(y_values) - min(y_values),
        1.0,
    )
    tolerance = max(_AREA_EPSILON, extent * extent * 1.0e-10)

    lower: list[_Position2D] = []
    for point in points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= tolerance:
            lower.pop()
        lower.append(point)

    upper: list[_Position2D] = []
    for point in reversed(points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= tolerance:
            upper.pop()
        upper.append(point)

    hull_positions = tuple(lower[:-1] + upper[:-1])
    if len(hull_positions) < 3:
        raise A1AttachmentProjectionError(
            "Depth attachment screen positions are collinear"
        )
    return tuple(representatives[position] for position in hull_positions)


def _attachment_edges(
    triangles: tuple[int, ...],
) -> tuple[int, ...]:
    pairs: set[tuple[int, int]] = set()
    for offset in range(0, len(triangles), 3):
        triangle = triangles[offset : offset + 3]
        for index, first in enumerate(triangle):
            second = triangle[(index + 1) % 3]
            pair = (first, second) if first < second else (second, first)
            pairs.add(pair)
    return tuple(component for pair in sorted(pairs) for component in pair)


def project_depth_camera_attachment(
    snapshot: MeshSnapshot,
    rig: LegacyRigBuildResult,
    settings: A1AttachmentProjectionSettings,
) -> A1AttachmentProjectionResult:
    """Build one full depth attachment with compensated vertex-bone setup positions."""

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    if not isinstance(rig, LegacyRigBuildResult):
        raise TypeError("rig must be LegacyRigBuildResult")
    if not isinstance(settings, A1AttachmentProjectionSettings):
        raise TypeError("settings must be A1AttachmentProjectionSettings")

    MeshSnapshotValidator().validate_or_raise(snapshot)
    rig.validate()
    loop_keys = _loop_keys(snapshot, settings.uv_layer_name)
    all_keys = _ordered_unique_keys(snapshot, loop_keys)
    z_by_vertex = _z_group_by_vertex(snapshot, settings, rig)
    setup_positions = _setup_position_by_key(snapshot, all_keys, settings, rig)
    hull_keys = _convex_hull_keys(all_keys, setup_positions)
    hull_set = set(hull_keys)
    ordered_keys = hull_keys + tuple(key for key in all_keys if key not in hull_set)
    key_to_index = {
        key: attachment_index for attachment_index, key in enumerate(ordered_keys)
    }

    z_offset_by_index = {
        group.index: float(group.y_offset_pixels) for group in rig.info.z_groups
    }
    vertices: list[LegacyAttachmentVertex] = []
    for attachment_index, key in enumerate(ordered_keys):
        z_group_index = z_by_vertex[key.vertex_id]
        setup_x, setup_y = setup_positions[key]
        parent_y = z_offset_by_index[z_group_index]
        local_y = setup_y - parent_y
        if not isfinite(local_y):
            raise A1AttachmentProjectionError(
                f"Depth vertex {attachment_index} local Y became non-finite"
            )
        vertices.append(
            LegacyAttachmentVertex(
                index=attachment_index,
                uv=key.uv,
                bone_position_pixels=(float(setup_x), float(local_y)),
                z_group_index=z_group_index,
            )
        )

    triangle_loop_ids = tuple(
        loop_id
        for face in sorted(snapshot.faces, key=lambda item: item.id.index)
        for loop_id in face.loop_ids
    )
    triangles = tuple(
        key_to_index[loop_keys[loop_id]] for loop_id in triangle_loop_ids
    )
    if len(triangles) % 3 != 0 or not triangles:
        raise A1AttachmentProjectionError(
            "Depth attachment did not produce complete triangles"
        )

    request = LegacyMeshAttachmentRequest(
        slot_name=settings.slot_name,
        attachment_name=settings.attachment_name,
        vertex_prefix=settings.vertex_prefix,
        image_path=settings.image_path,
        width=settings.attachment_width,
        height=settings.attachment_height,
        vertices=tuple(vertices),
        triangles=triangles,
        hull=len(hull_keys),
        edges=_attachment_edges(triangles),
        sequence=settings.sequence,
        skin_name=settings.skin_name,
    )
    projection = A1AttachmentProjectionResult(
        request=request,
        hull_vertex_keys=hull_keys,
        ordered_vertex_keys=ordered_keys,
        loop_to_attachment_index=tuple(zip(triangle_loop_ids, triangles, strict=True)),
    )

    for vertex in projection.request.vertices:
        setup_x, setup_y = setup_positions[projection.ordered_vertex_keys[vertex.index]]
        parent_y = z_offset_by_index[vertex.z_group_index]
        reconstructed = (
            float(vertex.bone_position_pixels[0]),
            float(vertex.bone_position_pixels[1]) + parent_y,
        )
        if (
            abs(reconstructed[0] - setup_x) > 1.0e-7
            or abs(reconstructed[1] - setup_y) > 1.0e-7
        ):
            raise A1AttachmentProjectionError(
                "Depth parent-offset compensation did not reconstruct screen setup "
                f"position for vertex {vertex.index}; reconstructed={reconstructed}, "
                f"expected={(setup_x, setup_y)}"
            )
    return projection


__all__ = ["project_depth_camera_attachment"]
