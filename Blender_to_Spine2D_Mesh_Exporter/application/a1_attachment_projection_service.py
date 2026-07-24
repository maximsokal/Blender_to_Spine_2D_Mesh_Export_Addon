"""Production A1 attachment projection with physical Spine hull normalization.

The loop-level projector owns UV-split identity and topological boundary traversal.
Spine's ``hull`` field has a different responsibility: it counts the vertices of
the physical convex hull, stored as the prefix of the final attachment vertex order.
This service keeps those concerns separate and remaps every dependent index exactly
once after the raw projection has been built.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

from ..domain.geometry import MeshSnapshot
from ..domain.spine import LegacyAttachmentVertex, LegacyMeshAttachmentRequest, LegacyRigBuildResult
from .a1_attachment_projection import (
    A1AttachmentProjectionError,
    A1AttachmentProjectionResult,
    A1AttachmentProjectionSettings,
    A1AttachmentVertexKey,
    A1VertexZBinding,
    project_triangulated_disk_attachment as _project_raw_attachment,
)


Position2D = Tuple[float, float]


def _position(vertex: LegacyAttachmentVertex) -> Position2D:
    if not isinstance(vertex, LegacyAttachmentVertex):
        raise TypeError("vertex must be LegacyAttachmentVertex")
    return (
        float(vertex.bone_position_pixels[0]),
        float(vertex.bone_position_pixels[1]),
    )


def _cross(first: Position2D, second: Position2D, third: Position2D) -> float:
    return (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _convex_hull_positions(
    vertices: Tuple[LegacyAttachmentVertex, ...],
) -> frozenset[Position2D]:
    """Return unique physical convex-hull positions using monotone chain."""

    if not isinstance(vertices, tuple) or not vertices:
        raise ValueError("vertices must be a non-empty tuple")
    if not all(isinstance(vertex, LegacyAttachmentVertex) for vertex in vertices):
        raise TypeError("vertices must contain LegacyAttachmentVertex values")

    points = tuple(sorted({_position(vertex) for vertex in vertices}))
    if len(points) < 3:
        raise A1AttachmentProjectionError(
            "Spine mesh attachment requires at least three unique physical positions"
        )

    lower: list[Position2D] = []
    for point in points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper: list[Position2D] = []
    for point in reversed(points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    hull = tuple(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        raise A1AttachmentProjectionError(
            "Spine mesh attachment physical positions are collinear"
        )
    return frozenset(hull)


def _remap_index_stream(
    values: Tuple[int, ...],
    mapping: dict[int, int],
    *,
    field_name: str,
    group_size: int,
) -> Tuple[int, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be tuple")
    if len(values) % group_size != 0:
        raise A1AttachmentProjectionError(
            f"{field_name} length must be divisible by {group_size}"
        )
    remapped: list[int] = []
    for value_index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field_name}[{value_index}] must be int")
        try:
            remapped.append(mapping[value])
        except KeyError as exc:
            raise A1AttachmentProjectionError(
                f"{field_name}[{value_index}] references unknown attachment index {value}"
            ) from exc
    return tuple(remapped)


def normalize_a1_attachment_projection_hull(
    projection: A1AttachmentProjectionResult,
) -> A1AttachmentProjectionResult:
    """Move the unique physical convex hull to the vertex prefix and remap indices."""

    if not isinstance(projection, A1AttachmentProjectionResult):
        raise TypeError("projection must be A1AttachmentProjectionResult")

    request = projection.request
    vertices = request.vertices
    convex_positions = _convex_hull_positions(vertices)

    # The raw projector places the complete topological boundary first. Filter that
    # stable cycle to the physical convex positions, retaining one UV representative
    # per position. UV duplicates and concave boundary vertices move to the tail.
    hull_indices: list[int] = []
    seen_positions: set[Position2D] = set()
    for old_index in range(request.hull):
        position = _position(vertices[old_index])
        if position not in convex_positions or position in seen_positions:
            continue
        seen_positions.add(position)
        hull_indices.append(old_index)

    missing_positions = convex_positions - seen_positions
    if missing_positions:
        raise A1AttachmentProjectionError(
            "Raw topological hull does not contain every physical convex-hull point; "
            f"missing={tuple(sorted(missing_positions))}"
        )
    if len(hull_indices) < 3:
        raise A1AttachmentProjectionError(
            "Normalized Spine convex hull must contain at least three vertices"
        )

    hull_index_set = set(hull_indices)
    old_order = tuple(range(len(vertices)))
    new_order = tuple(hull_indices) + tuple(
        old_index for old_index in old_order if old_index not in hull_index_set
    )
    old_to_new = {
        old_index: new_index for new_index, old_index in enumerate(new_order)
    }

    if new_order == old_order and request.hull == len(hull_indices):
        return projection

    normalized_vertices = tuple(
        replace(vertices[old_index], index=new_index)
        for new_index, old_index in enumerate(new_order)
    )
    normalized_triangles = _remap_index_stream(
        request.triangles,
        old_to_new,
        field_name="triangles",
        group_size=3,
    )
    normalized_edges = _remap_index_stream(
        request.edges,
        old_to_new,
        field_name="edges",
        group_size=2,
    )
    normalized_keys = tuple(
        projection.ordered_vertex_keys[old_index] for old_index in new_order
    )
    normalized_loop_mapping = tuple(
        (loop_id, old_to_new[attachment_index])
        for loop_id, attachment_index in projection.loop_to_attachment_index
    )

    normalized_request: LegacyMeshAttachmentRequest = replace(
        request,
        vertices=normalized_vertices,
        triangles=normalized_triangles,
        hull=len(hull_indices),
        edges=normalized_edges,
    )
    result = A1AttachmentProjectionResult(
        request=normalized_request,
        hull_vertex_keys=normalized_keys[: len(hull_indices)],
        ordered_vertex_keys=normalized_keys,
        loop_to_attachment_index=normalized_loop_mapping,
    )

    result_hull_positions = tuple(
        _position(vertex) for vertex in result.request.vertices[: result.request.hull]
    )
    if len(result_hull_positions) != len(set(result_hull_positions)):
        raise A1AttachmentProjectionError(
            "Normalized Spine hull still contains duplicate physical positions"
        )
    if set(result_hull_positions) != set(convex_positions):
        raise A1AttachmentProjectionError(
            "Normalized Spine hull does not match the physical convex hull"
        )
    return result


def project_triangulated_disk_attachment(
    snapshot: MeshSnapshot,
    rig: LegacyRigBuildResult,
    settings: A1AttachmentProjectionSettings,
) -> A1AttachmentProjectionResult:
    """Project loop-level UV identity, then enforce the Spine physical hull contract."""

    raw = _project_raw_attachment(snapshot, rig, settings)
    return normalize_a1_attachment_projection_hull(raw)


__all__ = [
    "A1AttachmentProjectionError",
    "A1AttachmentProjectionResult",
    "A1AttachmentProjectionSettings",
    "A1AttachmentVertexKey",
    "A1VertexZBinding",
    "normalize_a1_attachment_projection_hull",
    "project_triangulated_disk_attachment",
]
