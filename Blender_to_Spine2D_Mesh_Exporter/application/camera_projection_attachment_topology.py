"""Normalize camera-projection contours into Spine mesh attachment topology.

``CameraProjectionLayout.hull`` is a compatibility field name and may contain a
simple concave contour. Spine's mesh ``hull`` value has a stricter meaning: it is
the number of physical convex-hull vertices stored at the beginning of the vertex
array. This module preserves the complete contour triangulation while moving only
those convex boundary vertices to the required prefix and remapping every topology
index exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..domain.baking import (
    CameraProjectionLayout,
    ProjectionPixelPoint,
    ProjectionTriangle,
    convex_hull,
)


class CameraProjectionAttachmentTopologyError(ValueError):
    """Raised when a camera contour cannot become a valid Spine mesh topology."""


@dataclass(frozen=True, slots=True)
class CameraProjectionAttachmentTopology:
    """One deterministic Spine-ready ordering of a camera projection contour."""

    points: Tuple[ProjectionPixelPoint, ...]
    triangles: Tuple[ProjectionTriangle, ...]
    edges: Tuple[int, ...]
    hull_count: int
    source_indices: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.points, tuple) or len(self.points) < 3:
            raise ValueError("points must contain at least three contour points")
        if not all(isinstance(point, ProjectionPixelPoint) for point in self.points):
            raise TypeError("points must contain ProjectionPixelPoint values")
        if len(self.points) != len(set(self.points)):
            raise ValueError("points cannot contain duplicates")
        if not isinstance(self.triangles, tuple) or not self.triangles:
            raise ValueError("triangles must be a non-empty tuple")
        if not all(
            isinstance(triangle, tuple)
            and len(triangle) == 3
            and all(
                isinstance(index, int)
                and not isinstance(index, bool)
                and 0 <= index < len(self.points)
                for index in triangle
            )
            and len(set(triangle)) == 3
            for triangle in self.triangles
        ):
            raise ValueError("triangles must contain valid non-degenerate index triples")
        if not isinstance(self.edges, tuple) or len(self.edges) % 2 != 0:
            raise ValueError("edges must be a flat tuple of complete index pairs")
        if any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(self.points)
            for index in self.edges
        ):
            raise ValueError("edges reference an invalid attachment vertex")
        for offset in range(0, len(self.edges), 2):
            if self.edges[offset] == self.edges[offset + 1]:
                raise ValueError("edges cannot repeat one attachment vertex")
        if (
            not isinstance(self.hull_count, int)
            or isinstance(self.hull_count, bool)
            or self.hull_count < 3
            or self.hull_count > len(self.points)
        ):
            raise ValueError("hull_count must be between 3 and the vertex count")
        if (
            not isinstance(self.source_indices, tuple)
            or len(self.source_indices) != len(self.points)
            or set(self.source_indices) != set(range(len(self.points)))
        ):
            raise ValueError("source_indices must be a complete source permutation")


def _normalized_pair(first: int, second: int) -> tuple[int, int]:
    if first == second:
        raise CameraProjectionAttachmentTopologyError(
            f"camera projection edge repeats vertex {first}"
        )
    return (first, second) if first < second else (second, first)


def _validate_triangle_geometry(
    points: Tuple[ProjectionPixelPoint, ...],
    triangles: Tuple[ProjectionTriangle, ...],
) -> None:
    for triangle_index, (first, second, third) in enumerate(triangles):
        first_point = points[first]
        second_point = points[second]
        third_point = points[third]
        twice_area = (
            (second_point.x - first_point.x)
            * (third_point.y - first_point.y)
            - (second_point.y - first_point.y)
            * (third_point.x - first_point.x)
        )
        if twice_area == 0:
            raise CameraProjectionAttachmentTopologyError(
                f"camera projection triangle {triangle_index} has zero screen-space "
                f"area; indices={(first, second, third)}"
            )


def _source_topology_edges(
    vertex_count: int,
    triangles: Tuple[ProjectionTriangle, ...],
) -> Tuple[tuple[int, int], ...]:
    """Return original contour edges first and deterministic internal edges after."""

    boundary = tuple(
        (index, (index + 1) % vertex_count) for index in range(vertex_count)
    )
    resolved: list[tuple[int, int]] = list(boundary)
    seen = {_normalized_pair(first, second) for first, second in boundary}

    for triangle in triangles:
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            pair = _normalized_pair(first, second)
            if pair in seen:
                continue
            seen.add(pair)
            resolved.append((first, second))

    expected_edge_count = vertex_count + max(0, vertex_count - 3)
    if len(resolved) != expected_edge_count:
        raise CameraProjectionAttachmentTopologyError(
            "camera projection triangulation produced an unexpected unique edge "
            f"count; expected={expected_edge_count}, actual={len(resolved)}"
        )
    return tuple(resolved)


def build_camera_projection_attachment_topology(
    layout: CameraProjectionLayout,
) -> CameraProjectionAttachmentTopology:
    """Return a Spine-ready convex-hull prefix without changing rendered geometry."""

    if not isinstance(layout, CameraProjectionLayout):
        raise TypeError("layout must be CameraProjectionLayout")

    source_points = layout.contour
    source_triangles = layout.triangle_indices
    if len(source_triangles) != len(source_points) - 2:
        raise CameraProjectionAttachmentTopologyError(
            "camera projection contour triangulation must contain vertex_count - 2 "
            f"triangles; vertices={len(source_points)}, triangles={len(source_triangles)}"
        )
    _validate_triangle_geometry(source_points, source_triangles)

    convex_points = frozenset(convex_hull(source_points))
    hull_source_indices = tuple(
        index for index, point in enumerate(source_points) if point in convex_points
    )
    if len(hull_source_indices) != len(convex_points):
        raise CameraProjectionAttachmentTopologyError(
            "camera projection contour does not contain every computed convex-hull "
            "point exactly once"
        )

    hull_source_set = set(hull_source_indices)
    source_indices = hull_source_indices + tuple(
        index
        for index in range(len(source_points))
        if index not in hull_source_set
    )
    old_to_new = {
        source_index: new_index
        for new_index, source_index in enumerate(source_indices)
    }

    points = tuple(source_points[source_index] for source_index in source_indices)
    triangles = tuple(
        tuple(old_to_new[index] for index in triangle)
        for triangle in source_triangles
    )
    source_edges = _source_topology_edges(len(source_points), source_triangles)
    edges = tuple(
        remapped
        for first, second in source_edges
        for remapped in (old_to_new[first], old_to_new[second])
    )

    result = CameraProjectionAttachmentTopology(
        points=points,
        triangles=triangles,
        edges=edges,
        hull_count=len(hull_source_indices),
        source_indices=source_indices,
    )
    result_hull = result.points[: result.hull_count]
    if set(result_hull) != set(convex_points):
        raise CameraProjectionAttachmentTopologyError(
            "Spine camera attachment hull prefix does not match the physical convex hull"
        )
    _validate_triangle_geometry(result.points, result.triangles)
    return result


__all__ = [
    "CameraProjectionAttachmentTopology",
    "CameraProjectionAttachmentTopologyError",
    "build_camera_projection_attachment_topology",
]
