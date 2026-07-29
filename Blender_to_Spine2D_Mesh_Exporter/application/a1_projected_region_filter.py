"""Cull faces that have no physical area in Spine XY projection space.

A Blender mesh may contain legitimate three-dimensional side walls.  Normal UV export
projects every attachment into Spine's two-dimensional X/Y plane, where those edge-on
faces become lines.  They must not be serialized as zero-area Spine triangles, but the
source mesh, UV layers, source-lineage IDs, and remaining visible faces must stay
immutable and exact.
"""

from __future__ import annotations

from collections import deque
from math import isfinite
from typing import Tuple

from ..domain.geometry import (
    FaceId,
    MeshSnapshot,
    MeshSnapshotValidator,
    analyse_face_region,
    build_edge_to_faces,
    build_face_adjacency,
    extract_face_subset,
    is_simple_disk,
)


Position2D = Tuple[float, float]
_RELATIVE_AREA_EPSILON = 1.0e-10
_MINIMUM_AREA_EPSILON = 1.0e-12


class A1ProjectedRegionFilterError(ValueError):
    """Raised when visible projected faces cannot form safe Spine disk regions."""


def _require_finite_number(
    value: object,
    field_name: str,
    *,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    if positive and resolved <= 0.0:
        raise ValueError(f"{field_name} must be greater than zero")
    return resolved


def _cross(first: Position2D, second: Position2D, third: Position2D) -> float:
    return (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _area_tolerance(points: Tuple[Position2D, ...]) -> float:
    if not isinstance(points, tuple) or not points:
        raise ValueError("points must be a non-empty tuple")
    x_values = tuple(point[0] for point in points)
    y_values = tuple(point[1] for point in points)
    extent = max(
        max(x_values) - min(x_values),
        max(y_values) - min(y_values),
        1.0,
    )
    return max(
        _MINIMUM_AREA_EPSILON,
        extent * extent * _RELATIVE_AREA_EPSILON,
    )


def _projected_positions(
    snapshot: MeshSnapshot,
    *,
    uniform_scale: float,
    center_x: float,
    center_y: float,
) -> dict:
    return {
        vertex.id: (
            (float(vertex.position[0]) - center_x) * uniform_scale,
            -(float(vertex.position[1]) - center_y) * uniform_scale,
        )
        for vertex in snapshot.vertices
    }


def _visible_face_ids(
    snapshot: MeshSnapshot,
    projected_positions: dict,
    *,
    tolerance: float,
) -> tuple[FaceId, ...]:
    loop_map = snapshot.loop_by_id()
    visible: list[FaceId] = []

    for face in sorted(snapshot.faces, key=lambda item: item.id.index):
        if len(face.loop_ids) != 3:
            raise A1ProjectedRegionFilterError(
                f"Projected visibility requires triangulated input; "
                f"face {face.id.index} has {len(face.loop_ids)} corners"
            )
        positions = tuple(
            projected_positions[loop_map[loop_id].vertex_id]
            for loop_id in face.loop_ids
        )
        area_twice = _cross(positions[0], positions[1], positions[2])
        if abs(area_twice) > tolerance:
            visible.append(face.id)

    return tuple(visible)


def _connected_face_components(
    snapshot: MeshSnapshot,
    face_ids: tuple[FaceId, ...],
) -> tuple[tuple[FaceId, ...], ...]:
    if not face_ids:
        return ()

    edge_to_faces = build_edge_to_faces(snapshot)
    adjacency = build_face_adjacency(
        snapshot,
        face_ids,
        edge_to_faces=edge_to_faces,
    )
    remaining = set(face_ids)
    components: list[tuple[FaceId, ...]] = []

    while remaining:
        seed = min(remaining, key=lambda item: item.index)
        remaining.remove(seed)
        queue = deque([seed])
        component: list[FaceId] = []

        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbour in adjacency[current]:
                if neighbour not in remaining:
                    continue
                remaining.remove(neighbour)
                queue.append(neighbour)

        components.append(tuple(sorted(component, key=lambda item: item.index)))

    return tuple(components)


def split_xy_visible_region_snapshots(
    snapshot: MeshSnapshot,
    *,
    uniform_scale: float,
    center_x: float,
    center_y: float,
) -> tuple[MeshSnapshot, ...]:
    """Return visible projected disk snapshots without mutating the source snapshot.

    A completely edge-on region returns an empty tuple.  Partially visible regions are
    compacted through exact ``Source*Id``-preserving subset extraction.  If culling
    disconnects a region, each connected visible disk becomes an independent snapshot.
    """

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    MeshSnapshotValidator().validate_or_raise(snapshot)
    resolved_scale = _require_finite_number(
        uniform_scale,
        "uniform_scale",
        positive=True,
    )
    resolved_center_x = _require_finite_number(center_x, "center_x")
    resolved_center_y = _require_finite_number(center_y, "center_y")

    projected_positions = _projected_positions(
        snapshot,
        uniform_scale=resolved_scale,
        center_x=resolved_center_x,
        center_y=resolved_center_y,
    )
    tolerance = _area_tolerance(tuple(projected_positions.values()))
    visible_face_ids = _visible_face_ids(
        snapshot,
        projected_positions,
        tolerance=tolerance,
    )
    if not visible_face_ids:
        return ()

    all_face_ids = tuple(
        face.id for face in sorted(snapshot.faces, key=lambda item: item.id.index)
    )
    if visible_face_ids == all_face_ids:
        return (snapshot,)

    components = _connected_face_components(snapshot, visible_face_ids)
    visible_snapshots: list[MeshSnapshot] = []
    for component_index, component_face_ids in enumerate(components):
        component = extract_face_subset(
            snapshot,
            component_face_ids,
            snapshot_id=(
                f"{snapshot.snapshot_id}:xy-visible-{component_index:03d}"
            ),
            object_name=(
                f"{snapshot.object_name}_XYVisible_{component_index:03d}"
            ),
        )
        topology = analyse_face_region(
            component,
            tuple(face.id for face in component.faces),
        )
        if not is_simple_disk(topology):
            raise A1ProjectedRegionFilterError(
                "Removing edge-on faces produced a visible component that is not a "
                f"manifold disk for snapshot {snapshot.snapshot_id!r}; "
                f"component={component_index}, Euler={topology.euler_characteristic}, "
                f"boundaries={topology.boundary_component_count}, "
                f"manifold={topology.manifold}"
            )
        visible_snapshots.append(component)

    return tuple(visible_snapshots)


__all__ = [
    "A1ProjectedRegionFilterError",
    "split_xy_visible_region_snapshots",
]
