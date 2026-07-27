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
from ..domain.spine import (
    LegacyAttachmentVertex,
    LegacyMeshAttachmentRequest,
    LegacyRigBuildResult,
)
from .a1_attachment_projection import (
    A1AttachmentProjectionError,
    A1AttachmentProjectionResult,
    A1AttachmentProjectionSettings,
    A1AttachmentVertexKey,
    A1VertexZBinding,
    project_triangulated_disk_attachment as _project_raw_attachment,
)
from .a1_material_correspondence import (
    Position2D,
    validate_projection_material_correspondence,
)


_RELATIVE_AREA_EPSILON = 1.0e-10
_MINIMUM_AREA_EPSILON = 1.0e-12


def _position(vertex: LegacyAttachmentVertex) -> Position2D:
    """Return the legacy local position used by compatibility-only unit fixtures."""

    if not isinstance(vertex, LegacyAttachmentVertex):
        raise TypeError("vertex must be LegacyAttachmentVertex")
    return (
        float(vertex.bone_position_pixels[0]),
        float(vertex.bone_position_pixels[1]),
    )


def _resolved_setup_positions(
    projection: A1AttachmentProjectionResult,
    rig: LegacyRigBuildResult | None,
) -> Tuple[Position2D, ...]:
    if rig is None:
        return tuple(_position(vertex) for vertex in projection.request.vertices)
    return validate_projection_material_correspondence(projection, rig)


def _cross(first: Position2D, second: Position2D, third: Position2D) -> float:
    return (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _area_tolerance(points: Tuple[Position2D, ...]) -> float:
    """Return a scale-aware tolerance for twice-signed pixel-space areas."""

    if not isinstance(points, tuple) or not points:
        raise ValueError("points must be a non-empty tuple")
    if not all(
        isinstance(point, tuple)
        and len(point) == 2
        and all(isinstance(component, float) for component in point)
        for point in points
    ):
        raise TypeError("points must contain float coordinate pairs")

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


def _convex_hull_position_cycle(
    positions: Tuple[Position2D, ...],
) -> Tuple[Position2D, ...]:
    """Return the deterministic setup-pose convex-hull cycle."""

    if not isinstance(positions, tuple) or not positions:
        raise ValueError("positions must be a non-empty tuple")
    if not all(
        isinstance(position, tuple)
        and len(position) == 2
        and all(isinstance(component, float) for component in position)
        for position in positions
    ):
        raise TypeError("positions must contain float coordinate pairs")

    points = tuple(sorted(set(positions)))
    if len(points) < 3:
        raise A1AttachmentProjectionError(
            "Spine mesh attachment requires at least three unique setup-pose positions"
        )
    tolerance = _area_tolerance(points)

    lower: list[Position2D] = []
    for point in points:
        while (
            len(lower) >= 2
            and _cross(lower[-2], lower[-1], point) <= tolerance
        ):
            lower.pop()
        lower.append(point)

    upper: list[Position2D] = []
    for point in reversed(points):
        while (
            len(upper) >= 2
            and _cross(upper[-2], upper[-1], point) <= tolerance
        ):
            upper.pop()
        upper.append(point)

    hull = tuple(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        raise A1AttachmentProjectionError(
            "Spine mesh attachment setup-pose positions are collinear within "
            f"pixel-space tolerance {tolerance}"
        )
    if len(hull) != len(set(hull)):
        raise A1AttachmentProjectionError(
            "Setup-pose convex-hull cycle contains duplicate positions"
        )
    return hull


def _rotate_cycle(
    cycle: Tuple[Position2D, ...],
    start: Position2D,
) -> Tuple[Position2D, ...]:
    """Rotate one closed-cycle representation without changing its orientation."""

    if not isinstance(cycle, tuple) or not cycle:
        raise ValueError("cycle must be a non-empty tuple")
    try:
        start_index = cycle.index(start)
    except ValueError as exc:
        raise A1AttachmentProjectionError(
            f"Physical hull cycle does not contain requested start position {start}"
        ) from exc
    return cycle[start_index:] + cycle[:start_index]


def _order_inversion_count(
    candidate: Tuple[Position2D, ...],
    observed: Tuple[Position2D, ...],
) -> int:
    """Count pairwise order inversions for one cycle anchored at observed[0]."""

    rank = {position: index for index, position in enumerate(candidate)}
    try:
        observed_ranks = tuple(rank[position] for position in observed)
    except KeyError as exc:
        raise A1AttachmentProjectionError(
            "Observed topological hull contains a non-physical hull position"
        ) from exc
    return sum(
        first > second
        for first_index, first in enumerate(observed_ranks)
        for second in observed_ranks[first_index + 1 :]
    )


def _ordered_physical_hull_positions(
    positions: Tuple[Position2D, ...],
    topological_hull_count: int,
) -> Tuple[Position2D, ...]:
    """Align the setup-pose hull to the stable raw boundary whenever possible.

    The raw projector stores the topological boundary first. That order is preserved
    exactly when it already covers the complete setup-pose convex hull. When a
    topologically interior vertex becomes physically extreme after the Z-group parent
    translation is applied, the monotone-chain cycle supplies the missing point and
    the observed boundary order is used only to choose deterministic orientation.
    """

    if isinstance(topological_hull_count, bool) or not isinstance(
        topological_hull_count, int
    ):
        raise TypeError("topological_hull_count must be int")
    if topological_hull_count < 0 or topological_hull_count > len(positions):
        raise ValueError("topological_hull_count is outside the vertex range")

    physical_cycle = _convex_hull_position_cycle(positions)
    physical_positions = set(physical_cycle)
    observed: list[Position2D] = []
    seen: set[Position2D] = set()
    for old_index in range(topological_hull_count):
        position = positions[old_index]
        if position not in physical_positions or position in seen:
            continue
        seen.add(position)
        observed.append(position)

    observed_cycle = tuple(observed)
    if seen == physical_positions:
        return observed_cycle
    if not observed_cycle:
        return physical_cycle

    anchor = observed_cycle[0]
    forward = _rotate_cycle(physical_cycle, anchor)
    if len(observed_cycle) == 1:
        return forward

    reverse = _rotate_cycle(tuple(reversed(physical_cycle)), anchor)
    forward_inversions = _order_inversion_count(forward, observed_cycle)
    reverse_inversions = _order_inversion_count(reverse, observed_cycle)
    if forward_inversions < reverse_inversions:
        return forward
    if reverse_inversions < forward_inversions:
        return reverse

    return min(forward, reverse)


def _select_physical_hull_indices(
    positions: Tuple[Position2D, ...],
    topological_hull_count: int,
    ordered_positions: Tuple[Position2D, ...],
) -> Tuple[int, ...]:
    """Select one deterministic UV representative for every setup-pose hull point."""

    indices_by_position: dict[Position2D, list[int]] = {}
    for old_index, position in enumerate(positions):
        indices_by_position.setdefault(position, []).append(old_index)

    selected: list[int] = []
    for position in ordered_positions:
        candidates = indices_by_position.get(position)
        if not candidates:
            raise A1AttachmentProjectionError(
                f"Physical hull position {position} has no attachment vertex"
            )
        topological_candidates = tuple(
            index for index in candidates if index < topological_hull_count
        )
        selected.append(
            topological_candidates[0] if topological_candidates else candidates[0]
        )

    resolved = tuple(selected)
    if len(resolved) != len(set(resolved)):
        raise A1AttachmentProjectionError(
            "Physical hull representative selection contains duplicate indices"
        )
    return resolved


def _validate_projected_triangles(
    positions: Tuple[Position2D, ...],
    triangles: Tuple[int, ...],
) -> None:
    """Reject triangles that collapse in the effective Spine setup pose."""

    if not isinstance(positions, tuple) or not positions:
        raise ValueError("positions must be a non-empty tuple")
    if not isinstance(triangles, tuple) or not triangles:
        raise ValueError("triangles must be a non-empty tuple")
    if len(triangles) % 3 != 0:
        raise A1AttachmentProjectionError(
            "triangles must contain complete index triples"
        )

    tolerance = _area_tolerance(positions)
    for triangle_index in range(0, len(triangles), 3):
        indices = triangles[triangle_index : triangle_index + 3]
        try:
            first, second, third = (
                positions[vertex_index] for vertex_index in indices
            )
        except IndexError as exc:
            raise A1AttachmentProjectionError(
                f"Triangle {triangle_index // 3} references an unknown attachment vertex"
            ) from exc
        area_twice = _cross(first, second, third)
        if abs(area_twice) <= tolerance:
            raise A1AttachmentProjectionError(
                f"Triangle {triangle_index // 3} collapses within Spine setup-pose "
                f"area tolerance {tolerance}; indices={indices}, "
                f"positions={(first, second, third)}, twice_area={area_twice}"
            )


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
    rig: LegacyRigBuildResult | None = None,
) -> A1AttachmentProjectionResult:
    """Move the complete setup-pose convex hull to the prefix and remap all indices.

    ``rig`` is optional only for compatibility with low-level fixtures that construct
    already-resolved pixel positions. Production callers always provide it so Z-group
    parent translations participate in area and physical-hull calculations.
    """

    if not isinstance(projection, A1AttachmentProjectionResult):
        raise TypeError("projection must be A1AttachmentProjectionResult")
    if rig is not None and not isinstance(rig, LegacyRigBuildResult):
        raise TypeError("rig must be LegacyRigBuildResult or None")

    request = projection.request
    vertices = request.vertices
    setup_positions = _resolved_setup_positions(projection, rig)
    _validate_projected_triangles(setup_positions, request.triangles)

    ordered_hull_positions = _ordered_physical_hull_positions(
        setup_positions,
        request.hull,
    )
    hull_indices = _select_physical_hull_indices(
        setup_positions,
        request.hull,
        ordered_hull_positions,
    )
    if len(hull_indices) < 3:
        raise A1AttachmentProjectionError(
            "Normalized Spine convex hull must contain at least three vertices"
        )

    hull_index_set = set(hull_indices)
    old_order = tuple(range(len(vertices)))
    new_order = hull_indices + tuple(
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
    normalized_positions = tuple(setup_positions[old_index] for old_index in new_order)
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

    result_hull_positions = normalized_positions[: result.request.hull]
    if len(result_hull_positions) != len(set(result_hull_positions)):
        raise A1AttachmentProjectionError(
            "Normalized Spine hull still contains duplicate setup-pose positions"
        )
    if result_hull_positions != ordered_hull_positions:
        raise A1AttachmentProjectionError(
            "Normalized Spine hull order does not match the setup-pose convex-hull cycle"
        )
    _validate_projected_triangles(normalized_positions, result.request.triangles)
    if rig is not None:
        validate_projection_material_correspondence(result, rig)
    return result


def project_triangulated_disk_attachment(
    snapshot: MeshSnapshot,
    rig: LegacyRigBuildResult,
    settings: A1AttachmentProjectionSettings,
) -> A1AttachmentProjectionResult:
    """Project exact loop UV identity, then enforce the setup-pose hull contract."""

    raw = _project_raw_attachment(snapshot, rig, settings)
    return normalize_a1_attachment_projection_hull(raw, rig=rig)


__all__ = [
    "A1AttachmentProjectionError",
    "A1AttachmentProjectionResult",
    "A1AttachmentProjectionSettings",
    "A1AttachmentVertexKey",
    "A1VertexZBinding",
    "normalize_a1_attachment_projection_hull",
    "project_triangulated_disk_attachment",
]
