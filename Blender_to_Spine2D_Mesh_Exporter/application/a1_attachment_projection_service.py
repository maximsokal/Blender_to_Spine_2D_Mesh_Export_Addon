"""Production A1 attachment projection with physical Spine hull normalization.

The loop-level projector owns UV-split identity and topological boundary traversal.
Spine's ``hull`` field has a different responsibility: it counts the vertices of
the physical convex hull, stored as the prefix of the final attachment vertex order.
This service keeps those concerns separate and remaps every dependent index exactly
once after the raw projection has been built.

Normal / UV Segments rigs are deformable. A valid three-dimensional side surface can
therefore collapse to a line only in the selected Setup Pose and become visible again
when the generated X/Y controls rotate its vertex bones. Such setup-degenerate triangles
must remain in the attachment. Rigid ``PREPROJECTED_SCREEN`` geometry has no later
per-depth deformation that can restore them, so that route retains strict physical-area
validation.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

from ..domain.geometry import MeshSnapshot
from ..domain.spine import (
    A1RigSetupPoseMode,
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


Position2D = Tuple[float, float]
_RELATIVE_AREA_EPSILON = 1.0e-10
_MINIMUM_AREA_EPSILON = 1.0e-12
_DEFORMABLE_SETUP_MODES = frozenset(
    {
        A1RigSetupPoseMode.NORMALIZED_SINGLE,
        A1RigSetupPoseMode.PRESERVE_COMPOSITION,
        A1RigSetupPoseMode.CAMERA_VIEW_NORMAL,
        A1RigSetupPoseMode.CAMERA_DEPTH_SURFACE,
    }
)


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
    vertices: Tuple[LegacyAttachmentVertex, ...],
) -> Tuple[Position2D, ...]:
    """Return the deterministic physical convex-hull cycle using monotone chain."""

    if not isinstance(vertices, tuple) or not vertices:
        raise ValueError("vertices must be a non-empty tuple")
    if not all(isinstance(vertex, LegacyAttachmentVertex) for vertex in vertices):
        raise TypeError("vertices must contain LegacyAttachmentVertex values")

    points = tuple(sorted({_position(vertex) for vertex in vertices}))
    if len(points) < 3:
        raise A1AttachmentProjectionError(
            "Spine mesh attachment requires at least three unique physical positions"
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
            "Spine mesh attachment physical positions are collinear within "
            f"pixel-space tolerance {tolerance}"
        )
    if len(hull) != len(set(hull)):
        raise A1AttachmentProjectionError(
            "Physical convex-hull cycle contains duplicate positions"
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
    vertices: Tuple[LegacyAttachmentVertex, ...],
    topological_hull_count: int,
) -> Tuple[Position2D, ...]:
    """Align the physical hull to the stable raw boundary whenever possible.

    The raw projector stores the topological boundary first. That order is preserved
    exactly when it already covers the complete physical convex hull. When a
    topologically interior vertex becomes physically extreme after XY projection, the
    monotone-chain cycle supplies the missing point and the observed boundary order is
    used only to choose the deterministic orientation and rotation.
    """

    if isinstance(topological_hull_count, bool) or not isinstance(
        topological_hull_count, int
    ):
        raise TypeError("topological_hull_count must be int")
    if topological_hull_count < 0 or topological_hull_count > len(vertices):
        raise ValueError("topological_hull_count is outside the vertex range")

    physical_cycle = _convex_hull_position_cycle(vertices)
    physical_positions = set(physical_cycle)
    observed: list[Position2D] = []
    seen: set[Position2D] = set()
    for old_index in range(topological_hull_count):
        position = _position(vertices[old_index])
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
    vertices: Tuple[LegacyAttachmentVertex, ...],
    topological_hull_count: int,
    ordered_positions: Tuple[Position2D, ...],
) -> Tuple[int, ...]:
    """Select one deterministic UV representative for every physical hull point."""

    indices_by_position: dict[Position2D, list[int]] = {}
    for old_index, vertex in enumerate(vertices):
        indices_by_position.setdefault(_position(vertex), []).append(old_index)

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


def _projected_triangle_area_report(
    vertices: Tuple[LegacyAttachmentVertex, ...],
    triangles: Tuple[int, ...],
) -> tuple[float, Tuple[int, ...]]:
    """Validate index access and return setup-collapsed triangle indices."""

    if not isinstance(vertices, tuple) or not vertices:
        raise ValueError("vertices must be a non-empty tuple")
    if not all(isinstance(vertex, LegacyAttachmentVertex) for vertex in vertices):
        raise TypeError("vertices must contain LegacyAttachmentVertex values")
    if not isinstance(triangles, tuple) or not triangles:
        raise ValueError("triangles must be a non-empty tuple")
    if len(triangles) % 3 != 0:
        raise A1AttachmentProjectionError(
            "triangles must contain complete index triples"
        )

    all_positions = tuple(_position(vertex) for vertex in vertices)
    tolerance = _area_tolerance(all_positions)
    collapsed: list[int] = []
    for offset in range(0, len(triangles), 3):
        indices = triangles[offset : offset + 3]
        try:
            first, second, third = (
                all_positions[vertex_index] for vertex_index in indices
            )
        except IndexError as exc:
            raise A1AttachmentProjectionError(
                f"Triangle {offset // 3} references an unknown attachment vertex"
            ) from exc
        if abs(_cross(first, second, third)) <= tolerance:
            collapsed.append(offset // 3)
    return tolerance, tuple(collapsed)


def _raise_first_collapsed_triangle(
    vertices: Tuple[LegacyAttachmentVertex, ...],
    triangles: Tuple[int, ...],
    *,
    tolerance: float,
    triangle_index: int,
) -> None:
    """Raise the historical strict physical-area diagnostic for one triangle."""

    offset = triangle_index * 3
    indices = triangles[offset : offset + 3]
    positions = tuple(_position(vertices[index]) for index in indices)
    area_twice = _cross(positions[0], positions[1], positions[2])
    raise A1AttachmentProjectionError(
        f"Triangle {triangle_index} collapses within Spine pixel-space "
        f"area tolerance {tolerance}; indices={indices}, "
        f"positions={positions}, twice_area={area_twice}"
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
    *,
    allow_setup_degenerate: bool = False,
) -> A1AttachmentProjectionResult:
    """Normalize a physical hull without deleting deformable setup geometry.

    Strict callers reject every triangle that has no physical area in the current XY
    pose. Normal / UV Segments passes ``allow_setup_degenerate=True`` because its
    per-depth vertex bones can make those triangles visible after control rotation.

    When every triangle is setup-degenerate, no two-dimensional physical hull exists.
    The raw projector's deterministic topological boundary remains the only meaningful
    hull ordering and is returned unchanged. Mixed visible/edge-on regions still receive
    normal physical-hull normalization while retaining all triangle groups.
    """

    if not isinstance(projection, A1AttachmentProjectionResult):
        raise TypeError("projection must be A1AttachmentProjectionResult")
    if not isinstance(allow_setup_degenerate, bool):
        raise TypeError("allow_setup_degenerate must be bool")

    request = projection.request
    vertices = request.vertices
    tolerance, collapsed_triangles = _projected_triangle_area_report(
        vertices,
        request.triangles,
    )
    if collapsed_triangles and not allow_setup_degenerate:
        _raise_first_collapsed_triangle(
            vertices,
            request.triangles,
            tolerance=tolerance,
            triangle_index=collapsed_triangles[0],
        )

    triangle_count = len(request.triangles) // 3
    if allow_setup_degenerate and len(collapsed_triangles) == triangle_count:
        return projection

    ordered_hull_positions = _ordered_physical_hull_positions(
        vertices,
        request.hull,
    )
    hull_indices = _select_physical_hull_indices(
        vertices,
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
    if result_hull_positions != ordered_hull_positions:
        raise A1AttachmentProjectionError(
            "Normalized Spine hull order does not match the physical convex-hull cycle"
        )

    _, normalized_collapsed = _projected_triangle_area_report(
        result.request.vertices,
        result.request.triangles,
    )
    if normalized_collapsed != collapsed_triangles:
        raise A1AttachmentProjectionError(
            "Physical-hull remapping changed setup-degenerate triangle ownership; "
            f"before={collapsed_triangles}, after={normalized_collapsed}"
        )
    return result


def project_triangulated_disk_attachment(
    snapshot: MeshSnapshot,
    rig: LegacyRigBuildResult,
    settings: A1AttachmentProjectionSettings,
) -> A1AttachmentProjectionResult:
    """Project one disk and preserve valid edge-on Normal setup triangles."""

    if not isinstance(rig, LegacyRigBuildResult):
        raise TypeError("rig must be LegacyRigBuildResult")
    raw = _project_raw_attachment(snapshot, rig, settings)
    allow_setup_degenerate = (
        rig.request.setup_pose_mode in _DEFORMABLE_SETUP_MODES
    )
    return normalize_a1_attachment_projection_hull(
        raw,
        allow_setup_degenerate=allow_setup_degenerate,
    )


__all__ = [
    "A1AttachmentProjectionError",
    "A1AttachmentProjectionResult",
    "A1AttachmentProjectionSettings",
    "A1AttachmentVertexKey",
    "A1VertexZBinding",
    "normalize_a1_attachment_projection_hull",
    "project_triangulated_disk_attachment",
]
