"""Deterministic screen-space contour extraction, simplification, and triangulation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import hypot, isfinite
from typing import Iterable, Tuple


ProjectionTriangle = Tuple[int, int, int]


class ProjectionContourError(ValueError):
    """Raised when a binary alpha boundary cannot become a valid contour."""


class ProjectionContourMode(str, Enum):
    """How a binary alpha union becomes one Spine-compatible outer boundary."""

    CONVEX_HULL = "CONVEX_HULL"
    SIMPLIFIED_CONCAVE = "SIMPLIFIED_CONCAVE"


@dataclass(frozen=True, order=True, slots=True)
class ProjectionPixelPoint:
    """One full-frame pixel-boundary coordinate with a bottom-left origin."""

    x: int
    y: int

    def __post_init__(self) -> None:
        if not isinstance(self.x, int) or not isinstance(self.y, int):
            raise TypeError("ProjectionPixelPoint coordinates must be int")
        if self.x < 0 or self.y < 0:
            raise ValueError("ProjectionPixelPoint coordinates cannot be negative")


@dataclass(frozen=True, slots=True)
class ProjectionContourResult:
    points: Tuple[ProjectionPixelPoint, ...]
    mode: ProjectionContourMode
    source_vertex_count: int
    outer_component_count: int
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        validate_simple_contour(self.points)
        if not isinstance(self.mode, ProjectionContourMode):
            raise TypeError("mode must be ProjectionContourMode")
        if (
            not isinstance(self.source_vertex_count, int)
            or self.source_vertex_count < len(self.points)
        ):
            raise ValueError(
                "source_vertex_count must be an integer not smaller than points"
            )
        if not isinstance(self.outer_component_count, int) or self.outer_component_count <= 0:
            raise ValueError("outer_component_count must be a positive integer")
        if self.fallback_reason is not None and (
            not isinstance(self.fallback_reason, str)
            or not self.fallback_reason.strip()
        ):
            raise ValueError("fallback_reason must be non-empty str or None")


def cross(
    origin: ProjectionPixelPoint,
    first: ProjectionPixelPoint,
    second: ProjectionPixelPoint,
) -> int:
    return (first.x - origin.x) * (second.y - origin.y) - (
        first.y - origin.y
    ) * (second.x - origin.x)


def signed_double_area(points: Tuple[ProjectionPixelPoint, ...]) -> int:
    return sum(
        first.x * second.y - second.x * first.y
        for first, second in zip(points, points[1:] + points[:1])
    )


def _orientation(
    first: ProjectionPixelPoint,
    second: ProjectionPixelPoint,
    third: ProjectionPixelPoint,
) -> int:
    value = cross(first, second, third)
    return 1 if value > 0 else -1 if value < 0 else 0


def _point_on_segment(
    point: ProjectionPixelPoint,
    first: ProjectionPixelPoint,
    second: ProjectionPixelPoint,
) -> bool:
    return (
        _orientation(first, second, point) == 0
        and min(first.x, second.x) <= point.x <= max(first.x, second.x)
        and min(first.y, second.y) <= point.y <= max(first.y, second.y)
    )


def _segments_intersect(
    first_start: ProjectionPixelPoint,
    first_end: ProjectionPixelPoint,
    second_start: ProjectionPixelPoint,
    second_end: ProjectionPixelPoint,
) -> bool:
    orientations = (
        _orientation(first_start, first_end, second_start),
        _orientation(first_start, first_end, second_end),
        _orientation(second_start, second_end, first_start),
        _orientation(second_start, second_end, first_end),
    )
    if orientations[0] != orientations[1] and orientations[2] != orientations[3]:
        return True
    return any(
        orientation == 0 and _point_on_segment(point, segment_start, segment_end)
        for orientation, point, segment_start, segment_end in (
            (orientations[0], second_start, first_start, first_end),
            (orientations[1], second_end, first_start, first_end),
            (orientations[2], first_start, second_start, second_end),
            (orientations[3], first_end, second_start, second_end),
        )
    )


def validate_simple_contour(
    points: Tuple[ProjectionPixelPoint, ...],
) -> None:
    """Reject duplicate, clockwise, collinear, or self-intersecting contours."""

    if not isinstance(points, tuple) or len(points) < 3:
        raise ProjectionContourError("contour must contain at least three points")
    if not all(isinstance(point, ProjectionPixelPoint) for point in points):
        raise TypeError("contour must contain ProjectionPixelPoint values")
    if len(points) != len(set(points)):
        raise ProjectionContourError("contour cannot contain duplicate points")
    if signed_double_area(points) <= 0:
        raise ProjectionContourError(
            "contour must be counter-clockwise and non-degenerate"
        )

    count = len(points)
    for index, point in enumerate(points):
        if cross(points[index - 1], point, points[(index + 1) % count]) == 0:
            raise ProjectionContourError(
                "contour cannot contain collinear consecutive points"
            )

    for first_index in range(count):
        first_next = (first_index + 1) % count
        for second_index in range(first_index + 1, count):
            second_next = (second_index + 1) % count
            if (
                first_index == second_index
                or first_next == second_index
                or second_next == first_index
                or (first_index == 0 and second_next == 0)
            ):
                continue
            if _segments_intersect(
                points[first_index],
                points[first_next],
                points[second_index],
                points[second_next],
            ):
                raise ProjectionContourError(
                    "contour is self-intersecting between edges "
                    f"{first_index}-{first_next} and {second_index}-{second_next}"
                )


def _monotonic_convex_hull(
    points: Iterable[ProjectionPixelPoint],
) -> Tuple[ProjectionPixelPoint, ...]:
    unique = tuple(sorted(set(points)))
    if len(unique) < 3:
        return unique
    lower: list[ProjectionPixelPoint] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[ProjectionPixelPoint] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])


def convex_hull(
    points: Iterable[ProjectionPixelPoint],
) -> Tuple[ProjectionPixelPoint, ...]:
    """Return a deterministic strict CCW convex hull."""

    result = _monotonic_convex_hull(points)
    if len(result) < 3 or signed_double_area(result) <= 0:
        raise ProjectionContourError(
            "at least three non-collinear pixel-boundary points are required"
        )
    return result


def _validate_strict_convex_hull(
    points: Tuple[ProjectionPixelPoint, ...],
) -> None:
    validate_simple_contour(points)
    if any(
        cross(points[index - 1], point, points[(index + 1) % len(points)]) <= 0
        for index, point in enumerate(points)
    ):
        raise ProjectionContourError("hull must be strictly convex")


def _point_in_or_on_triangle(
    point: ProjectionPixelPoint,
    first: ProjectionPixelPoint,
    second: ProjectionPixelPoint,
    third: ProjectionPixelPoint,
) -> bool:
    return all(
        value >= 0
        for value in (
            cross(first, second, point),
            cross(second, third, point),
            cross(third, first, point),
        )
    )


def _validate_triangulation(
    points: Tuple[ProjectionPixelPoint, ...],
    triangles: Tuple[ProjectionTriangle, ...],
) -> None:
    if len(triangles) != len(points) - 2:
        raise ProjectionContourError(
            "triangulation must contain exactly vertex_count - 2 triangles"
        )
    areas = tuple(
        cross(points[first], points[second], points[third])
        for first, second, third in triangles
    )
    if any(area <= 0 for area in areas):
        raise ProjectionContourError(
            "triangulation contains a clockwise or degenerate triangle; "
            f"triangle_areas2={areas}"
        )
    polygon_area = signed_double_area(points)
    if sum(areas) != polygon_area:
        raise ProjectionContourError(
            "triangulation does not cover the contour exactly; "
            f"polygon_area2={polygon_area}, triangles_area2={sum(areas)}"
        )


def triangulate_convex_hull(
    points: Tuple[ProjectionPixelPoint, ...],
) -> Tuple[ProjectionTriangle, ...]:
    """Return the historical deterministic fan for a strict convex hull."""

    _validate_strict_convex_hull(points)
    triangles = tuple((0, index, index + 1) for index in range(1, len(points) - 1))
    _validate_triangulation(points, triangles)
    return triangles


def triangulate_simple_contour(
    points: Tuple[ProjectionPixelPoint, ...],
) -> Tuple[ProjectionTriangle, ...]:
    """Triangulate one simple CCW convex or concave contour."""

    validate_simple_contour(points)
    if all(
        cross(points[index - 1], point, points[(index + 1) % len(points)]) > 0
        for index, point in enumerate(points)
    ):
        return triangulate_convex_hull(points)

    remaining = list(range(len(points)))
    triangles: list[ProjectionTriangle] = []
    while len(remaining) > 3:
        ear_position: int | None = None
        for position, current in enumerate(remaining):
            previous = remaining[position - 1]
            following = remaining[(position + 1) % len(remaining)]
            if cross(points[previous], points[current], points[following]) <= 0:
                continue
            if any(
                _point_in_or_on_triangle(
                    points[candidate],
                    points[previous],
                    points[current],
                    points[following],
                )
                for candidate in remaining
                if candidate not in {previous, current, following}
            ):
                continue
            triangles.append((previous, current, following))
            ear_position = position
            break
        if ear_position is None:
            raise ProjectionContourError(
                "ear clipping found no valid contour ear"
            )
        del remaining[ear_position]

    final = tuple(remaining)
    if cross(points[final[0]], points[final[1]], points[final[2]]) <= 0:
        raise ProjectionContourError("final contour triangle is degenerate")
    triangles.append((final[0], final[1], final[2]))
    result = tuple(triangles)
    _validate_triangulation(points, result)
    return result


def _visible(mask: bytearray, width: int, height: int, x: int, y: int) -> bool:
    return 0 <= x < width and 0 <= y < height and bool(mask[y * width + x])


def _boundary_edges(
    mask: bytearray,
    *,
    width: int,
    height: int,
) -> Tuple[tuple[ProjectionPixelPoint, ProjectionPixelPoint], ...]:
    edges: list[tuple[ProjectionPixelPoint, ProjectionPixelPoint]] = []
    for y in range(height):
        for x in range(width):
            if not _visible(mask, width, height, x, y):
                continue
            if not _visible(mask, width, height, x, y - 1):
                edges.append((ProjectionPixelPoint(x, y), ProjectionPixelPoint(x + 1, y)))
            if not _visible(mask, width, height, x + 1, y):
                edges.append(
                    (
                        ProjectionPixelPoint(x + 1, y),
                        ProjectionPixelPoint(x + 1, y + 1),
                    )
                )
            if not _visible(mask, width, height, x, y + 1):
                edges.append(
                    (
                        ProjectionPixelPoint(x + 1, y + 1),
                        ProjectionPixelPoint(x, y + 1),
                    )
                )
            if not _visible(mask, width, height, x - 1, y):
                edges.append((ProjectionPixelPoint(x, y + 1), ProjectionPixelPoint(x, y)))
    return tuple(edges)


def _edge_direction(first: ProjectionPixelPoint, second: ProjectionPixelPoint) -> int:
    try:
        return {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}[
            (second.x - first.x, second.y - first.y)
        ]
    except KeyError as exc:
        raise ProjectionContourError(
            f"boundary edge is not unit axis-aligned: {first}->{second}"
        ) from exc


def _trace_boundary_loops(
    edges: Tuple[tuple[ProjectionPixelPoint, ProjectionPixelPoint], ...],
) -> Tuple[Tuple[ProjectionPixelPoint, ...], ...]:
    if not edges:
        raise ProjectionContourError("alpha union has no boundary edges")
    outgoing: dict[ProjectionPixelPoint, list[ProjectionPixelPoint]] = {}
    for first, second in edges:
        outgoing.setdefault(first, []).append(second)

    unvisited = set(edges)
    loops: list[Tuple[ProjectionPixelPoint, ...]] = []
    while unvisited:
        start, current = min(unvisited)
        previous = start
        points = [start]
        unvisited.remove((start, current))
        while current != start:
            points.append(current)
            incoming = _edge_direction(previous, current)
            candidates = tuple(
                candidate
                for candidate in outgoing.get(current, ())
                if (current, candidate) in unvisited
            )
            if not candidates:
                raise ProjectionContourError(
                    "boundary tracing reached an open edge chain"
                )
            priority = {
                (incoming + 1) % 4: 0,
                incoming: 1,
                (incoming - 1) % 4: 2,
                (incoming + 2) % 4: 3,
            }
            following = min(
                candidates,
                key=lambda candidate: (
                    priority[_edge_direction(current, candidate)],
                    candidate,
                ),
            )
            unvisited.remove((current, following))
            previous, current = current, following
        loops.append(tuple(points))
    return tuple(loops)


def _remove_collinear(
    points: Tuple[ProjectionPixelPoint, ...],
) -> Tuple[ProjectionPixelPoint, ...]:
    result = list(points)
    changed = True
    while changed and len(result) > 3:
        changed = False
        count = len(result)
        filtered = []
        for index, point in enumerate(result):
            if cross(result[index - 1], point, result[(index + 1) % count]) == 0:
                changed = True
            else:
                filtered.append(point)
        result = filtered
    return tuple(result)


def _canonicalize_ccw(
    points: Tuple[ProjectionPixelPoint, ...],
) -> Tuple[ProjectionPixelPoint, ...]:
    if signed_double_area(points) < 0:
        points = tuple(reversed(points))
    start = min(range(len(points)), key=lambda index: points[index])
    return points[start:] + points[:start]


def _point_segment_distance(
    point: ProjectionPixelPoint,
    first: ProjectionPixelPoint,
    second: ProjectionPixelPoint,
) -> float:
    delta_x = second.x - first.x
    delta_y = second.y - first.y
    denominator = delta_x * delta_x + delta_y * delta_y
    if denominator == 0:
        return hypot(point.x - first.x, point.y - first.y)
    parameter = (
        (point.x - first.x) * delta_x + (point.y - first.y) * delta_y
    ) / denominator
    parameter = max(0.0, min(1.0, parameter))
    return hypot(
        point.x - (first.x + parameter * delta_x),
        point.y - (first.y + parameter * delta_y),
    )


def _replacement_chord_is_clear(
    points: list[ProjectionPixelPoint],
    previous_index: int,
    following_index: int,
) -> bool:
    first = points[previous_index]
    second = points[following_index]
    count = len(points)
    for edge_index in range(count):
        edge_next = (edge_index + 1) % count
        if (
            edge_index in {previous_index, following_index}
            or edge_next in {previous_index, following_index}
        ):
            continue
        if _segments_intersect(
            first,
            second,
            points[edge_index],
            points[edge_next],
        ):
            return False
    return True


def simplify_concave_contour(
    points: Tuple[ProjectionPixelPoint, ...],
    tolerance_pixels: float,
) -> Tuple[ProjectionPixelPoint, ...]:
    """Fill only shallow reflex notches, never cutting visible contour corners."""

    if (
        isinstance(tolerance_pixels, bool)
        or not isinstance(tolerance_pixels, (int, float))
        or not isfinite(float(tolerance_pixels))
        or float(tolerance_pixels) < 0.0
    ):
        raise ValueError("tolerance_pixels must be finite and non-negative")
    result = list(_canonicalize_ccw(_remove_collinear(points)))
    validate_simple_contour(tuple(result))
    tolerance = float(tolerance_pixels)

    while len(result) > 3:
        candidates: list[tuple[float, ProjectionPixelPoint, int]] = []
        count = len(result)
        for index, point in enumerate(result):
            previous_index = (index - 1) % count
            following_index = (index + 1) % count
            previous = result[previous_index]
            following = result[following_index]
            if cross(previous, point, following) >= 0:
                continue
            distance = _point_segment_distance(point, previous, following)
            if (
                distance <= tolerance
                and _replacement_chord_is_clear(
                    result,
                    previous_index,
                    following_index,
                )
            ):
                candidates.append((distance, point, index))
        if not candidates:
            break
        del result[min(candidates)[2]]
        result = list(_canonicalize_ccw(_remove_collinear(tuple(result))))

    resolved = tuple(result)
    validate_simple_contour(resolved)
    triangulate_simple_contour(resolved)
    return resolved


def build_contour_from_mask(
    mask: bytearray,
    *,
    width: int,
    height: int,
    mode: ProjectionContourMode,
    simplify_tolerance_pixels: float,
) -> ProjectionContourResult:
    """Trace one outer component or use a lossless convex fallback for many islands."""

    if not isinstance(mask, bytearray) or len(mask) != width * height:
        raise TypeError("mask must be a bytearray matching width * height")
    if not isinstance(mode, ProjectionContourMode):
        raise TypeError("mode must be ProjectionContourMode")

    edges = _boundary_edges(mask, width=width, height=height)
    boundary_points = tuple(point for edge in edges for point in edge)
    convex = convex_hull(boundary_points)
    if mode is ProjectionContourMode.CONVEX_HULL:
        return ProjectionContourResult(
            convex,
            ProjectionContourMode.CONVEX_HULL,
            len(convex),
            1,
        )

    try:
        loops = _trace_boundary_loops(edges)
        outer_loops = tuple(
            _canonicalize_ccw(_remove_collinear(loop))
            for loop in loops
            if signed_double_area(loop) > 0
        )
        if len(outer_loops) != 1:
            return ProjectionContourResult(
                convex,
                ProjectionContourMode.CONVEX_HULL,
                max(len(convex), sum(len(loop) for loop in outer_loops)),
                max(1, len(outer_loops)),
                "MULTIPLE_OUTER_COMPONENTS",
            )
        source = outer_loops[0]
        simplified = simplify_concave_contour(
            source,
            simplify_tolerance_pixels,
        )
        return ProjectionContourResult(
            simplified,
            ProjectionContourMode.SIMPLIFIED_CONCAVE,
            len(source),
            1,
        )
    except ProjectionContourError as exc:
        return ProjectionContourResult(
            convex,
            ProjectionContourMode.CONVEX_HULL,
            len(convex),
            1,
            f"CONTOUR_EXTRACTION_FAILED:{type(exc).__name__}",
        )
