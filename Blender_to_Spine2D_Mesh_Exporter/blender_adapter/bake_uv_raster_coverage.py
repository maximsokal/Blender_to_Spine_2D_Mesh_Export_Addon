"""Resolution-aware raster footprint helpers for baked UV validation.

The helpers in this module are Blender-independent.  They operate in Spine PNG
file-space where ``(0, 0)`` is the top-left of the image and every integer pixel
``(x, y)`` owns the normalized cell::

    [x / width, (x + 1) / width] x [y / height, (y + 1) / height]

The production validator uses the footprint only as a fallback after its regular
interior UV samples found no alpha.  This keeps the common path cheap while avoiding
false negatives for very small triangles whose rasterized texels lie between the
four deterministic interior samples.
"""

from __future__ import annotations

from math import ceil, floor, isfinite
from typing import Iterable


_DEFAULT_MAX_CANDIDATE_CELLS = 4096
_SAT_EPSILON = 1.0e-9


class RasterFootprintError(ValueError):
    """Raised when a triangle cannot be represented as a finite raster footprint."""


def _validate_triangle(
    triangle: Iterable[Iterable[float]],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    try:
        vertices = tuple(tuple(vertex) for vertex in triangle)
    except Exception as exc:
        raise RasterFootprintError(
            "triangle must contain exactly three two-component vertices"
        ) from exc
    if len(vertices) != 3 or any(len(vertex) != 2 for vertex in vertices):
        raise RasterFootprintError(
            "triangle must contain exactly three two-component vertices"
        )

    resolved: list[tuple[float, float]] = []
    for vertex_index, vertex in enumerate(vertices):
        try:
            u, v = (float(value) for value in vertex)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RasterFootprintError(
                f"triangle vertex {vertex_index} contains a non-numeric component"
            ) from exc
        if not isfinite(u) or not isfinite(v):
            raise RasterFootprintError(
                f"triangle vertex {vertex_index} contains a non-finite component"
            )
        if u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0:
            raise RasterFootprintError(
                f"triangle vertex {vertex_index} is outside the unit square: {(u, v)!r}"
            )
        resolved.append((u, v))
    return resolved[0], resolved[1], resolved[2]


def _projection_interval(
    points: tuple[tuple[float, float], ...],
    axis: tuple[float, float],
) -> tuple[float, float]:
    values = tuple(point[0] * axis[0] + point[1] * axis[1] for point in points)
    return min(values), max(values)


def _intervals_overlap(
    first: tuple[float, float],
    second: tuple[float, float],
) -> bool:
    return not (
        first[1] < second[0] - _SAT_EPSILON
        or second[1] < first[0] - _SAT_EPSILON
    )


def _triangle_intersects_pixel_cell(
    triangle_pixels: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
    pixel_x: int,
    pixel_y: int,
) -> bool:
    """Return whether one triangle intersects the closed unit square of one texel."""

    rect = (
        (float(pixel_x), float(pixel_y)),
        (float(pixel_x + 1), float(pixel_y)),
        (float(pixel_x + 1), float(pixel_y + 1)),
        (float(pixel_x), float(pixel_y + 1)),
    )
    axes: list[tuple[float, float]] = [(1.0, 0.0), (0.0, 1.0)]
    for index in range(3):
        first = triangle_pixels[index]
        second = triangle_pixels[(index + 1) % 3]
        edge_x = second[0] - first[0]
        edge_y = second[1] - first[1]
        normal = (-edge_y, edge_x)
        if abs(normal[0]) <= _SAT_EPSILON and abs(normal[1]) <= _SAT_EPSILON:
            continue
        axes.append(normal)

    triangle_points = tuple(triangle_pixels)
    rect_points = tuple(rect)
    for axis in axes:
        if not _intervals_overlap(
            _projection_interval(triangle_points, axis),
            _projection_interval(rect_points, axis),
        ):
            return False
    return True


def _cell_range(
    minimum: float,
    maximum: float,
    *,
    size: int,
) -> tuple[int, int]:
    """Return inclusive texel indices touched by a non-empty continuous interval."""

    first = floor(minimum)
    # A maximum exactly on a texel boundary does not own area in the following cell.
    # The SAT test remains the final authority for slanted edges inside this coarse box.
    last = ceil(maximum - _SAT_EPSILON) - 1
    first = max(0, min(size - 1, first))
    last = max(0, min(size - 1, last))
    if last < first:
        last = first
    return first, last


def raster_footprint_pixels(
    triangle: Iterable[Iterable[float]],
    *,
    width: int,
    height: int,
    max_candidate_cells: int = _DEFAULT_MAX_CANDIDATE_CELLS,
) -> tuple[tuple[int, int], ...] | None:
    """Return deterministic top-down PNG texels whose cells intersect ``triangle``.

    ``None`` means the triangle's bounding box exceeds ``max_candidate_cells``.  The
    caller can then keep the original interior-sample failure without doing an expensive
    raster scan.  The fallback is intended for small boundary/sub-pixel triangles, not
    for rescuing large triangles whose interiors are genuinely empty.
    """

    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError("width must be a positive integer")
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise ValueError("height must be a positive integer")
    if (
        not isinstance(max_candidate_cells, int)
        or isinstance(max_candidate_cells, bool)
        or max_candidate_cells < 1
    ):
        raise ValueError("max_candidate_cells must be a positive integer")

    resolved = _validate_triangle(triangle)
    triangle_pixels = tuple(
        (u * float(width), v * float(height)) for u, v in resolved
    )

    minimum_x, maximum_x = _cell_range(
        min(point[0] for point in triangle_pixels),
        max(point[0] for point in triangle_pixels),
        size=width,
    )
    minimum_y, maximum_y = _cell_range(
        min(point[1] for point in triangle_pixels),
        max(point[1] for point in triangle_pixels),
        size=height,
    )

    candidate_count = (maximum_x - minimum_x + 1) * (maximum_y - minimum_y + 1)
    if candidate_count > max_candidate_cells:
        return None

    pixels: list[tuple[int, int]] = []
    for pixel_y in range(minimum_y, maximum_y + 1):
        for pixel_x in range(minimum_x, maximum_x + 1):
            if _triangle_intersects_pixel_cell(
                triangle_pixels,
                pixel_x,
                pixel_y,
            ):
                pixels.append((pixel_x, pixel_y))
    return tuple(pixels)


__all__ = [
    "RasterFootprintError",
    "raster_footprint_pixels",
]
