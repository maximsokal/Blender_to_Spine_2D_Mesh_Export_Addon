"""Resolution-aware raster sample helpers for baked UV validation.

Blender image baking writes discrete texels. For validation, the meaningful question is
not whether a continuous UV triangle intersects a pixel *cell*, but whether the triangle
contains at least one texel sample centre at the requested output resolution. A triangle
smaller than the sampling grid may intersect a cell while containing no sample centre at
all; Blender cannot produce direct coverage for such a triangle at that resolution.

The helpers in this module are Blender-independent and operate in Spine PNG file-space,
where ``(0, 0)`` is the top-left of the saved image. Pixel ``(x, y)`` has sample centre::

    ((x + 0.5) / width, (y + 0.5) / height)

Callers can therefore distinguish a genuine empty-bake mismatch from a triangle that is
mathematically unrepresentable by the selected texture resolution.
"""

from __future__ import annotations

from math import ceil, floor, isfinite
from typing import Iterable


_DEFAULT_MAX_CANDIDATE_PIXELS = 4096
_POINT_EPSILON = 1.0e-12


class RasterFootprintError(ValueError):
    """Raised when a triangle cannot be represented as finite UV coordinates."""


def _validate_triangle(
    triangle: Iterable[Iterable[float]],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return one finite unit-square UV triangle."""

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


def _validate_image_size(width: int, height: int) -> None:
    """Validate positive integer texture dimensions."""

    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError("width must be a positive integer")
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise ValueError("height must be a positive integer")


def _cross(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> float:
    """Return twice the signed area of one two-dimensional triangle."""

    return (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def triangle_twice_area_pixels(
    triangle: Iterable[Iterable[float]],
    *,
    width: int,
    height: int,
) -> float:
    """Return the absolute twice-area of ``triangle`` measured in output texel units."""

    _validate_image_size(width, height)
    resolved = _validate_triangle(triangle)
    pixels = tuple(
        (u * float(width), v * float(height))
        for u, v in resolved
    )
    return abs(_cross(pixels[0], pixels[1], pixels[2]))


def _inclusive_point_in_triangle(
    point: tuple[float, float],
    triangle: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
) -> bool:
    """Return whether ``point`` is inside or on the boundary of a non-degenerate triangle."""

    signed_area = _cross(triangle[0], triangle[1], triangle[2])
    if abs(signed_area) <= _POINT_EPSILON:
        return False

    first = _cross(triangle[0], triangle[1], point)
    second = _cross(triangle[1], triangle[2], point)
    third = _cross(triangle[2], triangle[0], point)

    # Scale the tolerance to the triangle itself while retaining an absolute floor for
    # very small but still finite triangles.
    tolerance = max(_POINT_EPSILON, abs(signed_area) * 1.0e-10)
    if signed_area > 0.0:
        return (
            first >= -tolerance
            and second >= -tolerance
            and third >= -tolerance
        )
    return (
        first <= tolerance
        and second <= tolerance
        and third <= tolerance
    )


def _candidate_center_range(
    minimum: float,
    maximum: float,
    *,
    size: int,
) -> tuple[int, int] | None:
    """Return pixel indices whose centres can lie inside one pixel-space interval."""

    # Pixel centres are n + 0.5 in pixel coordinates. Expand only by the numerical
    # epsilon used by the point-in-triangle test; do not expand by a pixel or margin.
    first = ceil(minimum - 0.5 - _POINT_EPSILON)
    last = floor(maximum - 0.5 + _POINT_EPSILON)

    first = max(0, first)
    last = min(size - 1, last)
    if first > last:
        return None
    return first, last


def raster_sample_pixels(
    triangle: Iterable[Iterable[float]],
    *,
    width: int,
    height: int,
    max_candidate_pixels: int = _DEFAULT_MAX_CANDIDATE_PIXELS,
) -> tuple[tuple[int, int], ...] | None:
    """Return texels whose *sample centres* lie inside ``triangle``.

    The returned coordinates are top-down Spine PNG pixel indices. An empty tuple means
    the triangle has no raster sample at the requested resolution. That is a meaningful
    result: Blender cannot directly bake alpha for such a triangle because no output
    texel centre belongs to it.

    ``None`` means the coarse bounding box exceeded ``max_candidate_pixels``. Validation
    callers intentionally remain fail-closed in that uncommon case rather than scanning
    an unbounded number of pixels.
    """

    _validate_image_size(width, height)
    if (
        not isinstance(max_candidate_pixels, int)
        or isinstance(max_candidate_pixels, bool)
        or max_candidate_pixels < 1
    ):
        raise ValueError("max_candidate_pixels must be a positive integer")

    resolved = _validate_triangle(triangle)
    triangle_pixels = tuple(
        (u * float(width), v * float(height))
        for u, v in resolved
    )

    x_range = _candidate_center_range(
        min(point[0] for point in triangle_pixels),
        max(point[0] for point in triangle_pixels),
        size=width,
    )
    y_range = _candidate_center_range(
        min(point[1] for point in triangle_pixels),
        max(point[1] for point in triangle_pixels),
        size=height,
    )
    if x_range is None or y_range is None:
        return ()

    minimum_x, maximum_x = x_range
    minimum_y, maximum_y = y_range
    candidate_count = (
        (maximum_x - minimum_x + 1)
        * (maximum_y - minimum_y + 1)
    )
    if candidate_count > max_candidate_pixels:
        return None

    pixels: list[tuple[int, int]] = []
    for pixel_y in range(minimum_y, maximum_y + 1):
        for pixel_x in range(minimum_x, maximum_x + 1):
            centre = (
                float(pixel_x) + 0.5,
                float(pixel_y) + 0.5,
            )
            if _inclusive_point_in_triangle(centre, triangle_pixels):
                pixels.append((pixel_x, pixel_y))

    return tuple(pixels)


__all__ = [
    "RasterFootprintError",
    "raster_sample_pixels",
    "triangle_twice_area_pixels",
]
