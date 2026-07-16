"""Pure sequence-union crop and screen-space hull planning for B4 renders."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Tuple


class CameraProjectionLayoutError(ValueError):
    """Raised when rendered alpha cannot produce a stable projection layout."""


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
class ProjectionCropBounds:
    """Exclusive full-frame crop rectangle using Blender pixel orientation."""

    minimum_x: int
    minimum_y: int
    maximum_x: int
    maximum_y: int

    def __post_init__(self) -> None:
        for field_name in ("minimum_x", "minimum_y", "maximum_x", "maximum_y"):
            if not isinstance(getattr(self, field_name), int):
                raise TypeError(f"{field_name} must be int")
        if self.minimum_x < 0 or self.minimum_y < 0:
            raise ValueError("crop minimum coordinates cannot be negative")
        if self.maximum_x <= self.minimum_x or self.maximum_y <= self.minimum_y:
            raise ValueError("crop bounds must have positive width and height")

    @property
    def width(self) -> int:
        return self.maximum_x - self.minimum_x

    @property
    def height(self) -> int:
        return self.maximum_y - self.minimum_y


@dataclass(frozen=True, slots=True)
class CameraProjectionLayout:
    """One stable crop and convex hull shared by every projection frame."""

    full_width: int
    full_height: int
    crop: ProjectionCropBounds
    hull: Tuple[ProjectionPixelPoint, ...]
    alpha_threshold: float
    padding_pixels: int
    frame_count: int
    visible_pixel_count: int

    def __post_init__(self) -> None:
        for field_name in ("full_width", "full_height", "frame_count"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if not isinstance(self.crop, ProjectionCropBounds):
            raise TypeError("crop must be ProjectionCropBounds")
        if self.crop.maximum_x > self.full_width or self.crop.maximum_y > self.full_height:
            raise ValueError("crop bounds exceed the full render dimensions")
        if not isinstance(self.hull, tuple) or len(self.hull) < 3:
            raise ValueError("hull must contain at least three points")
        if not all(isinstance(point, ProjectionPixelPoint) for point in self.hull):
            raise TypeError("hull must contain ProjectionPixelPoint values")
        if len(self.hull) != len(set(self.hull)):
            raise ValueError("hull cannot contain duplicate points")
        for point in self.hull:
            if not (
                self.crop.minimum_x <= point.x <= self.crop.maximum_x
                and self.crop.minimum_y <= point.y <= self.crop.maximum_y
            ):
                raise ValueError("hull point lies outside crop bounds")
        if (
            not isinstance(self.alpha_threshold, (int, float))
            or not isfinite(float(self.alpha_threshold))
            or not 0.0 <= float(self.alpha_threshold) <= 1.0
        ):
            raise ValueError("alpha_threshold must be finite and in [0, 1]")
        if not isinstance(self.padding_pixels, int) or self.padding_pixels < 0:
            raise ValueError("padding_pixels must be a non-negative integer")
        if not isinstance(self.visible_pixel_count, int) or self.visible_pixel_count <= 0:
            raise ValueError("visible_pixel_count must be a positive integer")
        if _signed_double_area(self.hull) <= 0:
            raise ValueError("hull must be counter-clockwise and non-degenerate")

    @property
    def cropped_width(self) -> int:
        return self.crop.width

    @property
    def cropped_height(self) -> int:
        return self.crop.height

    @property
    def cropped(self) -> bool:
        return (
            self.crop.minimum_x != 0
            or self.crop.minimum_y != 0
            or self.crop.maximum_x != self.full_width
            or self.crop.maximum_y != self.full_height
        )

    @property
    def offset_pixels(self) -> tuple[float, float]:
        return (
            (self.crop.minimum_x + self.crop.maximum_x) / 2.0 - self.full_width / 2.0,
            (self.crop.minimum_y + self.crop.maximum_y) / 2.0 - self.full_height / 2.0,
        )

    def crop_local_point(self, point: ProjectionPixelPoint) -> tuple[int, int]:
        if point not in self.hull:
            raise KeyError("point is not part of this layout hull")
        return point.x - self.crop.minimum_x, point.y - self.crop.minimum_y

    def spine_uv(self, point: ProjectionPixelPoint) -> tuple[float, float]:
        local_x, local_y = self.crop_local_point(point)
        return (
            float(local_x) / float(self.cropped_width),
            1.0 - float(local_y) / float(self.cropped_height),
        )

    def spine_position_pixels(self, point: ProjectionPixelPoint) -> tuple[float, float]:
        return (
            float(point.x) - float(self.full_width) / 2.0,
            float(point.y) - float(self.full_height) / 2.0,
        )


def _cross(
    origin: ProjectionPixelPoint,
    first: ProjectionPixelPoint,
    second: ProjectionPixelPoint,
) -> int:
    return (first.x - origin.x) * (second.y - origin.y) - (
        first.y - origin.y
    ) * (second.x - origin.x)


def _signed_double_area(points: Tuple[ProjectionPixelPoint, ...]) -> int:
    return sum(
        first.x * second.y - second.x * first.y
        for first, second in zip(points, points[1:] + points[:1])
    )


def convex_hull(
    points: Iterable[ProjectionPixelPoint],
) -> Tuple[ProjectionPixelPoint, ...]:
    """Return a deterministic counter-clockwise convex hull without repeated endpoint."""

    unique = tuple(sorted(set(points)))
    if len(unique) < 3:
        raise CameraProjectionLayoutError(
            "at least three unique pixel-boundary points are required for a hull"
        )

    lower: list[ProjectionPixelPoint] = []
    for point in unique:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[ProjectionPixelPoint] = []
    for point in reversed(unique):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    resolved = tuple(lower[:-1] + upper[:-1])
    if len(resolved) < 3 or _signed_double_area(resolved) <= 0:
        raise CameraProjectionLayoutError("visible alpha produced a degenerate hull")
    return resolved


def _validate_masks(
    alpha_masks: Tuple[bytes | bytearray, ...],
    width: int,
    height: int,
) -> None:
    if not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    if not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer")
    if not isinstance(alpha_masks, tuple) or not alpha_masks:
        raise ValueError("alpha_masks must be a non-empty tuple")
    expected = width * height
    for index, mask in enumerate(alpha_masks):
        if not isinstance(mask, (bytes, bytearray)):
            raise TypeError(f"alpha_masks[{index}] must be bytes or bytearray")
        if len(mask) != expected:
            raise ValueError(
                f"alpha_masks[{index}] has {len(mask)} entries; expected {expected}"
            )


def build_sequence_union_layout(
    alpha_masks: Tuple[bytes | bytearray, ...],
    *,
    width: int,
    height: int,
    alpha_threshold: float,
    padding_pixels: int,
) -> CameraProjectionLayout:
    """Build one crop and convex alpha hull from all rendered sequence frames.

    Masks contain 0 for transparent pixels and non-zero for pixels whose decoded alpha met
    ``alpha_threshold``. The union is computed before crop expansion, so every frame shares
    exactly the same texture dimensions, attachment offset, UVs, hull and triangulation.
    """

    _validate_masks(alpha_masks, width, height)
    if (
        not isinstance(alpha_threshold, (int, float))
        or not isfinite(float(alpha_threshold))
        or not 0.0 <= float(alpha_threshold) <= 1.0
    ):
        raise ValueError("alpha_threshold must be finite and in [0, 1]")
    if not isinstance(padding_pixels, int) or padding_pixels < 0:
        raise ValueError("padding_pixels must be a non-negative integer")

    union = bytearray(width * height)
    for mask in alpha_masks:
        for index, value in enumerate(mask):
            if value:
                union[index] = 1

    visible_count = sum(union)
    if visible_count == 0:
        raise CameraProjectionLayoutError(
            "camera projection sequence contains no pixels above the alpha threshold"
        )

    minimum_x = width
    minimum_y = height
    maximum_x = -1
    maximum_y = -1
    boundary_points: list[ProjectionPixelPoint] = []
    for y in range(height):
        row_start = y * width
        visible_x = tuple(x for x in range(width) if union[row_start + x])
        if not visible_x:
            continue
        left = visible_x[0]
        right_exclusive = visible_x[-1] + 1
        minimum_x = min(minimum_x, left)
        minimum_y = min(minimum_y, y)
        maximum_x = max(maximum_x, right_exclusive - 1)
        maximum_y = max(maximum_y, y)
        boundary_points.extend(
            (
                ProjectionPixelPoint(left, y),
                ProjectionPixelPoint(right_exclusive, y),
                ProjectionPixelPoint(right_exclusive, y + 1),
                ProjectionPixelPoint(left, y + 1),
            )
        )

    crop = ProjectionCropBounds(
        minimum_x=max(0, minimum_x - padding_pixels),
        minimum_y=max(0, minimum_y - padding_pixels),
        maximum_x=min(width, maximum_x + 1 + padding_pixels),
        maximum_y=min(height, maximum_y + 1 + padding_pixels),
    )
    hull = convex_hull(boundary_points)
    return CameraProjectionLayout(
        full_width=width,
        full_height=height,
        crop=crop,
        hull=hull,
        alpha_threshold=float(alpha_threshold),
        padding_pixels=padding_pixels,
        frame_count=len(alpha_masks),
        visible_pixel_count=visible_count,
    )


def build_full_frame_layout(
    width: int,
    height: int,
    *,
    frame_count: int = 1,
) -> CameraProjectionLayout:
    """Compatibility layout used before a render-derived union is available."""

    crop = ProjectionCropBounds(0, 0, width, height)
    return CameraProjectionLayout(
        full_width=width,
        full_height=height,
        crop=crop,
        hull=(
            ProjectionPixelPoint(0, 0),
            ProjectionPixelPoint(width, 0),
            ProjectionPixelPoint(width, height),
            ProjectionPixelPoint(0, height),
        ),
        alpha_threshold=0.0,
        padding_pixels=0,
        frame_count=frame_count,
        visible_pixel_count=width * height,
    )
