"""Pure sequence-union crop and screen-space hull planning for B4 renders."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Iterable, Tuple


ProjectionTriangle = Tuple[int, int, int]


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
    """One stable crop and strictly convex hull shared by every projection frame."""

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
        _validate_strict_convex_hull(self.hull)
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

    @property
    def triangle_indices(self) -> Tuple[ProjectionTriangle, ...]:
        """Return the deterministic fan used by the Spine projection mesh."""

        return triangulate_convex_hull(self.hull)

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


def _validate_dimensions(width: int, height: int) -> None:
    if not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    if not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer")


def _validate_layout_policy(alpha_threshold: float, padding_pixels: int) -> None:
    if (
        not isinstance(alpha_threshold, (int, float))
        or not isfinite(float(alpha_threshold))
        or not 0.0 <= float(alpha_threshold) <= 1.0
    ):
        raise ValueError("alpha_threshold must be finite and in [0, 1]")
    if not isinstance(padding_pixels, int) or padding_pixels < 0:
        raise ValueError("padding_pixels must be a non-negative integer")


def _validate_alpha_mask(
    alpha_mask: bytes | bytearray,
    *,
    expected_size: int,
    frame_index: int | None,
) -> None:
    label = "alpha_mask" if frame_index is None else f"alpha_masks[{frame_index}]"
    if not isinstance(alpha_mask, (bytes, bytearray)):
        raise TypeError(f"{label} must be bytes or bytearray")
    if len(alpha_mask) != expected_size:
        raise ValueError(
            f"{label} has {len(alpha_mask)} entries; expected {expected_size}"
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


def _validate_strict_convex_hull(
    points: Tuple[ProjectionPixelPoint, ...],
) -> None:
    if not isinstance(points, tuple) or len(points) < 3:
        raise ValueError("hull must contain at least three points")
    if not all(isinstance(point, ProjectionPixelPoint) for point in points):
        raise TypeError("hull must contain ProjectionPixelPoint values")
    if len(points) != len(set(points)):
        raise ValueError("hull cannot contain duplicate points")
    if _signed_double_area(points) <= 0:
        raise ValueError("hull must be counter-clockwise and non-degenerate")

    invalid_turns = tuple(
        index
        for index in range(len(points))
        if _cross(points[index - 1], points[index], points[(index + 1) % len(points)])
        <= 0
    )
    if invalid_turns:
        raise ValueError(
            "hull must be strictly convex without collinear or reflex vertices; "
            f"invalid vertex indices={invalid_turns}"
        )


def triangulate_convex_hull(
    points: Tuple[ProjectionPixelPoint, ...],
) -> Tuple[ProjectionTriangle, ...]:
    """Return a deterministic, non-degenerate fan for a strict CCW convex hull."""

    _validate_strict_convex_hull(points)
    triangles = tuple((0, index, index + 1) for index in range(1, len(points) - 1))
    fan_double_area = sum(
        _cross(points[first], points[second], points[third])
        for first, second, third in triangles
    )
    polygon_double_area = _signed_double_area(points)
    if fan_double_area != polygon_double_area:
        raise CameraProjectionLayoutError(
            "convex hull fan does not cover the polygon exactly; "
            f"polygon_area2={polygon_double_area}, fan_area2={fan_double_area}"
        )
    return triangles


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
    try:
        _validate_strict_convex_hull(resolved)
    except (TypeError, ValueError) as exc:
        raise CameraProjectionLayoutError(
            f"visible alpha produced an invalid convex hull: {exc}"
        ) from exc
    return resolved


def _layout_from_union_mask(
    union_mask: bytearray,
    *,
    width: int,
    height: int,
    alpha_threshold: float,
    padding_pixels: int,
    frame_count: int,
    visible_pixel_count: int,
) -> CameraProjectionLayout:
    if frame_count <= 0:
        raise ValueError("at least one alpha mask must be accumulated")
    if visible_pixel_count == 0:
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
        left: int | None = None
        right: int | None = None
        for x in range(width):
            if not union_mask[row_start + x]:
                continue
            if left is None:
                left = x
            right = x
        if left is None or right is None:
            continue

        right_exclusive = right + 1
        minimum_x = min(minimum_x, left)
        minimum_y = min(minimum_y, y)
        maximum_x = max(maximum_x, right)
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
    return CameraProjectionLayout(
        full_width=width,
        full_height=height,
        crop=crop,
        hull=convex_hull(boundary_points),
        alpha_threshold=float(alpha_threshold),
        padding_pixels=padding_pixels,
        frame_count=frame_count,
        visible_pixel_count=visible_pixel_count,
    )


@dataclass(slots=True)
class ProjectionAlphaUnionAccumulator:
    """Incrementally OR frame alpha masks into one fixed-size union buffer."""

    width: int
    height: int
    alpha_threshold: float
    padding_pixels: int
    _union_mask: bytearray = field(init=False, repr=False)
    _frame_count: int = field(init=False, default=0, repr=False)
    _visible_pixel_count: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        _validate_dimensions(self.width, self.height)
        _validate_layout_policy(self.alpha_threshold, self.padding_pixels)
        self.alpha_threshold = float(self.alpha_threshold)
        self._union_mask = bytearray(self.width * self.height)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def visible_pixel_count(self) -> int:
        return self._visible_pixel_count

    @property
    def allocated_mask_bytes(self) -> int:
        """Return the fixed union-buffer size, independent of accumulated frame count."""

        return len(self._union_mask)

    def add_mask(
        self,
        alpha_mask: bytes | bytearray,
        *,
        frame_index: int | None = None,
    ) -> int:
        """Merge one frame and return the number of newly visible union pixels."""

        if frame_index is not None and (
            not isinstance(frame_index, int) or frame_index < 0
        ):
            raise ValueError("frame_index must be a non-negative integer or None")
        _validate_alpha_mask(
            alpha_mask,
            expected_size=self.width * self.height,
            frame_index=frame_index,
        )

        newly_visible = 0
        union_mask = self._union_mask
        for index, value in enumerate(alpha_mask):
            if value and not union_mask[index]:
                union_mask[index] = 1
                newly_visible += 1
        self._frame_count += 1
        self._visible_pixel_count += newly_visible
        return newly_visible

    def build_layout(self) -> CameraProjectionLayout:
        """Finalize the stable crop and hull without copying the union mask."""

        return _layout_from_union_mask(
            self._union_mask,
            width=self.width,
            height=self.height,
            alpha_threshold=self.alpha_threshold,
            padding_pixels=self.padding_pixels,
            frame_count=self._frame_count,
            visible_pixel_count=self._visible_pixel_count,
        )


def build_sequence_union_layout(
    alpha_masks: Tuple[bytes | bytearray, ...],
    *,
    width: int,
    height: int,
    alpha_threshold: float,
    padding_pixels: int,
) -> CameraProjectionLayout:
    """Compatibility wrapper that builds a layout from an existing mask tuple.

    New render executors should feed ``ProjectionAlphaUnionAccumulator`` one frame at a time,
    so memory remains ``O(width * height)`` instead of ``O(frame_count * width * height)``.
    """

    if not isinstance(alpha_masks, tuple) or not alpha_masks:
        raise ValueError("alpha_masks must be a non-empty tuple")
    accumulator = ProjectionAlphaUnionAccumulator(
        width=width,
        height=height,
        alpha_threshold=alpha_threshold,
        padding_pixels=padding_pixels,
    )
    for frame_index, alpha_mask in enumerate(alpha_masks):
        accumulator.add_mask(alpha_mask, frame_index=frame_index)
    return accumulator.build_layout()


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
