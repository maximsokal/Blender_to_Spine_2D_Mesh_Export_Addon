"""Pure sequence-union coverage, crop, and screen-space contour planning for B4."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Tuple

from .projection_contour import (
    ProjectionContourError,
    ProjectionContourMode,
    ProjectionPixelPoint,
    ProjectionTriangle,
    build_contour_from_mask,
    convex_hull,
    cross,
    simplify_concave_contour,
    triangulate_convex_hull,
    triangulate_simple_contour,
    validate_simple_contour,
)
from .projection_coverage import (
    ProjectionCoverageMode,
    ProjectionCoveragePolicy,
    ProjectionCoverageResult,
    build_projection_coverage_mask,
)


_COMPATIBILITY_COVERAGE_POLICY = ProjectionCoveragePolicy(
    mode=ProjectionCoverageMode.BINARY_THRESHOLD,
    core_alpha_threshold=0.0,
    minimum_component_pixels=1,
    maximum_hole_pixels=0,
)


class CameraProjectionLayoutError(ValueError):
    """Raised when rendered alpha cannot produce a stable projection layout."""


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
    """One stable cleaned-alpha crop and simple contour shared by every frame.

    ``hull`` remains the compatibility field name. It can contain a simple concave
    contour; internal holes continue to be represented by texture alpha unless the
    configured morphology policy fills a sufficiently small enclosed pinhole.
    """

    full_width: int
    full_height: int
    crop: ProjectionCropBounds
    hull: Tuple[ProjectionPixelPoint, ...]
    alpha_threshold: float
    padding_pixels: int
    frame_count: int
    visible_pixel_count: int
    contour_mode: ProjectionContourMode = ProjectionContourMode.CONVEX_HULL
    source_contour_vertex_count: int = 0
    outer_component_count: int = 1
    simplify_tolerance_pixels: float = 0.0
    contour_fallback_reason: str | None = None
    coverage_mode: ProjectionCoverageMode = ProjectionCoverageMode.BINARY_THRESHOLD
    coverage_core_alpha_threshold: float = 0.0
    coverage_raw_nonzero_pixel_count: int = 0
    coverage_strong_pixel_count: int = 0
    coverage_component_count_before_cleanup: int = 0
    coverage_component_count_after_cleanup: int = 0
    coverage_removed_component_pixel_count: int = 0
    coverage_filled_hole_pixel_count: int = 0
    coverage_used_weak_only_fallback: bool = False

    def __post_init__(self) -> None:
        for field_name in ("full_width", "full_height", "frame_count"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if not isinstance(self.crop, ProjectionCropBounds):
            raise TypeError("crop must be ProjectionCropBounds")
        if self.crop.maximum_x > self.full_width or self.crop.maximum_y > self.full_height:
            raise ValueError("crop bounds exceed the full render dimensions")
        try:
            validate_simple_contour(self.hull)
        except ProjectionContourError as exc:
            raise ValueError(str(exc)) from exc
        for point in self.hull:
            if not (
                self.crop.minimum_x <= point.x <= self.crop.maximum_x
                and self.crop.minimum_y <= point.y <= self.crop.maximum_y
            ):
                raise ValueError("contour point lies outside crop bounds")
        if (
            isinstance(self.alpha_threshold, bool)
            or not isinstance(self.alpha_threshold, (int, float))
            or not isfinite(float(self.alpha_threshold))
            or not 0.0 <= float(self.alpha_threshold) <= 1.0
        ):
            raise ValueError("alpha_threshold must be finite and in [0, 1]")
        if not isinstance(self.padding_pixels, int) or self.padding_pixels < 0:
            raise ValueError("padding_pixels must be a non-negative integer")
        if not isinstance(self.visible_pixel_count, int) or self.visible_pixel_count <= 0:
            raise ValueError("visible_pixel_count must be a positive integer")
        if not isinstance(self.contour_mode, ProjectionContourMode):
            raise TypeError("contour_mode must be ProjectionContourMode")
        if not isinstance(self.source_contour_vertex_count, int):
            raise TypeError("source_contour_vertex_count must be int")
        if self.source_contour_vertex_count == 0:
            object.__setattr__(self, "source_contour_vertex_count", len(self.hull))
        elif self.source_contour_vertex_count < len(self.hull):
            raise ValueError(
                "source_contour_vertex_count cannot be smaller than contour size"
            )
        if not isinstance(self.outer_component_count, int) or self.outer_component_count <= 0:
            raise ValueError("outer_component_count must be a positive integer")
        if (
            isinstance(self.simplify_tolerance_pixels, bool)
            or not isinstance(self.simplify_tolerance_pixels, (int, float))
            or not isfinite(float(self.simplify_tolerance_pixels))
            or float(self.simplify_tolerance_pixels) < 0.0
        ):
            raise ValueError(
                "simplify_tolerance_pixels must be finite and non-negative"
            )
        if self.contour_fallback_reason is not None and (
            not isinstance(self.contour_fallback_reason, str)
            or not self.contour_fallback_reason.strip()
        ):
            raise ValueError("contour_fallback_reason must be non-empty str or None")
        if not isinstance(self.coverage_mode, ProjectionCoverageMode):
            raise TypeError("coverage_mode must be ProjectionCoverageMode")
        if (
            isinstance(self.coverage_core_alpha_threshold, bool)
            or not isinstance(self.coverage_core_alpha_threshold, (int, float))
            or not isfinite(float(self.coverage_core_alpha_threshold))
            or not 0.0 <= float(self.coverage_core_alpha_threshold) <= 1.0
        ):
            raise ValueError(
                "coverage_core_alpha_threshold must be finite and in [0, 1]"
            )
        coverage_integer_fields = (
            "coverage_raw_nonzero_pixel_count",
            "coverage_strong_pixel_count",
            "coverage_component_count_before_cleanup",
            "coverage_component_count_after_cleanup",
            "coverage_removed_component_pixel_count",
            "coverage_filled_hole_pixel_count",
        )
        for field_name in coverage_integer_fields:
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.coverage_raw_nonzero_pixel_count == 0:
            object.__setattr__(
                self,
                "coverage_raw_nonzero_pixel_count",
                self.visible_pixel_count,
            )
        if not isinstance(self.coverage_used_weak_only_fallback, bool):
            raise TypeError("coverage_used_weak_only_fallback must be bool")

    @property
    def contour(self) -> Tuple[ProjectionPixelPoint, ...]:
        return self.hull

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
    def concave(self) -> bool:
        return any(
            cross(
                self.hull[index - 1],
                point,
                self.hull[(index + 1) % len(self.hull)],
            )
            < 0
            for index, point in enumerate(self.hull)
        )

    @property
    def offset_pixels(self) -> tuple[float, float]:
        return (
            (self.crop.minimum_x + self.crop.maximum_x) / 2.0 - self.full_width / 2.0,
            (self.crop.minimum_y + self.crop.maximum_y) / 2.0 - self.full_height / 2.0,
        )

    @property
    def triangle_indices(self) -> Tuple[ProjectionTriangle, ...]:
        try:
            return triangulate_simple_contour(self.hull)
        except ProjectionContourError as exc:
            raise CameraProjectionLayoutError(str(exc)) from exc

    def crop_local_point(self, point: ProjectionPixelPoint) -> tuple[int, int]:
        if point not in self.hull:
            raise KeyError("point is not part of this layout contour")
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
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError("width must be a positive integer")
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise ValueError("height must be a positive integer")


def _validate_layout_policy(
    alpha_threshold: float,
    padding_pixels: int,
    contour_mode: ProjectionContourMode,
    simplify_tolerance_pixels: float,
    coverage_policy: ProjectionCoveragePolicy,
) -> None:
    if (
        isinstance(alpha_threshold, bool)
        or not isinstance(alpha_threshold, (int, float))
        or not isfinite(float(alpha_threshold))
        or not 0.0 <= float(alpha_threshold) <= 1.0
    ):
        raise ValueError("alpha_threshold must be finite and in [0, 1]")
    if not isinstance(padding_pixels, int) or padding_pixels < 0:
        raise ValueError("padding_pixels must be a non-negative integer")
    if not isinstance(contour_mode, ProjectionContourMode):
        raise TypeError("contour_mode must be ProjectionContourMode")
    if (
        isinstance(simplify_tolerance_pixels, bool)
        or not isinstance(simplify_tolerance_pixels, (int, float))
        or not isfinite(float(simplify_tolerance_pixels))
        or float(simplify_tolerance_pixels) < 0.0
    ):
        raise ValueError(
            "simplify_tolerance_pixels must be finite and non-negative"
        )
    if not isinstance(coverage_policy, ProjectionCoveragePolicy):
        raise TypeError("coverage_policy must be ProjectionCoveragePolicy")


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


def _layout_from_union_mask(
    union_mask: bytearray,
    *,
    width: int,
    height: int,
    alpha_threshold: float,
    padding_pixels: int,
    frame_count: int,
    coverage_result: ProjectionCoverageResult,
    contour_mode: ProjectionContourMode,
    simplify_tolerance_pixels: float,
    coverage_policy: ProjectionCoveragePolicy,
) -> CameraProjectionLayout:
    visible_pixel_count = coverage_result.visible_pixel_count
    if frame_count <= 0:
        raise ValueError("at least one alpha mask must be accumulated")
    if visible_pixel_count == 0:
        raise CameraProjectionLayoutError(
            "camera projection sequence contains no pixels after alpha coverage cleanup"
        )

    minimum_x = width
    minimum_y = height
    maximum_x = -1
    maximum_y = -1
    for y in range(height):
        row_start = y * width
        for x in range(width):
            if not union_mask[row_start + x]:
                continue
            minimum_x = min(minimum_x, x)
            minimum_y = min(minimum_y, y)
            maximum_x = max(maximum_x, x)
            maximum_y = max(maximum_y, y)

    crop = ProjectionCropBounds(
        minimum_x=max(0, minimum_x - padding_pixels),
        minimum_y=max(0, minimum_y - padding_pixels),
        maximum_x=min(width, maximum_x + 1 + padding_pixels),
        maximum_y=min(height, maximum_y + 1 + padding_pixels),
    )
    try:
        contour = build_contour_from_mask(
            union_mask,
            width=width,
            height=height,
            mode=contour_mode,
            simplify_tolerance_pixels=simplify_tolerance_pixels,
        )
    except ProjectionContourError as exc:
        raise CameraProjectionLayoutError(str(exc)) from exc
    return CameraProjectionLayout(
        full_width=width,
        full_height=height,
        crop=crop,
        hull=contour.points,
        alpha_threshold=float(alpha_threshold),
        padding_pixels=padding_pixels,
        frame_count=frame_count,
        visible_pixel_count=visible_pixel_count,
        contour_mode=contour.mode,
        source_contour_vertex_count=contour.source_vertex_count,
        outer_component_count=contour.outer_component_count,
        simplify_tolerance_pixels=float(simplify_tolerance_pixels),
        contour_fallback_reason=contour.fallback_reason,
        coverage_mode=coverage_result.mode,
        coverage_core_alpha_threshold=float(
            coverage_policy.core_alpha_threshold
        ),
        coverage_raw_nonzero_pixel_count=coverage_result.raw_nonzero_pixel_count,
        coverage_strong_pixel_count=coverage_result.strong_pixel_count,
        coverage_component_count_before_cleanup=(
            coverage_result.component_count_before_cleanup
        ),
        coverage_component_count_after_cleanup=(
            coverage_result.component_count_after_cleanup
        ),
        coverage_removed_component_pixel_count=(
            coverage_result.removed_component_pixel_count
        ),
        coverage_filled_hole_pixel_count=coverage_result.filled_hole_pixel_count,
        coverage_used_weak_only_fallback=(
            coverage_result.used_weak_only_fallback
        ),
    )


@dataclass(slots=True)
class ProjectionAlphaUnionAccumulator:
    """Max-union frame alpha coverage in one fixed-size byte buffer."""

    width: int
    height: int
    alpha_threshold: float
    padding_pixels: int
    contour_mode: ProjectionContourMode = ProjectionContourMode.SIMPLIFIED_CONCAVE
    simplify_tolerance_pixels: float = 1.0
    coverage_policy: ProjectionCoveragePolicy = _COMPATIBILITY_COVERAGE_POLICY
    _union_mask: bytearray = field(init=False, repr=False)
    _frame_count: int = field(init=False, default=0, repr=False)
    _visible_pixel_count: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        _validate_dimensions(self.width, self.height)
        _validate_layout_policy(
            self.alpha_threshold,
            self.padding_pixels,
            self.contour_mode,
            self.simplify_tolerance_pixels,
            self.coverage_policy,
        )
        self.alpha_threshold = float(self.alpha_threshold)
        self.simplify_tolerance_pixels = float(self.simplify_tolerance_pixels)
        self._union_mask = bytearray(self.width * self.height)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def visible_pixel_count(self) -> int:
        """Return raw max-union non-zero coverage before final cleanup."""

        return self._visible_pixel_count

    @property
    def allocated_mask_bytes(self) -> int:
        return len(self._union_mask)

    def add_coverage(
        self,
        alpha_coverage: bytes | bytearray,
        *,
        frame_index: int | None = None,
    ) -> int:
        """Max-union one 8-bit coverage frame and return newly non-zero pixels."""

        if frame_index is not None and (
            not isinstance(frame_index, int) or frame_index < 0
        ):
            raise ValueError("frame_index must be a non-negative integer or None")
        _validate_alpha_mask(
            alpha_coverage,
            expected_size=self.width * self.height,
            frame_index=frame_index,
        )
        newly_visible = 0
        for index, value in enumerate(alpha_coverage):
            resolved = int(value)
            previous = self._union_mask[index]
            if resolved > previous:
                if previous == 0 and resolved > 0:
                    newly_visible += 1
                self._union_mask[index] = resolved
        self._frame_count += 1
        self._visible_pixel_count += newly_visible
        return newly_visible

    def add_mask(
        self,
        alpha_mask: bytes | bytearray,
        *,
        frame_index: int | None = None,
    ) -> int:
        """Compatibility alias accepting binary or 8-bit coverage bytes."""

        return self.add_coverage(alpha_mask, frame_index=frame_index)

    def build_layout(self) -> CameraProjectionLayout:
        coverage_result = build_projection_coverage_mask(
            self._union_mask,
            width=self.width,
            height=self.height,
            fringe_alpha_threshold=self.alpha_threshold,
            policy=self.coverage_policy,
        )
        return _layout_from_union_mask(
            bytearray(coverage_result.mask),
            width=self.width,
            height=self.height,
            alpha_threshold=self.alpha_threshold,
            padding_pixels=self.padding_pixels,
            frame_count=self._frame_count,
            coverage_result=coverage_result,
            contour_mode=self.contour_mode,
            simplify_tolerance_pixels=self.simplify_tolerance_pixels,
            coverage_policy=self.coverage_policy,
        )


def build_sequence_union_layout(
    alpha_masks: Tuple[bytes | bytearray, ...],
    *,
    width: int,
    height: int,
    alpha_threshold: float,
    padding_pixels: int,
    contour_mode: ProjectionContourMode = ProjectionContourMode.SIMPLIFIED_CONCAVE,
    simplify_tolerance_pixels: float = 1.0,
    coverage_policy: ProjectionCoveragePolicy = _COMPATIBILITY_COVERAGE_POLICY,
) -> CameraProjectionLayout:
    """Build one stable layout from binary masks or 8-bit alpha coverage frames."""

    if not isinstance(alpha_masks, tuple) or not alpha_masks:
        raise ValueError("alpha_masks must be a non-empty tuple")
    accumulator = ProjectionAlphaUnionAccumulator(
        width=width,
        height=height,
        alpha_threshold=alpha_threshold,
        padding_pixels=padding_pixels,
        contour_mode=contour_mode,
        simplify_tolerance_pixels=simplify_tolerance_pixels,
        coverage_policy=coverage_policy,
    )
    for frame_index, alpha_mask in enumerate(alpha_masks):
        accumulator.add_coverage(alpha_mask, frame_index=frame_index)
    return accumulator.build_layout()


def build_full_frame_layout(
    width: int,
    height: int,
    *,
    frame_count: int = 1,
) -> CameraProjectionLayout:
    _validate_dimensions(width, height)
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count <= 0:
        raise ValueError("frame_count must be a positive integer")
    visible_pixels = width * height
    return CameraProjectionLayout(
        full_width=width,
        full_height=height,
        crop=ProjectionCropBounds(0, 0, width, height),
        hull=(
            ProjectionPixelPoint(0, 0),
            ProjectionPixelPoint(width, 0),
            ProjectionPixelPoint(width, height),
            ProjectionPixelPoint(0, height),
        ),
        alpha_threshold=0.0,
        padding_pixels=0,
        frame_count=frame_count,
        visible_pixel_count=visible_pixels,
        contour_mode=ProjectionContourMode.CONVEX_HULL,
        source_contour_vertex_count=4,
        outer_component_count=1,
        simplify_tolerance_pixels=0.0,
        coverage_mode=ProjectionCoverageMode.BINARY_THRESHOLD,
        coverage_core_alpha_threshold=0.0,
        coverage_raw_nonzero_pixel_count=visible_pixels,
        coverage_strong_pixel_count=visible_pixels,
        coverage_component_count_before_cleanup=1,
        coverage_component_count_after_cleanup=1,
    )


__all__ = [
    "CameraProjectionLayout",
    "CameraProjectionLayoutError",
    "ProjectionAlphaUnionAccumulator",
    "ProjectionContourMode",
    "ProjectionCoverageMode",
    "ProjectionCoveragePolicy",
    "ProjectionCropBounds",
    "ProjectionPixelPoint",
    "ProjectionTriangle",
    "build_full_frame_layout",
    "build_sequence_union_layout",
    "convex_hull",
    "simplify_concave_contour",
    "triangulate_convex_hull",
    "triangulate_simple_contour",
]
