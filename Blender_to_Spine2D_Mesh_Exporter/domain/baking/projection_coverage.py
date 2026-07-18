"""Coverage-aware alpha reconstruction and conservative binary morphology for B4."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import ceil, isfinite
from typing import Iterable, Tuple


class ProjectionCoverageError(ValueError):
    """Raised when alpha coverage cannot produce a valid cleaned mask."""


class ProjectionCoverageMode(str, Enum):
    """Policy used to convert sequence-union alpha coverage into geometry coverage."""

    BINARY_THRESHOLD = "BINARY_THRESHOLD"
    COVERAGE_THRESHOLD = "COVERAGE_THRESHOLD"
    HYSTERESIS_MORPHOLOGY = "HYSTERESIS_MORPHOLOGY"


@dataclass(frozen=True, slots=True)
class ProjectionCoveragePolicy:
    """Immutable output policy for antialias coverage and conservative morphology."""

    mode: ProjectionCoverageMode = ProjectionCoverageMode.HYSTERESIS_MORPHOLOGY
    core_alpha_threshold: float = 0.5
    minimum_component_pixels: int = 2
    maximum_hole_pixels: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ProjectionCoverageMode):
            raise TypeError("mode must be ProjectionCoverageMode")
        if (
            isinstance(self.core_alpha_threshold, bool)
            or not isinstance(self.core_alpha_threshold, (int, float))
            or not isfinite(float(self.core_alpha_threshold))
            or not 0.0 <= float(self.core_alpha_threshold) <= 1.0
        ):
            raise ValueError("core_alpha_threshold must be finite and in [0, 1]")
        if (
            not isinstance(self.minimum_component_pixels, int)
            or isinstance(self.minimum_component_pixels, bool)
            or self.minimum_component_pixels < 1
        ):
            raise ValueError("minimum_component_pixels must be a positive integer")
        if (
            not isinstance(self.maximum_hole_pixels, int)
            or isinstance(self.maximum_hole_pixels, bool)
            or self.maximum_hole_pixels < 0
        ):
            raise ValueError("maximum_hole_pixels must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ProjectionCoverageResult:
    """One cleaned binary mask plus diagnostics retained by the projection layout."""

    mask: bytes
    mode: ProjectionCoverageMode
    visible_pixel_count: int
    raw_nonzero_pixel_count: int
    strong_pixel_count: int
    component_count_before_cleanup: int
    component_count_after_cleanup: int
    removed_component_pixel_count: int
    filled_hole_pixel_count: int
    used_weak_only_fallback: bool

    def __post_init__(self) -> None:
        if not isinstance(self.mask, bytes):
            raise TypeError("mask must be bytes")
        if not isinstance(self.mode, ProjectionCoverageMode):
            raise TypeError("mode must be ProjectionCoverageMode")
        for field_name in (
            "visible_pixel_count",
            "raw_nonzero_pixel_count",
            "strong_pixel_count",
            "component_count_before_cleanup",
            "component_count_after_cleanup",
            "removed_component_pixel_count",
            "filled_hole_pixel_count",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.visible_pixel_count != sum(1 for value in self.mask if value):
            raise ValueError("visible_pixel_count does not match mask")
        if not isinstance(self.used_weak_only_fallback, bool):
            raise TypeError("used_weak_only_fallback must be bool")


def _validate_dimensions(width: int, height: int) -> None:
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError("width must be a positive integer")
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise ValueError("height must be a positive integer")


def _validate_alpha_threshold(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return float(value)


def _coverage_byte_threshold(value: float) -> int:
    """Return the smallest byte whose normalized coverage satisfies ``>= value``."""

    resolved = _validate_alpha_threshold(value, "alpha threshold")
    if resolved <= 0.0:
        return 0
    return min(255, max(1, int(ceil(resolved * 255.0 - 1e-12))))


def _neighbors4(index: int, width: int, height: int) -> Iterable[int]:
    x = index % width
    y = index // width
    if x > 0:
        yield index - 1
    if x + 1 < width:
        yield index + 1
    if y > 0:
        yield index - width
    if y + 1 < height:
        yield index + width


def _neighbors8(index: int, width: int, height: int) -> Iterable[int]:
    x = index % width
    y = index // width
    for delta_y in (-1, 0, 1):
        candidate_y = y + delta_y
        if not 0 <= candidate_y < height:
            continue
        for delta_x in (-1, 0, 1):
            if delta_x == 0 and delta_y == 0:
                continue
            candidate_x = x + delta_x
            if 0 <= candidate_x < width:
                yield candidate_y * width + candidate_x


def _label_components(
    mask: bytearray,
    *,
    width: int,
    height: int,
) -> Tuple[Tuple[int, ...], ...]:
    """Label foreground with 8-connectivity so diagonal antialias strokes survive."""

    visited = bytearray(len(mask))
    components: list[Tuple[int, ...]] = []
    for start, value in enumerate(mask):
        if not value or visited[start]:
            continue
        visited[start] = 1
        queue = deque((start,))
        indices: list[int] = []
        while queue:
            current = queue.popleft()
            indices.append(current)
            for neighbor in _neighbors8(current, width, height):
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    queue.append(neighbor)
        components.append(tuple(indices))
    return tuple(components)


def _hysteresis_reconstruct(
    weak_mask: bytearray,
    strong_mask: bytearray,
    *,
    width: int,
    height: int,
) -> tuple[bytearray, bool]:
    strong_indices = tuple(index for index, value in enumerate(strong_mask) if value)
    if not strong_indices:
        return bytearray(weak_mask), any(weak_mask)

    result = bytearray(len(weak_mask))
    queue = deque(strong_indices)
    for index in strong_indices:
        result[index] = 1
    while queue:
        current = queue.popleft()
        for neighbor in _neighbors8(current, width, height):
            if weak_mask[neighbor] and not result[neighbor]:
                result[neighbor] = 1
                queue.append(neighbor)
    return result, False


def _remove_small_components(
    mask: bytearray,
    *,
    width: int,
    height: int,
    minimum_pixels: int,
) -> tuple[bytearray, int, int, int]:
    components = _label_components(mask, width=width, height=height)
    if not components:
        return bytearray(mask), 0, 0, 0

    largest_index = min(
        range(len(components)),
        key=lambda index: (-len(components[index]), min(components[index])),
    )
    result = bytearray(len(mask))
    removed_pixels = 0
    retained_components = 0
    for component_index, component in enumerate(components):
        keep = component_index == largest_index or len(component) >= minimum_pixels
        if keep:
            retained_components += 1
            for index in component:
                result[index] = 1
        else:
            removed_pixels += len(component)
    return result, len(components), retained_components, removed_pixels


def _fill_small_enclosed_holes(
    mask: bytearray,
    *,
    width: int,
    height: int,
    maximum_pixels: int,
) -> tuple[bytearray, int]:
    if maximum_pixels == 0:
        return bytearray(mask), 0

    result = bytearray(mask)
    visited = bytearray(len(mask))
    filled_pixels = 0
    for start, value in enumerate(mask):
        if value or visited[start]:
            continue
        visited[start] = 1
        queue = deque((start,))
        region: list[int] = []
        touches_border = False
        while queue:
            current = queue.popleft()
            region.append(current)
            x = current % width
            y = current // width
            if x == 0 or y == 0 or x + 1 == width or y + 1 == height:
                touches_border = True
            for neighbor in _neighbors4(current, width, height):
                if not mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    queue.append(neighbor)
        if not touches_border and len(region) <= maximum_pixels:
            for index in region:
                result[index] = 1
            filled_pixels += len(region)
    return result, filled_pixels


def build_projection_coverage_mask(
    coverage: bytes | bytearray,
    *,
    width: int,
    height: int,
    fringe_alpha_threshold: float,
    policy: ProjectionCoveragePolicy,
) -> ProjectionCoverageResult:
    """Convert max-unioned 8-bit alpha coverage into a cleaned binary geometry mask.

    `BINARY_THRESHOLD` consumes an already-binary compatibility mask and treats every
    non-zero byte as visible. `COVERAGE_THRESHOLD` applies the configured fringe threshold
    directly to 8-bit coverage. `HYSTERESIS_MORPHOLOGY` retains weak antialias coverage only
    when connected to a strong core, with a weak-only fallback for translucent objects.
    Component cleanup always keeps the largest component, and hole cleanup fills only enclosed
    regions without bridging disconnected objects.
    """

    _validate_dimensions(width, height)
    if not isinstance(coverage, (bytes, bytearray)):
        raise TypeError("coverage must be bytes or bytearray")
    if len(coverage) != width * height:
        raise ValueError(
            f"coverage has {len(coverage)} entries; expected {width * height}"
        )
    if not isinstance(policy, ProjectionCoveragePolicy):
        raise TypeError("policy must be ProjectionCoveragePolicy")

    fringe_threshold = _coverage_byte_threshold(fringe_alpha_threshold)
    core_threshold = _coverage_byte_threshold(policy.core_alpha_threshold)
    raw_nonzero_count = sum(1 for value in coverage if value)

    if policy.mode is ProjectionCoverageMode.BINARY_THRESHOLD:
        candidate = bytearray(1 if value else 0 for value in coverage)
        strong_count = sum(candidate)
        used_weak_only_fallback = False
    elif policy.mode is ProjectionCoverageMode.COVERAGE_THRESHOLD:
        if fringe_threshold == 0:
            candidate = bytearray(b"\x01" * len(coverage))
        else:
            candidate = bytearray(
                1 if int(value) >= fringe_threshold else 0 for value in coverage
            )
        strong_count = sum(candidate)
        used_weak_only_fallback = False
    else:
        weak_mask = bytearray(
            1 if fringe_threshold == 0 or int(value) >= fringe_threshold else 0
            for value in coverage
        )
        strong_mask = bytearray(
            1 if core_threshold == 0 or int(value) >= core_threshold else 0
            for value in coverage
        )
        strong_count = sum(strong_mask)
        candidate, used_weak_only_fallback = _hysteresis_reconstruct(
            weak_mask,
            strong_mask,
            width=width,
            height=height,
        )

    cleaned, before_count, _, removed_pixels = _remove_small_components(
        candidate,
        width=width,
        height=height,
        minimum_pixels=policy.minimum_component_pixels,
    )
    cleaned, filled_pixels = _fill_small_enclosed_holes(
        cleaned,
        width=width,
        height=height,
        maximum_pixels=policy.maximum_hole_pixels,
    )
    final_components = _label_components(cleaned, width=width, height=height)
    return ProjectionCoverageResult(
        mask=bytes(cleaned),
        mode=policy.mode,
        visible_pixel_count=sum(cleaned),
        raw_nonzero_pixel_count=raw_nonzero_count,
        strong_pixel_count=strong_count,
        component_count_before_cleanup=before_count,
        component_count_after_cleanup=len(final_components),
        removed_component_pixel_count=removed_pixels,
        filled_hole_pixel_count=filled_pixels,
        used_weak_only_fallback=used_weak_only_fallback,
    )


__all__ = [
    "ProjectionCoverageError",
    "ProjectionCoverageMode",
    "ProjectionCoveragePolicy",
    "ProjectionCoverageResult",
    "build_projection_coverage_mask",
]
