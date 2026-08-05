"""Pure UV-envelope requirements for camera projection image crops.

Rendered alpha is intentionally allowed to be smaller than the prepared attachment
surface. Transparent source materials and compact parallax proxy geometry can therefore
own valid full-frame UVs outside the non-zero alpha crop. The final image crop must cover
both domains before the staged PNG is rewritten:

``alpha coverage crop ∪ prepared attachment UV bounds``.

This module is Blender-independent and never changes the alpha-derived contour. It only
expands the rectangular crop enough to preserve every required normalized UV.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, floor, isfinite
from typing import Iterable

from .projection_layout import CameraProjectionLayout, ProjectionCropBounds


_UV_EPSILON = 1.0e-7


class ProjectionCropRequirementError(ValueError):
    """Raised when prepared attachment UVs cannot define a valid render crop."""


def _unit_coordinate(value: float, *, field_name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved):
        raise ProjectionCropRequirementError(f"{field_name} must be finite")
    if resolved < -_UV_EPSILON or resolved > 1.0 + _UV_EPSILON:
        raise ProjectionCropRequirementError(
            f"{field_name}={resolved} lies outside the full camera frame"
        )
    return min(1.0, max(0.0, resolved))


def _positive_pixel_interval(
    minimum: int,
    maximum: int,
    *,
    limit: int,
) -> tuple[int, int]:
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    lower = min(limit, max(0, int(minimum)))
    upper = min(limit, max(0, int(maximum)))
    if upper > lower:
        return lower, upper

    # A degenerate UV interval still owns one attachment vertex. Preserve one physical
    # pixel around it so the resulting ProjectionCropBounds remains valid.
    anchor = min(limit - 1, max(0, lower))
    return anchor, anchor + 1


@dataclass(frozen=True, slots=True)
class ProjectionUvBounds:
    """Closed normalized UV rectangle required by one camera attachment view."""

    minimum_u: float
    minimum_v: float
    maximum_u: float
    maximum_v: float

    def __post_init__(self) -> None:
        minimum_u = _unit_coordinate(self.minimum_u, field_name="minimum_u")
        minimum_v = _unit_coordinate(self.minimum_v, field_name="minimum_v")
        maximum_u = _unit_coordinate(self.maximum_u, field_name="maximum_u")
        maximum_v = _unit_coordinate(self.maximum_v, field_name="maximum_v")
        if maximum_u < minimum_u:
            raise ProjectionCropRequirementError(
                "maximum_u cannot be smaller than minimum_u"
            )
        if maximum_v < minimum_v:
            raise ProjectionCropRequirementError(
                "maximum_v cannot be smaller than minimum_v"
            )
        object.__setattr__(self, "minimum_u", minimum_u)
        object.__setattr__(self, "minimum_v", minimum_v)
        object.__setattr__(self, "maximum_u", maximum_u)
        object.__setattr__(self, "maximum_v", maximum_v)

    @classmethod
    def from_uvs(
        cls,
        values: Iterable[tuple[float, float]],
        *,
        field_name: str = "uvs",
    ) -> "ProjectionUvBounds":
        resolved: list[tuple[float, float]] = []
        for index, uv in enumerate(values):
            if not isinstance(uv, tuple) or len(uv) != 2:
                raise TypeError(f"{field_name}[{index}] must contain two values")
            resolved.append(
                (
                    _unit_coordinate(
                        uv[0],
                        field_name=f"{field_name}[{index}][0]",
                    ),
                    _unit_coordinate(
                        uv[1],
                        field_name=f"{field_name}[{index}][1]",
                    ),
                )
            )
        if not resolved:
            raise ProjectionCropRequirementError(
                f"{field_name} must contain at least one UV coordinate"
            )
        return cls(
            minimum_u=min(uv[0] for uv in resolved),
            minimum_v=min(uv[1] for uv in resolved),
            maximum_u=max(uv[0] for uv in resolved),
            maximum_v=max(uv[1] for uv in resolved),
        )

    def pixel_crop(
        self,
        *,
        width: int,
        height: int,
        padding_pixels: int = 0,
    ) -> ProjectionCropBounds:
        """Convert Spine UV orientation to an exclusive Blender pixel rectangle."""

        for field_name, value in (("width", width), ("height", height)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if (
            not isinstance(padding_pixels, int)
            or isinstance(padding_pixels, bool)
            or padding_pixels < 0
        ):
            raise ValueError("padding_pixels must be a non-negative integer")

        minimum_x = floor(self.minimum_u * width) - padding_pixels
        maximum_x = ceil(self.maximum_u * width) + padding_pixels

        # Spine V grows upward; Blender image rows used by ProjectionCropBounds grow
        # downward. The upper UV edge therefore becomes the minimum pixel Y.
        minimum_y = floor((1.0 - self.maximum_v) * height) - padding_pixels
        maximum_y = ceil((1.0 - self.minimum_v) * height) + padding_pixels

        minimum_x, maximum_x = _positive_pixel_interval(
            minimum_x,
            maximum_x,
            limit=width,
        )
        minimum_y, maximum_y = _positive_pixel_interval(
            minimum_y,
            maximum_y,
            limit=height,
        )
        return ProjectionCropBounds(
            minimum_x=minimum_x,
            minimum_y=minimum_y,
            maximum_x=maximum_x,
            maximum_y=maximum_y,
        )


def expand_projection_layout_to_uv_bounds(
    layout: CameraProjectionLayout,
    required: ProjectionUvBounds | None,
) -> CameraProjectionLayout:
    """Return the alpha layout with a crop expanded to required attachment UVs."""

    if not isinstance(layout, CameraProjectionLayout):
        raise TypeError("layout must be CameraProjectionLayout")
    if required is None:
        return layout
    if not isinstance(required, ProjectionUvBounds):
        raise TypeError("required must be ProjectionUvBounds or None")

    uv_crop = required.pixel_crop(
        width=layout.full_width,
        height=layout.full_height,
        padding_pixels=layout.padding_pixels,
    )
    merged = ProjectionCropBounds(
        minimum_x=min(layout.crop.minimum_x, uv_crop.minimum_x),
        minimum_y=min(layout.crop.minimum_y, uv_crop.minimum_y),
        maximum_x=max(layout.crop.maximum_x, uv_crop.maximum_x),
        maximum_y=max(layout.crop.maximum_y, uv_crop.maximum_y),
    )
    if merged == layout.crop:
        return layout
    return replace(layout, crop=merged)


__all__ = [
    "ProjectionCropRequirementError",
    "ProjectionUvBounds",
    "expand_projection_layout_to_uv_bounds",
]
