"""Typed B4 dynamic-range, tone-mapping, and alpha-representation policy."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Iterable

from .model import TextureFormat


class ProjectionOutputPolicyError(ValueError):
    """Raised when one B4 output policy is incompatible with its texture format."""


class ProjectionDynamicRange(str, Enum):
    AUTO_BY_FORMAT = "AUTO_BY_FORMAT"
    DISPLAY_REFERRED_SDR = "DISPLAY_REFERRED_SDR"
    SCENE_LINEAR_HDR = "SCENE_LINEAR_HDR"


class ProjectionToneMapping(str, Enum):
    AUTO_BY_DYNAMIC_RANGE = "AUTO_BY_DYNAMIC_RANGE"
    SCENE_VIEW_TRANSFORM = "SCENE_VIEW_TRANSFORM"
    NONE = "NONE"


class ProjectionAlphaRepresentation(str, Enum):
    AUTO_BY_FORMAT = "AUTO_BY_FORMAT"
    STRAIGHT = "STRAIGHT"
    PREMULTIPLIED = "PREMULTIPLIED"


@dataclass(frozen=True, slots=True)
class ProjectionOutputPolicy:
    """User-facing policy resolved against the planned texture format."""

    dynamic_range: ProjectionDynamicRange = ProjectionDynamicRange.AUTO_BY_FORMAT
    tone_mapping: ProjectionToneMapping = ProjectionToneMapping.AUTO_BY_DYNAMIC_RANGE
    alpha_representation: ProjectionAlphaRepresentation = (
        ProjectionAlphaRepresentation.AUTO_BY_FORMAT
    )

    def __post_init__(self) -> None:
        if not isinstance(self.dynamic_range, ProjectionDynamicRange):
            raise TypeError("dynamic_range must be ProjectionDynamicRange")
        if not isinstance(self.tone_mapping, ProjectionToneMapping):
            raise TypeError("tone_mapping must be ProjectionToneMapping")
        if not isinstance(
            self.alpha_representation,
            ProjectionAlphaRepresentation,
        ):
            raise TypeError(
                "alpha_representation must be ProjectionAlphaRepresentation"
            )


@dataclass(frozen=True, slots=True)
class ResolvedProjectionOutputPolicy:
    texture_format: TextureFormat
    dynamic_range: ProjectionDynamicRange
    tone_mapping: ProjectionToneMapping
    alpha_representation: ProjectionAlphaRepresentation
    color_depth: str
    float_buffer: bool

    def __post_init__(self) -> None:
        if not isinstance(self.texture_format, TextureFormat):
            raise TypeError("texture_format must be TextureFormat")
        if self.dynamic_range is ProjectionDynamicRange.AUTO_BY_FORMAT:
            raise ValueError("resolved dynamic_range cannot remain AUTO_BY_FORMAT")
        if self.tone_mapping is ProjectionToneMapping.AUTO_BY_DYNAMIC_RANGE:
            raise ValueError("resolved tone_mapping cannot remain AUTO_BY_DYNAMIC_RANGE")
        if self.alpha_representation is ProjectionAlphaRepresentation.AUTO_BY_FORMAT:
            raise ValueError("resolved alpha_representation cannot remain AUTO_BY_FORMAT")
        if self.color_depth not in {"8", "16", "32"}:
            raise ValueError("color_depth must be 8, 16, or 32")
        if not isinstance(self.float_buffer, bool):
            raise TypeError("float_buffer must be bool")

    @property
    def blender_alpha_mode(self) -> str:
        return (
            "STRAIGHT"
            if self.alpha_representation is ProjectionAlphaRepresentation.STRAIGHT
            else "PREMUL"
        )


def resolve_projection_output_policy(
    policy: ProjectionOutputPolicy,
    texture_format: TextureFormat,
) -> ResolvedProjectionOutputPolicy:
    if not isinstance(policy, ProjectionOutputPolicy):
        raise TypeError("policy must be ProjectionOutputPolicy")
    if not isinstance(texture_format, TextureFormat):
        raise TypeError("texture_format must be TextureFormat")
    if texture_format is TextureFormat.JPEG:
        raise ProjectionOutputPolicyError(
            "B4 projection requires an alpha-capable output format; JPEG is unsupported"
        )

    dynamic_range = policy.dynamic_range
    if dynamic_range is ProjectionDynamicRange.AUTO_BY_FORMAT:
        dynamic_range = (
            ProjectionDynamicRange.SCENE_LINEAR_HDR
            if texture_format is TextureFormat.OPEN_EXR
            else ProjectionDynamicRange.DISPLAY_REFERRED_SDR
        )

    tone_mapping = policy.tone_mapping
    if tone_mapping is ProjectionToneMapping.AUTO_BY_DYNAMIC_RANGE:
        tone_mapping = (
            ProjectionToneMapping.NONE
            if dynamic_range is ProjectionDynamicRange.SCENE_LINEAR_HDR
            else ProjectionToneMapping.SCENE_VIEW_TRANSFORM
        )

    alpha_representation = policy.alpha_representation
    if alpha_representation is ProjectionAlphaRepresentation.AUTO_BY_FORMAT:
        alpha_representation = (
            ProjectionAlphaRepresentation.PREMULTIPLIED
            if texture_format is TextureFormat.OPEN_EXR
            else ProjectionAlphaRepresentation.STRAIGHT
        )

    if dynamic_range is ProjectionDynamicRange.SCENE_LINEAR_HDR:
        if texture_format is not TextureFormat.OPEN_EXR:
            raise ProjectionOutputPolicyError(
                "SCENE_LINEAR_HDR requires TextureFormat.OPEN_EXR"
            )
        if tone_mapping is not ProjectionToneMapping.NONE:
            raise ProjectionOutputPolicyError(
                "SCENE_LINEAR_HDR requires tone_mapping=NONE"
            )
        color_depth = "32"
        float_buffer = True
    else:
        if texture_format is TextureFormat.OPEN_EXR:
            raise ProjectionOutputPolicyError(
                "DISPLAY_REFERRED_SDR cannot be written as OPEN_EXR in B4; "
                "use PNG/WEBP or select SCENE_LINEAR_HDR"
            )
        if tone_mapping is not ProjectionToneMapping.SCENE_VIEW_TRANSFORM:
            raise ProjectionOutputPolicyError(
                "DISPLAY_REFERRED_SDR requires SCENE_VIEW_TRANSFORM"
            )
        color_depth = "8"
        float_buffer = False

    return ResolvedProjectionOutputPolicy(
        texture_format=texture_format,
        dynamic_range=dynamic_range,
        tone_mapping=tone_mapping,
        alpha_representation=alpha_representation,
        color_depth=color_depth,
        float_buffer=float_buffer,
    )


def _normalize_alpha_mode(value: str) -> ProjectionAlphaRepresentation:
    normalized = str(value or "").strip().upper()
    if normalized in {"STRAIGHT", "CHANNEL_PACKED", "NONE"}:
        return ProjectionAlphaRepresentation.STRAIGHT
    if normalized in {"PREMUL", "PREMULTIPLIED"}:
        return ProjectionAlphaRepresentation.PREMULTIPLIED
    raise ProjectionOutputPolicyError(
        f"unsupported Blender image alpha mode: {value!r}"
    )


def convert_rgba_alpha_representation(
    pixels: Iterable[float],
    *,
    source_alpha_mode: str,
    target: ProjectionAlphaRepresentation,
) -> array:
    """Convert RGBA pixels without clamping HDR RGB values.

    Alpha itself is clamped to `[0, 1]` because associated-alpha arithmetic outside this range
    is undefined for the exported texture contract. RGB remains finite and unbounded so EXR HDR
    values greater than one survive.
    """

    if not isinstance(target, ProjectionAlphaRepresentation) or target is (
        ProjectionAlphaRepresentation.AUTO_BY_FORMAT
    ):
        raise TypeError("target must be resolved STRAIGHT or PREMULTIPLIED")
    source = _normalize_alpha_mode(source_alpha_mode)
    values = array("f", (float(value) for value in pixels))
    if len(values) % 4:
        raise ValueError("RGBA pixel buffer length must be divisible by four")
    if source is target:
        if not all(isfinite(value) for value in values):
            raise ProjectionOutputPolicyError("RGBA pixel buffer contains non-finite values")
        return values

    result = array("f", [0.0]) * len(values)
    for offset in range(0, len(values), 4):
        red, green, blue, alpha = values[offset : offset + 4]
        if not all(isfinite(value) for value in (red, green, blue, alpha)):
            raise ProjectionOutputPolicyError(
                f"RGBA pixel buffer contains non-finite values at pixel {offset // 4}"
            )
        resolved_alpha = max(0.0, min(1.0, float(alpha)))
        if source is ProjectionAlphaRepresentation.STRAIGHT:
            result[offset] = red * resolved_alpha
            result[offset + 1] = green * resolved_alpha
            result[offset + 2] = blue * resolved_alpha
        elif resolved_alpha > 0.0:
            result[offset] = red / resolved_alpha
            result[offset + 1] = green / resolved_alpha
            result[offset + 2] = blue / resolved_alpha
        else:
            result[offset] = 0.0
            result[offset + 1] = 0.0
            result[offset + 2] = 0.0
        result[offset + 3] = resolved_alpha
    return result


__all__ = [
    "ProjectionAlphaRepresentation",
    "ProjectionDynamicRange",
    "ProjectionOutputPolicy",
    "ProjectionOutputPolicyError",
    "ProjectionToneMapping",
    "ResolvedProjectionOutputPolicy",
    "convert_rgba_alpha_representation",
    "resolve_projection_output_policy",
]
