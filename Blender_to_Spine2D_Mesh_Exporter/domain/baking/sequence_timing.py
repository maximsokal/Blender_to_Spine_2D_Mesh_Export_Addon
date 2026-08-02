"""Typed frame-rate timing for deterministic texture-sequence export."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


DEFAULT_SEQUENCE_FPS = 30.0
SEQUENCE_TIME_DECIMALS = 6


def _require_finite_number(
    value: object,
    field_name: str,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    if minimum is not None and resolved < minimum:
        raise ValueError(f"{field_name} must be greater than or equal to {minimum}")
    return resolved


@dataclass(frozen=True, slots=True)
class TextureSequenceTiming:
    """Resolve Scene FPS, optional override, and exact frame-index timestamps.

    ``override_fps == 0`` selects Blender Scene timing. Invalid or unavailable Scene
    timing falls back to 30 FPS. Timestamps are always derived directly from the frame
    index instead of accumulating rounded delays, so long sequences do not drift.
    """

    scene_fps: int = 30
    scene_fps_base: float = 1.0
    override_fps: float = 0.0
    fallback_fps: float = DEFAULT_SEQUENCE_FPS

    def __post_init__(self) -> None:
        if isinstance(self.scene_fps, bool) or not isinstance(self.scene_fps, int):
            raise TypeError("scene_fps must be int")
        if self.scene_fps < 0:
            raise ValueError("scene_fps must be non-negative")
        _require_finite_number(self.scene_fps_base, "scene_fps_base", minimum=0.0)
        _require_finite_number(self.override_fps, "override_fps", minimum=0.0)
        fallback = _require_finite_number(
            self.fallback_fps,
            "fallback_fps",
            minimum=0.0,
        )
        if fallback <= 0.0:
            raise ValueError("fallback_fps must be greater than zero")

    @property
    def scene_fps_value(self) -> float:
        base = float(self.scene_fps_base)
        if self.scene_fps > 0 and base > 0.0:
            resolved = float(self.scene_fps) / base
            if isfinite(resolved) and resolved > 0.0:
                return resolved
        return float(self.fallback_fps)

    @property
    def resolved_fps(self) -> float:
        override = float(self.override_fps)
        if override > 0.0:
            return override
        return self.scene_fps_value

    @property
    def frame_duration(self) -> float:
        return 1.0 / self.resolved_fps

    def time_for_frame_index(self, frame_index: int) -> float:
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise TypeError("frame_index must be int")
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        return round(
            float(frame_index) / self.resolved_fps,
            SEQUENCE_TIME_DECIMALS,
        )

    def duration_for_frame_count(self, frame_count: int) -> float:
        if isinstance(frame_count, bool) or not isinstance(frame_count, int):
            raise TypeError("frame_count must be int")
        if frame_count < 1:
            raise ValueError("frame_count must be greater than or equal to 1")
        return self.time_for_frame_index(frame_count)


__all__ = [
    "DEFAULT_SEQUENCE_FPS",
    "SEQUENCE_TIME_DECIMALS",
    "TextureSequenceTiming",
]
