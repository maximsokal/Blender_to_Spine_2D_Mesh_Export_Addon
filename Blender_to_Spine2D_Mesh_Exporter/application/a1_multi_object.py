"""Blender-independent settings for one A1 multi-object output transaction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path

from ..domain.baking import sanitize_filename_stem
from ..domain.spine import UniformScaleMode


class A1MultiObjectMode(str, Enum):
    STANDALONE = "STANDALONE"
    CONNECTED = "CONNECTED"


class A1MultiObjectStage(str, Enum):
    VALIDATE_REQUEST = "VALIDATE_REQUEST"
    PREPARE_OBJECTS = "PREPARE_OBJECTS"
    VALIDATE_OUTPUTS = "VALIDATE_OUTPUTS"
    COMPOSE_DOCUMENT = "COMPOSE_DOCUMENT"
    SERIALIZE_DOCUMENT = "SERIALIZE_DOCUMENT"
    STAGE_OUTPUTS = "STAGE_OUTPUTS"
    COMMIT_OUTPUTS = "COMMIT_OUTPUTS"

    @property
    def error_code(self) -> str:
        return f"A1_MULTI_{self.value}_FAILED"


@dataclass(frozen=True, slots=True)
class A1MultiObjectExportSettings:
    output_directory: Path
    output_stem: str
    mode: A1MultiObjectMode = A1MultiObjectMode.STANDALONE
    json_indent: int = 2
    namespace_animations: bool = True
    animation_separator: str = "/"
    connected_group_prefix: str = "all_objects"
    anchor_component_id: str | None = None
    z_tolerance: float = 1e-4
    connected_scale_mode: UniformScaleMode = UniformScaleMode.AVERAGE

    def __post_init__(self) -> None:
        if not isinstance(self.output_directory, Path):
            raise TypeError("output_directory must be pathlib.Path")
        if not isinstance(self.output_stem, str) or not self.output_stem.strip():
            raise ValueError("output_stem must be a non-empty string")
        sanitize_filename_stem(self.output_stem)
        if not isinstance(self.mode, A1MultiObjectMode):
            raise TypeError("mode must be A1MultiObjectMode")
        if not isinstance(self.json_indent, int) or not 0 <= self.json_indent <= 16:
            raise ValueError("json_indent must be an integer in [0, 16]")
        if not isinstance(self.namespace_animations, bool):
            raise TypeError("namespace_animations must be bool")
        if not isinstance(self.animation_separator, str) or not self.animation_separator:
            raise ValueError("animation_separator must be a non-empty string")
        if (
            not isinstance(self.connected_group_prefix, str)
            or not self.connected_group_prefix.strip()
        ):
            raise ValueError("connected_group_prefix must be a non-empty string")
        if self.anchor_component_id is not None and (
            not isinstance(self.anchor_component_id, str)
            or not self.anchor_component_id.strip()
        ):
            raise ValueError(
                "anchor_component_id must be a non-empty string or None"
            )
        if not isinstance(self.z_tolerance, (int, float)) or not isfinite(
            float(self.z_tolerance)
        ):
            raise ValueError("z_tolerance must be finite")
        if self.z_tolerance < 0.0:
            raise ValueError("z_tolerance cannot be negative")
        if not isinstance(self.connected_scale_mode, UniformScaleMode):
            raise TypeError("connected_scale_mode must be UniformScaleMode")

    @property
    def resolved_output_stem(self) -> str:
        return sanitize_filename_stem(self.output_stem)

    @property
    def json_path(self) -> Path:
        root = self.output_directory.expanduser().resolve(strict=False)
        return root / f"{self.resolved_output_stem}.json"
