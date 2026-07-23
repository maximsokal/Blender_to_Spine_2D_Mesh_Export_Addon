"""Blender-independent settings for one A1 multi-object output transaction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ..domain.baking import sanitize_filename_stem
from ..domain.spine import UniformScaleMode
from .a1_numeric_contracts import (
    require_finite_number,
    require_identity,
    require_integer,
    require_non_empty_string,
)


class A1MultiObjectMode(str, Enum):
    STANDALONE = "STANDALONE"
    CONNECTED = "CONNECTED"
    MIXED = "MIXED"


class ConnectedCameraRenderPolicy(str, Enum):
    """How connected camera-dependent objects become visible Spine layers."""

    INDIVIDUAL_LAYERS = "INDIVIDUAL_LAYERS"
    AUTO_GROUPED_CAMERA = "AUTO_GROUPED_CAMERA"
    GROUPED_CAMERA_REQUIRED = "GROUPED_CAMERA_REQUIRED"


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
    connected_camera_render_policy: ConnectedCameraRenderPolicy = (
        ConnectedCameraRenderPolicy.AUTO_GROUPED_CAMERA
    )

    def __post_init__(self) -> None:
        if not isinstance(self.output_directory, Path):
            raise TypeError("output_directory must be pathlib.Path")
        require_non_empty_string(self.output_stem, "output_stem")
        sanitize_filename_stem(self.output_stem)
        if not isinstance(self.mode, A1MultiObjectMode):
            raise TypeError("mode must be A1MultiObjectMode")
        require_integer(self.json_indent, "json_indent", minimum=0, maximum=16)
        if not isinstance(self.namespace_animations, bool):
            raise TypeError("namespace_animations must be bool")
        require_non_empty_string(self.animation_separator, "animation_separator")
        require_identity(self.connected_group_prefix, "connected_group_prefix")
        if self.anchor_component_id is not None:
            require_identity(self.anchor_component_id, "anchor_component_id")
        require_finite_number(
            self.z_tolerance,
            "z_tolerance",
            minimum=0.0,
        )
        if not isinstance(self.connected_scale_mode, UniformScaleMode):
            raise TypeError("connected_scale_mode must be UniformScaleMode")
        if not isinstance(
            self.connected_camera_render_policy,
            ConnectedCameraRenderPolicy,
        ):
            raise TypeError(
                "connected_camera_render_policy must be ConnectedCameraRenderPolicy"
            )

    @property
    def resolved_output_stem(self) -> str:
        return sanitize_filename_stem(self.output_stem)

    @property
    def json_path(self) -> Path:
        root = self.output_directory.expanduser().resolve(strict=False)
        return root / f"{self.resolved_output_stem}.json"


__all__ = [
    "A1MultiObjectExportSettings",
    "A1MultiObjectMode",
    "A1MultiObjectStage",
    "ConnectedCameraRenderPolicy",
]
