"""Typed input and output contracts for export use cases.

This module intentionally contains no Blender API imports. Blender operators will
translate Scene/Object properties into :class:`ExportRequest` objects and render
:class:`ExportResult` objects back to the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Tuple


class IssueSeverity(str, Enum):
    """Severity levels exposed to the Blender UI and export logs."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class ExportSettings:
    """Immutable settings used by one export transaction."""

    texture_width: int
    texture_height: int
    output_directory: Path
    images_relative_path: str = "images"
    spine_version: str = "4.2.43"
    rig_profile: str = "LEGACY_ROTATABLE_MESH"
    seam_mode: str = "AUTO"
    angle_limit_degrees: float = 30.0
    bake_margin: int = 4
    sequence_start_frame: int = 0
    sequence_frame_count: int = 0
    preserve_debug_artifacts: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.texture_width, int) or self.texture_width <= 0:
            raise ValueError("texture_width must be a positive integer")
        if not isinstance(self.texture_height, int) or self.texture_height <= 0:
            raise ValueError("texture_height must be a positive integer")
        if not isinstance(self.output_directory, Path):
            raise TypeError("output_directory must be pathlib.Path")
        if self.seam_mode not in {"AUTO", "CUSTOM"}:
            raise ValueError("seam_mode must be 'AUTO' or 'CUSTOM'")
        if self.angle_limit_degrees <= 0.0 or self.angle_limit_degrees > 180.0:
            raise ValueError("angle_limit_degrees must be in the range (0, 180]")
        if self.bake_margin < 0:
            raise ValueError("bake_margin cannot be negative")
        if self.sequence_start_frame < 0:
            raise ValueError("sequence_start_frame cannot be negative")
        if self.sequence_frame_count < 0:
            raise ValueError("sequence_frame_count cannot be negative")


@dataclass(frozen=True, slots=True)
class ExportRequest:
    """Application request independent from a live ``bpy.types.Object``."""

    source_object_ids: Tuple[str, ...]
    active_object_id: str
    settings: ExportSettings
    connected_object_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_object_ids:
            raise ValueError("source_object_ids cannot be empty")
        if self.active_object_id not in self.source_object_ids:
            raise ValueError("active_object_id must be present in source_object_ids")
        unknown_connected = set(self.connected_object_ids) - set(self.source_object_ids)
        if unknown_connected:
            raise ValueError(
                "connected_object_ids contain unknown object ids: "
                + ", ".join(sorted(unknown_connected))
            )


@dataclass(frozen=True, slots=True)
class ExportIssue:
    """Structured user-facing issue produced by an export stage."""

    severity: IssueSeverity
    stage: str
    code: str
    message: str
    object_id: str | None = None
    technical_details: str | None = None
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Complete result returned by single- and multi-object export use cases."""

    success: bool
    output_files: Tuple[Path, ...] = ()
    issues: Tuple[ExportIssue, ...] = ()
    statistics: Mapping[str, int | float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.success and any(issue.severity is IssueSeverity.ERROR for issue in self.issues):
            raise ValueError("A successful ExportResult cannot contain ERROR issues")
        if not self.success and not any(
            issue.severity is IssueSeverity.ERROR for issue in self.issues
        ):
            raise ValueError("A failed ExportResult must contain at least one ERROR issue")
