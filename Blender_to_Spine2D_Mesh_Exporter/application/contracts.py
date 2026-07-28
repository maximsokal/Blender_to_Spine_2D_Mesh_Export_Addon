"""Typed input and output contracts for export use cases.

This module intentionally contains no Blender API imports. Blender operators will
translate Scene/Object properties into :class:`ExportRequest` objects and render
:class:`ExportResult` objects back to the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Tuple

from ..domain.spine.rig_profiles import A1RigProfile, resolve_a1_rig_profile
from .a1_numeric_contracts import (
    require_finite_number,
    require_integer,
    require_non_empty_string,
)


class IssueSeverity(str, Enum):
    """Severity levels exposed to the Blender UI and export logs."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


def _validate_relative_output_directory(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    normalized = value.replace("\\", "/").strip()
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(normalized)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in posix_path.parts
    ):
        raise ValueError(f"{field_name} must be a safe relative directory")


@dataclass(frozen=True, slots=True)
class ExportSettings:
    """Immutable settings used by one export transaction."""

    texture_width: int
    texture_height: int
    output_directory: Path
    images_relative_path: str = "images"
    spine_version: str = "4.2.43"
    rig_profile: str = A1RigProfile.THREE_AXIS_ROTATION.value
    seam_mode: str = "AUTO"
    angle_limit_degrees: float = 30.0
    bake_margin: int = 4
    sequence_start_frame: int = 0
    sequence_frame_count: int = 0
    preserve_debug_artifacts: bool = False

    def __post_init__(self) -> None:
        require_integer(self.texture_width, "texture_width", minimum=1)
        require_integer(self.texture_height, "texture_height", minimum=1)
        if not isinstance(self.output_directory, Path):
            raise TypeError("output_directory must be pathlib.Path")
        _validate_relative_output_directory(
            self.images_relative_path,
            "images_relative_path",
        )
        require_non_empty_string(self.spine_version, "spine_version")
        require_non_empty_string(self.rig_profile, "rig_profile")
        resolve_a1_rig_profile(self.rig_profile)
        if not isinstance(self.seam_mode, str):
            raise TypeError("seam_mode must be str")
        if self.seam_mode not in {"AUTO", "CUSTOM"}:
            raise ValueError("seam_mode must be 'AUTO' or 'CUSTOM'")
        require_finite_number(
            self.angle_limit_degrees,
            "angle_limit_degrees",
            minimum=0.0,
            maximum=180.0,
            minimum_inclusive=False,
        )
        require_integer(self.bake_margin, "bake_margin", minimum=0)
        require_integer(
            self.sequence_start_frame,
            "sequence_start_frame",
            minimum=0,
        )
        require_integer(
            self.sequence_frame_count,
            "sequence_frame_count",
            minimum=0,
        )
        if not isinstance(self.preserve_debug_artifacts, bool):
            raise TypeError("preserve_debug_artifacts must be bool")


@dataclass(frozen=True, slots=True)
class ExportRequest:
    """Application request independent from a live ``bpy.types.Object``."""

    source_object_ids: Tuple[str, ...]
    active_object_id: str
    settings: ExportSettings
    connected_object_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source_object_ids, tuple) or not self.source_object_ids:
            raise ValueError("source_object_ids cannot be empty")
        if not all(isinstance(value, str) and value.strip() for value in self.source_object_ids):
            raise TypeError("source_object_ids must contain non-empty strings")
        if len(self.source_object_ids) != len(set(self.source_object_ids)):
            raise ValueError("source_object_ids cannot contain duplicates")
        require_non_empty_string(self.active_object_id, "active_object_id")
        if not isinstance(self.settings, ExportSettings):
            raise TypeError("settings must be ExportSettings")
        if not isinstance(self.connected_object_ids, tuple) or not all(
            isinstance(value, str) and value.strip()
            for value in self.connected_object_ids
        ):
            raise TypeError("connected_object_ids must contain non-empty strings")
        if len(self.connected_object_ids) != len(set(self.connected_object_ids)):
            raise ValueError("connected_object_ids cannot contain duplicates")
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
        if not isinstance(self.success, bool):
            raise TypeError("success must be bool")
        if self.success and any(
            issue.severity is IssueSeverity.ERROR for issue in self.issues
        ):
            raise ValueError("A successful ExportResult cannot contain ERROR issues")
        if not self.success and not any(
            issue.severity is IssueSeverity.ERROR for issue in self.issues
        ):
            raise ValueError("A failed ExportResult must contain at least one ERROR issue")
