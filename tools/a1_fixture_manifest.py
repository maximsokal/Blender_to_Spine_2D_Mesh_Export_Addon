"""Typed manifest for reproducible Legacy-versus-Rewrite Blender fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from math import isfinite
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Tuple


_SCHEMA_VERSION = 1
_SAFE_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class FixtureManifestError(ValueError):
    """Raised when a fixture manifest is missing or internally inconsistent."""


class FixtureMode(str, Enum):
    SINGLE = "single"
    MULTI = "multi"


@dataclass(frozen=True, slots=True)
class FixtureSequenceSettings:
    start_frame: int = 0
    frame_count: int = 0

    def __post_init__(self) -> None:
        for field_name in ("start_frame", "frame_count"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise FixtureManifestError(
                    f"{field_name} must be a non-negative integer"
                )


@dataclass(frozen=True, slots=True)
class FixtureExportSettings:
    texture_size: int = 1024
    images_path: str = "images"
    seam_mode: str = "AUTO"
    angle_limit: float = 30.0
    sequence: FixtureSequenceSettings = FixtureSequenceSettings()
    per_object_sequence: Mapping[str, FixtureSequenceSettings] = field(
        default_factory=dict
    )
    control_icons: bool = True
    preview_animation: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.texture_size, int)
            or isinstance(self.texture_size, bool)
            or self.texture_size <= 0
        ):
            raise FixtureManifestError("texture_size must be a positive integer")
        _validate_safe_relative_directory(self.images_path, "images_path")
        if self.seam_mode not in {"AUTO", "CUSTOM"}:
            raise FixtureManifestError("seam_mode must be AUTO or CUSTOM")
        if (
            isinstance(self.angle_limit, bool)
            or not isinstance(self.angle_limit, (int, float))
            or not isfinite(float(self.angle_limit))
            or not 0.0 < float(self.angle_limit) <= 180.0
        ):
            raise FixtureManifestError("angle_limit must be finite and in (0, 180]")
        if not isinstance(self.sequence, FixtureSequenceSettings):
            raise TypeError("sequence must be FixtureSequenceSettings")
        if not isinstance(self.per_object_sequence, Mapping):
            raise TypeError("per_object_sequence must be a mapping")
        for object_name, settings in self.per_object_sequence.items():
            if not isinstance(object_name, str) or not object_name.strip():
                raise FixtureManifestError(
                    "per_object_sequence keys must be non-empty object names"
                )
            if not isinstance(settings, FixtureSequenceSettings):
                raise TypeError(
                    "per_object_sequence values must be FixtureSequenceSettings"
                )
        for field_name in ("control_icons", "preview_animation"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")
        object.__setattr__(
            self,
            "per_object_sequence",
            MappingProxyType(dict(self.per_object_sequence)),
        )


@dataclass(frozen=True, slots=True)
class FixtureParitySettings:
    absolute_tolerance: float = 1e-4
    relative_tolerance: float = 1e-6
    compare_animations: bool = True
    strict_edges: bool = False
    ignore_paths: Tuple[str, ...] = ()
    image_absolute_tolerance: float = 1e-6
    image_max_differing_pixel_ratio: float = 0.0
    image_max_mean_absolute_delta: float = 0.0

    def __post_init__(self) -> None:
        for field_name in (
            "absolute_tolerance",
            "relative_tolerance",
            "image_absolute_tolerance",
            "image_max_differing_pixel_ratio",
            "image_max_mean_absolute_delta",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or float(value) < 0.0
            ):
                raise FixtureManifestError(f"{field_name} must be finite and non-negative")
        if not 0.0 <= float(self.image_max_differing_pixel_ratio) <= 1.0:
            raise FixtureManifestError(
                "image_max_differing_pixel_ratio must be in [0, 1]"
            )
        for field_name in ("compare_animations", "strict_edges"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")
        if not isinstance(self.ignore_paths, tuple) or not all(
            isinstance(value, str) and value.strip() for value in self.ignore_paths
        ):
            raise FixtureManifestError(
                "ignore_paths must be a tuple of non-empty strings"
            )


@dataclass(frozen=True, slots=True)
class A1FixtureCase:
    case_id: str
    blend_file: Path
    mode: FixtureMode
    active_object: str
    selected_objects: Tuple[str, ...]
    connected_objects: Tuple[str, ...] = ()
    settings: FixtureExportSettings = FixtureExportSettings()
    parity: FixtureParitySettings = FixtureParitySettings()
    expected_json_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not _SAFE_CASE_ID.fullmatch(
            self.case_id
        ):
            raise FixtureManifestError(
                "case_id must match [A-Za-z0-9][A-Za-z0-9._-]*"
            )
        if self.case_id in {".", ".."}:
            raise FixtureManifestError("case_id cannot be '.' or '..'")
        if not isinstance(self.blend_file, Path):
            raise TypeError("blend_file must be pathlib.Path")
        if self.blend_file.suffix.lower() != ".blend":
            raise FixtureManifestError("blend_file must have a .blend extension")
        if not self.blend_file.is_file():
            raise FixtureManifestError(f"blend_file does not exist: {self.blend_file}")
        if not isinstance(self.mode, FixtureMode):
            raise TypeError("mode must be FixtureMode")
        if not isinstance(self.active_object, str) or not self.active_object.strip():
            raise FixtureManifestError("active_object must be a non-empty string")
        _validate_unique_names(self.selected_objects, "selected_objects")
        _validate_unique_names(self.connected_objects, "connected_objects", allow_empty=True)
        if self.active_object not in self.selected_objects:
            raise FixtureManifestError(
                "active_object must be included in selected_objects"
            )
        unknown_connected = set(self.connected_objects) - set(self.selected_objects)
        if unknown_connected:
            raise FixtureManifestError(
                "connected_objects contain names outside selected_objects: "
                + ", ".join(sorted(unknown_connected))
            )
        if self.mode is FixtureMode.SINGLE:
            if len(self.selected_objects) != 1:
                raise FixtureManifestError(
                    "single fixture must contain exactly one selected object"
                )
            if self.connected_objects:
                raise FixtureManifestError(
                    "single fixture cannot define connected_objects"
                )
            if self.settings.per_object_sequence:
                raise FixtureManifestError(
                    "single fixture must use settings.sequence, not per_object_sequence"
                )
        elif len(self.selected_objects) < 2:
            raise FixtureManifestError(
                "multi fixture must contain at least two selected objects"
            )
        unknown_sequences = set(self.settings.per_object_sequence) - set(
            self.selected_objects
        )
        if unknown_sequences:
            raise FixtureManifestError(
                "per_object_sequence contains unselected objects: "
                + ", ".join(sorted(unknown_sequences))
            )
        if not isinstance(self.settings, FixtureExportSettings):
            raise TypeError("settings must be FixtureExportSettings")
        if not isinstance(self.parity, FixtureParitySettings):
            raise TypeError("parity must be FixtureParitySettings")
        if self.expected_json_name is not None:
            _validate_safe_json_filename(self.expected_json_name)


@dataclass(frozen=True, slots=True)
class A1FixtureManifest:
    schema_version: int
    cases: Tuple[A1FixtureCase, ...]
    blender_executable: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise FixtureManifestError(
                f"Unsupported schema_version {self.schema_version}; expected {_SCHEMA_VERSION}"
            )
        if not isinstance(self.cases, tuple) or not self.cases:
            raise FixtureManifestError("cases must be a non-empty array")
        if not all(isinstance(case, A1FixtureCase) for case in self.cases):
            raise TypeError("cases must contain A1FixtureCase values")
        case_ids = tuple(case.case_id for case in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise FixtureManifestError("case_id values must be unique")
        if self.blender_executable is not None and (
            not isinstance(self.blender_executable, str)
            or not self.blender_executable.strip()
        ):
            raise FixtureManifestError(
                "blender_executable must be a non-empty string or null"
            )


def _validate_safe_relative_directory(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    normalized = value.replace("\\", "/").strip()
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(normalized)
    if (
        not normalized
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in posix.parts
    ):
        raise FixtureManifestError(f"{field_name} must be a safe relative directory")


def _validate_unique_names(
    values: Tuple[str, ...],
    field_name: str,
    *,
    allow_empty: bool = False,
) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be a tuple")
    if not values and not allow_empty:
        raise FixtureManifestError(f"{field_name} cannot be empty")
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise FixtureManifestError(
            f"{field_name} must contain non-empty strings"
        )
    if len(values) != len(set(values)):
        raise FixtureManifestError(f"{field_name} cannot contain duplicates")


def _validate_safe_json_filename(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise FixtureManifestError("expected_json_name must be a non-empty string")
    path = PurePosixPath(value.replace("\\", "/"))
    if len(path.parts) != 1 or path.name != value or path.suffix.lower() != ".json":
        raise FixtureManifestError(
            "expected_json_name must be one safe .json filename without directories"
        )


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FixtureManifestError(f"{field_name} must be an object")
    return value


def _sequence_from_mapping(value: Any, field_name: str) -> FixtureSequenceSettings:
    data = _mapping(value, field_name)
    _reject_unknown_keys(data, {"start_frame", "frame_count"}, field_name)
    return FixtureSequenceSettings(
        start_frame=data.get("start_frame", 0),
        frame_count=data.get("frame_count", 0),
    )


def _settings_from_mapping(value: Any) -> FixtureExportSettings:
    data = _mapping(value, "settings")
    allowed = {
        "texture_size",
        "images_path",
        "seam_mode",
        "angle_limit",
        "sequence",
        "per_object_sequence",
        "control_icons",
        "preview_animation",
    }
    _reject_unknown_keys(data, allowed, "settings")
    raw_sequences = _mapping(data.get("per_object_sequence", {}), "per_object_sequence")
    per_object = {
        str(name): _sequence_from_mapping(settings, f"per_object_sequence.{name}")
        for name, settings in raw_sequences.items()
    }
    return FixtureExportSettings(
        texture_size=data.get("texture_size", 1024),
        images_path=data.get("images_path", "images"),
        seam_mode=data.get("seam_mode", "AUTO"),
        angle_limit=data.get("angle_limit", 30.0),
        sequence=_sequence_from_mapping(data.get("sequence", {}), "settings.sequence"),
        per_object_sequence=per_object,
        control_icons=data.get("control_icons", True),
        preview_animation=data.get("preview_animation", True),
    )


def _parity_from_mapping(value: Any) -> FixtureParitySettings:
    data = _mapping(value, "parity")
    allowed = {
        "absolute_tolerance",
        "relative_tolerance",
        "compare_animations",
        "strict_edges",
        "ignore_paths",
        "image_absolute_tolerance",
        "image_max_differing_pixel_ratio",
        "image_max_mean_absolute_delta",
    }
    _reject_unknown_keys(data, allowed, "parity")
    ignore_paths = data.get("ignore_paths", [])
    if not isinstance(ignore_paths, Sequence) or isinstance(ignore_paths, (str, bytes)):
        raise FixtureManifestError("parity.ignore_paths must be an array")
    return FixtureParitySettings(
        absolute_tolerance=data.get("absolute_tolerance", 1e-4),
        relative_tolerance=data.get("relative_tolerance", 1e-6),
        compare_animations=data.get("compare_animations", True),
        strict_edges=data.get("strict_edges", False),
        ignore_paths=tuple(ignore_paths),
        image_absolute_tolerance=data.get("image_absolute_tolerance", 1e-6),
        image_max_differing_pixel_ratio=data.get(
            "image_max_differing_pixel_ratio", 0.0
        ),
        image_max_mean_absolute_delta=data.get(
            "image_max_mean_absolute_delta", 0.0
        ),
    )


def _case_from_mapping(value: Any, base_directory: Path) -> A1FixtureCase:
    data = _mapping(value, "case")
    allowed = {
        "case_id",
        "blend_file",
        "mode",
        "active_object",
        "selected_objects",
        "connected_objects",
        "settings",
        "parity",
        "expected_json_name",
    }
    _reject_unknown_keys(data, allowed, "case")
    required = {"case_id", "blend_file", "mode", "active_object", "selected_objects"}
    missing = sorted(required - set(data))
    if missing:
        raise FixtureManifestError("case is missing required fields: " + ", ".join(missing))
    blend_file = Path(str(data["blend_file"])).expanduser()
    if not blend_file.is_absolute():
        blend_file = base_directory / blend_file
    selected = data["selected_objects"]
    connected = data.get("connected_objects", [])
    if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes)):
        raise FixtureManifestError("selected_objects must be an array")
    if not isinstance(connected, Sequence) or isinstance(connected, (str, bytes)):
        raise FixtureManifestError("connected_objects must be an array")
    try:
        mode = FixtureMode(str(data["mode"]).lower())
    except ValueError as exc:
        raise FixtureManifestError("mode must be 'single' or 'multi'") from exc
    return A1FixtureCase(
        case_id=str(data["case_id"]),
        blend_file=blend_file.resolve(strict=False),
        mode=mode,
        active_object=str(data["active_object"]),
        selected_objects=tuple(str(name) for name in selected),
        connected_objects=tuple(str(name) for name in connected),
        settings=_settings_from_mapping(data.get("settings", {})),
        parity=_parity_from_mapping(data.get("parity", {})),
        expected_json_name=(
            None
            if data.get("expected_json_name") is None
            else str(data["expected_json_name"])
        ),
    )


def _reject_unknown_keys(
    data: Mapping[str, Any],
    allowed: set[str],
    field_name: str,
) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise FixtureManifestError(
            f"{field_name} contains unknown fields: " + ", ".join(unknown)
        )


def load_fixture_manifest(path: Path) -> A1FixtureManifest:
    """Load and validate one manifest relative to its own directory."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    resolved = path.expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise FixtureManifestError(f"Manifest does not exist: {resolved}")
    try:
        data = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise FixtureManifestError(f"Unable to read manifest {resolved}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise FixtureManifestError(
            f"Invalid manifest JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    root = _mapping(data, "manifest")
    _reject_unknown_keys(root, {"schema_version", "blender_executable", "cases"}, "manifest")
    cases = root.get("cases")
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        raise FixtureManifestError("cases must be an array")
    return A1FixtureManifest(
        schema_version=root.get("schema_version"),
        blender_executable=(
            None
            if root.get("blender_executable") is None
            else str(root["blender_executable"])
        ),
        cases=tuple(_case_from_mapping(case, resolved.parent) for case in cases),
    )


def case_to_worker_payload(case: A1FixtureCase, output_directory: Path) -> dict[str, Any]:
    """Convert a validated case into the JSON payload consumed inside Blender."""

    if not isinstance(case, A1FixtureCase):
        raise TypeError("case must be A1FixtureCase")
    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be pathlib.Path")
    return {
        "case_id": case.case_id,
        "mode": case.mode.value,
        "active_object": case.active_object,
        "selected_objects": list(case.selected_objects),
        "connected_objects": list(case.connected_objects),
        "expected_json_name": case.expected_json_name,
        "output_directory": str(output_directory.expanduser().resolve(strict=False)),
        "settings": {
            "texture_size": case.settings.texture_size,
            "images_path": case.settings.images_path,
            "seam_mode": case.settings.seam_mode,
            "angle_limit": case.settings.angle_limit,
            "sequence": {
                "start_frame": case.settings.sequence.start_frame,
                "frame_count": case.settings.sequence.frame_count,
            },
            "per_object_sequence": {
                name: {
                    "start_frame": settings.start_frame,
                    "frame_count": settings.frame_count,
                }
                for name, settings in case.settings.per_object_sequence.items()
            },
            "control_icons": case.settings.control_icons,
            "preview_animation": case.settings.preview_animation,
        },
    }
