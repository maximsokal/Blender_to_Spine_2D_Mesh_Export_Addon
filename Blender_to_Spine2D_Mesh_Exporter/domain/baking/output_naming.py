"""Windows-safe deterministic naming for rewrite texture outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from .model import BakePlanError, BakeSettings


_WINDOWS_RESERVED_BASENAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
        "com¹",
        "com²",
        "com³",
        "lpt¹",
        "lpt²",
        "lpt³",
    }
)
_WINDOWS_INVALID_CHARACTERS = frozenset('<>:"/\\|?*')


def _windows_reserved_basename(value: str) -> bool:
    """Return whether the first filename component is a DOS device name."""

    if not isinstance(value, str):
        raise TypeError("value must be str")
    basename = value.partition(".")[0].rstrip(" .").casefold()
    return basename in _WINDOWS_RESERVED_BASENAMES


def sanitize_filename_stem(value: str) -> str:
    """Return a deterministic stem safe for ordinary Windows file APIs.

    Invalid characters and ASCII control characters are replaced with underscores.
    Trailing spaces/periods are removed. Reserved DOS device basenames are suffixed
    with an underscore before any remaining dot-separated suffix.
    """

    if not isinstance(value, str):
        raise TypeError("value must be str")
    sanitized = "".join(
        "_"
        if character in _WINDOWS_INVALID_CHARACTERS or ord(character) < 32
        else character
        for character in value.strip()
    )
    sanitized = sanitized.rstrip(" .")
    if not sanitized:
        raise BakePlanError("output filename stem is empty after sanitization")

    if _windows_reserved_basename(sanitized):
        basename, separator, suffix = sanitized.partition(".")
        basename = basename.rstrip(" .")
        sanitized = f"{basename}_{separator}{suffix}" if separator else f"{basename}_"
    return sanitized


def windows_path_identity(path: Path) -> Tuple[str, ...]:
    """Return a host-independent case-normalized identity for a Windows path.

    The check intentionally runs with Windows semantics even when tests execute on
    Linux. Every resolved path component is compared case-insensitively and without
    trailing spaces/periods, which ordinary Windows file APIs do not preserve safely.
    """

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    resolved = path.expanduser().resolve(strict=False)
    return tuple(part.rstrip(" .").casefold() for part in resolved.parts)


def predict_bake_output_paths(settings: BakeSettings) -> Tuple[Path, ...]:
    """Predict every final texture path without reading geometry or Blender state."""

    if not isinstance(settings, BakeSettings):
        raise TypeError("settings must be BakeSettings")
    stem = sanitize_filename_stem(settings.output_stem)
    extension = settings.texture_format.extension
    output_directory = settings.output_directory.expanduser().resolve(strict=False)

    if settings.sequence_frame_count == 0:
        return (output_directory / f"{stem}_Baked{extension}",)

    return tuple(
        output_directory
        / (
            f"{stem}_Baked_"
            f"{settings.sequence_start_frame + task_index:0{settings.sequence_frame_digits}d}"
            f"{extension}"
        )
        for task_index in range(settings.sequence_frame_count)
    )


__all__ = [
    "predict_bake_output_paths",
    "sanitize_filename_stem",
    "windows_path_identity",
]
