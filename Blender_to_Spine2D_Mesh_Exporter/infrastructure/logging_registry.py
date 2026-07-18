"""Pure discovery and naming helpers for per-module addon logging."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


_VALID_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


@dataclass(frozen=True, slots=True)
class ModuleLogLevel:
    module_name: str
    level: str

    def __post_init__(self) -> None:
        if not isinstance(self.module_name, str) or not self.module_name.strip():
            raise ValueError("module_name must be a non-empty string")
        normalized = str(self.level).upper()
        if normalized not in _VALID_LEVELS:
            raise ValueError(f"unsupported logging level: {self.level!r}")
        object.__setattr__(self, "level", normalized)


def discover_python_modules(
    package_directory: Path,
    *,
    root_display_name: str,
) -> tuple[str, ...]:
    """Return deterministic relative logger names for every Python source file."""

    if not isinstance(package_directory, Path):
        raise TypeError("package_directory must be pathlib.Path")
    root = package_directory.expanduser().resolve(strict=False)
    if not root.is_dir():
        raise ValueError(f"package_directory is not a directory: {root}")
    if not isinstance(root_display_name, str) or not root_display_name.strip():
        raise ValueError("root_display_name must be a non-empty string")

    modules: set[str] = {root_display_name.strip()}
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        without_suffix = relative.with_suffix("")
        if without_suffix.name == "__init__":
            parts = without_suffix.parts[:-1]
        else:
            parts = without_suffix.parts
        if parts:
            modules.add(".".join(parts))

    root_name = root_display_name.strip()
    return (root_name, *sorted(modules - {root_name}, key=lambda value: value.casefold()))


def normalize_existing_levels(
    existing: Mapping[str, str] | Iterable[tuple[str, str]],
    *,
    package_root: str,
    root_display_name: str,
) -> dict[str, str]:
    """Normalize old relative/full logger keys while preserving valid user levels."""

    if not isinstance(package_root, str) or not package_root.strip():
        raise ValueError("package_root must be a non-empty string")
    items = existing.items() if isinstance(existing, Mapping) else existing
    result: dict[str, str] = {}
    prefix = package_root.strip() + "."
    for raw_name, raw_level in items:
        name = str(raw_name or "").strip()
        level = str(raw_level or "ERROR").upper()
        if not name or level not in _VALID_LEVELS:
            continue
        if name == package_root or name == root_display_name:
            normalized_name = root_display_name
        elif name.startswith(prefix):
            normalized_name = name[len(prefix) :]
        else:
            normalized_name = name
        result[normalized_name] = level
    return result


def merge_module_levels(
    discovered_modules: tuple[str, ...],
    existing_levels: Mapping[str, str] | Iterable[tuple[str, str]],
    *,
    package_root: str,
    root_display_name: str,
    default_level: str = "ERROR",
) -> tuple[ModuleLogLevel, ...]:
    """Merge a fresh module scan with persisted user levels."""

    default = str(default_level).upper()
    if default not in _VALID_LEVELS:
        raise ValueError(f"unsupported default logging level: {default_level!r}")
    normalized = normalize_existing_levels(
        existing_levels,
        package_root=package_root,
        root_display_name=root_display_name,
    )
    return tuple(
        ModuleLogLevel(name, normalized.get(name, default))
        for name in discovered_modules
    )


def resolve_logger_name(
    module_name: str,
    *,
    package_root: str,
    root_display_name: str,
) -> str:
    """Translate one persisted relative module name to its runtime logger name."""

    name = str(module_name or "").strip()
    if not name:
        raise ValueError("module_name must be a non-empty string")
    root = str(package_root or "").strip()
    if not root:
        raise ValueError("package_root must be a non-empty string")
    if name in {root, root_display_name}:
        return root
    if name.startswith(root + "."):
        return name
    return f"{root}.{name}"


__all__ = [
    "ModuleLogLevel",
    "discover_python_modules",
    "merge_module_levels",
    "normalize_existing_levels",
    "resolve_logger_name",
]
