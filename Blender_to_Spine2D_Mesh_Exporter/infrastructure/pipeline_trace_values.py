"""Bounded JSON-safe value summaries and module discovery for pipeline tracing."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import json
from pathlib import Path
from types import FrameType
from typing import Any, Mapping


PRODUCTION_LAYERS = frozenset({"application", "domain", "blender_adapter", "infrastructure"})


def qualified_type_name(value: Any) -> str:
    cls = type(value)
    module = getattr(cls, "__module__", "")
    name = getattr(cls, "__qualname__", getattr(cls, "__name__", str(cls)))
    if not module or module == "builtins":
        return str(name)
    return f"{module}.{name}"


def safe_length(value: Any) -> int | None:
    try:
        return len(value)
    except (TypeError, AttributeError, RuntimeError):
        return None


def shape_summary(value: Any, *, capture_values: bool, depth: int = 0) -> dict[str, Any]:
    """Return a bounded summary without serializing arbitrary object contents."""

    type_name = qualified_type_name(value)
    if value is None:
        return {"type": "NoneType"}
    if isinstance(value, bool):
        return {"type": "bool", **({"value": value} if capture_values else {})}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"type": "int", **({"value": value} if capture_values else {})}
    if isinstance(value, float):
        return {"type": "float", **({"value": value} if capture_values else {})}
    if isinstance(value, str):
        result: dict[str, Any] = {"type": "str", "length": len(value)}
        if capture_values:
            result["value"] = value if len(value) <= 160 else value[:157] + "..."
        return result
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}
    if isinstance(value, Path):
        result = {"type": "pathlib.Path"}
        if capture_values:
            result["value"] = str(value)
        return result
    if isinstance(value, Enum):
        result = {"type": type_name}
        if capture_values:
            result["value"] = value.value
        return result

    if isinstance(value, Mapping):
        keys = list(value.keys())[:8]
        result = {
            "type": type_name,
            "length": safe_length(value),
            "key_types": sorted({qualified_type_name(item) for item in keys}),
        }
        if capture_values:
            result["keys"] = [str(item)[:80] for item in keys]
        return result

    if isinstance(value, (tuple, list, set, frozenset)):
        sample = list(value)[:8]
        return {
            "type": type_name,
            "length": safe_length(value),
            "item_types": sorted({qualified_type_name(item) for item in sample}),
        }

    if is_dataclass(value) and depth == 0:
        result = {"type": type_name, "dataclass": True}
        field_summaries: dict[str, Any] = {}
        for field in fields(value)[:10]:
            try:
                field_value = getattr(value, field.name)
            except Exception:  # pipeline-audit: ignore=BROAD_EXCEPTION_WITHOUT_VISIBILITY
                field_summaries[field.name] = {"type": "<unavailable>"}
                continue
            field_summaries[field.name] = shape_summary(
                field_value,
                capture_values=False,
                depth=depth + 1,
            )
        result["fields"] = field_summaries
        return result

    result = {"type": type_name}
    for attribute in ("name_full", "name", "type"):
        try:
            attribute_value = getattr(value, attribute)
        except Exception:  # pipeline-audit: ignore=BROAD_EXCEPTION_WITHOUT_VISIBILITY
            continue
        if isinstance(attribute_value, (str, int, float, bool)):
            result[attribute] = attribute_value

    for attribute in (
        "vertices", "edges", "faces", "polygons", "loops", "regions",
        "segments", "groups", "slots", "bones", "skins", "frame_tasks",
        "passes", "issues", "warnings",
    ):
        try:
            attribute_value = getattr(value, attribute)
        except Exception:  # pipeline-audit: ignore=BROAD_EXCEPTION_WITHOUT_VISIBILITY
            continue
        length = safe_length(attribute_value)
        if length is not None:
            result[f"{attribute}_count"] = length
    return result


def argument_names(frame: FrameType) -> tuple[str, ...]:
    code = frame.f_code
    positional_count = code.co_argcount
    keyword_only_count = code.co_kwonlyargcount
    names = list(code.co_varnames[: positional_count + keyword_only_count])
    if code.co_flags & 0x04:  # CO_VARARGS
        names.append(code.co_varnames[positional_count + keyword_only_count])
    if code.co_flags & 0x08:  # CO_VARKEYWORDS
        index = positional_count + keyword_only_count + (1 if code.co_flags & 0x04 else 0)
        names.append(code.co_varnames[index])
    return tuple(names)


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def layer_for_relative_path(relative_path: Path) -> str:
    if not relative_path.parts:
        return "root"
    first = relative_path.parts[0]
    return first if first in PRODUCTION_LAYERS else "root"


def module_name(relative_path: Path) -> str:
    without_suffix = relative_path.with_suffix("")
    parts = without_suffix.parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def discover_pipeline_modules(package_directory: Path) -> dict[Path, dict[str, str]]:
    """Discover production Python files and their relative module names."""

    if not isinstance(package_directory, Path):
        raise TypeError("package_directory must be pathlib.Path")
    root = package_directory.expanduser().resolve(strict=False)
    if not root.is_dir():
        raise ValueError(f"package_directory is not a directory: {root}")

    result: dict[Path, dict[str, str]] = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        layer = layer_for_relative_path(relative)
        if layer == "root" and len(relative.parts) > 1:
            continue
        module = module_name(relative) or "<package>"
        result[path.resolve(strict=False)] = {
            "module": module,
            "relative_path": relative.as_posix(),
            "layer": layer,
        }
    return result


__all__ = [
    "PRODUCTION_LAYERS",
    "argument_names",
    "canonical_json",
    "discover_pipeline_modules",
    "shape_summary",
]
