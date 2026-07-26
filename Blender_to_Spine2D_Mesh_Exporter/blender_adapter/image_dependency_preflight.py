"""Validate reachable Blender material images before object bake or render."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import glob
import os
from typing import Any, Iterator

from ..domain.baking import ObjectMaterialAnalysis


class ImageDependencyPreflightError(RuntimeError):
    """Raised when a reachable material image cannot be loaded deterministically."""


@dataclass(frozen=True, slots=True)
class ImageDependencyFailure:
    image_name: str
    source: str
    raw_path: str | None
    resolved_paths: tuple[str, ...]
    reason: str


def _rna_identity(value: Any) -> int:
    pointer = getattr(value, "as_pointer", None)
    if callable(pointer):
        try:
            return int(pointer())
        except Exception:
            pass
    return id(value)


def _iter_node_trees(obj: Any) -> Iterator[Any]:
    pending: list[Any] = []
    try:
        slots = tuple(getattr(obj, "material_slots", ()) or ())
    except Exception:
        slots = ()
    for slot in slots:
        material = getattr(slot, "material", None)
        tree = getattr(material, "node_tree", None)
        if tree is not None:
            pending.append(tree)

    visited: set[int] = set()
    while pending:
        tree = pending.pop()
        identity = _rna_identity(tree)
        if identity in visited:
            continue
        visited.add(identity)
        yield tree
        try:
            nodes = tuple(getattr(tree, "nodes", ()) or ())
        except Exception:
            nodes = ()
        for node in nodes:
            nested = getattr(node, "node_tree", None)
            if nested is not None:
                pending.append(nested)


def _reachable_dependency_keys(
    analysis: ObjectMaterialAnalysis,
) -> set[tuple[str, str, str | None, int]]:
    return {
        (
            dependency.image_name,
            dependency.source,
            dependency.filepath,
            dependency.frame_duration,
        )
        for slot in analysis.slots
        for dependency in slot.image_dependencies
    }


def _image_key(image: Any) -> tuple[str, str, str | None, int]:
    name = str(
        getattr(image, "name_full", None)
        or getattr(image, "name", None)
        or ""
    ).strip()
    source = str(getattr(image, "source", "FILE") or "FILE").strip().upper()
    filepath_value = (
        getattr(image, "filepath_raw", None)
        or getattr(image, "filepath", None)
    )
    filepath = None if filepath_value in (None, "") else str(filepath_value)
    try:
        frame_duration = int(getattr(image, "frame_duration", 1) or 1)
    except (TypeError, ValueError, OverflowError):
        frame_duration = 1
    return name, source, filepath, max(1, frame_duration)


def _reachable_images(
    obj: Any,
    analysis: ObjectMaterialAnalysis,
) -> tuple[Any, ...]:
    keys = _reachable_dependency_keys(analysis)
    if not keys:
        return ()
    resolved: dict[tuple[str, str, str | None, int], Any] = {}
    for tree in _iter_node_trees(obj):
        try:
            nodes = tuple(getattr(tree, "nodes", ()) or ())
        except Exception:
            nodes = ()
        for node in nodes:
            node_type = str(
                getattr(node, "type", None)
                or getattr(node, "bl_idname", None)
                or ""
            ).strip().upper()
            if node_type not in {"TEX_IMAGE", "SHADERNODETEXIMAGE"}:
                continue
            image = getattr(node, "image", None)
            if image is None:
                continue
            key = _image_key(image)
            if key in keys:
                resolved[key] = image
    return tuple(resolved[key] for key in sorted(resolved))


def _is_packed(image: Any) -> bool:
    if getattr(image, "packed_file", None) is not None:
        return True
    try:
        return bool(tuple(getattr(image, "packed_files", ()) or ()))
    except Exception:
        return False


def _abspath(raw_path: str, image: Any) -> str:
    try:
        import bpy

        try:
            return str(
                bpy.path.abspath(
                    raw_path,
                    library=getattr(image, "library", None),
                )
            )
        except TypeError:
            return str(bpy.path.abspath(raw_path))
    except Exception:
        return os.path.abspath(os.path.expanduser(raw_path))


def _filepath_from_user(image: Any, *, frame: int | None, tile: int | None) -> str | None:
    callback = getattr(image, "filepath_from_user", None)
    if not callable(callback):
        return None
    keyword_sets: list[dict[str, int]] = []
    if frame is not None and tile is not None:
        keyword_sets.append({"frame": frame, "tile": tile})
    if frame is not None:
        keyword_sets.append({"frame": frame})
    if tile is not None:
        keyword_sets.append({"tile": tile})
    keyword_sets.append({})
    for keywords in keyword_sets:
        try:
            value = str(callback(**keywords) or "").strip()
        except Exception:
            continue
        if value:
            return value
    return None


def _expand_tokenized_path(path: str) -> tuple[str, ...]:
    if "<UDIM>" in path:
        pattern = path.replace("<UDIM>", "[0-9][0-9][0-9][0-9]")
        return tuple(sorted(glob.glob(pattern)))
    if "####" in path:
        pattern = path.replace("####", "[0-9][0-9][0-9][0-9]")
        return tuple(sorted(glob.glob(pattern)))
    return (path,)


def _candidate_paths(image: Any, scene: Any | None) -> tuple[str, ...]:
    source = str(getattr(image, "source", "FILE") or "FILE").strip().upper()
    frame = None
    if scene is not None:
        try:
            frame = int(getattr(scene, "frame_current", 1) or 1)
        except (TypeError, ValueError, OverflowError):
            frame = 1

    generated: list[str] = []
    if source == "TILED":
        try:
            tiles = tuple(getattr(image, "tiles", ()) or ())
        except Exception:
            tiles = ()
        for tile_entry in tiles or (None,):
            tile_number = None if tile_entry is None else int(tile_entry.number)
            value = _filepath_from_user(image, frame=frame, tile=tile_number)
            if value:
                generated.append(_abspath(value, image))
    elif source == "SEQUENCE":
        value = _filepath_from_user(image, frame=frame, tile=None)
        if value:
            generated.append(_abspath(value, image))

    if not generated:
        raw = str(
            getattr(image, "filepath_raw", None)
            or getattr(image, "filepath", None)
            or ""
        ).strip()
        if raw:
            generated.append(_abspath(raw, image))

    expanded: list[str] = []
    for path in generated:
        expanded.extend(_expand_tokenized_path(path))
    return tuple(dict.fromkeys(expanded))


def preflight_object_image_dependencies(
    obj: Any,
    analysis: ObjectMaterialAnalysis,
    *,
    scene: Any | None = None,
) -> tuple[str, ...]:
    """Require every reachable external image to be packed or loadable on disk."""

    if obj is None:
        raise ImageDependencyPreflightError("obj cannot be None")
    if not isinstance(analysis, ObjectMaterialAnalysis):
        raise TypeError("analysis must be ObjectMaterialAnalysis")

    expected_keys = _reachable_dependency_keys(analysis)
    actual_images = _reachable_images(obj, analysis)
    actual_by_key = {_image_key(image): image for image in actual_images}
    failures: list[ImageDependencyFailure] = []
    valid_names: list[str] = []

    for key in sorted(expected_keys):
        image_name, source, raw_path, _frame_duration = key
        image = actual_by_key.get(key)
        if image is None:
            failures.append(
                ImageDependencyFailure(
                    image_name=image_name,
                    source=source,
                    raw_path=raw_path,
                    resolved_paths=(),
                    reason="reachable image datablock could not be resolved from the material graph",
                )
            )
            continue
        if _is_packed(image) or source in {"GENERATED", "VIEWER"}:
            valid_names.append(image_name)
            continue

        paths = _candidate_paths(image, scene)
        existing = tuple(path for path in paths if Path(path).is_file())
        if not paths:
            failures.append(
                ImageDependencyFailure(
                    image_name=image_name,
                    source=source,
                    raw_path=raw_path,
                    resolved_paths=(),
                    reason="image has no resolvable file path and is not packed",
                )
            )
            continue
        if not existing:
            failures.append(
                ImageDependencyFailure(
                    image_name=image_name,
                    source=source,
                    raw_path=raw_path,
                    resolved_paths=paths,
                    reason="no resolved image file exists",
                )
            )
            continue
        valid_names.append(image_name)

    if failures:
        detail = "\n".join(
            f"- {failure.image_name} [{failure.source}]: {failure.reason}; "
            f"raw={failure.raw_path!r}; resolved={failure.resolved_paths!r}"
            for failure in failures
        )
        raise ImageDependencyPreflightError(
            "Material image dependency preflight failed. Relink or pack every image "
            "before export:\n" + detail
        )

    return tuple(sorted(dict.fromkeys(valid_names)))


__all__ = [
    "ImageDependencyFailure",
    "ImageDependencyPreflightError",
    "preflight_object_image_dependencies",
]
