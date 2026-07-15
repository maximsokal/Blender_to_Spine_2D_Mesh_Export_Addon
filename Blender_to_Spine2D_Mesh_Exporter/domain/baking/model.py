"""Typed material analysis and deterministic texture bake planning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Tuple


class MaterialKind(str, Enum):
    EMPTY = "EMPTY"
    IMAGE = "IMAGE"
    SOLID_COLOR = "SOLID_COLOR"
    PROCEDURAL = "PROCEDURAL"
    MIXED = "MIXED"
    UNSUPPORTED = "UNSUPPORTED"


class BakeMode(str, Enum):
    DIFFUSE = "DIFFUSE"
    COMBINED = "COMBINED"
    EMIT = "EMIT"


class TextureFormat(str, Enum):
    PNG = "PNG"
    JPEG = "JPEG"
    WEBP = "WEBP"
    OPEN_EXR = "OPEN_EXR"

    @property
    def extension(self) -> str:
        return {
            TextureFormat.PNG: ".png",
            TextureFormat.JPEG: ".jpg",
            TextureFormat.WEBP: ".webp",
            TextureFormat.OPEN_EXR: ".exr",
        }[self]


@dataclass(frozen=True, slots=True)
class ImageDependency:
    image_name: str
    source: str
    filepath: str | None = None
    frame_duration: int = 1
    generated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.image_name, str) or not self.image_name.strip():
            raise ValueError("image_name must be a non-empty string")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")
        if self.filepath is not None and not isinstance(self.filepath, str):
            raise TypeError("filepath must be str or None")
        if not isinstance(self.frame_duration, int) or self.frame_duration < 1:
            raise ValueError("frame_duration must be a positive integer")
        if not isinstance(self.generated, bool):
            raise TypeError("generated must be bool")

    @property
    def animated(self) -> bool:
        return self.source.upper() in {"SEQUENCE", "MOVIE"} or self.frame_duration > 1


@dataclass(frozen=True, slots=True)
class MaterialAnalysis:
    slot_index: int
    material_name: str | None
    kind: MaterialKind
    node_types: Tuple[str, ...] = ()
    image_dependencies: Tuple[ImageDependency, ...] = ()
    issues: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.slot_index, int) or self.slot_index < 0:
            raise ValueError("slot_index must be a non-negative integer")
        if self.material_name is not None and (
            not isinstance(self.material_name, str) or not self.material_name.strip()
        ):
            raise ValueError("material_name must be a non-empty string or None")
        if not isinstance(self.kind, MaterialKind):
            raise TypeError("kind must be MaterialKind")
        for field_name in ("node_types", "image_dependencies", "issues"):
            if not isinstance(getattr(self, field_name), tuple):
                raise TypeError(f"{field_name} must be tuple")

    @property
    def animated(self) -> bool:
        return any(dependency.animated for dependency in self.image_dependencies)


@dataclass(frozen=True, slots=True)
class ObjectMaterialAnalysis:
    source_object_id: str
    slots: Tuple[MaterialAnalysis, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_object_id, str) or not self.source_object_id.strip():
            raise ValueError("source_object_id must be a non-empty string")
        if not isinstance(self.slots, tuple):
            raise TypeError("slots must be tuple")
        actual_indices = tuple(slot.slot_index for slot in self.slots)
        expected_indices = tuple(range(len(self.slots)))
        if actual_indices != expected_indices:
            raise ValueError("material slot indices must be ordered and dense from zero")

    @property
    def has_animated_dependencies(self) -> bool:
        return any(slot.animated for slot in self.slots)

    @property
    def material_names(self) -> Tuple[str, ...]:
        return tuple(
            slot.material_name for slot in self.slots if slot.material_name is not None
        )


@dataclass(frozen=True, slots=True)
class BakeSettings:
    width: int
    height: int
    output_directory: Path
    output_stem: str
    uv_layer_name: str = "SpineBakeUV"
    texture_format: TextureFormat = TextureFormat.PNG
    margin_pixels: int = 4
    selected_to_active: bool = False
    cage_extrusion: float = 0.1
    diffuse_mode: BakeMode = BakeMode.DIFFUSE
    procedural_mode: BakeMode = BakeMode.COMBINED
    sequence_start_frame: int = 0
    sequence_frame_count: int = 0
    sequence_frame_digits: int = 4

    def __post_init__(self) -> None:
        for field_name in ("width", "height"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if not isinstance(self.output_directory, Path):
            raise TypeError("output_directory must be pathlib.Path")
        if not isinstance(self.output_stem, str) or not self.output_stem.strip():
            raise ValueError("output_stem must be a non-empty string")
        if not isinstance(self.uv_layer_name, str) or not self.uv_layer_name.strip():
            raise ValueError("uv_layer_name must be a non-empty string")
        if not isinstance(self.texture_format, TextureFormat):
            raise TypeError("texture_format must be TextureFormat")
        if not isinstance(self.margin_pixels, int) or self.margin_pixels < 0:
            raise ValueError("margin_pixels must be a non-negative integer")
        if not isinstance(self.selected_to_active, bool):
            raise TypeError("selected_to_active must be bool")
        if not isinstance(self.cage_extrusion, (int, float)) or not isfinite(
            float(self.cage_extrusion)
        ):
            raise ValueError("cage_extrusion must be finite")
        if self.cage_extrusion < 0.0:
            raise ValueError("cage_extrusion cannot be negative")
        for field_name in ("diffuse_mode", "procedural_mode"):
            if not isinstance(getattr(self, field_name), BakeMode):
                raise TypeError(f"{field_name} must be BakeMode")
        if not isinstance(self.sequence_start_frame, int) or self.sequence_start_frame < 0:
            raise ValueError("sequence_start_frame must be a non-negative integer")
        if not isinstance(self.sequence_frame_count, int) or self.sequence_frame_count < 0:
            raise ValueError("sequence_frame_count must be a non-negative integer")
        if not isinstance(self.sequence_frame_digits, int) or not 1 <= self.sequence_frame_digits <= 12:
            raise ValueError("sequence_frame_digits must be in [1, 12]")


@dataclass(frozen=True, slots=True)
class BakeFrameTask:
    task_index: int
    timeline_frame: int | None
    image_name: str
    output_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.task_index, int) or self.task_index < 0:
            raise ValueError("task_index must be a non-negative integer")
        if self.timeline_frame is not None and (
            not isinstance(self.timeline_frame, int) or self.timeline_frame < 0
        ):
            raise ValueError("timeline_frame must be a non-negative integer or None")
        if not isinstance(self.image_name, str) or not self.image_name.strip():
            raise ValueError("image_name must be a non-empty string")
        if not isinstance(self.output_path, Path):
            raise TypeError("output_path must be pathlib.Path")


@dataclass(frozen=True, slots=True)
class BakePlan:
    source_object_id: str
    settings: BakeSettings
    material_analysis: ObjectMaterialAnalysis
    bake_mode: BakeMode
    frame_tasks: Tuple[BakeFrameTask, ...]
    representative_task_index: int = 0

    def __post_init__(self) -> None:
        if self.source_object_id != self.material_analysis.source_object_id:
            raise ValueError("source_object_id and material_analysis disagree")
        if not isinstance(self.bake_mode, BakeMode):
            raise TypeError("bake_mode must be BakeMode")
        if not self.frame_tasks:
            raise ValueError("frame_tasks cannot be empty")
        actual_indices = tuple(task.task_index for task in self.frame_tasks)
        if actual_indices != tuple(range(len(self.frame_tasks))):
            raise ValueError("frame task indices must be ordered and dense from zero")
        if not 0 <= self.representative_task_index < len(self.frame_tasks):
            raise ValueError("representative_task_index is out of range")

    @property
    def representative_task(self) -> BakeFrameTask:
        return self.frame_tasks[self.representative_task_index]

    @property
    def sequence(self) -> bool:
        return any(task.timeline_frame is not None for task in self.frame_tasks)


class BakePlanError(ValueError):
    """Raised when material analysis cannot produce a safe deterministic bake plan."""


def sanitize_filename_stem(value: str) -> str:
    """Return a Windows-safe, deterministic filename stem."""

    if not isinstance(value, str):
        raise TypeError("value must be str")
    invalid = '<>:"/\\|?*'
    sanitized = "".join(
        "_" if character in invalid or ord(character) < 32 else character
        for character in value.strip()
    )
    sanitized = sanitized.rstrip(" .")
    if not sanitized:
        raise BakePlanError("output filename stem is empty after sanitization")
    return sanitized


def _select_bake_mode(
    analysis: ObjectMaterialAnalysis,
    settings: BakeSettings,
) -> BakeMode:
    if not analysis.slots:
        raise BakePlanError("object has no material slots")
    unsupported = tuple(
        slot for slot in analysis.slots if slot.kind is MaterialKind.UNSUPPORTED
    )
    if unsupported:
        names = tuple(slot.material_name or f"slot-{slot.slot_index}" for slot in unsupported)
        raise BakePlanError(f"unsupported materials cannot be baked safely: {names}")
    usable = tuple(slot for slot in analysis.slots if slot.kind is not MaterialKind.EMPTY)
    if not usable:
        raise BakePlanError("object has no usable materials")
    if any(slot.kind in {MaterialKind.PROCEDURAL, MaterialKind.MIXED} for slot in usable):
        return settings.procedural_mode
    return settings.diffuse_mode


def _build_frame_tasks(settings: BakeSettings) -> Tuple[BakeFrameTask, ...]:
    stem = sanitize_filename_stem(settings.output_stem)
    extension = settings.texture_format.extension
    if settings.sequence_frame_count == 0:
        image_name = f"{stem}_Baked"
        return (
            BakeFrameTask(
                task_index=0,
                timeline_frame=None,
                image_name=image_name,
                output_path=settings.output_directory / f"{image_name}{extension}",
            ),
        )

    tasks = []
    for task_index in range(settings.sequence_frame_count):
        timeline_frame = settings.sequence_start_frame + task_index
        suffix = f"{timeline_frame:0{settings.sequence_frame_digits}d}"
        image_name = f"{stem}_Baked_{suffix}"
        tasks.append(
            BakeFrameTask(
                task_index=task_index,
                timeline_frame=timeline_frame,
                image_name=image_name,
                output_path=settings.output_directory / f"{image_name}{extension}",
            )
        )
    return tuple(tasks)


def build_bake_plan(
    analysis: ObjectMaterialAnalysis,
    settings: BakeSettings,
) -> BakePlan:
    """Build one complete bake plan without reading Blender scene globals."""

    if not isinstance(analysis, ObjectMaterialAnalysis):
        raise TypeError("analysis must be ObjectMaterialAnalysis")
    if not isinstance(settings, BakeSettings):
        raise TypeError("settings must be BakeSettings")
    bake_mode = _select_bake_mode(analysis, settings)
    tasks = _build_frame_tasks(settings)
    return BakePlan(
        source_object_id=analysis.source_object_id,
        settings=settings,
        material_analysis=analysis,
        bake_mode=bake_mode,
        frame_tasks=tasks,
        representative_task_index=0,
    )
