"""Typed material analysis and deterministic multi-pass texture bake planning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Tuple

from .context import ObjectBakeContext, SceneBakeContext
from .contracts import (
    require_finite_number,
    require_integer,
    require_non_empty_string,
)
from .graph import (
    MaterialDependencyKind,
    MaterialGraphSnapshot,
    MaterialSemanticChannel,
)


class MaterialKind(str, Enum):
    EMPTY = "EMPTY"
    IMAGE = "IMAGE"
    SOLID_COLOR = "SOLID_COLOR"
    PROCEDURAL = "PROCEDURAL"
    MIXED = "MIXED"
    UNSUPPORTED = "UNSUPPORTED"


class BakeMode(str, Enum):
    """Bake types accepted by Blender 4.4 ``bpy.ops.object.bake``."""

    DIFFUSE = "DIFFUSE"
    COMBINED = "COMBINED"
    EMIT = "EMIT"


class BakeMaterialPolicy(str, Enum):
    """Legacy compatibility policy used by the surface-color strategy."""

    LEGACY_ANY_IMAGE = "LEGACY_ANY_IMAGE"
    CONSERVATIVE_MIXED = "CONSERVATIVE_MIXED"


class BakeEvaluationScope(str, Enum):
    """Context required to evaluate one material appearance strategy."""

    LOCAL = "LOCAL"
    SCENE = "SCENE"
    CAMERA = "CAMERA"
    AUXILIARY = "AUXILIARY"


class BakeStrategyId(str, Enum):
    """Stable identifiers for independently executable bake strategies."""

    CAMERA_COMBINED = "CAMERA_COMBINED"
    SCENE_COMBINED = "SCENE_COMBINED"
    SURFACE_COLOR = "SURFACE_COLOR"
    EMISSION = "EMISSION"
    ALPHA = "ALPHA"
    LEGACY_SINGLE_PASS = "LEGACY_SINGLE_PASS"


class MaterialPreparationMode(str, Enum):
    """How one temporary copied material is prepared for a specific bake pass."""

    PRESERVE = "PRESERVE"
    ZERO_TO_EMISSION = "ZERO_TO_EMISSION"
    EXTRACT_ALPHA_TO_EMISSION = "EXTRACT_ALPHA_TO_EMISSION"
    OPAQUE_ALPHA_TO_EMISSION = "OPAQUE_ALPHA_TO_EMISSION"


class BakeCompositeMode(str, Enum):
    """How one or more pass images become one exported RGBA texture."""

    SINGLE = "SINGLE"
    ADD_RGB_MAX_ALPHA = "ADD_RGB_MAX_ALPHA"
    ADD_RGB_REPLACE_ALPHA = "ADD_RGB_REPLACE_ALPHA"


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
        require_non_empty_string(self.image_name, "image_name")
        require_non_empty_string(self.source, "source")
        if self.filepath is not None and not isinstance(self.filepath, str):
            raise TypeError("filepath must be str or None")
        require_integer(self.frame_duration, "frame_duration", minimum=1)
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
    graph: MaterialGraphSnapshot | None = None

    def __post_init__(self) -> None:
        require_integer(self.slot_index, "slot_index", minimum=0)
        if self.material_name is not None:
            require_non_empty_string(self.material_name, "material_name")
        if not isinstance(self.kind, MaterialKind):
            raise TypeError("kind must be MaterialKind")
        for field_name in ("node_types", "image_dependencies", "issues"):
            if not isinstance(getattr(self, field_name), tuple):
                raise TypeError(f"{field_name} must be tuple")
        if self.graph is not None and not isinstance(self.graph, MaterialGraphSnapshot):
            raise TypeError("graph must be MaterialGraphSnapshot or None")
        if not all(isinstance(value, str) and value.strip() for value in self.node_types):
            raise TypeError("node_types must contain non-empty strings")
        if not all(isinstance(value, ImageDependency) for value in self.image_dependencies):
            raise TypeError("image_dependencies must contain ImageDependency")
        if not all(isinstance(value, str) and value.strip() for value in self.issues):
            raise TypeError("issues must contain non-empty strings")

    @property
    def animated(self) -> bool:
        if any(dependency.animated for dependency in self.image_dependencies):
            return True
        return (
            self.graph is not None
            and MaterialDependencyKind.TIME in self.graph.dependencies
        )

    @property
    def has_image_dependency(self) -> bool:
        return bool(self.image_dependencies)

    @property
    def semantic_channels(self) -> Tuple[MaterialSemanticChannel, ...]:
        """Return precise graph channels, with a legacy fallback for synthetic tests."""

        if self.graph is not None:
            return self.graph.semantic_channels
        if self.kind in {MaterialKind.EMPTY, MaterialKind.UNSUPPORTED}:
            return ()

        node_types = set(self.node_types)
        channels: list[MaterialSemanticChannel] = []
        pure_emission = "EMISSION" in node_types and "BSDF_PRINCIPLED" not in node_types
        if not pure_emission and "BSDF_TRANSPARENT" not in node_types:
            channels.append(MaterialSemanticChannel.SURFACE_COLOR)
        if "EMISSION" in node_types:
            channels.append(MaterialSemanticChannel.SURFACE_EMISSION)
        if "BSDF_TRANSPARENT" in node_types:
            channels.append(MaterialSemanticChannel.ALPHA)
        if pure_emission:
            channels.append(MaterialSemanticChannel.SURFACE_EMISSION)
        return tuple(dict.fromkeys(channels))

    @property
    def dependencies(self) -> Tuple[MaterialDependencyKind, ...]:
        if self.graph is not None:
            return self.graph.dependencies
        result: list[MaterialDependencyKind] = []
        if self.image_dependencies:
            result.append(MaterialDependencyKind.IMAGE)
        if self.animated:
            result.append(MaterialDependencyKind.TIME)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ObjectMaterialAnalysis:
    source_object_id: str
    slots: Tuple[MaterialAnalysis, ...]

    def __post_init__(self) -> None:
        require_non_empty_string(self.source_object_id, "source_object_id")
        if not isinstance(self.slots, tuple):
            raise TypeError("slots must be tuple")
        if not all(isinstance(slot, MaterialAnalysis) for slot in self.slots):
            raise TypeError("slots must contain MaterialAnalysis")
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
    material_policy: BakeMaterialPolicy = BakeMaterialPolicy.LEGACY_ANY_IMAGE
    sequence_start_frame: int = 0
    sequence_frame_count: int = 0
    sequence_frame_digits: int = 4

    def __post_init__(self) -> None:
        require_integer(self.width, "width", minimum=1)
        require_integer(self.height, "height", minimum=1)
        if not isinstance(self.output_directory, Path):
            raise TypeError("output_directory must be pathlib.Path")
        require_non_empty_string(self.output_stem, "output_stem")
        require_non_empty_string(self.uv_layer_name, "uv_layer_name")
        if not isinstance(self.texture_format, TextureFormat):
            raise TypeError("texture_format must be TextureFormat")
        require_integer(self.margin_pixels, "margin_pixels", minimum=0)
        if not isinstance(self.selected_to_active, bool):
            raise TypeError("selected_to_active must be bool")
        require_finite_number(
            self.cage_extrusion,
            "cage_extrusion",
            minimum=0.0,
        )
        for field_name in ("diffuse_mode", "procedural_mode"):
            if not isinstance(getattr(self, field_name), BakeMode):
                raise TypeError(f"{field_name} must be BakeMode")
        if not isinstance(self.material_policy, BakeMaterialPolicy):
            raise TypeError("material_policy must be BakeMaterialPolicy")
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
        require_integer(
            self.sequence_frame_digits,
            "sequence_frame_digits",
            minimum=1,
            maximum=12,
        )


@dataclass(frozen=True, slots=True)
class BakeFrameTask:
    task_index: int
    timeline_frame: int | None
    image_name: str
    output_path: Path

    def __post_init__(self) -> None:
        require_integer(self.task_index, "task_index", minimum=0)
        if self.timeline_frame is not None:
            require_integer(self.timeline_frame, "timeline_frame", minimum=0)
        require_non_empty_string(self.image_name, "image_name")
        if not isinstance(self.output_path, Path):
            raise TypeError("output_path must be pathlib.Path")


@dataclass(frozen=True, slots=True)
class MaterialSlotPreparation:
    slot_index: int
    mode: MaterialPreparationMode

    def __post_init__(self) -> None:
        require_integer(self.slot_index, "slot_index", minimum=0)
        if not isinstance(self.mode, MaterialPreparationMode):
            raise TypeError("mode must be MaterialPreparationMode")


@dataclass(frozen=True, slots=True)
class BakePassPlan:
    pass_index: int
    strategy_id: BakeStrategyId
    bake_mode: BakeMode
    material_slot_indices: Tuple[int, ...]
    semantic_channels: Tuple[MaterialSemanticChannel, ...]
    evaluation_scope: BakeEvaluationScope = BakeEvaluationScope.LOCAL
    material_preparations: Tuple[MaterialSlotPreparation, ...] = ()

    def __post_init__(self) -> None:
        require_integer(self.pass_index, "pass_index", minimum=0)
        if not isinstance(self.strategy_id, BakeStrategyId):
            raise TypeError("strategy_id must be BakeStrategyId")
        if not isinstance(self.bake_mode, BakeMode):
            raise TypeError("bake_mode must be BakeMode")
        if not isinstance(self.evaluation_scope, BakeEvaluationScope):
            raise TypeError("evaluation_scope must be BakeEvaluationScope")
        if not isinstance(self.material_slot_indices, tuple) or not self.material_slot_indices:
            raise ValueError("material_slot_indices must be a non-empty tuple")
        for index, slot_index in enumerate(self.material_slot_indices):
            require_integer(
                slot_index,
                f"material_slot_indices[{index}]",
                minimum=0,
            )
        if tuple(sorted(set(self.material_slot_indices))) != self.material_slot_indices:
            raise ValueError("material_slot_indices must be sorted and unique")
        if not isinstance(self.semantic_channels, tuple) or not self.semantic_channels:
            raise ValueError("semantic_channels must be a non-empty tuple")
        if not all(
            isinstance(channel, MaterialSemanticChannel)
            for channel in self.semantic_channels
        ):
            raise TypeError("semantic_channels must contain MaterialSemanticChannel")
        if not isinstance(self.material_preparations, tuple):
            raise TypeError("material_preparations must be tuple")
        if not all(
            isinstance(item, MaterialSlotPreparation)
            for item in self.material_preparations
        ):
            raise TypeError("material_preparations must contain MaterialSlotPreparation")
        preparation_indices = tuple(item.slot_index for item in self.material_preparations)
        if preparation_indices != tuple(sorted(set(preparation_indices))):
            raise ValueError("material_preparations must be ordered and unique by slot")


@dataclass(frozen=True, slots=True)
class BakeCompositePlan:
    mode: BakeCompositeMode = BakeCompositeMode.SINGLE
    clamp_rgb: bool = True
    color_pass_indices: Tuple[int, ...] = ()
    alpha_pass_index: int | None = None
    unpremultiply_color_by_alpha: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.mode, BakeCompositeMode):
            raise TypeError("mode must be BakeCompositeMode")
        if not isinstance(self.clamp_rgb, bool):
            raise TypeError("clamp_rgb must be bool")
        if not isinstance(self.unpremultiply_color_by_alpha, bool):
            raise TypeError("unpremultiply_color_by_alpha must be bool")
        if not isinstance(self.color_pass_indices, tuple):
            raise TypeError("color_pass_indices must be tuple")
        for index, pass_index in enumerate(self.color_pass_indices):
            require_integer(
                pass_index,
                f"color_pass_indices[{index}]",
                minimum=0,
            )
        if self.color_pass_indices != tuple(sorted(set(self.color_pass_indices))):
            raise ValueError("color_pass_indices must be sorted and unique")
        if self.alpha_pass_index is not None:
            require_integer(self.alpha_pass_index, "alpha_pass_index", minimum=0)
        if self.mode is BakeCompositeMode.SINGLE:
            if self.alpha_pass_index is not None or self.color_pass_indices:
                raise ValueError("SINGLE composition cannot declare pass routing")
            if self.unpremultiply_color_by_alpha:
                raise ValueError("SINGLE composition cannot request alpha unpremultiplication")
        elif self.mode is BakeCompositeMode.ADD_RGB_MAX_ALPHA:
            if self.alpha_pass_index is not None:
                raise ValueError("ADD_RGB_MAX_ALPHA cannot declare alpha_pass_index")
            if self.unpremultiply_color_by_alpha:
                raise ValueError(
                    "ADD_RGB_MAX_ALPHA cannot request explicit alpha unpremultiplication"
                )
        elif self.mode is BakeCompositeMode.ADD_RGB_REPLACE_ALPHA:
            if self.alpha_pass_index is None:
                raise ValueError("ADD_RGB_REPLACE_ALPHA requires alpha_pass_index")


@dataclass(frozen=True, slots=True)
class BakePlan:
    source_object_id: str
    settings: BakeSettings
    material_analysis: ObjectMaterialAnalysis
    bake_mode: BakeMode
    frame_tasks: Tuple[BakeFrameTask, ...]
    representative_task_index: int = 0
    passes: Tuple[BakePassPlan, ...] = ()
    composite: BakeCompositePlan = BakeCompositePlan()
    object_context: ObjectBakeContext | None = None
    scene_context: SceneBakeContext | None = None

    def __post_init__(self) -> None:
        require_non_empty_string(self.source_object_id, "source_object_id")
        if not isinstance(self.settings, BakeSettings):
            raise TypeError("settings must be BakeSettings")
        if not isinstance(self.material_analysis, ObjectMaterialAnalysis):
            raise TypeError("material_analysis must be ObjectMaterialAnalysis")
        if self.source_object_id != self.material_analysis.source_object_id:
            raise ValueError("source_object_id and material_analysis disagree")
        if not isinstance(self.bake_mode, BakeMode):
            raise TypeError("bake_mode must be BakeMode")
        if self.object_context is not None:
            if not isinstance(self.object_context, ObjectBakeContext):
                raise TypeError("object_context must be ObjectBakeContext or None")
            if self.object_context.source_object_id != self.source_object_id:
                raise ValueError("object_context source_object_id disagrees with BakePlan")
        if self.scene_context is not None and not isinstance(
            self.scene_context,
            SceneBakeContext,
        ):
            raise TypeError("scene_context must be SceneBakeContext or None")
        if not isinstance(self.frame_tasks, tuple) or not self.frame_tasks:
            raise ValueError("frame_tasks cannot be empty")
        if not all(isinstance(task, BakeFrameTask) for task in self.frame_tasks):
            raise TypeError("frame_tasks must contain BakeFrameTask values")
        actual_indices = tuple(task.task_index for task in self.frame_tasks)
        if actual_indices != tuple(range(len(self.frame_tasks))):
            raise ValueError("frame task indices must be ordered and dense from zero")
        require_integer(
            self.representative_task_index,
            "representative_task_index",
            minimum=0,
        )
        if self.representative_task_index >= len(self.frame_tasks):
            raise ValueError("representative_task_index is out of range")
        if not isinstance(self.passes, tuple):
            raise TypeError("passes must be tuple")
        if not isinstance(self.composite, BakeCompositePlan):
            raise TypeError("composite must be BakeCompositePlan")

        if not self.passes:
            usable_slots = tuple(
                slot.slot_index
                for slot in self.material_analysis.slots
                if slot.kind is not MaterialKind.EMPTY
            )
            if not usable_slots:
                raise ValueError("BakePlan requires at least one usable material slot")
            object.__setattr__(
                self,
                "passes",
                (
                    BakePassPlan(
                        pass_index=0,
                        strategy_id=BakeStrategyId.LEGACY_SINGLE_PASS,
                        bake_mode=self.bake_mode,
                        material_slot_indices=usable_slots,
                        semantic_channels=(MaterialSemanticChannel.SURFACE_COLOR,),
                    ),
                ),
            )
        if not all(isinstance(item, BakePassPlan) for item in self.passes):
            raise TypeError("passes must contain BakePassPlan values")
        pass_indices = tuple(item.pass_index for item in self.passes)
        if pass_indices != tuple(range(len(self.passes))):
            raise ValueError("bake pass indices must be ordered and dense from zero")
        if self.bake_mode is not self.passes[0].bake_mode:
            raise ValueError("bake_mode must match the first compatibility bake pass")
        if self.composite.mode is BakeCompositeMode.SINGLE and len(self.passes) != 1:
            raise ValueError("SINGLE composition requires exactly one pass")

        slot_count = len(self.material_analysis.slots)
        for item in self.passes:
            if max(item.material_slot_indices) >= slot_count:
                raise ValueError("bake pass references a material slot outside analysis")
            if item.material_preparations and max(
                prep.slot_index for prep in item.material_preparations
            ) >= slot_count:
                raise ValueError("material preparation references a slot outside analysis")

        referenced_pass_indices = set(self.composite.color_pass_indices)
        if self.composite.alpha_pass_index is not None:
            referenced_pass_indices.add(self.composite.alpha_pass_index)
        if referenced_pass_indices and max(referenced_pass_indices) >= len(self.passes):
            raise ValueError("composite plan references a missing bake pass")
        if self.scene_aware and self.scene_context is None:
            raise ValueError("scene-aware BakePlan requires scene_context")
        if any(
            item.evaluation_scope is BakeEvaluationScope.CAMERA for item in self.passes
        ) and (self.scene_context is None or self.scene_context.camera is None):
            raise ValueError("camera-aware BakePlan requires an active camera snapshot")

    @property
    def representative_task(self) -> BakeFrameTask:
        return self.frame_tasks[self.representative_task_index]

    @property
    def sequence(self) -> bool:
        return any(task.timeline_frame is not None for task in self.frame_tasks)

    @property
    def multipass(self) -> bool:
        return len(self.passes) > 1

    @property
    def requires_composition(self) -> bool:
        return self.composite.mode is not BakeCompositeMode.SINGLE

    @property
    def scene_aware(self) -> bool:
        return any(
            item.evaluation_scope in {BakeEvaluationScope.SCENE, BakeEvaluationScope.CAMERA}
            for item in self.passes
        )


class BakePlanError(ValueError):
    """Raised when material analysis cannot produce a deterministic bake plan."""


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
    *,
    object_context: ObjectBakeContext | None = None,
    scene_context: SceneBakeContext | None = None,
) -> BakePlan:
    """Build a complete plan through the deterministic strategy registry."""

    if not isinstance(analysis, ObjectMaterialAnalysis):
        raise TypeError("analysis must be ObjectMaterialAnalysis")
    if not isinstance(settings, BakeSettings):
        raise TypeError("settings must be BakeSettings")
    if object_context is not None and not isinstance(object_context, ObjectBakeContext):
        raise TypeError("object_context must be ObjectBakeContext or None")
    if scene_context is not None and not isinstance(scene_context, SceneBakeContext):
        raise TypeError("scene_context must be SceneBakeContext or None")

    from .strategies import resolve_bake_strategy_plan

    passes, composite = resolve_bake_strategy_plan(
        analysis,
        settings,
        object_context=object_context,
        scene_context=scene_context,
    )
    tasks = _build_frame_tasks(settings)
    return BakePlan(
        source_object_id=analysis.source_object_id,
        settings=settings,
        material_analysis=analysis,
        bake_mode=passes[0].bake_mode,
        frame_tasks=tasks,
        representative_task_index=0,
        passes=passes,
        composite=composite,
        object_context=object_context,
        scene_context=scene_context,
    )
