"""Blender-independent object and scene snapshots used by scene-aware baking."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Tuple


def _validate_name(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_matrix(value: Tuple[float, ...], field_name: str) -> None:
    if not isinstance(value, tuple) or len(value) != 16:
        raise ValueError(f"{field_name} must contain sixteen values")
    if not all(isinstance(item, (int, float)) and isfinite(float(item)) for item in value):
        raise ValueError(f"{field_name} must contain finite numeric values")


def _validate_color(value: Tuple[float, ...], field_name: str) -> None:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{field_name} must contain three values")
    if not all(isinstance(item, (int, float)) and isfinite(float(item)) for item in value):
        raise ValueError(f"{field_name} must contain finite numeric values")


@dataclass(frozen=True, slots=True)
class ObjectBakeContext:
    """Immutable facts about the source object that affect a scene-aware bake."""

    source_object_id: str
    object_type: str
    world_matrix: Tuple[float, ...]
    collection_names: Tuple[str, ...] = ()
    hide_render: bool = False
    visible_camera: bool = True
    visible_shadow: bool = True
    animated: bool = False

    def __post_init__(self) -> None:
        _validate_name(self.source_object_id, "source_object_id")
        _validate_name(self.object_type, "object_type")
        _validate_matrix(self.world_matrix, "world_matrix")
        if not isinstance(self.collection_names, tuple) or not all(
            isinstance(value, str) and value.strip() for value in self.collection_names
        ):
            raise TypeError("collection_names must contain non-empty strings")
        if len(self.collection_names) != len(set(self.collection_names)):
            raise ValueError("collection_names cannot contain duplicates")
        for field_name in ("hide_render", "visible_camera", "visible_shadow", "animated"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")


@dataclass(frozen=True, slots=True)
class WorldBakeSnapshot:
    world_name: str
    color: Tuple[float, float, float]
    use_nodes: bool
    node_types: Tuple[str, ...] = ()
    background_strength: float | None = None
    animated: bool = False

    def __post_init__(self) -> None:
        _validate_name(self.world_name, "world_name")
        _validate_color(self.color, "color")
        if not isinstance(self.use_nodes, bool):
            raise TypeError("use_nodes must be bool")
        if not isinstance(self.node_types, tuple) or not all(
            isinstance(value, str) and value.strip() for value in self.node_types
        ):
            raise TypeError("node_types must contain non-empty strings")
        if self.background_strength is not None and (
            not isinstance(self.background_strength, (int, float))
            or not isfinite(float(self.background_strength))
            or float(self.background_strength) < 0.0
        ):
            raise ValueError("background_strength must be finite, non-negative, or None")
        if not isinstance(self.animated, bool):
            raise TypeError("animated must be bool")

    @property
    def effective(self) -> bool:
        color_energy = max(float(value) for value in self.color)
        if self.background_strength is None:
            return self.use_nodes or color_energy > 1e-8
        return float(self.background_strength) > 1e-8 and (
            self.use_nodes or color_energy > 1e-8
        )


@dataclass(frozen=True, slots=True)
class LightBakeSnapshot:
    object_id: str
    light_type: str
    energy: float
    color: Tuple[float, float, float]
    world_matrix: Tuple[float, ...]
    use_shadow: bool = True
    animated: bool = False

    def __post_init__(self) -> None:
        _validate_name(self.object_id, "object_id")
        _validate_name(self.light_type, "light_type")
        if not isinstance(self.energy, (int, float)) or not isfinite(float(self.energy)):
            raise ValueError("energy must be finite")
        if float(self.energy) < 0.0:
            raise ValueError("energy cannot be negative")
        _validate_color(self.color, "color")
        _validate_matrix(self.world_matrix, "world_matrix")
        if not isinstance(self.use_shadow, bool):
            raise TypeError("use_shadow must be bool")
        if not isinstance(self.animated, bool):
            raise TypeError("animated must be bool")

    @property
    def effective(self) -> bool:
        return float(self.energy) > 1e-8 and max(float(value) for value in self.color) > 1e-8


@dataclass(frozen=True, slots=True)
class CameraBakeSnapshot:
    object_id: str
    camera_type: str
    world_matrix: Tuple[float, ...]
    lens: float
    ortho_scale: float
    clip_start: float
    clip_end: float
    animated: bool = False

    def __post_init__(self) -> None:
        _validate_name(self.object_id, "object_id")
        _validate_name(self.camera_type, "camera_type")
        _validate_matrix(self.world_matrix, "world_matrix")
        for field_name in ("lens", "ortho_scale", "clip_start", "clip_end"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or not isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
            if float(value) <= 0.0:
                raise ValueError(f"{field_name} must be positive")
        if float(self.clip_end) <= float(self.clip_start):
            raise ValueError("clip_end must be greater than clip_start")
        if not isinstance(self.animated, bool):
            raise TypeError("animated must be bool")


@dataclass(frozen=True, slots=True)
class ColorManagementSnapshot:
    view_transform: str
    look: str
    exposure: float
    gamma: float

    def __post_init__(self) -> None:
        _validate_name(self.view_transform, "view_transform")
        if not isinstance(self.look, str):
            raise TypeError("look must be str")
        for field_name in ("exposure", "gamma"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or not isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        if float(self.gamma) <= 0.0:
            raise ValueError("gamma must be positive")


@dataclass(frozen=True, slots=True)
class SceneBakeContext:
    """Diagnostic snapshot of scene resources available to scene/camera strategies."""

    scene_name: str
    render_engine: str
    analysis_frame: int
    world: WorldBakeSnapshot | None
    camera: CameraBakeSnapshot | None
    lights: Tuple[LightBakeSnapshot, ...]
    visible_object_ids: Tuple[str, ...]
    shadow_caster_ids: Tuple[str, ...]
    color_management: ColorManagementSnapshot

    def __post_init__(self) -> None:
        _validate_name(self.scene_name, "scene_name")
        _validate_name(self.render_engine, "render_engine")
        if not isinstance(self.analysis_frame, int):
            raise TypeError("analysis_frame must be int")
        if self.world is not None and not isinstance(self.world, WorldBakeSnapshot):
            raise TypeError("world must be WorldBakeSnapshot or None")
        if self.camera is not None and not isinstance(self.camera, CameraBakeSnapshot):
            raise TypeError("camera must be CameraBakeSnapshot or None")
        if not isinstance(self.lights, tuple) or not all(
            isinstance(value, LightBakeSnapshot) for value in self.lights
        ):
            raise TypeError("lights must contain LightBakeSnapshot values")
        light_ids = tuple(value.object_id for value in self.lights)
        if light_ids != tuple(sorted(set(light_ids), key=str.casefold)):
            raise ValueError("lights must be sorted and unique by object_id")
        for field_name in ("visible_object_ids", "shadow_caster_ids"):
            values = getattr(self, field_name)
            if not isinstance(values, tuple) or not all(
                isinstance(value, str) and value.strip() for value in values
            ):
                raise TypeError(f"{field_name} must contain non-empty strings")
            if values != tuple(sorted(set(values), key=str.casefold)):
                raise ValueError(f"{field_name} must be sorted and unique")
        if not isinstance(self.color_management, ColorManagementSnapshot):
            raise TypeError("color_management must be ColorManagementSnapshot")

    @property
    def has_camera(self) -> bool:
        return self.camera is not None

    @property
    def has_effective_lighting(self) -> bool:
        return any(light.effective for light in self.lights) or (
            self.world is not None and self.world.effective
        )

    @property
    def animated_dependency_ids(self) -> Tuple[str, ...]:
        values = [light.object_id for light in self.lights if light.animated]
        if self.camera is not None and self.camera.animated:
            values.append(self.camera.object_id)
        if self.world is not None and self.world.animated:
            values.append(self.world.world_name)
        return tuple(sorted(set(values), key=str.casefold))
