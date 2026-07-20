"""Fail-closed planning/execution parity for scene-aware baking."""

from __future__ import annotations

from typing import Any

from ..domain.baking.context import (
    CameraBakeSnapshot,
    LightBakeSnapshot,
    ObjectBakeContext,
    SceneBakeContext,
    WorldBakeSnapshot,
)
from .scene_bake_capture import analyse_bake_contexts
from .scene_bake_error import SceneBakeAnalysisError


def _world_structure(
    value: WorldBakeSnapshot | None,
) -> tuple[str, bool, tuple[str, ...], bool] | None:
    if value is None:
        return None
    return value.world_name, value.use_nodes, value.node_types, value.animated


def _camera_structure(
    value: CameraBakeSnapshot | None,
) -> tuple[str, str, bool] | None:
    if value is None:
        return None
    return value.object_id, value.camera_type, value.animated


def _light_structure(
    values: tuple[LightBakeSnapshot, ...],
) -> tuple[tuple[str, str, bool, bool], ...]:
    return tuple(
        (value.object_id, value.light_type, value.use_shadow, value.animated)
        for value in values
    )


def validate_runtime_scene_context(
    source_obj: Any,
    expected_object: ObjectBakeContext | None,
    expected_scene: SceneBakeContext | None,
    *,
    scene: Any,
    context: Any | None = None,
) -> None:
    """Reject structural scene changes while allowing frame-evaluated numerics."""

    if expected_object is None and expected_scene is None:
        return

    current_object, current_scene = analyse_bake_contexts(
        source_obj,
        scene=scene,
        context=context,
    )
    failures: list[str] = []

    if expected_object is not None:
        if current_object.source_object_id != expected_object.source_object_id:
            failures.append("source object identity changed")
        if current_object.object_type != expected_object.object_type:
            failures.append("source object type changed")
        if current_object.collection_names != expected_object.collection_names:
            failures.append("source collection membership changed")
        if current_object.hide_render != expected_object.hide_render:
            failures.append("source render visibility changed")
        if current_object.visible_camera != expected_object.visible_camera:
            failures.append("source camera visibility changed")
        if current_object.visible_shadow != expected_object.visible_shadow:
            failures.append("source shadow visibility changed")
        if current_object.animated != expected_object.animated:
            failures.append("source animation status changed")

    if expected_scene is not None:
        if current_scene.scene_name != expected_scene.scene_name:
            failures.append("scene identity changed")
        if current_scene.render_engine != expected_scene.render_engine:
            failures.append("scene render engine changed")
        if _world_structure(current_scene.world) != _world_structure(expected_scene.world):
            failures.append("World structure changed")
        if _camera_structure(current_scene.camera) != _camera_structure(expected_scene.camera):
            failures.append("active camera structure changed")
        if _light_structure(current_scene.lights) != _light_structure(expected_scene.lights):
            failures.append("visible light structure changed")
        if current_scene.visible_object_ids != expected_scene.visible_object_ids:
            failures.append("render-visible object set changed")
        if current_scene.shadow_caster_ids != expected_scene.shadow_caster_ids:
            failures.append("shadow-caster set changed")
        if current_scene.color_management != expected_scene.color_management:
            failures.append("color management changed")

    if failures:
        raise SceneBakeAnalysisError(
            "Scene bake context changed after planning: " + "; ".join(failures)
        )


__all__ = ["validate_runtime_scene_context"]
