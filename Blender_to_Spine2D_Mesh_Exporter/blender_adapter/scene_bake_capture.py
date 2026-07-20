"""Assemble deterministic object and Scene contexts for bake planning."""

from __future__ import annotations

import logging
from typing import Any

from ..domain.baking.context import ObjectBakeContext, SceneBakeContext
from .scene_bake_error import SceneBakeAnalysisError
from .scene_bake_resources import (
    analyse_camera,
    analyse_color_management,
    analyse_light,
    analyse_object_bake_context,
)
from .scene_bake_rna import (
    name,
    object_render_visible,
    resolve_scene_inputs,
    visible_boolean,
)
from .scene_bake_world import analyse_world

logger = logging.getLogger(__name__)

SHADOW_CASTER_TYPES = frozenset(
    {"MESH", "CURVE", "SURFACE", "META", "FONT", "VOLUME"}
)


def analyse_scene_bake_context(
    *,
    scene: Any | None = None,
    context: Any | None = None,
) -> SceneBakeContext:
    _, _, resolved_scene = resolve_scene_inputs(scene=scene, context=context)
    scene_name = name(resolved_scene)
    if not scene_name:
        raise SceneBakeAnalysisError("Scene name is empty")

    try:
        objects = tuple(getattr(resolved_scene, "objects", ()))
    except Exception as exc:
        raise SceneBakeAnalysisError("Unable to iterate scene objects") from exc

    visible = tuple(obj for obj in objects if object_render_visible(obj))
    lights = tuple(
        sorted(
            (
                analyse_light(obj)
                for obj in visible
                if getattr(obj, "type", None) == "LIGHT"
            ),
            key=lambda item: (item.object_id.casefold(), item.object_id),
        )
    )

    visible_names = tuple(
        object_name
        for obj in visible
        if (object_name := name(obj))
    )
    visible_object_ids = tuple(sorted(set(visible_names), key=str.casefold))

    shadow_names = tuple(
        object_name
        for obj in visible
        if getattr(obj, "type", None) in SHADOW_CASTER_TYPES
        and visible_boolean(obj, "visible_shadow", True)
        and (object_name := name(obj))
    )
    shadow_caster_ids = tuple(sorted(set(shadow_names), key=str.casefold))

    try:
        analysis_frame = int(getattr(resolved_scene, "frame_current", 0) or 0)
    except Exception as exc:
        raise SceneBakeAnalysisError("Scene frame_current is not an integer") from exc

    try:
        return SceneBakeContext(
            scene_name=scene_name,
            render_engine=str(
                getattr(getattr(resolved_scene, "render", None), "engine", "CYCLES")
                or "CYCLES"
            ),
            analysis_frame=analysis_frame,
            world=analyse_world(resolved_scene),
            camera=analyse_camera(resolved_scene),
            lights=lights,
            visible_object_ids=visible_object_ids,
            shadow_caster_ids=shadow_caster_ids,
            color_management=analyse_color_management(resolved_scene),
        )
    except SceneBakeAnalysisError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise SceneBakeAnalysisError(
            f"Unable to build Scene snapshot for '{scene_name}': {exc}"
        ) from exc


def analyse_bake_contexts(
    source_obj: Any,
    *,
    scene: Any | None = None,
    context: Any | None = None,
) -> tuple[ObjectBakeContext, SceneBakeContext]:
    """Capture source-object and Scene facts in one deterministic adapter call."""

    object_context = analyse_object_bake_context(source_obj)
    scene_context = analyse_scene_bake_context(scene=scene, context=context)
    if (
        object_context.source_object_id not in scene_context.visible_object_ids
        and not object_context.hide_render
    ):
        logger.warning(
            "Source object '%s' is not present in the scene visible-object snapshot",
            object_context.source_object_id,
        )
    return object_context, scene_context


__all__ = [
    "SHADOW_CASTER_TYPES",
    "analyse_bake_contexts",
    "analyse_scene_bake_context",
]
