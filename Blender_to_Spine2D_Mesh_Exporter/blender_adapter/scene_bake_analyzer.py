"""Compatibility facade for decomposed scene-bake capture and parity."""

from .scene_bake_capture import (
    analyse_bake_contexts,
    analyse_scene_bake_context,
)
from .scene_bake_error import SceneBakeAnalysisError
from .scene_bake_resources import (
    analyse_camera as _analyse_camera,
    analyse_color_management as _analyse_color_management,
    analyse_light as _analyse_light,
    analyse_object_bake_context,
)
from .scene_bake_rna import (
    animated as _animated,
    color_tuple as _color_tuple,
    load_bpy as _load_bpy,
    matrix_tuple as _matrix_tuple,
    name as _name,
    object_render_visible as _object_render_visible,
    visible_boolean as _visible_boolean,
)
from .scene_bake_runtime import validate_runtime_scene_context
from .scene_bake_world import (
    active_world_output as _active_world_output,
    analyse_world as _analyse_world,
    background_strength as _background_strength,
    input_socket as _input_socket,
)


__all__ = [
    "SceneBakeAnalysisError",
    "_active_world_output",
    "_analyse_camera",
    "_analyse_color_management",
    "_analyse_light",
    "_analyse_world",
    "_animated",
    "_background_strength",
    "_color_tuple",
    "_input_socket",
    "_load_bpy",
    "_matrix_tuple",
    "_name",
    "_object_render_visible",
    "_visible_boolean",
    "analyse_bake_contexts",
    "analyse_object_bake_context",
    "analyse_scene_bake_context",
    "validate_runtime_scene_context",
]
