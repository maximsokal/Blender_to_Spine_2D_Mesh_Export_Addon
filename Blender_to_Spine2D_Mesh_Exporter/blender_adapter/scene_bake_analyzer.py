"""Read Blender object/scene state into immutable scene-aware bake contexts."""

from __future__ import annotations

import logging
from typing import Any, Iterable, Tuple

from ..domain.baking.context import (
    CameraBakeSnapshot,
    ColorManagementSnapshot,
    LightBakeSnapshot,
    ObjectBakeContext,
    SceneBakeContext,
    WorldBakeSnapshot,
)

logger = logging.getLogger(__name__)


class SceneBakeAnalysisError(RuntimeError):
    """Raised when required Blender scene state cannot be inspected safely."""


def _load_bpy() -> Any:
    try:
        import bpy
    except Exception as exc:
        raise SceneBakeAnalysisError("Blender bpy module is unavailable") from exc
    return bpy


def _name(value: Any, *, fallback: str = "") -> str:
    return str(
        getattr(value, "name_full", None)
        or getattr(value, "name", None)
        or fallback
        or ""
    ).strip()


def _matrix_tuple(value: Any) -> Tuple[float, ...]:
    try:
        return tuple(float(value[row][column]) for row in range(4) for column in range(4))
    except Exception as exc:
        raise SceneBakeAnalysisError("Unable to read a 4x4 Blender matrix") from exc


def _color_tuple(value: Any, *, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    try:
        return float(value[0]), float(value[1]), float(value[2])
    except Exception:
        return default


def _animated(*datablocks: Any) -> bool:
    for datablock in datablocks:
        if datablock is None:
            continue
        animation_data = getattr(datablock, "animation_data", None)
        if animation_data is None:
            continue
        if getattr(animation_data, "action", None) is not None:
            return True
        try:
            if len(getattr(animation_data, "drivers", ())) > 0:
                return True
        except Exception:
            return True
    return False


def _visible_boolean(obj: Any, property_name: str, default: bool) -> bool:
    try:
        return bool(getattr(obj, property_name))
    except Exception:
        return default


def _object_render_visible(obj: Any) -> bool:
    try:
        return not bool(getattr(obj, "hide_render", False))
    except Exception:
        return True


def _active_world_output(node_tree: Any) -> Any | None:
    try:
        outputs = tuple(node for node in node_tree.nodes if str(getattr(node, "type", "")) == "OUTPUT_WORLD")
    except Exception:
        return None
    active = tuple(node for node in outputs if bool(getattr(node, "is_active_output", False)))
    return active[0] if active else (outputs[0] if outputs else None)


def _input_socket(node: Any, name: str) -> Any | None:
    inputs = getattr(node, "inputs", None)
    if inputs is None:
        return None
    getter = getattr(inputs, "get", None)
    if callable(getter):
        try:
            result = getter(name)
            if result is not None:
                return result
        except Exception:
            logger.debug("World socket lookup failed", exc_info=True)
    try:
        return next(socket for socket in inputs if str(getattr(socket, "name", "")) == name)
    except Exception:
        return None


def _background_strength(world: Any) -> float | None:
    node_tree = getattr(world, "node_tree", None)
    if node_tree is None:
        return 1.0
    output = _active_world_output(node_tree)
    if output is None:
        return None
    surface = _input_socket(output, "Surface")
    try:
        links = tuple(getattr(surface, "links", ())) if surface is not None else ()
    except Exception:
        links = ()
    if not links:
        return 0.0
    source_node = getattr(links[0], "from_node", None)
    if str(getattr(source_node, "type", "")) != "BACKGROUND":
        return None
    strength = _input_socket(source_node, "Strength")
    if strength is None or bool(getattr(strength, "is_linked", False)):
        return None
    try:
        return max(0.0, float(getattr(strength, "default_value", 0.0)))
    except Exception:
        return None


def _analyse_world(scene: Any) -> WorldBakeSnapshot | None:
    world = getattr(scene, "world", None)
    if world is None:
        return None
    world_name = _name(world)
    if not world_name:
        raise SceneBakeAnalysisError("Scene World has an empty name")
    use_nodes = bool(getattr(world, "use_nodes", False))
    node_tree = getattr(world, "node_tree", None)
    try:
        node_types = tuple(
            sorted(
                {
                    str(getattr(node, "type", "") or "UNKNOWN")
                    for node in getattr(node_tree, "nodes", ())
                }
            )
        )
    except Exception:
        node_types = ()
    return WorldBakeSnapshot(
        world_name=world_name,
        color=_color_tuple(
            getattr(world, "color", (0.0, 0.0, 0.0)),
            default=(0.0, 0.0, 0.0),
        ),
        use_nodes=use_nodes,
        node_types=node_types,
        background_strength=_background_strength(world),
        animated=_animated(world, node_tree),
    )


def _analyse_light(obj: Any) -> LightBakeSnapshot:
    data = getattr(obj, "data", None)
    object_id = _name(obj)
    if not object_id or data is None:
        raise SceneBakeAnalysisError("Visible light object is missing name or data")
    return LightBakeSnapshot(
        object_id=object_id,
        light_type=str(getattr(data, "type", "POINT") or "POINT"),
        energy=max(0.0, float(getattr(data, "energy", 0.0) or 0.0)),
        color=_color_tuple(
            getattr(data, "color", (1.0, 1.0, 1.0)),
            default=(1.0, 1.0, 1.0),
        ),
        world_matrix=_matrix_tuple(getattr(obj, "matrix_world", None)),
        use_shadow=bool(getattr(data, "use_shadow", True)),
        animated=_animated(obj, data),
    )


def _analyse_camera(scene: Any) -> CameraBakeSnapshot | None:
    obj = getattr(scene, "camera", None)
    if obj is None:
        return None
    data = getattr(obj, "data", None)
    object_id = _name(obj)
    if not object_id or data is None or getattr(obj, "type", None) != "CAMERA":
        raise SceneBakeAnalysisError("scene.camera is not a valid Blender Camera object")
    return CameraBakeSnapshot(
        object_id=object_id,
        camera_type=str(getattr(data, "type", "PERSP") or "PERSP"),
        world_matrix=_matrix_tuple(getattr(obj, "matrix_world", None)),
        lens=max(1e-8, float(getattr(data, "lens", 50.0) or 50.0)),
        ortho_scale=max(1e-8, float(getattr(data, "ortho_scale", 6.0) or 6.0)),
        clip_start=max(1e-8, float(getattr(data, "clip_start", 0.1) or 0.1)),
        clip_end=max(1e-7, float(getattr(data, "clip_end", 1000.0) or 1000.0)),
        animated=_animated(obj, data),
    )


def _analyse_color_management(scene: Any) -> ColorManagementSnapshot:
    view = getattr(scene, "view_settings", None)
    if view is None:
        raise SceneBakeAnalysisError("Scene view_settings are unavailable")
    return ColorManagementSnapshot(
        view_transform=str(getattr(view, "view_transform", "Standard") or "Standard"),
        look=str(getattr(view, "look", "") or ""),
        exposure=float(getattr(view, "exposure", 0.0) or 0.0),
        gamma=float(getattr(view, "gamma", 1.0) or 1.0),
    )


def analyse_object_bake_context(obj: Any) -> ObjectBakeContext:
    if obj is None or getattr(obj, "type", None) != "MESH":
        raise SceneBakeAnalysisError("obj must be a Blender MESH object")
    object_id = _name(obj)
    if not object_id:
        raise SceneBakeAnalysisError("Source object name is empty")
    collection_names = tuple(
        sorted(
            {
                name
                for collection in tuple(getattr(obj, "users_collection", ()))
                if (name := _name(collection))
            },
            key=str.casefold,
        )
    )
    data = getattr(obj, "data", None)
    return ObjectBakeContext(
        source_object_id=object_id,
        object_type=str(getattr(obj, "type", "MESH") or "MESH"),
        world_matrix=_matrix_tuple(getattr(obj, "matrix_world", None)),
        collection_names=collection_names,
        hide_render=bool(getattr(obj, "hide_render", False)),
        visible_camera=_visible_boolean(obj, "visible_camera", True),
        visible_shadow=_visible_boolean(obj, "visible_shadow", True),
        animated=_animated(obj, data),
    )


def analyse_scene_bake_context(
    *,
    scene: Any | None = None,
    context: Any | None = None,
) -> SceneBakeContext:
    bpy_module = _load_bpy()
    resolved_context = context or getattr(bpy_module, "context", None)
    resolved_scene = scene or getattr(resolved_context, "scene", None)
    if resolved_scene is None:
        raise SceneBakeAnalysisError("A Blender Scene is required")
    scene_name = _name(resolved_scene)
    if not scene_name:
        raise SceneBakeAnalysisError("Scene name is empty")

    try:
        objects = tuple(getattr(resolved_scene, "objects", ()))
    except Exception as exc:
        raise SceneBakeAnalysisError("Unable to iterate scene objects") from exc

    visible = tuple(obj for obj in objects if _object_render_visible(obj))
    lights = tuple(
        sorted(
            (_analyse_light(obj) for obj in visible if getattr(obj, "type", None) == "LIGHT"),
            key=lambda item: item.object_id.casefold(),
        )
    )
    visible_object_ids = tuple(
        sorted(
            {_name(obj) for obj in visible if _name(obj)},
            key=str.casefold,
        )
    )
    caster_types = {"MESH", "CURVE", "SURFACE", "META", "FONT", "VOLUME"}
    shadow_caster_ids = tuple(
        sorted(
            {
                _name(obj)
                for obj in visible
                if getattr(obj, "type", None) in caster_types
                and _visible_boolean(obj, "visible_shadow", True)
                and _name(obj)
            },
            key=str.casefold,
        )
    )
    return SceneBakeContext(
        scene_name=scene_name,
        render_engine=str(
            getattr(getattr(resolved_scene, "render", None), "engine", "CYCLES")
            or "CYCLES"
        ),
        analysis_frame=int(getattr(resolved_scene, "frame_current", 0) or 0),
        world=_analyse_world(resolved_scene),
        camera=_analyse_camera(resolved_scene),
        lights=lights,
        visible_object_ids=visible_object_ids,
        shadow_caster_ids=shadow_caster_ids,
        color_management=_analyse_color_management(resolved_scene),
    )


def analyse_bake_contexts(
    source_obj: Any,
    *,
    scene: Any | None = None,
    context: Any | None = None,
) -> tuple[ObjectBakeContext, SceneBakeContext]:
    """Capture source-object and scene facts in one deterministic adapter call."""

    object_context = analyse_object_bake_context(source_obj)
    scene_context = analyse_scene_bake_context(scene=scene, context=context)
    if object_context.source_object_id not in scene_context.visible_object_ids and not object_context.hide_render:
        logger.warning(
            "Source object '%s' is not present in the scene visible-object snapshot",
            object_context.source_object_id,
        )
    return object_context, scene_context


def validate_runtime_scene_context(
    source_obj: Any,
    expected_object: ObjectBakeContext | None,
    expected_scene: SceneBakeContext | None,
    *,
    scene: Any,
    context: Any | None = None,
) -> None:
    """Validate stable scene identities before executing a planned scene-aware bake.

    Animated numeric values and transforms are deliberately not compared: Blender evaluates
    those at every frame task. Object, scene, World, light and camera identities must remain
    stable between planning and execution.
    """

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
    if expected_scene is not None:
        if current_scene.scene_name != expected_scene.scene_name:
            failures.append("scene identity changed")
        expected_world = None if expected_scene.world is None else expected_scene.world.world_name
        current_world = None if current_scene.world is None else current_scene.world.world_name
        if current_world != expected_world:
            failures.append("World identity changed")
        expected_camera = None if expected_scene.camera is None else expected_scene.camera.object_id
        current_camera = None if current_scene.camera is None else current_scene.camera.object_id
        if current_camera != expected_camera:
            failures.append("active camera identity changed")
        expected_lights = tuple(item.object_id for item in expected_scene.lights)
        current_lights = tuple(item.object_id for item in current_scene.lights)
        if current_lights != expected_lights:
            failures.append("visible light set changed")
    if failures:
        raise SceneBakeAnalysisError(
            "Scene bake context changed after planning: " + "; ".join(failures)
        )
