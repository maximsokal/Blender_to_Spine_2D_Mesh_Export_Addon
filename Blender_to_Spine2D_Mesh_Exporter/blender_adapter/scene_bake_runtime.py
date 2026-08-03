"""Fail-closed planning/execution parity for scene-aware baking."""

from __future__ import annotations

from math import isfinite
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
from .scene_bake_resources import analyse_object_bake_context
from .scene_bake_rna import matrix_tuple


_MATRIX_RELATIVE_TOLERANCE = 1.0e-9
_MATRIX_ABSOLUTE_TOLERANCE = 1.0e-10


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


def _matrix_difference(
    expected: tuple[float, ...],
    actual: tuple[float, ...],
) -> tuple[bool, float, int]:
    """Return ``(equal, maximum_delta, maximum_delta_index)`` for affine matrices."""

    if not isinstance(expected, tuple) or not isinstance(actual, tuple):
        raise TypeError("expected and actual matrices must be tuples")
    if len(expected) != 16 or len(actual) != 16:
        raise ValueError("expected and actual matrices must contain 16 values")

    maximum_delta = 0.0
    maximum_index = 0
    equal = True
    for index, (expected_value, actual_value) in enumerate(
        zip(expected, actual, strict=True)
    ):
        expected_float = float(expected_value)
        actual_float = float(actual_value)
        if not isfinite(expected_float) or not isfinite(actual_float):
            return False, float("inf"), index
        delta = abs(expected_float - actual_float)
        if delta > maximum_delta:
            maximum_delta = delta
            maximum_index = index
        tolerance = _MATRIX_ABSOLUTE_TOLERANCE + _MATRIX_RELATIVE_TOLERANCE * max(
            1.0,
            abs(expected_float),
            abs(actual_float),
        )
        if delta > tolerance:
            equal = False
    return equal, maximum_delta, maximum_index


def validate_runtime_object_transform(
    source_obj: Any,
    expected_object: ObjectBakeContext | None,
    *,
    timeline_frame: int | None,
    allow_sequence_transform: bool = False,
) -> ObjectBakeContext | None:
    """Validate source identity and return the current frame-evaluated transform.

    Static object-bake callers keep the historical fail-closed matrix check. Sequence
    execution may opt into frame-evaluated transforms because its temporary UV target
    is synchronized to the source before every bake operation.
    """

    if expected_object is None:
        return None
    if not isinstance(expected_object, ObjectBakeContext):
        raise TypeError("expected_object must be ObjectBakeContext or None")
    if not isinstance(allow_sequence_transform, bool):
        raise TypeError("allow_sequence_transform must be bool")

    current = analyse_object_bake_context(source_obj)
    if current.source_object_id != expected_object.source_object_id:
        raise SceneBakeAnalysisError(
            "Object bake source identity changed while evaluating a frame"
        )

    equal, maximum_delta, maximum_index = _matrix_difference(
        expected_object.world_matrix,
        current.world_matrix,
    )
    if equal:
        return current
    if allow_sequence_transform and timeline_frame is not None:
        return current

    raise SceneBakeAnalysisError(
        "Object bake source matrix_world changed after planning and no synchronized "
        "sequence target is active; "
        f"source={expected_object.source_object_id!r}, frame={timeline_frame!r}, "
        f"maximum_matrix_delta={maximum_delta}, matrix_index={maximum_index}"
    )


def synchronize_runtime_object_transform(
    source_obj: Any,
    target_obj: Any,
    expected_object: ObjectBakeContext | None,
    *,
    context: Any,
    timeline_frame: int | None,
) -> ObjectBakeContext | None:
    """Copy the current source ``matrix_world`` to one temporary UV bake target.

    The target mesh stores fixed local-space geometry and UVs. Synchronizing only its
    world matrix preserves that topology while allowing Camera, Reflection, Generated,
    Object, Fresnel, and similar material inputs to evaluate at each sequence frame.
    Vertex deformation is intentionally outside this contract.
    """

    if target_obj is None:
        raise SceneBakeAnalysisError("Temporary object-bake target is unavailable")
    if context is None:
        raise SceneBakeAnalysisError("Blender context is required for transform sync")

    current = validate_runtime_object_transform(
        source_obj,
        expected_object,
        timeline_frame=timeline_frame,
        allow_sequence_transform=True,
    )
    if current is None:
        return None

    source_matrix = getattr(source_obj, "matrix_world", None)
    copy_matrix = getattr(source_matrix, "copy", None)
    if not callable(copy_matrix):
        raise SceneBakeAnalysisError(
            "Source object matrix_world does not provide copy()"
        )
    try:
        target_obj.matrix_world = copy_matrix()
    except Exception as exc:
        raise SceneBakeAnalysisError(
            "Unable to synchronize temporary UV target matrix_world from source"
        ) from exc

    view_layer = getattr(context, "view_layer", None)
    update = getattr(view_layer, "update", None)
    if not callable(update):
        raise SceneBakeAnalysisError(
            "Active View Layer does not provide update() after transform sync"
        )
    try:
        update()
    except Exception as exc:
        raise SceneBakeAnalysisError(
            "Unable to update the View Layer after transform sync"
        ) from exc

    try:
        target_matrix = matrix_tuple(getattr(target_obj, "matrix_world", None))
    except SceneBakeAnalysisError:
        raise
    except Exception as exc:
        raise SceneBakeAnalysisError(
            "Unable to inspect synchronized UV target matrix_world"
        ) from exc

    equal, maximum_delta, maximum_index = _matrix_difference(
        current.world_matrix,
        target_matrix,
    )
    if not equal:
        raise SceneBakeAnalysisError(
            "Temporary UV target matrix_world differs from the frame-evaluated source; "
            f"source={current.source_object_id!r}, frame={timeline_frame!r}, "
            f"maximum_matrix_delta={maximum_delta}, matrix_index={maximum_index}"
        )
    return current


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


__all__ = [
    "synchronize_runtime_object_transform",
    "validate_runtime_object_transform",
    "validate_runtime_scene_context",
]
