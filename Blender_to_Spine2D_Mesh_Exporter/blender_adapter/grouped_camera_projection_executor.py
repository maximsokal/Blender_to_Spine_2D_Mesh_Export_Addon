"""Atomic depth-correct camera rendering for one connected group of B4 objects."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Tuple

from ..domain.baking import (
    BakeExecutionSettings,
    CameraProjectionLayout,
    CameraProjectionLayoutError,
    GroupedCameraProjectionPlan,
    ProjectionAlphaUnionAccumulator,
    resolve_projection_output_policy,
)
from ..infrastructure import AtomicFileTransaction, AtomicOutputReservation
from .camera_projection_image import (
    read_staged_alpha_coverage,
    rewrite_staged_image_with_crop,
)
from .camera_projection_state import (
    CameraProjectionExecutionError,
    call_public_render_operator,
    configure_scene_for_camera_projection,
    preserve_camera_projection_state,
    set_timeline_frame,
    validate_projection_runtime,
)

logger = logging.getLogger(__name__)
_RENDERABLE_TYPES = frozenset({"MESH", "CURVE", "SURFACE", "META", "FONT", "VOLUME"})


@dataclass(frozen=True, slots=True)
class GroupedCameraProjectionStageResult:
    reservations: Tuple[AtomicOutputReservation, ...]
    layout: CameraProjectionLayout
    source_object_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.reservations, tuple) or not self.reservations:
            raise ValueError("reservations must be a non-empty tuple")
        if not all(isinstance(item, AtomicOutputReservation) for item in self.reservations):
            raise TypeError("reservations must contain AtomicOutputReservation values")
        if not isinstance(self.layout, CameraProjectionLayout):
            raise TypeError("layout must be CameraProjectionLayout")
        if len(self.reservations) != self.layout.frame_count:
            raise ValueError("reservation count must match layout.frame_count")
        if (
            not isinstance(self.source_object_ids, tuple)
            or len(self.source_object_ids) < 2
            or not all(
                isinstance(value, str) and value.strip()
                for value in self.source_object_ids
            )
        ):
            raise ValueError("source_object_ids must contain at least two names")


def _object_name(obj: Any) -> str:
    value = str(
        getattr(obj, "name_full", None)
        or getattr(obj, "name", None)
        or ""
    ).strip()
    if not value:
        raise CameraProjectionExecutionError("grouped B4 source has an empty name")
    return value


def _rna_identity(value: Any) -> tuple[str, object]:
    pointer = getattr(value, "as_pointer", None)
    if callable(pointer):
        try:
            resolved = int(pointer())
            if resolved:
                return ("RNA_POINTER", resolved)
        except Exception:
            logger.debug("Unable to read Blender RNA pointer", exc_info=True)
    name = str(
        getattr(value, "name_full", None)
        or getattr(value, "name", None)
        or ""
    ).strip()
    if name:
        return ("RNA_NAME", name)
    return ("PYTHON_ID", id(value))


def _validate_group_runtime(
    source_objects: Tuple[Any, ...],
    plan: GroupedCameraProjectionPlan,
    *,
    context: Any | None,
    scene: Any | None,
) -> tuple[Any, Any, Any]:
    if not isinstance(plan, GroupedCameraProjectionPlan):
        raise TypeError("plan must be GroupedCameraProjectionPlan")
    if (
        not isinstance(source_objects, tuple)
        or len(source_objects) != len(plan.source_plans)
        or len(source_objects) < 2
    ):
        raise ValueError("source_objects must match grouped source plans")
    source_identities = tuple(_rna_identity(obj) for obj in source_objects)
    if len(source_identities) != len(set(source_identities)):
        raise CameraProjectionExecutionError(
            "grouped B4 source_objects contain duplicate Blender objects"
        )

    resolved_bpy = None
    resolved_context = context
    resolved_scene = scene
    expected_scene_identity = None
    names: list[str] = []
    for source_obj, source_plan in zip(source_objects, plan.source_plans):
        bpy_module, current_context, current_scene = validate_projection_runtime(
            source_obj,
            source_plan,
            context=resolved_context,
            scene=resolved_scene,
        )
        current_scene_identity = _rna_identity(current_scene)
        if resolved_bpy is None:
            resolved_bpy = bpy_module
            resolved_context = current_context
            resolved_scene = current_scene
            expected_scene_identity = current_scene_identity
        elif current_scene_identity != expected_scene_identity:
            raise CameraProjectionExecutionError(
                "grouped B4 sources resolved to different Blender scenes"
            )
        names.append(_object_name(source_obj))

    if tuple(names) != plan.source_object_ids:
        raise CameraProjectionExecutionError(
            "grouped B4 source object order differs from the immutable plan; "
            f"expected={plan.source_object_ids}, actual={tuple(names)}"
        )
    assert resolved_bpy is not None
    assert resolved_context is not None
    assert resolved_scene is not None
    return resolved_bpy, resolved_context, resolved_scene


def _configure_group_camera_visibility(
    source_objects: Tuple[Any, ...],
    scene: Any,
) -> None:
    try:
        scene_objects = tuple(scene.objects)
    except Exception as exc:
        raise CameraProjectionExecutionError(
            "Unable to inspect scene objects for grouped B4 visibility"
        ) from exc

    source_identities = {_rna_identity(obj) for obj in source_objects}
    scene_identities = {_rna_identity(obj) for obj in scene_objects}
    missing = tuple(
        obj
        for obj in source_objects
        if _rna_identity(obj) not in scene_identities
    )
    if missing:
        raise CameraProjectionExecutionError(
            "grouped B4 source objects are not linked to the render scene: "
            + str(tuple(_object_name(obj) for obj in missing))
        )

    for obj in scene_objects:
        if _rna_identity(obj) in source_identities:
            try:
                obj.hide_render = False
                if hasattr(obj, "visible_camera"):
                    obj.visible_camera = True
            except Exception as exc:
                raise CameraProjectionExecutionError(
                    f"Unable to make grouped source '{_object_name(obj)}' camera-visible"
                ) from exc
            continue
        if (
            str(getattr(obj, "type", "") or "") in _RENDERABLE_TYPES
            and hasattr(obj, "visible_camera")
        ):
            try:
                obj.visible_camera = False
            except Exception as exc:
                raise CameraProjectionExecutionError(
                    f"Unable to isolate grouped B4 camera layer from '{_object_name(obj)}'"
                ) from exc


def _reserve_group_outputs(
    plan: GroupedCameraProjectionPlan,
    transaction: AtomicFileTransaction,
) -> Tuple[AtomicOutputReservation, ...]:
    if not isinstance(transaction, AtomicFileTransaction):
        raise TypeError("transaction must be AtomicFileTransaction")
    reservations = tuple(
        transaction.reserve(task.output_path) for task in plan.frame_tasks
    )
    if len(reservations) != len(plan.frame_tasks):
        raise CameraProjectionExecutionError(
            "grouped B4 reservation count does not match frame tasks"
        )
    return reservations


def stage_grouped_camera_projection_outputs(
    source_objects: Tuple[Any, ...],
    plan: GroupedCameraProjectionPlan,
    output_transaction: AtomicFileTransaction,
    execution_settings: BakeExecutionSettings,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> GroupedCameraProjectionStageResult:
    """Render all connected sources together and return one shared cropped layout."""

    if not isinstance(execution_settings, BakeExecutionSettings):
        raise TypeError("execution_settings must be BakeExecutionSettings")
    bpy_module, resolved_context, resolved_scene = _validate_group_runtime(
        source_objects,
        plan,
        context=context,
        scene=scene,
    )
    reservations = _reserve_group_outputs(plan, output_transaction)
    output_policy = resolve_projection_output_policy(
        execution_settings.projection_output_policy,
        plan.settings.texture_format,
    )
    accumulator = ProjectionAlphaUnionAccumulator(
        width=plan.settings.width,
        height=plan.settings.height,
        alpha_threshold=float(execution_settings.projection_alpha_threshold),
        padding_pixels=plan.settings.margin_pixels,
        contour_mode=execution_settings.projection_contour_mode,
        simplify_tolerance_pixels=float(
            execution_settings.projection_contour_simplify_tolerance_pixels
        ),
        coverage_policy=execution_settings.projection_coverage_policy,
    )

    try:
        with preserve_camera_projection_state(resolved_scene):
            _configure_group_camera_visibility(source_objects, resolved_scene)
            for task, reservation in zip(plan.frame_tasks, reservations):
                set_timeline_frame(
                    resolved_scene,
                    resolved_context,
                    task.timeline_frame,
                )
                configure_scene_for_camera_projection(
                    resolved_scene,
                    plan.representative_plan,
                    execution_settings,
                    reservation.staged_path,
                )
                logger.info(
                    "Rendering grouped B4 '%s' frame %d/%d camera='%s' sources=%s "
                    "dynamic_range=%s tone_mapping=%s alpha=%s",
                    plan.group_id,
                    task.task_index + 1,
                    len(plan.frame_tasks),
                    plan.camera_object_id,
                    plan.source_object_ids,
                    output_policy.dynamic_range.value,
                    output_policy.tone_mapping.value,
                    output_policy.alpha_representation.value,
                )
                call_public_render_operator(bpy_module)
                if (
                    not reservation.staged_path.is_file()
                    or reservation.staged_path.stat().st_size <= 0
                ):
                    raise CameraProjectionExecutionError(
                        "Grouped B4 staged output is missing or empty: "
                        f"{reservation.staged_path}"
                    )
                coverage = read_staged_alpha_coverage(
                    bpy_module,
                    reservation.staged_path,
                    width=plan.settings.width,
                    height=plan.settings.height,
                )
                accumulator.add_coverage(
                    coverage,
                    frame_index=task.task_index,
                )
                del coverage

            try:
                layout = accumulator.build_layout()
            except CameraProjectionLayoutError as exc:
                raise CameraProjectionExecutionError(str(exc)) from exc
            for reservation in reservations:
                rewrite_staged_image_with_crop(
                    bpy_module,
                    plan,
                    reservation,
                    layout,
                    output_policy,
                )
    except CameraProjectionExecutionError:
        raise
    except Exception as exc:
        logger.exception("Unexpected grouped B4 failure for '%s'", plan.group_id)
        raise CameraProjectionExecutionError(
            f"Grouped camera projection failed for '{plan.group_id}': {exc}"
        ) from exc

    logger.info(
        "Grouped B4 layout '%s': sources=%d crop=%dx%d contour=%s vertices=%d "
        "components=%d coverage=%s final_visible=%d dynamic_range=%s "
        "tone_mapping=%s alpha=%s",
        plan.group_id,
        len(plan.source_object_ids),
        layout.cropped_width,
        layout.cropped_height,
        layout.contour_mode.value,
        len(layout.hull),
        layout.outer_component_count,
        layout.coverage_mode.value,
        layout.visible_pixel_count,
        output_policy.dynamic_range.value,
        output_policy.tone_mapping.value,
        output_policy.alpha_representation.value,
    )
    return GroupedCameraProjectionStageResult(
        reservations=reservations,
        layout=layout,
        source_object_ids=plan.source_object_ids,
    )


__all__ = [
    "GroupedCameraProjectionStageResult",
    "stage_grouped_camera_projection_outputs",
]
