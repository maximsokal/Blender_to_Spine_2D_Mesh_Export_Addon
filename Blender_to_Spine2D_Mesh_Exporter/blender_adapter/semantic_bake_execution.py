"""Execute already validated semantic bake requests into caller-owned reservations."""

from __future__ import annotations

import logging
from typing import Any, Tuple

from ..domain.baking import TextureFormat
from ..domain.baking.generated_materials import GeneratedBakePlan
from ..infrastructure import AtomicOutputReservation
from .bake_compositor import (
    BakePixelBuffer,
    compose_bake_passes,
    read_bake_image_pixels,
    write_bake_image_pixels,
)
from .bake_execution_error import BakeExecutionError
from .bake_materials import temporary_bake_materials
from .bake_scene_state import configure_scene_for_bake, preserve_bake_scene_state
from .context_state import activate_object_for_operator
from .mesh_writer import temporary_mesh_object
from .scene_bake_execution import temporarily_exclude_source_from_render
from .semantic_bake_image_io import (
    _activate_uv_layer,
    _create_bake_image,
    _remove_image,
    _save_bake_image,
    _set_timeline_frame,
)
from .semantic_bake_validation import (
    SemanticBakeRuntime,
    validate_semantic_bake_reservations,
)


logger = logging.getLogger(__name__)


def _call_public_bake_operator(bpy_module: Any, bake_type: str) -> None:
    """Use the stable facade so existing failure-injection tests remain valid."""

    from . import bake_executor as public_executor

    public_executor._call_bake_operator(bpy_module, bake_type)


def _bake_pass_to_buffer(
    *,
    runtime: SemanticBakeRuntime,
    task: Any,
    pass_plan: Any,
    prepared_materials: Any,
) -> BakePixelBuffer:
    image = None
    try:
        configure_scene_for_bake(
            runtime.scene,
            runtime.plan,
            runtime.execution_settings,
            bake_mode=pass_plan.bake_mode,
        )
        image = _create_bake_image(
            runtime.bpy_module,
            runtime.plan,
            runtime.execution_settings,
            f"{task.image_name}__pass_{pass_plan.pass_index}_{pass_plan.strategy_id.value}",
            force_float_buffer=True,
        )
        try:
            image.alpha_mode = "STRAIGHT"
        except Exception:
            logger.debug("Pass image alpha_mode is not writable", exc_info=True)

        with prepared_materials.prepare_pass(pass_plan):
            prepared_materials.assign_image(image)
            logger.info(
                "Baking semantic pass %d/%d for '%s': strategy=%s scope=%s "
                "mode=%s slots=%s",
                pass_plan.pass_index + 1,
                len(runtime.plan.passes),
                runtime.plan.source_object_id,
                pass_plan.strategy_id.value,
                pass_plan.evaluation_scope.value,
                pass_plan.bake_mode.value,
                pass_plan.material_slot_indices,
            )
            _call_public_bake_operator(
                runtime.bpy_module,
                pass_plan.bake_mode.value,
            )
            return read_bake_image_pixels(image)
    finally:
        _remove_image(runtime.bpy_module, image)


def _bake_single_frame(
    *,
    runtime: SemanticBakeRuntime,
    task: Any,
    reservation: AtomicOutputReservation,
    prepared_materials: Any,
) -> None:
    image = None
    pass_plan = runtime.plan.passes[0]
    try:
        configure_scene_for_bake(
            runtime.scene,
            runtime.plan,
            runtime.execution_settings,
            bake_mode=pass_plan.bake_mode,
        )
        image = _create_bake_image(
            runtime.bpy_module,
            runtime.plan,
            runtime.execution_settings,
            task.image_name,
        )
        with prepared_materials.prepare_pass(pass_plan):
            prepared_materials.assign_image(image)
            _call_public_bake_operator(
                runtime.bpy_module,
                pass_plan.bake_mode.value,
            )
            _save_bake_image(image, reservation, runtime.plan)
    finally:
        _remove_image(runtime.bpy_module, image)


def _bake_composed_frame(
    *,
    runtime: SemanticBakeRuntime,
    task: Any,
    reservation: AtomicOutputReservation,
    prepared_materials: Any,
) -> None:
    buffers = tuple(
        _bake_pass_to_buffer(
            runtime=runtime,
            task=task,
            pass_plan=pass_plan,
            prepared_materials=prepared_materials,
        )
        for pass_plan in runtime.plan.passes
    )
    composed = compose_bake_passes(buffers, runtime.plan.composite)

    final_image = None
    try:
        final_image = _create_bake_image(
            runtime.bpy_module,
            runtime.plan,
            runtime.execution_settings,
            task.image_name,
            force_float_buffer=(
                runtime.plan.settings.texture_format is TextureFormat.OPEN_EXR
            ),
        )
        try:
            final_image.alpha_mode = "STRAIGHT"
        except Exception:
            logger.debug("Final image alpha_mode is not writable", exc_info=True)
        write_bake_image_pixels(final_image, composed)
        _save_bake_image(final_image, reservation, runtime.plan)
    finally:
        _remove_image(runtime.bpy_module, final_image)


def _bake_frame_task(
    *,
    runtime: SemanticBakeRuntime,
    task: Any,
    reservation: AtomicOutputReservation,
    prepared_materials: Any,
) -> None:
    _set_timeline_frame(
        runtime.scene,
        runtime.context,
        task.timeline_frame,
    )
    if runtime.plan.requires_composition:
        _bake_composed_frame(
            runtime=runtime,
            task=task,
            reservation=reservation,
            prepared_materials=prepared_materials,
        )
        return
    _bake_single_frame(
        runtime=runtime,
        task=task,
        reservation=reservation,
        prepared_materials=prepared_materials,
    )


def run_semantic_bake(
    runtime: SemanticBakeRuntime,
    reservations: Tuple[AtomicOutputReservation, ...],
) -> None:
    """Write every planned frame to pre-reserved staged paths without committing."""

    if not isinstance(runtime, SemanticBakeRuntime):
        raise TypeError("runtime must be SemanticBakeRuntime")
    resolved_reservations = validate_semantic_bake_reservations(
        runtime.plan,
        reservations,
    )

    with preserve_bake_scene_state(runtime.scene):
        with temporary_mesh_object(
            runtime.target_snapshot,
            scene=runtime.scene,
            name_prefix="__Spine2D_BakeTarget",
        ) as temporary:
            _activate_uv_layer(
                temporary.mesh,
                runtime.plan.settings.uv_layer_name,
            )
            with temporarily_exclude_source_from_render(
                runtime.source_object,
                enabled=(
                    runtime.plan.scene_aware
                    and not runtime.plan.settings.selected_to_active
                ),
                context=runtime.context,
            ):
                generated_material = (
                    runtime.plan.generated_material
                    if isinstance(runtime.plan, GeneratedBakePlan)
                    else None
                )
                with temporary_bake_materials(
                    runtime.source_object,
                    temporary.object,
                    used_material_indices=runtime.used_material_indices,
                    face_material_indices=runtime.face_material_indices,
                    render_target=runtime.renderer.shader_target,
                    generated_material=generated_material,
                ) as prepared_materials:
                    with activate_object_for_operator(
                        temporary.object,
                        context=runtime.context,
                    ):
                        if runtime.plan.settings.selected_to_active:
                            try:
                                runtime.source_object.select_set(True)
                                runtime.context.view_layer.objects.active = (
                                    temporary.object
                                )
                            except Exception as exc:
                                raise BakeExecutionError(
                                    "Unable to prepare selected-to-active bake selection"
                                ) from exc

                        for task, reservation in zip(
                            runtime.plan.frame_tasks,
                            resolved_reservations,
                            strict=True,
                        ):
                            logger.info(
                                "Staging semantic bake '%s' frame %d/%d "
                                "(timeline=%s passes=%d composite=%s scene_aware=%s)",
                                runtime.plan.source_object_id,
                                task.task_index + 1,
                                len(runtime.plan.frame_tasks),
                                task.timeline_frame,
                                len(runtime.plan.passes),
                                runtime.plan.composite.mode.value,
                                runtime.plan.scene_aware,
                            )
                            _bake_frame_task(
                                runtime=runtime,
                                task=task,
                                reservation=reservation,
                                prepared_materials=prepared_materials,
                            )


__all__ = ["BakeExecutionError", "run_semantic_bake"]
