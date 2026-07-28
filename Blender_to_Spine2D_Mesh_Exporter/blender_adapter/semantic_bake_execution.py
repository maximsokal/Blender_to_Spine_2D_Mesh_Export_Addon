"""Execute validated Blender 5.2 semantic bake requests into staged reservations."""

from __future__ import annotations

import logging
from typing import Any, Tuple

from ..application import A1ExportProgressCallback, emit_a1_frame_progress
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
from .material_uv_binding import bind_materials_implicit_uv_sampling
from .mesh_writer import temporary_mesh_object
from .scene_bake_execution import temporarily_exclude_source_from_render
from .scene_bake_runtime import validate_runtime_object_transform
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


def _call_bake_operator(
    bpy_module: Any,
    bake_type: str,
    *,
    uv_layer_name: str,
) -> None:
    """Invoke Blender 5.2 baking with an explicit destination UV layer.

    The bake destination must not be inferred from ``active_render`` because that
    role belongs to source material sampling. Passing ``uv_layer`` lets material
    nodes continue to read the original render UV while Blender writes the result
    into the generated Spine bake layout.
    """

    if bpy_module is None:
        raise BakeExecutionError("bpy_module cannot be None")
    if not isinstance(bake_type, str) or not bake_type.strip():
        raise ValueError("bake_type must be a non-empty string")
    if not isinstance(uv_layer_name, str) or not uv_layer_name.strip():
        raise ValueError("uv_layer_name must be a non-empty string")

    resolved_type = bake_type.strip().upper()
    resolved_uv_layer = uv_layer_name.strip()
    try:
        operator = bpy_module.ops.object.bake
    except Exception as exc:
        raise BakeExecutionError("bpy.ops.object.bake is unavailable") from exc
    poll = getattr(operator, "poll", None)
    if callable(poll):
        try:
            available = bool(poll())
        except Exception as exc:
            raise BakeExecutionError(
                "bpy.ops.object.bake.poll() failed"
            ) from exc
        if not available:
            raise BakeExecutionError("bpy.ops.object.bake.poll() returned False")
    try:
        result = operator(
            type=resolved_type,
            uv_layer=resolved_uv_layer,
        )
    except Exception as exc:
        raise BakeExecutionError(
            "bpy.ops.object.bake failed with "
            f"type={resolved_type!r}, uv_layer={resolved_uv_layer!r}"
        ) from exc
    try:
        finished = "FINISHED" in result
    except Exception as exc:
        raise BakeExecutionError(
            f"bpy.ops.object.bake returned an invalid result: {result!r}"
        ) from exc
    if not finished:
        raise BakeExecutionError(
            "bpy.ops.object.bake did not finish for "
            f"type={resolved_type!r}, uv_layer={resolved_uv_layer!r}: {result!r}"
        )


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
            _call_bake_operator(
                runtime.bpy_module,
                pass_plan.bake_mode.value,
                uv_layer_name=runtime.plan.settings.uv_layer_name,
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
            _call_bake_operator(
                runtime.bpy_module,
                pass_plan.bake_mode.value,
                uv_layer_name=runtime.plan.settings.uv_layer_name,
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
    validate_runtime_object_transform(
        runtime.source_object,
        runtime.plan.object_context,
        timeline_frame=task.timeline_frame,
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
    *,
    progress_callback: A1ExportProgressCallback | None = None,
) -> None:
    """Write every planned frame to reserved staged paths without committing."""

    if not isinstance(runtime, SemanticBakeRuntime):
        raise TypeError("runtime must be SemanticBakeRuntime")
    resolved_reservations = validate_semantic_bake_reservations(
        runtime.plan,
        reservations,
    )
    frame_count = len(runtime.plan.frame_tasks)

    with preserve_bake_scene_state(runtime.scene):
        with temporary_mesh_object(
            runtime.target_snapshot,
            scene=runtime.scene,
            name_prefix="__Spine2D_BakeTarget",
        ) as temporary:
            _activate_uv_layer(
                temporary.mesh,
                runtime.plan.settings.uv_layer_name,
                render_layer_name=runtime.target_snapshot.render_uv_layer,
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
                    target_snapshot=runtime.target_snapshot,
                    used_material_indices=runtime.used_material_indices,
                    face_material_indices=runtime.face_material_indices,
                    render_target=runtime.renderer.shader_target,
                    generated_material=generated_material,
                ) as prepared_materials:
                    source_uv_name = runtime.target_snapshot.render_uv_layer
                    if source_uv_name is None:
                        raise BakeExecutionError(
                            "Semantic bake target has no source render UV layer"
                        )
                    bind_materials_implicit_uv_sampling(
                        prepared_materials.materials,
                        source_uv_name,
                        used_material_indices=prepared_materials.used_material_indices,
                        excluded_nodes=prepared_materials.image_nodes,
                    )
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

                        for frame_index, (task, reservation) in enumerate(
                            zip(
                                runtime.plan.frame_tasks,
                                resolved_reservations,
                                strict=True,
                            ),
                            start=1,
                        ):
                            emit_a1_frame_progress(
                                progress_callback,
                                stage="BAKE_FRAME",
                                action="Baking",
                                frame_index=frame_index,
                                frame_count=frame_count,
                                completed=False,
                                object_id=runtime.plan.source_object_id,
                            )
                            logger.info(
                                "Staging semantic bake '%s' frame %d/%d "
                                "(timeline=%s passes=%d composite=%s scene_aware=%s)",
                                runtime.plan.source_object_id,
                                frame_index,
                                frame_count,
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
                            emit_a1_frame_progress(
                                progress_callback,
                                stage="BAKE_FRAME",
                                action="Baked",
                                frame_index=frame_index,
                                frame_count=frame_count,
                                completed=True,
                                object_id=runtime.plan.source_object_id,
                            )


__all__ = [
    "BakeExecutionError",
    "_call_bake_operator",
    "run_semantic_bake",
]
