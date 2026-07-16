"""Execute semantic bake strategies using the stable low-level bake primitives.

``bake_executor`` remains the owner of Blender context validation, temporary mesh/image
creation, scene state, atomic reservations, and the only ``bpy.ops.object.bake`` call.
This module owns pass-specific material preparation and final RGBA composition.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

from ..domain.baking import (
    BakeArtifact,
    BakeExecutionResult,
    BakeExecutionSettings,
    BakePlan,
    TextureFormat,
)
from ..domain.geometry import MeshSnapshot
from ..infrastructure import AtomicFileTransaction, atomic_file_transaction
from . import bake_executor as core
from .bake_compositor import (
    BakeCompositeError,
    BakePixelBuffer,
    compose_bake_passes,
    read_bake_image_pixels,
    write_bake_image_pixels,
)
from .bake_material_preparation import BakeMaterialPreparationError
from .bake_materials import BakeMaterialError, temporary_bake_materials
from .bake_scene_state import (
    BakeSceneStateError,
    configure_scene_for_bake,
    preserve_bake_scene_state,
)
from .context_state import BlenderContextError, activate_object_for_operator
from .mesh_writer import MeshWriteError, temporary_mesh_object

logger = logging.getLogger(__name__)

BakeExecutionError = core.BakeExecutionError


def _bake_pass_to_buffer(
    *,
    bpy_module: Any,
    scene: Any,
    plan: BakePlan,
    execution_settings: BakeExecutionSettings,
    task: Any,
    pass_plan: Any,
    prepared_materials: Any,
) -> BakePixelBuffer:
    image = None
    try:
        configure_scene_for_bake(
            scene,
            plan,
            execution_settings,
            bake_mode=pass_plan.bake_mode,
        )
        image = core._create_bake_image(
            bpy_module,
            plan,
            execution_settings,
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
                "Baking semantic pass %d/%d for '%s': strategy=%s mode=%s slots=%s",
                pass_plan.pass_index + 1,
                len(plan.passes),
                plan.source_object_id,
                pass_plan.strategy_id.value,
                pass_plan.bake_mode.value,
                pass_plan.material_slot_indices,
            )
            core._call_bake_operator(bpy_module, pass_plan.bake_mode.value)
            return read_bake_image_pixels(image)
    finally:
        core._remove_image(bpy_module, image)


def _bake_single_frame(
    *,
    bpy_module: Any,
    scene: Any,
    plan: BakePlan,
    execution_settings: BakeExecutionSettings,
    task: Any,
    reservation: Any,
    prepared_materials: Any,
) -> None:
    image = None
    pass_plan = plan.passes[0]
    try:
        configure_scene_for_bake(
            scene,
            plan,
            execution_settings,
            bake_mode=pass_plan.bake_mode,
        )
        image = core._create_bake_image(
            bpy_module,
            plan,
            execution_settings,
            task.image_name,
        )
        with prepared_materials.prepare_pass(pass_plan):
            prepared_materials.assign_image(image)
            core._call_bake_operator(bpy_module, pass_plan.bake_mode.value)
            core._save_bake_image(image, reservation, plan)
    finally:
        core._remove_image(bpy_module, image)


def _bake_composed_frame(
    *,
    bpy_module: Any,
    scene: Any,
    plan: BakePlan,
    execution_settings: BakeExecutionSettings,
    task: Any,
    reservation: Any,
    prepared_materials: Any,
) -> None:
    buffers = tuple(
        _bake_pass_to_buffer(
            bpy_module=bpy_module,
            scene=scene,
            plan=plan,
            execution_settings=execution_settings,
            task=task,
            pass_plan=pass_plan,
            prepared_materials=prepared_materials,
        )
        for pass_plan in plan.passes
    )
    composed = compose_bake_passes(buffers, plan.composite)

    final_image = None
    try:
        final_image = core._create_bake_image(
            bpy_module,
            plan,
            execution_settings,
            task.image_name,
            force_float_buffer=plan.settings.texture_format is TextureFormat.OPEN_EXR,
        )
        try:
            final_image.alpha_mode = "STRAIGHT"
        except Exception:
            logger.debug("Final image alpha_mode is not writable", exc_info=True)
        write_bake_image_pixels(final_image, composed)
        core._save_bake_image(final_image, reservation, plan)
    finally:
        core._remove_image(bpy_module, final_image)


def _bake_frame_task(
    *,
    bpy_module: Any,
    context: Any,
    scene: Any,
    plan: BakePlan,
    execution_settings: BakeExecutionSettings,
    task: Any,
    reservation: Any,
    prepared_materials: Any,
) -> None:
    core._set_timeline_frame(scene, context, task.timeline_frame)
    if plan.requires_composition:
        _bake_composed_frame(
            bpy_module=bpy_module,
            scene=scene,
            plan=plan,
            execution_settings=execution_settings,
            task=task,
            reservation=reservation,
            prepared_materials=prepared_materials,
        )
        return
    _bake_single_frame(
        bpy_module=bpy_module,
        scene=scene,
        plan=plan,
        execution_settings=execution_settings,
        task=task,
        reservation=reservation,
        prepared_materials=prepared_materials,
    )


def _run_bake_to_reservations(
    source_obj: Any,
    target_snapshot: MeshSnapshot,
    plan: BakePlan,
    execution_settings: BakeExecutionSettings,
    reservations: tuple[Any, ...],
    *,
    context: Any | None,
    scene: Any | None,
) -> None:
    used_material_indices, face_material_indices = core._validate_execution_input(
        source_obj,
        target_snapshot,
        plan,
    )
    resolved_reservations = core._require_reservations(plan, reservations)
    bpy_module = core._load_bpy()
    resolved_context = context or bpy_module.context
    resolved_scene = scene or getattr(resolved_context, "scene", None)
    if resolved_scene is None:
        raise BakeExecutionError("A Blender Scene is required for texture baking")

    with preserve_bake_scene_state(resolved_scene):
        with temporary_mesh_object(
            target_snapshot,
            scene=resolved_scene,
            name_prefix="__Spine2D_BakeTarget",
        ) as temporary:
            core._activate_uv_layer(temporary.mesh, plan.settings.uv_layer_name)
            with temporary_bake_materials(
                source_obj,
                temporary.object,
                used_material_indices=used_material_indices,
                face_material_indices=face_material_indices,
            ) as prepared_materials:
                with activate_object_for_operator(
                    temporary.object,
                    context=resolved_context,
                ):
                    if plan.settings.selected_to_active:
                        try:
                            source_obj.select_set(True)
                            resolved_context.view_layer.objects.active = temporary.object
                        except Exception as exc:
                            raise BakeExecutionError(
                                "Unable to prepare selected-to-active bake selection"
                            ) from exc

                    for task, reservation in zip(
                        plan.frame_tasks,
                        resolved_reservations,
                    ):
                        logger.info(
                            "Staging semantic bake '%s' frame %d/%d "
                            "(timeline=%s passes=%d composite=%s)",
                            plan.source_object_id,
                            task.task_index + 1,
                            len(plan.frame_tasks),
                            task.timeline_frame,
                            len(plan.passes),
                            plan.composite.mode.value,
                        )
                        _bake_frame_task(
                            bpy_module=bpy_module,
                            context=resolved_context,
                            scene=resolved_scene,
                            plan=plan,
                            execution_settings=execution_settings,
                            task=task,
                            reservation=reservation,
                            prepared_materials=prepared_materials,
                        )


def stage_bake_plan_outputs(
    source_obj: Any,
    target_snapshot: MeshSnapshot,
    plan: BakePlan,
    output_transaction: AtomicFileTransaction,
    execution_settings: BakeExecutionSettings | None = None,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> tuple[Any, ...]:
    """Bake semantic passes into a caller-owned transaction without committing."""

    if not isinstance(output_transaction, AtomicFileTransaction):
        raise TypeError("output_transaction must be AtomicFileTransaction")
    resolved_execution_settings = execution_settings or BakeExecutionSettings()
    if not isinstance(resolved_execution_settings, BakeExecutionSettings):
        raise TypeError("execution_settings must be BakeExecutionSettings")

    try:
        reservations = tuple(
            output_transaction.reserve(task.output_path)
            for task in plan.frame_tasks
        )
        _run_bake_to_reservations(
            source_obj,
            target_snapshot,
            plan,
            resolved_execution_settings,
            reservations,
            context=context,
            scene=scene,
        )
        return reservations
    except BakeExecutionError:
        raise
    except (
        BakeCompositeError,
        BakeMaterialError,
        BakeMaterialPreparationError,
        BakeSceneStateError,
        BlenderContextError,
        MeshWriteError,
    ) as exc:
        raise BakeExecutionError(
            f"Texture bake transaction failed for '{plan.source_object_id}': {exc}"
        ) from exc
    except Exception as exc:
        logger.exception(
            "Unexpected semantic texture bake failure for '%s'",
            plan.source_object_id,
        )
        raise BakeExecutionError(
            f"Unexpected texture bake failure for '{plan.source_object_id}': {exc}"
        ) from exc


def build_bake_execution_result(
    plan: BakePlan,
    committed_paths: Iterable[Path],
) -> BakeExecutionResult:
    return core.build_bake_execution_result(plan, committed_paths)


def execute_bake_plan(
    source_obj: Any,
    target_snapshot: MeshSnapshot,
    plan: BakePlan,
    execution_settings: BakeExecutionSettings | None = None,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> BakeExecutionResult:
    """Bake semantic passes and atomically commit only final texture outputs."""

    try:
        with atomic_file_transaction() as output_transaction:
            reservations = stage_bake_plan_outputs(
                source_obj,
                target_snapshot,
                plan,
                output_transaction,
                execution_settings,
                context=context,
                scene=scene,
            )
            committed_paths = output_transaction.commit()
        result = build_bake_execution_result(
            plan,
            tuple(
                path
                for reservation, path in zip(reservations, committed_paths)
                if path == reservation.final_path
            ),
        )
        logger.info(
            "Committed %d semantic baked texture files for '%s'",
            len(result.artifacts),
            plan.source_object_id,
        )
        return result
    except BakeExecutionError:
        raise
    except Exception as exc:
        logger.exception(
            "Unable to commit semantic texture outputs for '%s'",
            plan.source_object_id,
        )
        raise BakeExecutionError(
            f"Unable to commit texture outputs for '{plan.source_object_id}': {exc}"
        ) from exc


__all__ = [
    "BakeExecutionError",
    "build_bake_execution_result",
    "execute_bake_plan",
    "stage_bake_plan_outputs",
]
