"""Execute immutable BakePlans with complete Blender and filesystem cleanup."""

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
from ..domain.geometry import MeshSnapshot, MeshSnapshotValidator
from ..infrastructure import (
    AtomicFileTransaction,
    AtomicOutputReservation,
    atomic_file_transaction,
)
from .bake_compositor import (
    BakeCompositeError,
    BakePixelBuffer,
    compose_bake_passes,
    read_bake_image_pixels,
    write_bake_image_pixels,
)
from .bake_materials import BakeMaterialError, temporary_bake_materials
from .bake_scene_state import (
    BakeSceneStateError,
    configure_scene_for_bake,
    preserve_bake_scene_state,
)
from .context_state import BlenderContextError, activate_object_for_operator
from .mesh_writer import MeshWriteError, temporary_mesh_object

logger = logging.getLogger(__name__)


class BakeExecutionError(RuntimeError):
    """Raised when a planned Blender bake fails before atomic output commit."""


def _load_bpy() -> Any:
    try:
        import bpy
    except Exception as exc:
        raise BakeExecutionError("Blender bpy module is unavailable") from exc
    return bpy


def _validate_execution_input(
    source_obj: Any,
    target_snapshot: MeshSnapshot,
    plan: BakePlan,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate Blender/domain inputs and return used slots plus per-face bindings."""

    if source_obj is None or getattr(source_obj, "type", None) != "MESH":
        raise BakeExecutionError("source_obj must be a Blender MESH object")
    if not isinstance(target_snapshot, MeshSnapshot):
        raise TypeError("target_snapshot must be MeshSnapshot")
    if not isinstance(plan, BakePlan):
        raise TypeError("plan must be BakePlan")
    MeshSnapshotValidator().validate_or_raise(target_snapshot)
    if target_snapshot.source_object_id != plan.source_object_id:
        raise BakeExecutionError(
            "target_snapshot.source_object_id does not match BakePlan.source_object_id"
        )
    if plan.settings.uv_layer_name not in target_snapshot.uv_layer_names:
        raise BakeExecutionError(
            f"Target snapshot is missing bake UV layer '{plan.settings.uv_layer_name}'"
        )

    source_slots = tuple(getattr(source_obj, "material_slots", ()))
    if len(source_slots) != len(plan.material_analysis.slots):
        raise BakeExecutionError(
            f"Source object has {len(source_slots)} material slots but BakePlan was "
            f"built from {len(plan.material_analysis.slots)} slots"
        )
    face_material_indices = tuple(
        int(face.material_index) for face in target_snapshot.faces
    )
    used_material_indices = tuple(sorted(set(face_material_indices)))
    if not used_material_indices:
        raise BakeExecutionError("Target snapshot contains no material references")
    if max(used_material_indices) >= len(source_slots):
        raise BakeExecutionError(
            f"Target snapshot references material slot {max(used_material_indices)}, "
            f"but source object has only {len(source_slots)} slots"
        )

    planned_slots = {
        slot_index
        for pass_plan in plan.passes
        for slot_index in pass_plan.material_slot_indices
    }
    missing_plans = tuple(
        index for index in used_material_indices if index not in planned_slots
    )
    if missing_plans:
        raise BakeExecutionError(
            "BakePlan does not cover used material slots: " + str(missing_plans)
        )
    return used_material_indices, face_material_indices


def _activate_uv_layer(mesh: Any, layer_name: str) -> None:
    layers = getattr(mesh, "uv_layers", None)
    layer = layers.get(layer_name) if layers is not None else None
    if layer is None:
        raise BakeExecutionError(
            f"Temporary target mesh is missing UV layer '{layer_name}'"
        )
    try:
        layers.active = layer
    except Exception:
        try:
            layers.active_index = next(
                index for index, candidate in enumerate(layers) if candidate == layer
            )
        except Exception as exc:
            raise BakeExecutionError(
                f"Unable to activate bake UV layer '{layer_name}'"
            ) from exc
    try:
        layer.active_render = True
    except Exception:
        logger.debug("UV active_render flag is not writable", exc_info=True)


def _create_bake_image(
    bpy_module: Any,
    plan: BakePlan,
    execution_settings: BakeExecutionSettings,
    image_name: str,
    *,
    force_float_buffer: bool = False,
) -> Any:
    float_buffer = force_float_buffer or (
        plan.settings.texture_format is TextureFormat.OPEN_EXR
    )
    try:
        image = bpy_module.data.images.new(
            name=f"__Spine2D_{image_name}",
            width=plan.settings.width,
            height=plan.settings.height,
            alpha=execution_settings.color_mode == "RGBA",
            float_buffer=float_buffer,
        )
        image.generated_color = execution_settings.generated_color
        image.file_format = plan.settings.texture_format.value
        return image
    except Exception as exc:
        raise BakeExecutionError(
            f"Unable to create bake image '{image_name}'"
        ) from exc


def _remove_image(bpy_module: Any, image: Any | None) -> None:
    if image is None:
        return
    try:
        bpy_module.data.images.remove(image)
    except Exception:
        logger.exception("Failed to remove temporary bake image")


def _set_timeline_frame(scene: Any, context: Any, frame: int | None) -> None:
    if frame is None:
        return
    try:
        scene.frame_set(frame)
        update = getattr(context.view_layer, "update", None)
        if callable(update):
            update()
    except Exception as exc:
        raise BakeExecutionError(f"Unable to set timeline frame {frame}") from exc


def _call_bake_operator(bpy_module: Any, bake_type: str) -> None:
    operator = bpy_module.ops.object.bake
    poll = getattr(operator, "poll", None)
    if callable(poll) and not poll():
        raise BakeExecutionError("bpy.ops.object.bake.poll() returned False")
    try:
        result = operator(type=bake_type)
    except Exception as exc:
        raise BakeExecutionError(
            f"bpy.ops.object.bake(type='{bake_type}') failed"
        ) from exc
    try:
        finished = "FINISHED" in result
    except Exception as exc:
        raise BakeExecutionError(
            f"bpy.ops.object.bake returned an invalid result: {result!r}"
        ) from exc
    if not finished:
        raise BakeExecutionError(
            f"bpy.ops.object.bake did not finish: {result!r}"
        )


def _save_bake_image(
    image: Any,
    reservation: AtomicOutputReservation,
    plan: BakePlan,
) -> None:
    staged_path = reservation.staged_path
    try:
        image.filepath_raw = str(staged_path)
        image.file_format = plan.settings.texture_format.value
        image.save()
    except Exception as exc:
        raise BakeExecutionError(
            f"Unable to save staged bake image '{staged_path}'"
        ) from exc
    if not staged_path.is_file():
        raise BakeExecutionError(
            f"Blender reported a successful save but staged file is missing: {staged_path}"
        )


def _bake_one_pass_to_buffer(
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
        image = _create_bake_image(
            bpy_module,
            plan,
            execution_settings,
            f"{task.image_name}__pass_{pass_plan.pass_index}_{pass_plan.strategy_id.value}",
            force_float_buffer=True,
        )
        prepared_materials.assign_image(image)
        logger.info(
            "Baking pass %d/%d for '%s': strategy=%s mode=%s slots=%s",
            pass_plan.pass_index + 1,
            len(plan.passes),
            plan.source_object_id,
            pass_plan.strategy_id.value,
            pass_plan.bake_mode.value,
            pass_plan.material_slot_indices,
        )
        _call_bake_operator(bpy_module, pass_plan.bake_mode.value)
        return read_bake_image_pixels(image)
    finally:
        _remove_image(bpy_module, image)


def _bake_single_pass_frame(
    *,
    bpy_module: Any,
    scene: Any,
    plan: BakePlan,
    execution_settings: BakeExecutionSettings,
    task: Any,
    reservation: AtomicOutputReservation,
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
        image = _create_bake_image(
            bpy_module,
            plan,
            execution_settings,
            task.image_name,
        )
        prepared_materials.assign_image(image)
        _call_bake_operator(bpy_module, pass_plan.bake_mode.value)
        _save_bake_image(image, reservation, plan)
    finally:
        _remove_image(bpy_module, image)


def _bake_multi_pass_frame(
    *,
    bpy_module: Any,
    scene: Any,
    plan: BakePlan,
    execution_settings: BakeExecutionSettings,
    task: Any,
    reservation: AtomicOutputReservation,
    prepared_materials: Any,
) -> None:
    buffers = tuple(
        _bake_one_pass_to_buffer(
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
        final_image = _create_bake_image(
            bpy_module,
            plan,
            execution_settings,
            task.image_name,
            force_float_buffer=plan.settings.texture_format is TextureFormat.OPEN_EXR,
        )
        write_bake_image_pixels(final_image, composed)
        _save_bake_image(final_image, reservation, plan)
    finally:
        _remove_image(bpy_module, final_image)


def _bake_frame_task(
    *,
    bpy_module: Any,
    context: Any,
    scene: Any,
    plan: BakePlan,
    execution_settings: BakeExecutionSettings,
    task: Any,
    reservation: AtomicOutputReservation,
    prepared_materials: Any,
) -> None:
    _set_timeline_frame(scene, context, task.timeline_frame)
    if plan.multipass:
        _bake_multi_pass_frame(
            bpy_module=bpy_module,
            scene=scene,
            plan=plan,
            execution_settings=execution_settings,
            task=task,
            reservation=reservation,
            prepared_materials=prepared_materials,
        )
        return
    _bake_single_pass_frame(
        bpy_module=bpy_module,
        scene=scene,
        plan=plan,
        execution_settings=execution_settings,
        task=task,
        reservation=reservation,
        prepared_materials=prepared_materials,
    )


def _require_reservations(
    plan: BakePlan,
    reservations: Iterable[AtomicOutputReservation],
) -> tuple[AtomicOutputReservation, ...]:
    resolved = tuple(reservations)
    if len(resolved) != len(plan.frame_tasks):
        raise BakeExecutionError(
            f"Expected {len(plan.frame_tasks)} bake output reservations, "
            f"received {len(resolved)}"
        )
    for task, reservation in zip(plan.frame_tasks, resolved):
        if not isinstance(reservation, AtomicOutputReservation):
            raise TypeError(
                "reservations must contain AtomicOutputReservation values"
            )
        expected_path = task.output_path.expanduser().resolve(strict=False)
        if reservation.final_path != expected_path:
            raise BakeExecutionError(
                f"Bake task {task.task_index} expected output '{expected_path}', "
                f"reservation targets '{reservation.final_path}'"
            )
    return resolved


def _run_bake_to_reservations(
    source_obj: Any,
    target_snapshot: MeshSnapshot,
    plan: BakePlan,
    execution_settings: BakeExecutionSettings,
    reservations: tuple[AtomicOutputReservation, ...],
    *,
    context: Any | None,
    scene: Any | None,
) -> None:
    """Render every frame into pre-reserved staged files without committing them."""

    used_material_indices, face_material_indices = _validate_execution_input(
        source_obj,
        target_snapshot,
        plan,
    )
    resolved_reservations = _require_reservations(plan, reservations)
    bpy_module = _load_bpy()
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
            _activate_uv_layer(
                temporary.mesh,
                plan.settings.uv_layer_name,
            )
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
                            "Staging bake '%s' frame task %d/%d (timeline=%s passes=%d)",
                            plan.source_object_id,
                            task.task_index + 1,
                            len(plan.frame_tasks),
                            task.timeline_frame,
                            len(plan.passes),
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
) -> tuple[AtomicOutputReservation, ...]:
    """Bake into a caller-owned transaction without committing outputs."""

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
        BakeSceneStateError,
        BlenderContextError,
        MeshWriteError,
    ) as exc:
        raise BakeExecutionError(
            f"Texture bake transaction failed for '{plan.source_object_id}': {exc}"
        ) from exc
    except Exception as exc:
        logger.exception(
            "Unexpected texture bake failure for '%s'",
            plan.source_object_id,
        )
        raise BakeExecutionError(
            f"Unexpected texture bake failure for '{plan.source_object_id}': {exc}"
        ) from exc


def build_bake_execution_result(
    plan: BakePlan,
    committed_paths: Iterable[Path],
) -> BakeExecutionResult:
    """Create the typed result after a transaction committed bake files."""

    if not isinstance(plan, BakePlan):
        raise TypeError("plan must be BakePlan")
    resolved_paths = tuple(
        path.expanduser().resolve(strict=False) for path in committed_paths
    )
    if len(resolved_paths) != len(plan.frame_tasks):
        raise BakeExecutionError(
            f"Expected {len(plan.frame_tasks)} committed bake paths, "
            f"received {len(resolved_paths)}"
        )
    artifacts = tuple(
        BakeArtifact(
            task_index=task.task_index,
            timeline_frame=task.timeline_frame,
            image_name=task.image_name,
            output_path=committed_path,
            width=plan.settings.width,
            height=plan.settings.height,
        )
        for task, committed_path in zip(plan.frame_tasks, resolved_paths)
    )
    return BakeExecutionResult(plan=plan, artifacts=artifacts)


def execute_bake_plan(
    source_obj: Any,
    target_snapshot: MeshSnapshot,
    plan: BakePlan,
    execution_settings: BakeExecutionSettings | None = None,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> BakeExecutionResult:
    """Bake and atomically commit only the texture outputs in ``plan``."""

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
            "Committed %d baked texture files for '%s'",
            len(result.artifacts),
            plan.source_object_id,
        )
        return result
    except BakeExecutionError:
        raise
    except Exception as exc:
        logger.exception(
            "Unable to commit texture outputs for '%s'",
            plan.source_object_id,
        )
        raise BakeExecutionError(
            f"Unable to commit texture outputs for '{plan.source_object_id}': {exc}"
        ) from exc
