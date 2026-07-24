"""Validate semantic object-bake requests before filesystem or Blender mutation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Tuple

from ..domain.baking import BakeExecutionSettings, BakePlan
from ..domain.baking.generated_materials import GeneratedBakePlan
from ..domain.geometry import MeshSnapshot, MeshSnapshotValidator
from ..infrastructure import AtomicOutputReservation
from .bake_execution_error import BakeExecutionError
from .render_engine_contract import (
    RenderEngineContract,
    render_engine_contract_from_execution,
)
from .scene_bake_runtime import validate_runtime_scene_context
from .scene_context_contract import (
    BlenderSceneContextError,
    require_context_scene_consistency,
)


def _load_bpy() -> Any:
    """Import Blender lazily so the package remains importable outside Blender."""

    try:
        import bpy
    except Exception as exc:
        raise BakeExecutionError("Blender bpy module is unavailable") from exc
    return bpy


def _planned_material_slots(plan: BakePlan) -> set[int]:
    return {
        slot_index
        for pass_plan in plan.passes
        for slot_index in pass_plan.material_slot_indices
    }


def _validate_execution_input(
    source_obj: Any,
    target_snapshot: MeshSnapshot,
    plan: BakePlan,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate Blender/domain inputs and return used slots plus face bindings."""

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

    if isinstance(plan, GeneratedBakePlan):
        if target_snapshot != plan.generated_material.target_snapshot:
            raise BakeExecutionError(
                "Generated bake target does not match GeneratedMaterialPlan.target_snapshot"
            )
        face_material_indices = tuple(
            int(face.material_index) for face in target_snapshot.faces
        )
        if not face_material_indices:
            raise BakeExecutionError(
                "Generated target snapshot contains no material references"
            )
        if set(face_material_indices) != {0}:
            raise BakeExecutionError(
                "Generated target snapshot must reference only synthetic slot zero"
            )
        if 0 not in _planned_material_slots(plan):
            raise BakeExecutionError(
                "Generated BakePlan does not cover synthetic material slot zero"
            )
        return (0,), face_material_indices

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

    planned_slots = _planned_material_slots(plan)
    missing_plans = tuple(
        index for index in used_material_indices if index not in planned_slots
    )
    if missing_plans:
        raise BakeExecutionError(
            "BakePlan does not cover used material slots: " + str(missing_plans)
        )
    return used_material_indices, face_material_indices


def validate_semantic_bake_reservations(
    plan: BakePlan,
    reservations: Iterable[AtomicOutputReservation],
) -> Tuple[AtomicOutputReservation, ...]:
    """Require one correctly ordered reservation for every planned frame task."""

    if not isinstance(plan, BakePlan):
        raise TypeError("plan must be BakePlan")

    resolved = tuple(reservations)
    if len(resolved) != len(plan.frame_tasks):
        raise BakeExecutionError(
            f"Expected {len(plan.frame_tasks)} bake output reservations, "
            f"received {len(resolved)}"
        )

    for task, reservation in zip(plan.frame_tasks, resolved, strict=True):
        if not isinstance(reservation, AtomicOutputReservation):
            raise TypeError(
                "reservations must contain AtomicOutputReservation values"
            )
        expected_path = task.output_path.expanduser().resolve(strict=False)
        actual_path = Path(reservation.final_path).expanduser().resolve(strict=False)
        if actual_path != expected_path:
            raise BakeExecutionError(
                f"Bake task {task.task_index} expected output '{expected_path}', "
                f"reservation targets '{actual_path}'"
            )
    return resolved


@dataclass(frozen=True, slots=True)
class SemanticBakeRuntime:
    """Fully validated immutable values required by semantic bake execution."""

    source_object: Any
    target_snapshot: MeshSnapshot
    plan: BakePlan
    execution_settings: BakeExecutionSettings
    used_material_indices: Tuple[int, ...]
    face_material_indices: Tuple[int, ...]
    bpy_module: Any
    context: Any
    scene: Any
    renderer: RenderEngineContract

    def __post_init__(self) -> None:
        if self.source_object is None:
            raise ValueError("source_object cannot be None")
        if not isinstance(self.target_snapshot, MeshSnapshot):
            raise TypeError("target_snapshot must be MeshSnapshot")
        if not isinstance(self.plan, BakePlan):
            raise TypeError("plan must be BakePlan")
        if not isinstance(self.execution_settings, BakeExecutionSettings):
            raise TypeError("execution_settings must be BakeExecutionSettings")
        if not isinstance(self.used_material_indices, tuple):
            raise TypeError("used_material_indices must be tuple")
        if not isinstance(self.face_material_indices, tuple):
            raise TypeError("face_material_indices must be tuple")
        if not isinstance(self.renderer, RenderEngineContract):
            raise TypeError("renderer must be RenderEngineContract")
        if self.context is None or self.scene is None or self.bpy_module is None:
            raise ValueError("bpy_module, context, and scene cannot be None")


def validate_semantic_bake_request(
    source_obj: Any,
    target_snapshot: MeshSnapshot,
    plan: BakePlan,
    execution_settings: BakeExecutionSettings | None = None,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> SemanticBakeRuntime:
    """Resolve and validate all runtime inputs before reserving outputs."""

    if source_obj is None:
        raise ValueError("source_obj cannot be None")
    if not isinstance(target_snapshot, MeshSnapshot):
        raise TypeError("target_snapshot must be MeshSnapshot")
    if not isinstance(plan, BakePlan):
        raise TypeError("plan must be BakePlan")

    resolved_settings = (
        BakeExecutionSettings()
        if execution_settings is None
        else execution_settings
    )
    if not isinstance(resolved_settings, BakeExecutionSettings):
        raise TypeError("execution_settings must be BakeExecutionSettings or None")
    if (
        isinstance(plan, GeneratedBakePlan)
        and resolved_settings.render_engine != "CYCLES"
    ):
        resolved_settings = replace(resolved_settings, render_engine="CYCLES")

    used_material_indices, face_material_indices = _validate_execution_input(
        source_obj,
        target_snapshot,
        plan,
    )
    renderer = render_engine_contract_from_execution(resolved_settings)
    if renderer.uses_eevee:
        raise BakeExecutionError(
            "Blender object baking is restricted to Cycles; Eevee materials must "
            "use camera-render projection"
        )

    bpy_module = _load_bpy()
    resolved_context = context if context is not None else bpy_module.context
    resolved_scene = (
        scene
        if scene is not None
        else getattr(resolved_context, "scene", None)
    )
    if resolved_scene is None:
        raise BakeExecutionError("A Blender Scene is required for texture baking")
    try:
        require_context_scene_consistency(resolved_context, resolved_scene)
    except BlenderSceneContextError as exc:
        raise BakeExecutionError(
            f"Texture bake context and scene disagree: {exc}"
        ) from exc

    if plan.scene_aware:
        validate_runtime_scene_context(
            source_obj,
            plan.object_context,
            plan.scene_context,
            scene=resolved_scene,
            context=resolved_context,
        )

    return SemanticBakeRuntime(
        source_object=source_obj,
        target_snapshot=target_snapshot,
        plan=plan,
        execution_settings=resolved_settings,
        used_material_indices=used_material_indices,
        face_material_indices=face_material_indices,
        bpy_module=bpy_module,
        context=resolved_context,
        scene=resolved_scene,
        renderer=renderer,
    )


__all__ = [
    "BakeExecutionError",
    "SemanticBakeRuntime",
    "validate_semantic_bake_request",
    "validate_semantic_bake_reservations",
]
