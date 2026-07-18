"""Validate semantic object-bake requests before filesystem or Blender mutation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from ..domain.baking import BakeExecutionSettings, BakePlan
from ..domain.geometry import MeshSnapshot
from . import bake_executor_core as core
from .render_engine_contract import (
    RenderEngineContract,
    render_engine_contract_from_execution,
)
from .scene_bake_analyzer import validate_runtime_scene_context


BakeExecutionError = core.BakeExecutionError


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

    used_material_indices, face_material_indices = core._validate_execution_input(
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

    bpy_module = core._load_bpy()
    resolved_context = context if context is not None else bpy_module.context
    resolved_scene = (
        scene
        if scene is not None
        else getattr(resolved_context, "scene", None)
    )
    if resolved_scene is None:
        raise BakeExecutionError("A Blender Scene is required for texture baking")

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
]
