"""Own semantic bake reservations, atomic commit, rollback, and typed results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Tuple

from ..domain.baking import (
    BakeArtifact,
    BakeExecutionResult,
    BakeExecutionSettings,
    BakePlan,
)
from ..domain.geometry import MeshSnapshot
from ..infrastructure import (
    AtomicFileTransaction,
    AtomicOutputReservation,
    atomic_file_transaction,
)
from . import bake_executor_core as core
from .bake_compositor import BakeCompositeError
from .bake_material_preparation import BakeMaterialPreparationError
from .bake_materials import BakeMaterialError
from .bake_scene_state import BakeSceneStateError
from .context_state import BlenderContextError
from .mesh_writer import MeshWriteError
from .scene_bake_analyzer import SceneBakeAnalysisError
from .scene_bake_execution import SceneBakeExecutionError
from .semantic_bake_execution import run_semantic_bake
from .semantic_bake_validation import (
    SemanticBakeRuntime,
    validate_semantic_bake_request,
)


logger = logging.getLogger(__name__)
BakeExecutionError = core.BakeExecutionError


def _plan_identifier(plan: object) -> str:
    value = getattr(plan, "source_object_id", None)
    return str(value) if value is not None else "<unvalidated-plan>"


def _reserve_outputs(
    runtime: SemanticBakeRuntime,
    transaction: AtomicFileTransaction,
) -> Tuple[AtomicOutputReservation, ...]:
    return tuple(
        transaction.reserve(task.output_path)
        for task in runtime.plan.frame_tasks
    )


def _stage_validated_request(
    runtime: SemanticBakeRuntime,
    transaction: AtomicFileTransaction,
) -> Tuple[AtomicOutputReservation, ...]:
    reservations = _reserve_outputs(runtime, transaction)
    run_semantic_bake(runtime, reservations)
    return reservations


def stage_bake_plan_outputs(
    source_obj: Any,
    target_snapshot: MeshSnapshot,
    plan: BakePlan,
    output_transaction: AtomicFileTransaction,
    execution_settings: BakeExecutionSettings | None = None,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> Tuple[AtomicOutputReservation, ...]:
    """Validate first, then bake into a caller-owned transaction without commit."""

    if not isinstance(output_transaction, AtomicFileTransaction):
        raise TypeError("output_transaction must be AtomicFileTransaction")
    runtime = validate_semantic_bake_request(
        source_obj,
        target_snapshot,
        plan,
        execution_settings,
        context=context,
        scene=scene,
    )
    try:
        return _stage_validated_request(runtime, output_transaction)
    except BakeExecutionError:
        raise
    except (
        BakeCompositeError,
        BakeMaterialError,
        BakeMaterialPreparationError,
        BakeSceneStateError,
        BlenderContextError,
        MeshWriteError,
        SceneBakeAnalysisError,
        SceneBakeExecutionError,
    ) as exc:
        raise BakeExecutionError(
            f"Texture bake transaction failed for '{runtime.plan.source_object_id}': {exc}"
        ) from exc
    except Exception as exc:
        logger.exception(
            "Unexpected semantic texture bake failure for '%s'",
            runtime.plan.source_object_id,
        )
        raise BakeExecutionError(
            f"Unexpected texture bake failure for '{runtime.plan.source_object_id}': {exc}"
        ) from exc


def build_bake_execution_result(
    plan: BakePlan,
    committed_paths: Iterable[Path],
) -> BakeExecutionResult:
    """Build a result only when commit order exactly matches frame-task order."""

    if not isinstance(plan, BakePlan):
        raise TypeError("plan must be BakePlan")
    resolved_paths = tuple(
        Path(path).expanduser().resolve(strict=False)
        for path in committed_paths
    )
    expected_paths = tuple(
        task.output_path.expanduser().resolve(strict=False)
        for task in plan.frame_tasks
    )
    if resolved_paths != expected_paths:
        raise BakeExecutionError(
            "Committed bake paths do not match frame task order; "
            f"expected={expected_paths}, got={resolved_paths}"
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
        for task, committed_path in zip(
            plan.frame_tasks,
            resolved_paths,
            strict=True,
        )
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
    """Validate before transaction creation, then stage and commit exactly once."""

    runtime = validate_semantic_bake_request(
        source_obj,
        target_snapshot,
        plan,
        execution_settings,
        context=context,
        scene=scene,
    )
    try:
        with atomic_file_transaction(
            operation_name="semantic-object-bake"
        ) as output_transaction:
            reservations = _stage_validated_request(
                runtime,
                output_transaction,
            )
            expected_commit_order = tuple(
                reservation.final_path for reservation in reservations
            )
            committed_paths = output_transaction.commit()
        if committed_paths != expected_commit_order:
            raise BakeExecutionError(
                "Atomic transaction changed semantic bake output order; "
                f"expected={expected_commit_order}, got={committed_paths}"
            )
        result = build_bake_execution_result(
            runtime.plan,
            committed_paths,
        )
        logger.info(
            "Committed %d semantic baked texture files for '%s'",
            len(result.artifacts),
            runtime.plan.source_object_id,
        )
        return result
    except BakeExecutionError:
        raise
    except Exception as exc:
        plan_id = _plan_identifier(runtime.plan)
        logger.exception(
            "Unable to commit semantic texture outputs for '%s'",
            plan_id,
        )
        raise BakeExecutionError(
            f"Unable to commit texture outputs for '{plan_id}': {exc}"
        ) from exc


__all__ = [
    "BakeExecutionError",
    "build_bake_execution_result",
    "execute_bake_plan",
    "stage_bake_plan_outputs",
]
