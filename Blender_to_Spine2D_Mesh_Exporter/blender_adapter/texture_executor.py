"""Dispatch immutable texture plans to Blender 5.2 execution owners.

Every entry point first builds one typed ``TextureExecutionRequest`` so invalid
domain values fail before filesystem reservations or Blender Scene mutation. Depth
parallax stages the front and every reserve view in one caller-owned transaction and
retains an independent crop layout and required UV envelope for each reserve view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from ..application import A1ExportProgressCallback
from ..domain.baking import (
    BakeExecutionResult,
    BakeExecutionSettings,
    BakePlan,
    CameraProjectionPlan,
    ProjectionUvBounds,
)
from ..domain.baking.projection_layout import CameraProjectionLayout
from ..domain.geometry import MeshSnapshot
from ..infrastructure import (
    AtomicFileTransaction,
    AtomicOutputReservation,
)
from .camera_projection_output import (
    execute_camera_projection_plan,
    stage_camera_projection_outputs_detailed,
)
from .semantic_bake_output import (
    execute_bake_plan as execute_object_bake_plan,
    stage_bake_plan_outputs as stage_object_bake_outputs,
)


_FRONT_VIEW_ID = "FRONT"


@dataclass(frozen=True, slots=True)
class TextureExecutionRequest:
    """One validated texture execution request shared by all dispatch routes."""

    source_object: Any
    target_snapshot: MeshSnapshot
    plan: BakePlan
    execution_settings: BakeExecutionSettings
    required_uv_bounds: ProjectionUvBounds | None = None

    def __post_init__(self) -> None:
        if self.source_object is None:
            raise ValueError("source_object cannot be None")
        if not isinstance(self.target_snapshot, MeshSnapshot):
            raise TypeError("target_snapshot must be MeshSnapshot")
        if not isinstance(self.plan, BakePlan):
            raise TypeError("plan must be BakePlan")
        if not isinstance(self.execution_settings, BakeExecutionSettings):
            raise TypeError("execution_settings must be BakeExecutionSettings")
        if self.target_snapshot.source_object_id != self.plan.source_object_id:
            raise ValueError(
                "target_snapshot.source_object_id must match plan.source_object_id"
            )
        if self.required_uv_bounds is not None and not isinstance(
            self.required_uv_bounds,
            ProjectionUvBounds,
        ):
            raise TypeError(
                "required_uv_bounds must be ProjectionUvBounds or None"
            )
        if self.required_uv_bounds is not None and not isinstance(
            self.plan,
            CameraProjectionPlan,
        ):
            raise ValueError(
                "required_uv_bounds require a CameraProjectionPlan"
            )

    @classmethod
    def capture(
        cls,
        source_object: Any,
        target_snapshot: MeshSnapshot,
        plan: BakePlan,
        execution_settings: BakeExecutionSettings | None = None,
        *,
        required_uv_bounds: ProjectionUvBounds | None = None,
    ) -> "TextureExecutionRequest":
        resolved_settings = (
            BakeExecutionSettings()
            if execution_settings is None
            else execution_settings
        )
        return cls(
            source_object=source_object,
            target_snapshot=target_snapshot,
            plan=plan,
            execution_settings=resolved_settings,
            required_uv_bounds=required_uv_bounds,
        )


@dataclass(frozen=True, slots=True)
class ProjectionViewStage:
    """One camera view's staged images and crop layout."""

    view_id: str
    plan: CameraProjectionPlan
    reservations: Tuple[AtomicOutputReservation, ...]
    layout: CameraProjectionLayout

    def __post_init__(self) -> None:
        if not isinstance(self.view_id, str) or not self.view_id.strip():
            raise ValueError("view_id must be a non-empty string")
        object.__setattr__(self, "view_id", self.view_id.strip().upper())
        if not isinstance(self.plan, CameraProjectionPlan):
            raise TypeError("plan must be CameraProjectionPlan")
        if self.plan.view_id != self.view_id:
            raise ValueError("plan.view_id must match view_id")
        if not isinstance(self.reservations, tuple) or not self.reservations:
            raise ValueError("reservations must be a non-empty tuple")
        if not all(
            isinstance(item, AtomicOutputReservation)
            for item in self.reservations
        ):
            raise TypeError(
                "reservations must contain AtomicOutputReservation values"
            )
        if not isinstance(self.layout, CameraProjectionLayout):
            raise TypeError("layout must be CameraProjectionLayout")
        if self.layout.frame_count != len(self.reservations):
            raise ValueError(
                "view layout frame count must match reservations"
            )


@dataclass(frozen=True, slots=True)
class TextureStageResult:
    reservations: Tuple[AtomicOutputReservation, ...]
    projection_layout: CameraProjectionLayout | None = None
    primary_reservations: Tuple[AtomicOutputReservation, ...] = ()
    reserve_projection_stages: Tuple[ProjectionViewStage, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.reservations, tuple) or not self.reservations:
            raise ValueError("reservations must be a non-empty tuple")
        if not all(
            isinstance(item, AtomicOutputReservation)
            for item in self.reservations
        ):
            raise TypeError(
                "reservations must contain AtomicOutputReservation values"
            )
        if self.projection_layout is not None and not isinstance(
            self.projection_layout,
            CameraProjectionLayout,
        ):
            raise TypeError(
                "projection_layout must be CameraProjectionLayout or None"
            )
        primary = self.primary_reservations or self.reservations
        if not isinstance(primary, tuple) or not primary:
            raise ValueError("primary_reservations must be a non-empty tuple")
        if not all(
            isinstance(item, AtomicOutputReservation) for item in primary
        ):
            raise TypeError(
                "primary_reservations must contain AtomicOutputReservation values"
            )
        object.__setattr__(self, "primary_reservations", primary)
        if (
            self.projection_layout is not None
            and self.projection_layout.frame_count != len(primary)
        ):
            raise ValueError(
                "primary projection layout frame count must match primary reservations"
            )
        if not isinstance(self.reserve_projection_stages, tuple) or not all(
            isinstance(item, ProjectionViewStage)
            for item in self.reserve_projection_stages
        ):
            raise TypeError(
                "reserve_projection_stages must contain ProjectionViewStage values"
            )
        view_ids = tuple(item.view_id for item in self.reserve_projection_stages)
        if len(view_ids) != len(set(view_ids)):
            raise ValueError("reserve_projection_stages contain duplicate view ids")
        expected = primary + tuple(
            reservation
            for stage in self.reserve_projection_stages
            for reservation in stage.reservations
        )
        if expected != self.reservations:
            raise ValueError(
                "reservations must equal primary followed by reserve view reservations"
            )

    @property
    def view_layouts(self) -> dict[str, CameraProjectionLayout]:
        layouts = {}
        if self.projection_layout is not None:
            layouts[_FRONT_VIEW_ID] = self.projection_layout
        layouts.update(
            {stage.view_id: stage.layout for stage in self.reserve_projection_stages}
        )
        return layouts


def _require_transaction(value: Any) -> AtomicFileTransaction:
    if not isinstance(value, AtomicFileTransaction):
        raise TypeError("output_transaction must be AtomicFileTransaction")
    return value


def _projection_uv_bounds(
    values: Mapping[str, ProjectionUvBounds] | None,
    *,
    expected_view_ids: tuple[str, ...],
) -> Mapping[str, ProjectionUvBounds]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError(
            "projection_uv_bounds_by_view must be a mapping or None"
        )
    resolved: dict[str, ProjectionUvBounds] = {}
    for raw_view_id, bounds in values.items():
        view_id = str(raw_view_id).strip().upper()
        if not view_id:
            raise ValueError(
                "projection_uv_bounds_by_view contains an empty view id"
            )
        if not isinstance(bounds, ProjectionUvBounds):
            raise TypeError(
                f"projection_uv_bounds_by_view[{view_id!r}] must be ProjectionUvBounds"
            )
        if view_id in resolved:
            raise ValueError(
                f"projection_uv_bounds_by_view contains duplicate view {view_id}"
            )
        resolved[view_id] = bounds

    expected = set(expected_view_ids)
    missing = tuple(sorted(expected - set(resolved)))
    unknown = tuple(sorted(set(resolved) - expected))
    if missing or unknown:
        raise ValueError(
            "projection UV view bounds do not match reserve camera plans; "
            f"missing={missing}, unknown={unknown}"
        )
    return resolved


def _stage_camera_plan(
    request: TextureExecutionRequest,
    transaction: AtomicFileTransaction,
    *,
    context: Any | None,
    scene: Any | None,
    progress_callback: A1ExportProgressCallback | None,
) -> ProjectionViewStage:
    if not isinstance(request.plan, CameraProjectionPlan):
        raise TypeError("request.plan must be CameraProjectionPlan")
    staged = stage_camera_projection_outputs_detailed(
        request.source_object,
        request.plan,
        transaction,
        request.execution_settings,
        required_uv_bounds=request.required_uv_bounds,
        context=context,
        scene=scene,
        progress_callback=progress_callback,
    )
    return ProjectionViewStage(
        view_id=request.plan.view_id,
        plan=request.plan,
        reservations=staged.reservations,
        layout=staged.layout,
    )


def stage_texture_plan_outputs(
    source_obj: Any,
    target_snapshot: MeshSnapshot,
    plan: BakePlan,
    output_transaction: AtomicFileTransaction,
    execution_settings: BakeExecutionSettings | None = None,
    *,
    reserve_plans: Tuple[CameraProjectionPlan, ...] = (),
    projection_uv_bounds_by_view: Mapping[str, ProjectionUvBounds] | None = None,
    context: Any | None = None,
    scene: Any | None = None,
    progress_callback: A1ExportProgressCallback | None = None,
) -> TextureStageResult:
    """Stage a primary texture plan and optional parallax reserve views."""

    transaction = _require_transaction(output_transaction)
    if not isinstance(reserve_plans, tuple) or not all(
        isinstance(item, CameraProjectionPlan) for item in reserve_plans
    ):
        raise TypeError("reserve_plans must contain CameraProjectionPlan values")

    expected_reserve_view_ids = tuple(
        reserve_plan.view_id.strip().upper()
        for reserve_plan in reserve_plans
    )
    bounds_by_view = _projection_uv_bounds(
        projection_uv_bounds_by_view,
        expected_view_ids=expected_reserve_view_ids,
    )
    request = TextureExecutionRequest.capture(
        source_obj,
        target_snapshot,
        plan,
        execution_settings,
    )

    if isinstance(request.plan, CameraProjectionPlan):
        if request.plan.virtual_view:
            raise ValueError("primary plan must own the FRONT view")
        primary_stage = _stage_camera_plan(
            request,
            transaction,
            context=context,
            scene=scene,
            progress_callback=progress_callback,
        )
        reserve_stages = []
        for reserve_plan in reserve_plans:
            if not reserve_plan.virtual_view:
                raise ValueError("reserve_plans cannot contain the FRONT plan")
            if reserve_plan.source_object_id != request.plan.source_object_id:
                raise ValueError(
                    "reserve plan source_object_id must match the primary plan"
                )
            view_id = reserve_plan.view_id.strip().upper()
            reserve_request = TextureExecutionRequest.capture(
                source_obj,
                target_snapshot,
                reserve_plan,
                request.execution_settings,
                required_uv_bounds=bounds_by_view.get(view_id),
            )
            reserve_stages.append(
                _stage_camera_plan(
                    reserve_request,
                    transaction,
                    context=context,
                    scene=scene,
                    progress_callback=progress_callback,
                )
            )
        all_reservations = primary_stage.reservations + tuple(
            reservation
            for stage in reserve_stages
            for reservation in stage.reservations
        )
        return TextureStageResult(
            reservations=all_reservations,
            projection_layout=primary_stage.layout,
            primary_reservations=primary_stage.reservations,
            reserve_projection_stages=tuple(reserve_stages),
        )

    if reserve_plans:
        raise ValueError(
            "Parallax reserve plans require a CameraProjectionPlan primary route"
        )
    if bounds_by_view:
        raise ValueError(
            "projection UV view bounds require a CameraProjectionPlan route"
        )
    reservations = stage_object_bake_outputs(
        request.source_object,
        request.target_snapshot,
        request.plan,
        transaction,
        request.execution_settings,
        context=context,
        scene=scene,
        progress_callback=progress_callback,
    )
    return TextureStageResult(tuple(reservations))


def execute_bake_plan(
    source_obj: Any,
    target_snapshot: MeshSnapshot,
    plan: BakePlan,
    execution_settings: BakeExecutionSettings | None = None,
    *,
    required_uv_bounds: ProjectionUvBounds | None = None,
    context: Any | None = None,
    scene: Any | None = None,
    progress_callback: A1ExportProgressCallback | None = None,
) -> BakeExecutionResult:
    """Execute one validated texture plan and atomically commit its outputs."""

    request = TextureExecutionRequest.capture(
        source_obj,
        target_snapshot,
        plan,
        execution_settings,
        required_uv_bounds=required_uv_bounds,
    )
    if isinstance(request.plan, CameraProjectionPlan):
        return execute_camera_projection_plan(
            request.source_object,
            request.plan,
            request.execution_settings,
            required_uv_bounds=request.required_uv_bounds,
            context=context,
            scene=scene,
            progress_callback=progress_callback,
        )
    return execute_object_bake_plan(
        request.source_object,
        request.target_snapshot,
        request.plan,
        request.execution_settings,
        context=context,
        scene=scene,
        progress_callback=progress_callback,
    )


__all__ = [
    "ProjectionViewStage",
    "TextureExecutionRequest",
    "TextureStageResult",
    "execute_bake_plan",
    "stage_texture_plan_outputs",
]
