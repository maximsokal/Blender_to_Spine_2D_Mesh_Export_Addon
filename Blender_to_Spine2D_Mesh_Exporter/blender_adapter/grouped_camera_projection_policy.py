"""Resolve whether one connected A1 export can use a grouped camera layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from ..application import A1MultiObjectExportSettings, ConnectedCameraRenderPolicy
from ..domain.baking import (
    BakeExecutionSettings,
    CameraProjectionPlan,
    GroupedCameraProjectionPlan,
    GroupedCameraProjectionPlanError,
    build_grouped_camera_projection_plan,
)
from .a1_object_preparation import PreparedA1Object


class GroupedCameraProjectionPolicyError(ValueError):
    """Raised when strict grouped rendering cannot satisfy its contract."""


@dataclass(frozen=True, slots=True)
class GroupedCameraProjectionRequest:
    plan: GroupedCameraProjectionPlan
    source_objects: Tuple[Any, ...]
    execution_settings: BakeExecutionSettings
    visual_slot_names: Tuple[str, ...]
    image_relative_directory: str
    slot_name: str
    attachment_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.plan, GroupedCameraProjectionPlan):
            raise TypeError("plan must be GroupedCameraProjectionPlan")
        if (
            not isinstance(self.source_objects, tuple)
            or len(self.source_objects) != len(self.plan.source_object_ids)
        ):
            raise ValueError("source_objects must match grouped plan sources")
        if not isinstance(self.execution_settings, BakeExecutionSettings):
            raise TypeError("execution_settings must be BakeExecutionSettings")
        if (
            not isinstance(self.visual_slot_names, tuple)
            or len(self.visual_slot_names) != len(self.plan.source_object_ids)
            or not all(
                isinstance(value, str) and value.strip()
                for value in self.visual_slot_names
            )
        ):
            raise ValueError("visual_slot_names must match grouped sources")
        if not isinstance(self.image_relative_directory, str):
            raise TypeError("image_relative_directory must be str")
        for field_name in ("slot_name", "attachment_name"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")


def _failure(
    settings: A1MultiObjectExportSettings,
    reason: str,
) -> None:
    if (
        settings.connected_camera_render_policy
        is ConnectedCameraRenderPolicy.GROUPED_CAMERA_REQUIRED
    ):
        raise GroupedCameraProjectionPolicyError(reason)


def resolve_grouped_camera_projection_request(
    prepared_objects: Tuple[PreparedA1Object, ...],
    settings: A1MultiObjectExportSettings,
) -> GroupedCameraProjectionRequest | None:
    """Return one grouped request or a deterministic automatic/individual fallback.

    Grouping is valid only when every connected object uses camera projection.
    Mixing local UV-baked layers with one grouped camera layer would leave
    unresolved per-pixel depth between coordinate spaces, so AUTO falls back and
    REQUIRED fails explicitly.
    """

    if not isinstance(settings, A1MultiObjectExportSettings):
        raise TypeError("settings must be A1MultiObjectExportSettings")
    if (
        settings.connected_camera_render_policy
        is ConnectedCameraRenderPolicy.INDIVIDUAL_LAYERS
    ):
        return None
    if (
        not isinstance(prepared_objects, tuple)
        or len(prepared_objects) < 2
        or not all(isinstance(item, PreparedA1Object) for item in prepared_objects)
    ):
        _failure(
            settings,
            "grouped camera rendering requires at least two prepared connected objects",
        )
        return None

    camera_objects = tuple(
        item
        for item in prepared_objects
        if isinstance(item.bake_plan, CameraProjectionPlan)
    )
    if len(camera_objects) != len(prepared_objects):
        _failure(
            settings,
            "grouped camera rendering requires every connected component to use "
            "CameraProjectionPlan; "
            f"camera={len(camera_objects)}, total={len(prepared_objects)}",
        )
        return None

    execution_settings = prepared_objects[0].settings.bake_execution
    incompatible_execution = tuple(
        item.object_id
        for item in prepared_objects[1:]
        if item.settings.bake_execution != execution_settings
    )
    if incompatible_execution:
        _failure(
            settings,
            "grouped camera sources use different BakeExecutionSettings: "
            f"{incompatible_execution}",
        )
        return None

    relative_directory = prepared_objects[0].output_paths.image_relative_directory
    incompatible_directories = tuple(
        item.object_id
        for item in prepared_objects[1:]
        if item.output_paths.image_relative_directory != relative_directory
    )
    if incompatible_directories:
        _failure(
            settings,
            "grouped camera sources use different image-relative directories: "
            f"{incompatible_directories}",
        )
        return None

    visual_slot_names: list[str] = []
    for item in prepared_objects:
        projections = item.document_assembly.projections
        if len(projections) != 1:
            _failure(
                settings,
                "grouped camera rendering requires exactly one camera projection "
                "attachment per source; "
                f"object={item.object_id!r}, projections={len(projections)}",
            )
            return None
        visual_slot_names.append(projections[0].request.slot_name)

    output_stem = f"{settings.resolved_output_stem}_grouped_camera"
    try:
        grouped_plan = build_grouped_camera_projection_plan(
            tuple(item.bake_plan for item in prepared_objects),
            group_id=settings.connected_group_prefix,
            output_stem=output_stem,
        )
    except GroupedCameraProjectionPlanError as exc:
        _failure(settings, str(exc))
        return None

    safe_prefix = settings.connected_group_prefix.strip()
    return GroupedCameraProjectionRequest(
        plan=grouped_plan,
        source_objects=tuple(item.source_object for item in prepared_objects),
        execution_settings=execution_settings,
        visual_slot_names=tuple(visual_slot_names),
        image_relative_directory=relative_directory,
        slot_name=f"{safe_prefix}_grouped_camera_slot",
        attachment_name=f"{safe_prefix}_grouped_camera_attachment",
    )


__all__ = [
    "GroupedCameraProjectionPolicyError",
    "GroupedCameraProjectionRequest",
    "resolve_grouped_camera_projection_request",
]
