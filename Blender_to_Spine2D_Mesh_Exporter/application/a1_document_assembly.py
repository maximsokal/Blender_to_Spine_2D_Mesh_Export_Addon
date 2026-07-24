"""Assemble UV-ready A1 regions into one validated in-memory Spine document."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Mapping, Tuple

from ..domain.geometry import MeshSnapshot, MeshSnapshotValidator
from ..domain.spine import (
    LegacyAttachmentSequence,
    LegacyMeshDocumentBuildResult,
    LegacyRigBuildResult,
    apply_attachment_sequence_animations,
    apply_legacy_visual_options,
    build_legacy_mesh_document,
)
from ..domain.uv import UvRangePolicy, enforce_uv_range
from .a1_attachment_projection_service import (
    A1AttachmentProjectionResult,
    A1AttachmentProjectionSettings,
    project_triangulated_disk_attachment,
)
from .a1_z_groups import A1ZGroupAssignmentPlan


class A1DocumentAssemblyError(ValueError):
    """Raised when prepared regions cannot form one coherent A1 document."""


@dataclass(frozen=True, slots=True)
class A1DocumentAssemblySettings:
    prefix: str
    uv_layer_name: str
    image_path: str
    attachment_width: float
    attachment_height: float
    center_x: float
    center_y: float
    sequence: LegacyAttachmentSequence | None = None
    skin_name: str = "default"
    segment_index_base: int = 0
    include_control_icons: bool = False
    include_preview_animation: bool = False
    # Appended to preserve positional compatibility with existing callers.
    uv_range_policy: UvRangePolicy = UvRangePolicy.REQUIRE_UNIT_SQUARE
    uv_range_epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        for field_name in ("prefix", "uv_layer_name", "image_path", "skin_name"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        for field_name in (
            "attachment_width",
            "attachment_height",
            "center_x",
            "center_y",
        ):
            value = getattr(self, field_name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not isfinite(float(value))
            ):
                raise ValueError(f"{field_name} must be finite")
        if self.attachment_width <= 0.0 or self.attachment_height <= 0.0:
            raise ValueError("attachment dimensions must be positive")
        if self.sequence is not None and not isinstance(
            self.sequence, LegacyAttachmentSequence
        ):
            raise TypeError("sequence must be LegacyAttachmentSequence or None")
        if (
            not isinstance(self.segment_index_base, int)
            or isinstance(self.segment_index_base, bool)
            or self.segment_index_base < 0
        ):
            raise ValueError("segment_index_base must be a non-negative integer")
        for field_name in ("include_control_icons", "include_preview_animation"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")
        if type(self.uv_range_policy) is not UvRangePolicy:
            raise TypeError("uv_range_policy must be UvRangePolicy")
        if (
            isinstance(self.uv_range_epsilon, bool)
            or not isinstance(self.uv_range_epsilon, (int, float))
            or not isfinite(float(self.uv_range_epsilon))
        ):
            raise TypeError("uv_range_epsilon must be a finite number")
        if float(self.uv_range_epsilon) < 0.0:
            raise ValueError("uv_range_epsilon cannot be negative")


@dataclass(frozen=True, slots=True)
class A1DocumentAssemblyResult:
    settings: A1DocumentAssemblySettings
    rig: LegacyRigBuildResult
    z_groups: A1ZGroupAssignmentPlan
    projections: Tuple[A1AttachmentProjectionResult, ...]
    document_build: LegacyMeshDocumentBuildResult

    @property
    def document(self):
        return self.document_build.document


def assemble_a1_document(
    rig: LegacyRigBuildResult,
    z_groups: A1ZGroupAssignmentPlan,
    region_snapshots: Tuple[MeshSnapshot, ...],
    settings: A1DocumentAssemblySettings,
    *,
    skeleton_metadata: Mapping[str, object] | None = None,
) -> A1DocumentAssemblyResult:
    """Project ordered UV-ready regions and compose one final Spine document."""

    if not isinstance(rig, LegacyRigBuildResult):
        raise TypeError("rig must be LegacyRigBuildResult")
    if not isinstance(z_groups, A1ZGroupAssignmentPlan):
        raise TypeError("z_groups must be A1ZGroupAssignmentPlan")
    if not isinstance(region_snapshots, tuple) or not region_snapshots:
        raise ValueError("region_snapshots must be a non-empty tuple")
    if not all(isinstance(snapshot, MeshSnapshot) for snapshot in region_snapshots):
        raise TypeError("region_snapshots must contain MeshSnapshot values")
    if not isinstance(settings, A1DocumentAssemblySettings):
        raise TypeError("settings must be A1DocumentAssemblySettings")
    if settings.prefix.strip() != rig.request.prefix.strip():
        raise A1DocumentAssemblyError(
            f"Assembly prefix '{settings.prefix}' does not match rig prefix "
            f"'{rig.request.prefix}'"
        )
    if tuple(rig.request.z_groups) != tuple(z_groups.groups):
        raise A1DocumentAssemblyError(
            "Rig Z groups do not match the source-lineage Z assignment plan"
        )

    projections: list[A1AttachmentProjectionResult] = []
    for region_offset, snapshot in enumerate(region_snapshots):
        MeshSnapshotValidator().validate_or_raise(snapshot)
        enforce_uv_range(
            snapshot,
            settings.uv_layer_name,
            policy=settings.uv_range_policy,
            epsilon=settings.uv_range_epsilon,
        )
        segment_index = settings.segment_index_base + region_offset
        segment_name = rig.profile.segment_slot(settings.prefix, segment_index)
        projection = project_triangulated_disk_attachment(
            snapshot,
            rig,
            A1AttachmentProjectionSettings(
                slot_name=segment_name,
                attachment_name=segment_name,
                vertex_prefix=segment_name,
                image_path=settings.image_path,
                uv_layer_name=settings.uv_layer_name,
                attachment_width=settings.attachment_width,
                attachment_height=settings.attachment_height,
                center_x=settings.center_x,
                center_y=settings.center_y,
                z_bindings=z_groups.projection_bindings(snapshot),
                sequence=settings.sequence,
                skin_name=settings.skin_name,
            ),
        )
        projections.append(projection)

    resolved_projections = tuple(projections)
    try:
        document_build = build_legacy_mesh_document(
            rig,
            tuple(projection.request for projection in resolved_projections),
            skeleton_metadata=skeleton_metadata,
        )
        document = apply_legacy_visual_options(
            document_build.document,
            prefix=settings.prefix,
            include_control_icons=settings.include_control_icons,
            include_preview_animation=settings.include_preview_animation,
        )
        document = apply_attachment_sequence_animations(document)
        document_build = replace(document_build, document=document)
    except Exception as exc:
        raise A1DocumentAssemblyError(
            f"Unable to compose A1 document for '{settings.prefix}': {exc}"
        ) from exc
    return A1DocumentAssemblyResult(
        settings=settings,
        rig=rig,
        z_groups=z_groups,
        projections=resolved_projections,
        document_build=document_build,
    )
