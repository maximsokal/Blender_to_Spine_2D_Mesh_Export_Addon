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
    build_legacy_mesh_document,
)
from ..domain.spine.preprojected_setup import ensure_preprojected_screen_rig
from ..domain.spine.rig_profiles import (
    A1RigProfile,
    A1RigSetupPoseMode,
    resolve_a1_rig_profile,
)
from ..domain.spine.rig_visuals import apply_rig_visual_options
from ..domain.spine.vertex_bone_optimizer import optimize_shared_vertex_bones
from ..domain.uv import UvRangePolicy, enforce_uv_range
from .a1_attachment_projection_service import (
    A1AttachmentProjectionResult,
    A1AttachmentProjectionSettings,
    project_triangulated_disk_attachment,
)
from .a1_material_correspondence import validate_document_material_correspondence
from .a1_projected_region_filter import split_xy_visible_region_snapshots
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
    # Camera-relative documents validate that the rigid orbital depth layer and the
    # projected Object Origin base cancel their setup offset exactly. The historical
    # field name is retained for positional/API compatibility.
    compensate_depth_setup_y: bool = False

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
        for field_name in (
            "include_control_icons",
            "include_preview_animation",
            "compensate_depth_setup_y",
        ):
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


def _xy_visible_region_snapshots(
    rig: LegacyRigBuildResult,
    region_snapshots: Tuple[MeshSnapshot, ...],
    settings: A1DocumentAssemblySettings,
) -> Tuple[MeshSnapshot, ...]:
    """Cull edge-on faces and return dense visible disk regions for Spine export."""

    visible: list[MeshSnapshot] = []
    for snapshot in region_snapshots:
        visible.extend(
            split_xy_visible_region_snapshots(
                snapshot,
                uniform_scale=rig.info.uniform_scale,
                center_x=float(settings.center_x),
                center_y=float(settings.center_y),
            )
        )

    resolved = tuple(visible)
    if not resolved:
        raise A1DocumentAssemblyError(
            f"All prepared regions for '{settings.prefix}' collapse in Spine XY "
            "projection space. Rotate or flatten the source object so at least one "
            "face has visible two-dimensional area."
        )
    return resolved


def _compensate_projection_depth_setup_y(
    projection: A1AttachmentProjectionResult,
    rig: LegacyRigBuildResult,
) -> A1AttachmentProjectionResult:
    """Validate the rigid camera-layer setup without changing vertex coordinates.

    In the camera-relative hierarchy the depth helper is above ``base``. ``base.y`` is
    already authored as ``projected_origin_y - depth_helper_y``, so its world setup
    position is the exact projected Blender Object Origin. Vertex bones are children of
    ``base`` and remain object-local; subtracting depth from every vertex here would move
    the mesh twice and move the local scale pivot away from Object Origin.

    The historical function name is kept because callers and fault-matrix tests already
    use it. It now owns fail-closed validation rather than coordinate mutation.
    """

    if not isinstance(projection, A1AttachmentProjectionResult):
        raise TypeError("projection must be A1AttachmentProjectionResult")
    if not isinstance(rig, LegacyRigBuildResult):
        raise TypeError("rig must be LegacyRigBuildResult")

    resolved_profile = resolve_a1_rig_profile(rig.profile.profile_id)
    if resolved_profile is not A1RigProfile.TWO_AXIS_ROTATION_SCALE:
        raise A1DocumentAssemblyError(
            "Camera-layer setup validation requires TWO_AXIS_ROTATION_SCALE"
        )
    if rig.request.setup_pose_mode is not A1RigSetupPoseMode.PREPROJECTED_SCREEN:
        raise A1DocumentAssemblyError(
            "Active Camera depth compensation requires PREPROJECTED_SCREEN setup"
        )
    if len(rig.info.z_groups) != 1:
        raise A1DocumentAssemblyError(
            "Camera-relative setup requires exactly one depth group"
        )

    target_group = rig.info.z_groups[0]
    invalid_indices = tuple(
        sorted(
            {
                vertex.z_group_index
                for vertex in projection.request.vertices
                if vertex.z_group_index != target_group.index
            }
        )
    )
    if invalid_indices:
        raise A1DocumentAssemblyError(
            "Attachment vertices reference depth groups outside the rigid camera layer: "
            f"{invalid_indices}"
        )

    bones_by_name = {bone.name: bone for bone in rig.bones}
    base = bones_by_name.get(rig.info.base_bone_name)
    if base is None:
        raise A1DocumentAssemblyError("Camera-relative rig is missing object base bone")
    if base.parent != target_group.bone_name:
        raise A1DocumentAssemblyError(
            "Camera-relative object base must be parented to the orbital depth bone"
        )

    expected_world_x = float(rig.request.main_position_pixels[0])
    expected_world_y = float(rig.request.main_position_pixels[1])
    actual_world_x = float(base.x or 0.0)
    actual_world_y = float(target_group.y_offset_pixels) + float(base.y or 0.0)
    tolerance = 0.011
    if (
        abs(actual_world_x - expected_world_x) > tolerance
        or abs(actual_world_y - expected_world_y) > tolerance
    ):
        raise A1DocumentAssemblyError(
            "Camera-relative base setup does not reconstruct projected Object Origin; "
            f"actual={(actual_world_x, actual_world_y)}, "
            f"expected={(expected_world_x, expected_world_y)}"
        )

    return projection


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

    resolved_profile = resolve_a1_rig_profile(rig.profile.profile_id)
    resolved_rig = (
        ensure_preprojected_screen_rig(rig)
        if (
            settings.compensate_depth_setup_y
            and resolved_profile is A1RigProfile.TWO_AXIS_ROTATION_SCALE
        )
        else rig
    )
    if settings.prefix.strip() != resolved_rig.request.prefix.strip():
        raise A1DocumentAssemblyError(
            f"Assembly prefix '{settings.prefix}' does not match rig prefix "
            f"'{resolved_rig.request.prefix}'"
        )
    if tuple(resolved_rig.request.z_groups) != tuple(z_groups.groups):
        raise A1DocumentAssemblyError(
            "Rig Z groups do not match the source-lineage Z assignment plan"
        )

    visible_region_snapshots = _xy_visible_region_snapshots(
        resolved_rig,
        region_snapshots,
        settings,
    )

    projections: list[A1AttachmentProjectionResult] = []
    for region_offset, snapshot in enumerate(visible_region_snapshots):
        MeshSnapshotValidator().validate_or_raise(snapshot)
        enforce_uv_range(
            snapshot,
            settings.uv_layer_name,
            policy=settings.uv_range_policy,
            epsilon=settings.uv_range_epsilon,
        )
        segment_index = settings.segment_index_base + region_offset
        segment_name = resolved_rig.profile.segment_slot(
            settings.prefix,
            segment_index,
        )
        projection = project_triangulated_disk_attachment(
            snapshot,
            resolved_rig,
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
        if settings.compensate_depth_setup_y:
            projection = _compensate_projection_depth_setup_y(
                projection,
                resolved_rig,
            )
        projections.append(projection)

    resolved_projections = tuple(projections)
    try:
        document_build = build_legacy_mesh_document(
            resolved_rig,
            tuple(projection.request for projection in resolved_projections),
            skeleton_metadata=skeleton_metadata,
        )
        # Segmentation duplicates shared source points at attachment boundaries. Share
        # the corresponding generated bones before downstream correspondence checks,
        # while preserving every attachment vertex, UV, triangle, and local weight.
        document_build = optimize_shared_vertex_bones(document_build)
        validate_document_material_correspondence(
            resolved_projections,
            document_build,
        )
        document = apply_rig_visual_options(
            document_build.document,
            prefix=settings.prefix,
            rig_profile=resolved_rig.profile.profile_id,
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
        rig=resolved_rig,
        z_groups=z_groups,
        projections=resolved_projections,
        document_build=document_build,
    )
