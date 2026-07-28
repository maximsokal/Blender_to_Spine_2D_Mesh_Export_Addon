"""Typed Spine 4.2 document model and validation utilities."""

from .attachment_sequence_animation import (
    AttachmentSequenceAnimationError,
    DEFAULT_SEQUENCE_FRAME_DELAY,
    apply_attachment_sequence_animations,
    build_attachment_sequence_timeline,
)
from .composition_service import (
    AnimationNameAssignment,
    ComponentBoneIndexMap,
    ConstraintOrderAssignment,
    ConstraintOrderPolicy,
    SpineCompositionError,
    SpineCompositionSettings,
    SpineDocumentComponent,
    SpineDocumentCompositionResult,
    compose_spine_documents,
)
from .connected_group_assembly import build_connected_group_document
from .connected_group_contracts import (
    ConnectedConstraintSchedule,
    ConnectedGroupBuildResult,
    ConnectedGroupSettings,
    ConnectedObjectDocument,
    ConnectedObjectPlacement,
    ConnectedPlacementSpace,
    ConnectedZLayer,
)
from .connected_group_error import ConnectedGroupBuildError
from .golden import LegacyCompatibilityFingerprint, build_legacy_fingerprint
from .legacy_attachment_builder import (
    LegacyAttachmentComponent,
    LegacyAttachmentSequence,
    LegacyAttachmentVertex,
    LegacyMeshAttachmentBuildError,
    LegacyMeshAttachmentBuildResult,
    LegacyMeshAttachmentRequest,
    LegacyMeshDocumentBuildResult,
    build_legacy_mesh_attachment,
    build_legacy_mesh_document,
)
from .legacy_profile import LegacyRigProfile
from .legacy_rig_assembly import build_legacy_rig
from .legacy_rig_contracts import (
    LegacyRigBuildRequest,
    LegacyRigBuildResult,
    LegacyRigInfo,
    LegacyZGroup,
    LegacyZGroupBuildInfo,
    UniformScaleMode,
)
from .legacy_rig_error import LegacyRigBuildError
from .legacy_rig_scale import calculate_uniform_scale
from .legacy_visuals import (
    apply_legacy_visual_options as _apply_legacy_visual_options,
    build_legacy_control_slots_and_attachments,
    build_legacy_preview_animation,
)
from .model import (
    Bone,
    IKConstraint,
    MeshAttachment,
    Skin,
    Slot,
    SpineDocument,
    TransformConstraint,
)
from .parity import (
    A1ParityError,
    A1ParityIssue,
    A1ParityReport,
    A1ParitySettings,
    A1ParitySeverity,
    compare_a1_exports,
)
from .rig_builder import build_rig
from .rig_profiles import (
    A1RigProfile,
    A1RigSetupPoseMode,
    resolve_a1_rig_profile,
)
from .rig_visuals import (
    apply_rig_visual_options,
    build_two_axis_scale_control_slots_and_attachments,
    build_two_axis_scale_preview_animation,
)
from .serializer import SpineSerializer
from .two_axis_scale_profile import TwoAxisScaleRigProfile
from .two_axis_scale_rig import (
    TwoAxisScaleRigLayout,
    build_two_axis_scale_layout,
    build_two_axis_scale_rig,
    validate_two_axis_scale_rig_result,
)
from .validator import SpineValidationError, SpineValidationIssue, SpineValidator
from .weighted_vertices import (
    WeightedVertex,
    WeightedVertexInfluence,
    decode_weighted_vertices,
    encode_weighted_vertices,
)


def apply_legacy_visual_options(
    document: SpineDocument,
    *,
    prefix: str,
    include_control_icons: bool,
    include_preview_animation: bool,
) -> SpineDocument:
    """Compatibility facade that recognizes the generated control namespace.

    Existing application modules historically import this name from the package root.
    The facade keeps those callers working while camera-projection documents can use the
    new two-axis profile before every call site is renamed to ``apply_rig_visual_options``.
    """

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    normalized = prefix.strip() if isinstance(prefix, str) else ""
    if not normalized:
        raise ValueError("prefix must be a non-empty string")
    bone_names = {bone.name for bone in document.bones}
    rotation_z = f"{normalized}_rotation_Z"
    scale_control = f"{normalized}_scale"
    if rotation_z in bone_names:
        return _apply_legacy_visual_options(
            document,
            prefix=normalized,
            include_control_icons=include_control_icons,
            include_preview_animation=include_preview_animation,
        )
    if scale_control in bone_names:
        return apply_rig_visual_options(
            document,
            prefix=normalized,
            rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE,
            include_control_icons=include_control_icons,
            include_preview_animation=include_preview_animation,
        )
    raise ValueError(
        "Unable to identify rig profile from generated controls for "
        f"prefix {normalized!r}"
    )


__all__ = [
    "A1ParityError",
    "A1ParityIssue",
    "A1ParityReport",
    "A1ParitySettings",
    "A1ParitySeverity",
    "A1RigProfile",
    "A1RigSetupPoseMode",
    "AnimationNameAssignment",
    "AttachmentSequenceAnimationError",
    "Bone",
    "ComponentBoneIndexMap",
    "ConnectedConstraintSchedule",
    "ConnectedGroupBuildError",
    "ConnectedGroupBuildResult",
    "ConnectedGroupSettings",
    "ConnectedObjectDocument",
    "ConnectedObjectPlacement",
    "ConnectedPlacementSpace",
    "ConnectedZLayer",
    "ConstraintOrderAssignment",
    "ConstraintOrderPolicy",
    "DEFAULT_SEQUENCE_FRAME_DELAY",
    "IKConstraint",
    "LegacyAttachmentComponent",
    "LegacyAttachmentSequence",
    "LegacyAttachmentVertex",
    "LegacyCompatibilityFingerprint",
    "LegacyMeshAttachmentBuildError",
    "LegacyMeshAttachmentBuildResult",
    "LegacyMeshAttachmentRequest",
    "LegacyMeshDocumentBuildResult",
    "LegacyRigBuildError",
    "LegacyRigBuildRequest",
    "LegacyRigBuildResult",
    "LegacyRigInfo",
    "LegacyRigProfile",
    "LegacyZGroup",
    "LegacyZGroupBuildInfo",
    "MeshAttachment",
    "Skin",
    "Slot",
    "SpineCompositionError",
    "SpineCompositionSettings",
    "SpineDocument",
    "SpineDocumentComponent",
    "SpineDocumentCompositionResult",
    "SpineSerializer",
    "SpineValidationError",
    "SpineValidationIssue",
    "SpineValidator",
    "TransformConstraint",
    "TwoAxisScaleRigLayout",
    "TwoAxisScaleRigProfile",
    "UniformScaleMode",
    "WeightedVertex",
    "WeightedVertexInfluence",
    "apply_attachment_sequence_animations",
    "apply_legacy_visual_options",
    "apply_rig_visual_options",
    "build_attachment_sequence_timeline",
    "build_connected_group_document",
    "build_legacy_control_slots_and_attachments",
    "build_legacy_fingerprint",
    "build_legacy_mesh_attachment",
    "build_legacy_mesh_document",
    "build_legacy_preview_animation",
    "build_legacy_rig",
    "build_rig",
    "build_two_axis_scale_control_slots_and_attachments",
    "build_two_axis_scale_layout",
    "build_two_axis_scale_preview_animation",
    "build_two_axis_scale_rig",
    "calculate_uniform_scale",
    "compare_a1_exports",
    "compose_spine_documents",
    "decode_weighted_vertices",
    "encode_weighted_vertices",
    "resolve_a1_rig_profile",
    "validate_two_axis_scale_rig_result",
]
