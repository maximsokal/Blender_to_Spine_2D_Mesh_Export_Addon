"""Typed Spine 4.2 document model and validation utilities."""

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
    apply_legacy_visual_options,
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
from .serializer import SpineSerializer
from .validator import SpineValidationError, SpineValidationIssue, SpineValidator
from .weighted_vertices import (
    WeightedVertex,
    WeightedVertexInfluence,
    decode_weighted_vertices,
    encode_weighted_vertices,
)

__all__ = [
    "A1ParityError",
    "A1ParityIssue",
    "A1ParityReport",
    "A1ParitySettings",
    "A1ParitySeverity",
    "AnimationNameAssignment",
    "Bone",
    "ComponentBoneIndexMap",
    "ConnectedConstraintSchedule",
    "ConnectedGroupBuildError",
    "ConnectedGroupBuildResult",
    "ConnectedGroupSettings",
    "ConnectedObjectDocument",
    "ConnectedObjectPlacement",
    "ConnectedZLayer",
    "ConstraintOrderAssignment",
    "ConstraintOrderPolicy",
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
    "UniformScaleMode",
    "WeightedVertex",
    "WeightedVertexInfluence",
    "apply_legacy_visual_options",
    "build_connected_group_document",
    "build_legacy_control_slots_and_attachments",
    "build_legacy_fingerprint",
    "build_legacy_mesh_attachment",
    "build_legacy_mesh_document",
    "build_legacy_preview_animation",
    "build_legacy_rig",
    "calculate_uniform_scale",
    "compare_a1_exports",
    "compose_spine_documents",
    "decode_weighted_vertices",
    "encode_weighted_vertices",
]
