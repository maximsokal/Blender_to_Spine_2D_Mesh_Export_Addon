"""Typed Spine 4.2 document model and validation utilities."""

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
from .legacy_rig_builder import (
    LegacyRigBuildError,
    LegacyRigBuildRequest,
    LegacyRigBuildResult,
    LegacyRigInfo,
    LegacyZGroup,
    LegacyZGroupBuildInfo,
    UniformScaleMode,
    build_legacy_rig,
    calculate_uniform_scale,
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
from .serializer import SpineSerializer
from .validator import SpineValidationError, SpineValidationIssue, SpineValidator
from .weighted_vertices import (
    WeightedVertex,
    WeightedVertexInfluence,
    decode_weighted_vertices,
    encode_weighted_vertices,
)

__all__ = [
    "Bone",
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
    "SpineDocument",
    "SpineSerializer",
    "SpineValidationError",
    "SpineValidationIssue",
    "SpineValidator",
    "TransformConstraint",
    "UniformScaleMode",
    "WeightedVertex",
    "WeightedVertexInfluence",
    "build_legacy_fingerprint",
    "build_legacy_mesh_attachment",
    "build_legacy_mesh_document",
    "build_legacy_rig",
    "calculate_uniform_scale",
    "decode_weighted_vertices",
    "encode_weighted_vertices",
]
