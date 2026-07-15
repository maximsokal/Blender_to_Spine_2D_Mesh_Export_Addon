"""Typed Spine 4.2 document model and validation utilities."""

from .golden import LegacyCompatibilityFingerprint, build_legacy_fingerprint
from .legacy_profile import LegacyRigProfile
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
    "LegacyCompatibilityFingerprint",
    "LegacyRigProfile",
    "MeshAttachment",
    "Skin",
    "Slot",
    "SpineDocument",
    "SpineSerializer",
    "SpineValidationError",
    "SpineValidationIssue",
    "SpineValidator",
    "TransformConstraint",
    "WeightedVertex",
    "WeightedVertexInfluence",
    "build_legacy_fingerprint",
    "decode_weighted_vertices",
    "encode_weighted_vertices",
]
