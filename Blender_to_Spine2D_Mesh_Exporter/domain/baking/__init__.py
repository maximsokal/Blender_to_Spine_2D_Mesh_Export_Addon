"""Blender-independent material analysis and texture baking plans."""

from .execution import BakeArtifact, BakeExecutionResult
from .model import (
    BakeFrameTask,
    BakeMaterialPolicy,
    BakeMode,
    BakePlan,
    BakePlanError,
    BakeSettings,
    ImageDependency,
    MaterialAnalysis,
    MaterialKind,
    ObjectMaterialAnalysis,
    TextureFormat,
    build_bake_plan,
    sanitize_filename_stem,
)

__all__ = [
    "BakeArtifact",
    "BakeExecutionResult",
    "BakeFrameTask",
    "BakeMaterialPolicy",
    "BakeMode",
    "BakePlan",
    "BakePlanError",
    "BakeSettings",
    "ImageDependency",
    "MaterialAnalysis",
    "MaterialKind",
    "ObjectMaterialAnalysis",
    "TextureFormat",
    "build_bake_plan",
    "sanitize_filename_stem",
]
