"""Blender-independent material analysis and texture baking plans."""

from .model import (
    BakeFrameTask,
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
    "BakeFrameTask",
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
