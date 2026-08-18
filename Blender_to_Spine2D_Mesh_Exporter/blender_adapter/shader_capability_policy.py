"""Immutable policy tables for Blender shader capability auditing."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

from ..domain.baking.capabilities import ShaderBakeCapability
from ..domain.baking.graph import MaterialDependencyKind


RENDER_TARGETS: Final = frozenset({"ALL", "CYCLES", "EEVEE"})
CAMERA_DEPENDENCIES: Final = frozenset(
    {
        MaterialDependencyKind.CAMERA,
        MaterialDependencyKind.VIEW,
        MaterialDependencyKind.REFLECTION,
        MaterialDependencyKind.TRANSMISSION,
    }
)
SCENE_DEPENDENCIES: Final = frozenset(
    {
        MaterialDependencyKind.WORLD,
        MaterialDependencyKind.LIGHTING,
        MaterialDependencyKind.OCCLUSION,
        MaterialDependencyKind.SCENE_OBJECTS,
    }
)

# These nodes remain stable on the reconstructed UV target when no stronger dependency is
# present. Source sampling UV and bake target UV are separate, so tangent-space nodes may
# use the preserved source render UV while Cycles writes into SpineBakeUV.
LOCAL_SAFE_NODE_TYPES: Final = frozenset(
    {
        "ADD_SHADER",
        "BLACKBODY",
        "BRIGHTCONTRAST",
        "BSDF_DIFFUSE",
        "BSDF_PRINCIPLED",
        "BSDF_TRANSPARENT",
        "BUMP",
        "CHECKER",
        "CLAMP",
        "COMBHSV",
        "COMBRGB",
        "COMBXYZ",
        "COMBINE_COLOR",
        "CURVE_RGB",
        "CURVE_VEC",
        "DISPLACEMENT",
        "EMISSION",
        "FLOAT_CURVE",
        "GAMMA",
        "GROUP",
        "GROUP_INPUT",
        "GROUP_OUTPUT",
        "HUE_SAT",
        "INVERT",
        "MAP_RANGE",
        "MAPPING",
        "MATH",
        "MIX",
        "MIX_RGB",
        "MIX_SHADER",
        "NORMAL",
        "NORMAL_MAP",
        "OUTPUT_MATERIAL",
        "PRINCIPLED_VOLUME",
        "RGB",
        "RGBTOBW",
        "REROUTE",
        "SEPARATE_COLOR",
        "SEPHSV",
        "SEPRGB",
        "SEPXYZ",
        "TANGENT",
        "TEX_BRICK",
        "TEX_CHECKER",
        "TEX_GABOR",
        "TEX_GRADIENT",
        "TEX_IMAGE",
        "TEX_IES",
        "TEX_MAGIC",
        "TEX_MUSGRAVE",
        "TEX_NOISE",
        "TEX_SKY",
        "TEX_VORONOI",
        "TEX_WAVE",
        "TEX_WHITE_NOISE",
        "UVMAP",
        "VALUE",
        "VALTORGB",
        "VECT_MATH",
        "VECTOR_DISPLACEMENT",
        "VECTOR_ROTATE",
        "VOLUME_ABSORPTION",
        "VOLUME_INFO",
        "VOLUME_SCATTER",
        "WAVELENGTH",
    }
)
SCENE_NODE_TYPES: Final = frozenset(
    {
        "AMBIENT_OCCLUSION",
        "BSDF_TOON",
        "BSDF_TRANSLUCENT",
        "LIGHT_FALLOFF",
        "SUBSURFACE_SCATTERING",
    }
)
CAMERA_NODE_TYPES: Final = frozenset(
    {
        "BACKGROUND",
        "BEVEL",
        "BSDF_ANISOTROPIC",
        "BSDF_GLASS",
        "BSDF_GLOSSY",
        "BSDF_HAIR",
        "BSDF_HAIR_PRINCIPLED",
        "BSDF_REFRACTION",
        "BSDF_SHEEN",
        "BSDF_VELVET",
        "CAMERA",
        "FRESNEL",
        "HOLDOUT",
        "LAYER_WEIGHT",
        "LIGHT_PATH",
        "OBJECT_INFO",
        "TEX_ENVIRONMENT",
        "VECT_TRANSFORM",
        "VECTOR_TRANSFORM",
    }
)
GROUP_NODE_TYPES: Final = frozenset(
    {
        "CURVES_INFO",
        "HAIR_INFO",
        "PARTICLE_INFO",
        "TEX_POINTDENSITY",
    }
)
SOURCE_ATTRIBUTE_NODE_TYPES: Final = frozenset(
    {
        "ATTRIBUTE",
        "VERTEX_COLOR",
    }
)

# Blender's Texture Coordinate "From Instancer" switch is a node property, not an
# output socket. It is therefore intentionally absent here and handled by
# texture_coordinate_findings() using ShaderNodeSnapshot.from_instancer.
TEXTURE_COORD_CAPABILITIES: Mapping[str, ShaderBakeCapability] = MappingProxyType(
    {
        "uv": ShaderBakeCapability.LOCAL_UV_SAFE,
        "camera": ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
        "window": ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
        "reflection": ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
        "object": ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
        "generated": ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
        "normal": ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
    }
)
GEOMETRY_OUTPUT_CAPABILITIES: Mapping[str, ShaderBakeCapability] = MappingProxyType(
    {
        "incoming": ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
        "backfacing": ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
        "pointiness": ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
        "random per island": ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
    }
)


def normalise_render_target(value: str) -> str:
    """Normalize Blender render-engine identifiers for capability policy lookup."""

    target = str(value or "ALL").strip().upper()
    if target in RENDER_TARGETS:
        return target
    if target == "BLENDER_EEVEE":
        return "EEVEE"
    raise ValueError(f"Unsupported render_target: {value!r}")


__all__ = [
    "CAMERA_DEPENDENCIES",
    "CAMERA_NODE_TYPES",
    "GEOMETRY_OUTPUT_CAPABILITIES",
    "GROUP_NODE_TYPES",
    "LOCAL_SAFE_NODE_TYPES",
    "RENDER_TARGETS",
    "SCENE_DEPENDENCIES",
    "SCENE_NODE_TYPES",
    "SOURCE_ATTRIBUTE_NODE_TYPES",
    "TEXTURE_COORD_CAPABILITIES",
    "normalise_render_target",
]
