"""Derive semantic channels and dependencies from a frozen reachable shader graph."""

from __future__ import annotations

from enum import Enum
from typing import Any

from ..domain.baking.graph import (
    MaterialDependencyKind,
    MaterialSemanticChannel,
)
from .shader_graph_rna import (
    color_nonzero,
    first_input_socket,
    input_socket,
    node_type,
    numeric_default,
    socket_enabled,
)
from .shader_graph_traversal import (
    GROUP_INPUT_TYPES,
    GROUP_NODE_TYPES,
    GROUP_OUTPUT_TYPES,
    RecursiveShaderGraphWalker,
    ShaderGraphTraversalResult,
)


VIEW_NODE_TYPES = frozenset({"FRESNEL", "LAYER_WEIGHT", "LIGHT_PATH"})
OBJECT_NODE_TYPES = frozenset({"OBJECT_INFO", "TEX_COORD"})
GEOMETRY_NODE_TYPES = frozenset(
    {"NEW_GEOMETRY", "NORMAL", "NORMAL_MAP", "BUMP", "TANGENT", "BEVEL"}
)
CAMERA_SHADER_TYPES = frozenset(
    {"BSDF_GLASS", "BSDF_REFRACTION", "BSDF_GLOSSY"}
)
SCENE_LIGHTING_SHADER_TYPES = frozenset(
    {
        "BSDF_TRANSLUCENT",
        "BSDF_TOON",
        "SUBSURFACE_SCATTERING",
        "BSDF_HAIR",
        "BSDF_HAIR_PRINCIPLED",
    }
)
SURFACE_SHADER_TYPES = frozenset(
    {
        "BSDF_PRINCIPLED",
        "BSDF_DIFFUSE",
        "BSDF_GLOSSY",
        "BSDF_GLASS",
        "BSDF_REFRACTION",
        "BSDF_TRANSLUCENT",
        "BSDF_TOON",
        "SUBSURFACE_SCATTERING",
        "BSDF_HAIR",
        "BSDF_HAIR_PRINCIPLED",
        "HOLDOUT",
    }
)


class PrincipledCameraFeature(str, Enum):
    """Camera-sensitive Principled BSDF features with distinct bake semantics."""

    METALLIC = "METALLIC"
    COAT = "COAT"
    TRANSMISSION = "TRANSMISSION"


def _coerce_traversal(
    value: ShaderGraphTraversalResult | RecursiveShaderGraphWalker,
) -> ShaderGraphTraversalResult:
    if isinstance(value, ShaderGraphTraversalResult):
        return value
    if isinstance(value, RecursiveShaderGraphWalker):
        return value.build_result()
    raise TypeError(
        "value must be ShaderGraphTraversalResult or RecursiveShaderGraphWalker"
    )


def principled_emission_enabled(node: Any) -> bool:
    if node_type(node) != "BSDF_PRINCIPLED":
        return False
    color = first_input_socket(node, ("Emission Color", "Emission"))
    strength = input_socket(node, "Emission Strength")
    return color_nonzero(color) and numeric_default(strength, 1.0) > 1e-8


def principled_alpha_enabled(node: Any) -> bool:
    if node_type(node) != "BSDF_PRINCIPLED":
        return False
    alpha = input_socket(node, "Alpha")
    if alpha is None:
        return False
    return bool(getattr(alpha, "is_linked", False)) or numeric_default(
        alpha,
        1.0,
    ) < 0.999999


def principled_camera_features(node: Any) -> frozenset[PrincipledCameraFeature]:
    """Return enabled camera-sensitive Principled features from current RNA state.

    Blender 5.2 names are preferred while the historical socket names are retained as
    deterministic fallbacks for compatible saved node trees. Linked sockets count as
    enabled even when their displayed default value is zero.
    """

    if node_type(node) != "BSDF_PRINCIPLED":
        return frozenset()

    features: set[PrincipledCameraFeature] = set()
    transmission = first_input_socket(
        node,
        ("Transmission Weight", "Transmission"),
    )
    metallic = input_socket(node, "Metallic")
    coat = first_input_socket(node, ("Coat Weight", "Clearcoat"))

    if socket_enabled(metallic):
        features.add(PrincipledCameraFeature.METALLIC)
    if socket_enabled(coat):
        features.add(PrincipledCameraFeature.COAT)
    if socket_enabled(transmission):
        features.add(PrincipledCameraFeature.TRANSMISSION)
    return frozenset(features)


def principled_dependencies(node: Any) -> set[MaterialDependencyKind]:
    result: set[MaterialDependencyKind] = set()
    if node_type(node) != "BSDF_PRINCIPLED":
        return result

    camera_features = principled_camera_features(node)
    subsurface = first_input_socket(
        node,
        ("Subsurface Weight", "Subsurface"),
    )
    sheen = first_input_socket(node, ("Sheen Weight", "Sheen"))

    if PrincipledCameraFeature.TRANSMISSION in camera_features:
        result.update(
            {
                MaterialDependencyKind.CAMERA,
                MaterialDependencyKind.VIEW,
                MaterialDependencyKind.WORLD,
                MaterialDependencyKind.SCENE_OBJECTS,
                MaterialDependencyKind.REFLECTION,
                MaterialDependencyKind.TRANSMISSION,
            }
        )
    if camera_features & {
        PrincipledCameraFeature.METALLIC,
        PrincipledCameraFeature.COAT,
    }:
        result.update(
            {
                MaterialDependencyKind.CAMERA,
                MaterialDependencyKind.VIEW,
                MaterialDependencyKind.WORLD,
                MaterialDependencyKind.SCENE_OBJECTS,
                MaterialDependencyKind.REFLECTION,
            }
        )
    if socket_enabled(subsurface) or socket_enabled(sheen):
        result.add(MaterialDependencyKind.LIGHTING)
    return result


def derive_semantic_channels(
    value: ShaderGraphTraversalResult | RecursiveShaderGraphWalker,
) -> tuple[MaterialSemanticChannel, ...]:
    traversal = _coerce_traversal(value)
    surface_nodes = tuple(
        traversal.nodes[node_id].node
        for node_id in traversal.channel_nodes.get("SURFACE", ())
        if not bool(getattr(traversal.nodes[node_id].node, "mute", False))
    )
    surface_types = {node_type(node) for node in surface_nodes}
    channels: set[MaterialSemanticChannel] = set()
    if surface_types & SURFACE_SHADER_TYPES:
        channels.add(MaterialSemanticChannel.SURFACE_COLOR)
    if "EMISSION" in surface_types or any(
        principled_emission_enabled(node) for node in surface_nodes
    ):
        channels.add(MaterialSemanticChannel.SURFACE_EMISSION)
    if (
        "BSDF_TRANSPARENT" in surface_types
        or "HOLDOUT" in surface_types
        or any(principled_alpha_enabled(node) for node in surface_nodes)
    ):
        channels.add(MaterialSemanticChannel.ALPHA)

    volume_types = {
        node_type(traversal.nodes[node_id].node)
        for node_id in traversal.channel_nodes.get("VOLUME", ())
        if not bool(getattr(traversal.nodes[node_id].node, "mute", False))
    }
    if volume_types - {"OUTPUT_MATERIAL", "GROUP_OUTPUT", "GROUP"}:
        channels.add(MaterialSemanticChannel.VOLUME)

    displacement_types = {
        node_type(traversal.nodes[node_id].node)
        for node_id in traversal.channel_nodes.get("DISPLACEMENT", ())
        if not bool(getattr(traversal.nodes[node_id].node, "mute", False))
    }
    if displacement_types - {"OUTPUT_MATERIAL", "GROUP_OUTPUT", "GROUP"}:
        channels.add(MaterialSemanticChannel.DISPLACEMENT)

    known_non_surface = {
        "EMISSION",
        "OUTPUT_MATERIAL",
        "GROUP",
        "GROUP_INPUT",
        "GROUP_OUTPUT",
        "MIX_SHADER",
        "ADD_SHADER",
        "BSDF_TRANSPARENT",
        "HOLDOUT",
    }
    if (
        surface_nodes
        and not channels
        and not surface_types.issubset(known_non_surface)
    ):
        channels.add(MaterialSemanticChannel.SURFACE_COLOR)
    return tuple(sorted(channels, key=lambda item: item.value))


def derive_material_dependencies(
    material: Any,
    value: ShaderGraphTraversalResult | RecursiveShaderGraphWalker,
) -> tuple[MaterialDependencyKind, ...]:
    traversal = _coerce_traversal(value)
    result: set[MaterialDependencyKind] = set()
    reachable_nodes = tuple(
        item.node
        for item in traversal.nodes.values()
        if not bool(getattr(item.node, "mute", False))
    )
    node_types = {node_type(node) for node in reachable_nodes}

    if "TEX_IMAGE" in node_types or "TEX_ENVIRONMENT" in node_types:
        result.add(MaterialDependencyKind.IMAGE)
    if node_types & VIEW_NODE_TYPES:
        result.update(
            {
                MaterialDependencyKind.VIEW,
                MaterialDependencyKind.CAMERA,
            }
        )
    if "FRESNEL" in node_types or "LAYER_WEIGHT" in node_types:
        result.add(MaterialDependencyKind.REFLECTION)
    if "LIGHT_PATH" in node_types:
        result.add(MaterialDependencyKind.LIGHTING)
    if node_types & OBJECT_NODE_TYPES:
        result.add(MaterialDependencyKind.OBJECT)
    if node_types & GEOMETRY_NODE_TYPES:
        result.add(MaterialDependencyKind.GEOMETRY)
    if "AMBIENT_OCCLUSION" in node_types:
        result.update(
            {
                MaterialDependencyKind.OCCLUSION,
                MaterialDependencyKind.SCENE_OBJECTS,
            }
        )
    if node_types & SCENE_LIGHTING_SHADER_TYPES:
        result.add(MaterialDependencyKind.LIGHTING)
    if node_types & CAMERA_SHADER_TYPES:
        result.update(
            {
                MaterialDependencyKind.CAMERA,
                MaterialDependencyKind.VIEW,
                MaterialDependencyKind.WORLD,
                MaterialDependencyKind.SCENE_OBJECTS,
                MaterialDependencyKind.REFLECTION,
            }
        )
    if "BSDF_GLASS" in node_types or "BSDF_REFRACTION" in node_types:
        result.add(MaterialDependencyKind.TRANSMISSION)
    if "TEX_ENVIRONMENT" in node_types:
        result.update(
            {
                MaterialDependencyKind.VIEW,
                MaterialDependencyKind.CAMERA,
            }
        )
    if node_types & (
        GROUP_NODE_TYPES | GROUP_INPUT_TYPES | GROUP_OUTPUT_TYPES
    ):
        result.add(MaterialDependencyKind.NODE_GROUP)

    for node in reachable_nodes:
        result.update(principled_dependencies(node))
        if node_type(node) != "TEX_IMAGE":
            continue
        image = getattr(node, "image", None)
        source = str(getattr(image, "source", "") or "").upper()
        duration = (
            int(getattr(image, "frame_duration", 1) or 1)
            if image
            else 1
        )
        if source in {"SEQUENCE", "MOVIE"} or duration > 1:
            result.add(MaterialDependencyKind.TIME)

    if getattr(material, "animation_data", None) is not None:
        result.add(MaterialDependencyKind.TIME)
    for node_tree in traversal.node_trees:
        if getattr(node_tree, "animation_data", None) is not None:
            result.add(MaterialDependencyKind.TIME)
    return tuple(sorted(result, key=lambda item: item.value))


__all__ = [
    "CAMERA_SHADER_TYPES",
    "GEOMETRY_NODE_TYPES",
    "OBJECT_NODE_TYPES",
    "PrincipledCameraFeature",
    "SCENE_LIGHTING_SHADER_TYPES",
    "SURFACE_SHADER_TYPES",
    "VIEW_NODE_TYPES",
    "derive_material_dependencies",
    "derive_semantic_channels",
    "principled_alpha_enabled",
    "principled_camera_features",
    "principled_dependencies",
    "principled_emission_enabled",
]
