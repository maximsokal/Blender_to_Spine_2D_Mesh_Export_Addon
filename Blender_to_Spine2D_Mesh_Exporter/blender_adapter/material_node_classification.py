"""Legacy material-kind and image-dependency classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from ..domain.baking import ImageDependency, MaterialKind
from .material_analysis_rna import is_temporary_node, node_type


PROCEDURAL_NODE_TYPES = frozenset(
    {
        "TEX_BRICK",
        "TEX_CHECKER",
        "TEX_GABOR",
        "TEX_GRADIENT",
        "TEX_MAGIC",
        "TEX_MUSGRAVE",
        "TEX_NOISE",
        "TEX_SKY",
        "TEX_VORONOI",
        "TEX_WAVE",
        "TEX_WHITE_NOISE",
        "SCRIPT",
    }
)

ImageDependencyKey = tuple[str, str, str | None, int]
ImageDependencySortKey = tuple[str, str, str, str, int, str, str, int, int]


@dataclass(frozen=True, slots=True)
class MaterialNodeClassification:
    """Typed result of classifying one already-selected node set."""

    kind: MaterialKind
    node_types: Tuple[str, ...]
    image_dependencies: Tuple[ImageDependency, ...]
    issues: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MaterialKind):
            raise TypeError("kind must be MaterialKind")
        if not isinstance(self.node_types, tuple) or not all(
            isinstance(value, str) and value for value in self.node_types
        ):
            raise TypeError("node_types must be a tuple of non-empty strings")
        if not isinstance(self.image_dependencies, tuple) or not all(
            isinstance(value, ImageDependency) for value in self.image_dependencies
        ):
            raise TypeError(
                "image_dependencies must be a tuple of ImageDependency values"
            )
        if not isinstance(self.issues, tuple) or not all(
            isinstance(value, str) and value for value in self.issues
        ):
            raise TypeError("issues must be a tuple of non-empty strings")

    def as_legacy_tuple(
        self,
    ) -> tuple[
        MaterialKind,
        tuple[str, ...],
        tuple[ImageDependency, ...],
        tuple[str, ...],
    ]:
        """Return the historical private ``_classify_nodes`` result."""

        return (
            self.kind,
            self.node_types,
            self.image_dependencies,
            self.issues,
        )


def read_image_dependency(
    node: Any,
) -> tuple[ImageDependency | None, str | None]:
    """Read one Image Texture datablock without mutating Blender state."""

    image = getattr(node, "image", None)
    node_name_value = str(getattr(node, "name", "") or "TEX_IMAGE")
    if image is None:
        return None, f"Image Texture node '{node_name_value}' has no image"

    image_name = str(
        getattr(image, "name_full", None)
        or getattr(image, "name", None)
        or ""
    )
    if not image_name:
        return None, f"Image Texture node '{node_name_value}' references an unnamed image"

    source = str(getattr(image, "source", "FILE") or "FILE")
    filepath_value = (
        getattr(image, "filepath_raw", None)
        or getattr(image, "filepath", None)
    )
    filepath = None if filepath_value in (None, "") else str(filepath_value)
    frame_duration = int(getattr(image, "frame_duration", 1) or 1)
    return (
        ImageDependency(
            image_name=image_name,
            source=source,
            filepath=filepath,
            frame_duration=max(1, frame_duration),
            generated=source.upper() == "GENERATED",
        ),
        None,
    )


def image_dependency_key(dependency: ImageDependency) -> ImageDependencyKey:
    """Return the historical dependency deduplication key."""

    if not isinstance(dependency, ImageDependency):
        raise TypeError("dependency must be ImageDependency")
    return (
        dependency.image_name,
        dependency.source,
        dependency.filepath,
        dependency.frame_duration,
    )


def image_dependency_sort_key(
    dependency: ImageDependency,
) -> ImageDependencySortKey:
    """Return a total ordering key that safely handles optional file paths."""

    if not isinstance(dependency, ImageDependency):
        raise TypeError("dependency must be ImageDependency")
    filepath = dependency.filepath
    filepath_value = "" if filepath is None else filepath
    return (
        dependency.image_name.casefold(),
        dependency.image_name,
        dependency.source.casefold(),
        dependency.source,
        0 if filepath is None else 1,
        filepath_value.casefold(),
        filepath_value,
        dependency.frame_duration,
        int(dependency.generated),
    )


def classify_material_nodes(
    nodes: tuple[Any, ...],
) -> MaterialNodeClassification:
    """Classify one already-resolved reachable or compatibility node set."""

    if not isinstance(nodes, tuple):
        nodes = tuple(nodes)

    relevant_nodes = tuple(
        node
        for node in nodes
        if not is_temporary_node(node)
        and not bool(getattr(node, "mute", False))
    )
    node_types = tuple(sorted({node_type(node) for node in relevant_nodes}))
    procedural = any(
        node_type(node) in PROCEDURAL_NODE_TYPES for node in relevant_nodes
    )

    dependencies_by_key: dict[ImageDependencyKey, ImageDependency] = {}
    issues: list[str] = []
    invalid_image_count = 0
    for node in relevant_nodes:
        if node_type(node) != "TEX_IMAGE":
            continue
        dependency, issue = read_image_dependency(node)
        if issue is not None:
            invalid_image_count += 1
            issues.append(issue)
            continue
        assert dependency is not None
        dependencies_by_key[image_dependency_key(dependency)] = dependency

    dependencies = tuple(
        sorted(
            dependencies_by_key.values(),
            key=image_dependency_sort_key,
        )
    )

    if invalid_image_count:
        kind = MaterialKind.UNSUPPORTED
    elif dependencies and procedural:
        kind = MaterialKind.MIXED
    elif dependencies:
        kind = MaterialKind.IMAGE
    elif procedural:
        kind = MaterialKind.PROCEDURAL
    else:
        kind = MaterialKind.SOLID_COLOR

    return MaterialNodeClassification(
        kind=kind,
        node_types=node_types,
        image_dependencies=dependencies,
        issues=tuple(issues),
    )


def classify_nodes_legacy(
    nodes: tuple[Any, ...],
) -> tuple[
    MaterialKind,
    tuple[str, ...],
    tuple[ImageDependency, ...],
    tuple[str, ...],
]:
    """Compatibility wrapper retaining the private ``_classify_nodes`` tuple."""

    return classify_material_nodes(nodes).as_legacy_tuple()


__all__ = [
    "ImageDependencyKey",
    "ImageDependencySortKey",
    "MaterialNodeClassification",
    "PROCEDURAL_NODE_TYPES",
    "classify_material_nodes",
    "classify_nodes_legacy",
    "image_dependency_key",
    "image_dependency_sort_key",
    "read_image_dependency",
]
