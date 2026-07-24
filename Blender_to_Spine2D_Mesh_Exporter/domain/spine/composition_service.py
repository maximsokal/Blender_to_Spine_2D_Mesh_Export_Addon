"""Safe public composition boundary for typed and raw Spine documents.

The low-level composer can remap weighted bone indices only for typed
:class:`MeshAttachment` values. Raw JSON mappings have no ownership marker telling us
whether their weighted indices are local to a component or already global. Public
composition therefore rejects raw weighted meshes instead of silently copying invalid
indices into a combined rotatable-mesh rig.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Tuple

from .composition import (
    AnimationNameAssignment,
    ComponentBoneIndexMap,
    ConstraintOrderAssignment,
    ConstraintOrderPolicy,
    SpineCompositionError,
    SpineCompositionSettings,
    SpineDocumentComponent,
    SpineDocumentCompositionResult,
    compose_spine_documents as _compose_spine_documents,
)
from .model import MeshAttachment


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    )


def _reject_raw_weighted_meshes(
    components: Tuple[SpineDocumentComponent, ...],
) -> None:
    for component in components:
        for skin_index, skin in enumerate(component.document.skins):
            for slot_name, slot_attachments in skin.attachments.items():
                for attachment_name, attachment in slot_attachments.items():
                    if isinstance(attachment, MeshAttachment):
                        continue
                    if not isinstance(attachment, Mapping):
                        continue
                    if str(attachment.get("type", "region")) != "mesh":
                        continue
                    if attachment.get("parent"):
                        # Linked meshes inherit geometry from their parent and carry no
                        # local weighted stream to rebase at this boundary.
                        continue

                    uvs = attachment.get("uvs")
                    vertices = attachment.get("vertices")
                    if not _is_sequence(uvs) or not _is_sequence(vertices):
                        continue
                    if len(uvs) % 2 != 0:
                        continue
                    vertex_count = len(uvs) // 2
                    if vertex_count <= 0:
                        continue
                    if len(vertices) == vertex_count * 2:
                        # Raw unweighted meshes contain x/y pairs and no bone indices.
                        continue

                    path = (
                        f"component={component.component_id!r}, skin={skin.name!r}, "
                        f"skin_index={skin_index}, slot={str(slot_name)!r}, "
                        f"attachment={str(attachment_name)!r}"
                    )
                    raise SpineCompositionError(
                        "Raw weighted mesh mappings cannot be composed safely because "
                        "their bone-index ownership is ambiguous. Convert the mapping "
                        f"to typed MeshAttachment before composition; {path}"
                    )


def compose_spine_documents(
    components: Tuple[SpineDocumentComponent, ...],
    settings: SpineCompositionSettings | None = None,
) -> SpineDocumentCompositionResult:
    """Compose validated documents while requiring typed weighted mesh ownership."""

    if not isinstance(components, tuple) or not components:
        raise ValueError("components must be a non-empty tuple")
    if not all(isinstance(item, SpineDocumentComponent) for item in components):
        raise TypeError("components must contain SpineDocumentComponent values")
    if settings is not None and not isinstance(settings, SpineCompositionSettings):
        raise TypeError("settings must be SpineCompositionSettings or None")

    _reject_raw_weighted_meshes(components)
    return _compose_spine_documents(
        components,
        SpineCompositionSettings() if settings is None else settings,
    )


__all__ = [
    "AnimationNameAssignment",
    "ComponentBoneIndexMap",
    "ConstraintOrderAssignment",
    "ConstraintOrderPolicy",
    "SpineCompositionError",
    "SpineCompositionSettings",
    "SpineDocumentComponent",
    "SpineDocumentCompositionResult",
    "compose_spine_documents",
]
