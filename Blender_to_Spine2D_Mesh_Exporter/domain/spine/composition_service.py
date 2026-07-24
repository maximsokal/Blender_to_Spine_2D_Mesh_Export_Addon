"""Safe public composition boundary for typed and raw Spine documents.

The low-level composer owns logical mesh topology and can rebase weighted bone indices
only for typed :class:`MeshAttachment` values. Raw JSON mappings may already contain
serialized edge offsets and provide no marker telling whether weighted bone indices are
local to a component or already global. Public composition rejects those ambiguous raw
mesh forms instead of silently corrupting a combined rotatable-mesh rig.
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


def _raw_attachment_path(
    component: SpineDocumentComponent,
    *,
    skin_name: str,
    skin_index: int,
    slot_name: object,
    attachment_name: object,
) -> str:
    return (
        f"component={component.component_id!r}, skin={skin_name!r}, "
        f"skin_index={skin_index}, slot={str(slot_name)!r}, "
        f"attachment={str(attachment_name)!r}"
    )


def _reject_ambiguous_raw_meshes(
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
                        # local topology or weighted stream to rebase here.
                        continue

                    path = _raw_attachment_path(
                        component,
                        skin_name=skin.name,
                        skin_index=skin_index,
                        slot_name=slot_name,
                        attachment_name=attachment_name,
                    )
                    edges = attachment.get("edges")
                    if _is_sequence(edges) and len(edges) > 0:
                        raise SpineCompositionError(
                            "Raw mesh mappings with edges cannot be composed safely: "
                            "the mapping may already use Spine serialized coordinate "
                            "offsets while the composer owns logical vertex indices. "
                            f"Convert it to typed MeshAttachment first; {path}"
                        )

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

                    raise SpineCompositionError(
                        "Raw weighted mesh mappings cannot be composed safely because "
                        "their bone-index ownership is ambiguous. Convert the mapping "
                        f"to typed MeshAttachment before composition; {path}"
                    )


def compose_spine_documents(
    components: Tuple[SpineDocumentComponent, ...],
    settings: SpineCompositionSettings | None = None,
) -> SpineDocumentCompositionResult:
    """Compose documents while requiring typed ownership for ambiguous mesh data."""

    if not isinstance(components, tuple) or not components:
        raise ValueError("components must be a non-empty tuple")
    if not all(isinstance(item, SpineDocumentComponent) for item in components):
        raise TypeError("components must contain SpineDocumentComponent values")
    if settings is not None and not isinstance(settings, SpineCompositionSettings):
        raise TypeError("settings must be SpineCompositionSettings or None")

    _reject_ambiguous_raw_meshes(components)
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
