"""Deduplicate generated A1 vertex bones without changing mesh geometry.

The pre-Rewrite JSON merger compacted vertex bones after segment composition and then
rewrote every weighted-mesh bone index.  This module preserves that behaviour at the
typed document boundary instead of mutating serialized JSON.

Only generated component vertex bones are candidates.  Two candidates are shared when
their final serialized setup data is identical except for the generated name.  Parent
identity is part of that data, so coincident XY points in different Z groups remain
independent and continue to deform through their own depth parent.
"""

from __future__ import annotations

from dataclasses import replace
import logging
from typing import Any, Mapping, Tuple

from .legacy_attachment_builder import (
    LegacyAttachmentComponent,
    LegacyMeshDocumentBuildResult,
)
from .model import Bone, MeshAttachment, Skin, SpineDocument
from .validator import SpineValidator
from .weighted_vertices import (
    WeightedVertex,
    WeightedVertexInfluence,
    decode_weighted_vertices,
    encode_weighted_vertices,
)


logger = logging.getLogger(__name__)


class VertexBoneOptimizationError(ValueError):
    """Raised when generated vertex-bone compaction cannot preserve the document."""


def _freeze_json_value(value: Any) -> Any:
    """Return a deterministic hashable representation of one JSON-compatible value."""

    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_json_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, tuple):
        return tuple(_freeze_json_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _bone_semantic_key(bone: Bone) -> tuple[Any, ...]:
    """Return merge-relevant setup data while deliberately excluding the bone name."""

    if not isinstance(bone, Bone):
        raise TypeError("bone must be Bone")
    if bone.parent is None:
        raise VertexBoneOptimizationError(
            f"Generated vertex bone {bone.name!r} has no parent"
        )
    if bone.x is None or bone.y is None:
        raise VertexBoneOptimizationError(
            f"Generated vertex bone {bone.name!r} has no XY position"
        )

    # The attachment builder already rounds X/Y to the exact values emitted to Spine.
    # Comparing this final representation avoids merging visibly distinct bones while
    # treating 0.0 and -0.0 as the same serialized location.
    return (
        bone.parent,
        float(bone.x),
        float(bone.y),
        bone.length,
        bone.rotation,
        bone.scale_x,
        bone.scale_y,
        bone.color,
        bone.icon,
        _freeze_json_value(bone.extras),
    )


def _external_bone_references(document: SpineDocument) -> frozenset[str]:
    """Collect name-based references that prevent deleting a generated bone name."""

    references: set[str] = set()
    references.update(
        bone.parent for bone in document.bones if bone.parent is not None
    )
    references.update(slot.bone for slot in document.slots)
    for constraint in document.ik:
        references.update(constraint.bones)
        references.add(constraint.target)
    for constraint in document.transform:
        references.update(constraint.bones)
        references.add(constraint.target)
    for skin in document.skins:
        references.update(skin.bones)

    for animation in document.animations.values():
        if not isinstance(animation, Mapping):
            continue
        bone_timelines = animation.get("bones")
        if isinstance(bone_timelines, Mapping):
            references.update(
                name for name in bone_timelines if isinstance(name, str)
            )
    return frozenset(references)


def _decode_component_bindings(
    build: LegacyMeshDocumentBuildResult,
) -> Tuple[Tuple[int, ...], ...]:
    """Decode and validate each component's one-bone-per-vertex binding."""

    document_bones = build.document.bones
    result: list[Tuple[int, ...]] = []

    for component_index, component in enumerate(build.components):
        decoded = decode_weighted_vertices(
            component.attachment.vertices,
            expected_vertex_count=len(component.request.vertices),
        )
        if len(component.vertex_bones) != len(decoded):
            raise VertexBoneOptimizationError(
                f"Component {component_index} vertex-bone count differs from its "
                "weighted vertex count"
            )

        indices: list[int] = []
        for vertex_index, weighted_vertex in enumerate(decoded):
            if len(weighted_vertex.influences) != 1:
                raise VertexBoneOptimizationError(
                    f"Component {component_index} vertex {vertex_index} has "
                    f"{len(weighted_vertex.influences)} influences; generated A1 "
                    "vertices must have exactly one"
                )
            influence = weighted_vertex.influences[0]
            bone_index = influence.bone_index
            if bone_index >= len(document_bones):
                raise VertexBoneOptimizationError(
                    f"Component {component_index} vertex {vertex_index} references "
                    f"bone index {bone_index}, but the document contains only "
                    f"{len(document_bones)} bones"
                )
            if (influence.x, influence.y, influence.weight) != (0.0, 0.0, 1.0):
                raise VertexBoneOptimizationError(
                    f"Component {component_index} vertex {vertex_index} has "
                    "non-canonical generated weight data"
                )

            document_bone = document_bones[bone_index]
            component_bone = component.vertex_bones[vertex_index]
            if document_bone != component_bone:
                raise VertexBoneOptimizationError(
                    f"Component {component_index} vertex {vertex_index} does not "
                    "match its referenced document bone"
                )
            indices.append(bone_index)
        result.append(tuple(indices))

    return tuple(result)


def _duplicate_to_master_map(
    document: SpineDocument,
    component_indices: Tuple[Tuple[int, ...], ...],
) -> dict[int, int]:
    """Choose one deterministic master for every equivalent generated bone group."""

    candidate_indices = tuple(
        sorted({index for component in component_indices for index in component})
    )
    groups: dict[tuple[Any, ...], list[int]] = {}
    for bone_index in candidate_indices:
        bone = document.bones[bone_index]
        groups.setdefault(_bone_semantic_key(bone), []).append(bone_index)

    external_references = _external_bone_references(document)
    duplicate_to_master: dict[int, int] = {}
    for indices in groups.values():
        if len(indices) < 2:
            continue

        referenced = tuple(
            index
            for index in indices
            if document.bones[index].name in external_references
        )
        if len(referenced) > 1:
            # Two independently referenced names cannot be collapsed safely.
            logger.warning(
                "Skipping shared vertex-bone optimization for externally referenced "
                "bones: %s",
                tuple(document.bones[index].name for index in referenced),
            )
            continue

        master = referenced[0] if referenced else indices[0]
        for index in indices:
            if index != master:
                duplicate_to_master[index] = master
    return duplicate_to_master


def _compact_bones(
    bones: Tuple[Bone, ...],
    duplicate_to_master: Mapping[int, int],
) -> tuple[Tuple[Bone, ...], dict[int, int]]:
    """Remove duplicate bones and build a complete old-index to new-index map."""

    removed = frozenset(duplicate_to_master)
    compacted: list[Bone] = []
    old_to_new: dict[int, int] = {}

    for old_index, bone in enumerate(bones):
        if old_index in removed:
            continue
        old_to_new[old_index] = len(compacted)
        compacted.append(bone)

    for duplicate_index, master_index in duplicate_to_master.items():
        try:
            old_to_new[duplicate_index] = old_to_new[master_index]
        except KeyError as exc:
            raise VertexBoneOptimizationError(
                f"Duplicate bone {duplicate_index} references missing master "
                f"{master_index}"
            ) from exc

    if len(old_to_new) != len(bones):
        missing = tuple(sorted(set(range(len(bones))) - set(old_to_new)))
        raise VertexBoneOptimizationError(
            f"Bone compaction did not map every original index; missing={missing}"
        )
    return tuple(compacted), old_to_new


def _remap_weighted_stream(
    attachment: MeshAttachment,
    old_to_new: Mapping[int, int],
) -> tuple[MeshAttachment, int]:
    """Remap only bone indices while preserving every local coordinate and weight."""

    decoded = decode_weighted_vertices(
        attachment.vertices,
        expected_vertex_count=len(attachment.uvs) // 2,
    )
    remapped_count = 0
    remapped_vertices: list[WeightedVertex] = []

    for vertex in decoded:
        influences: list[WeightedVertexInfluence] = []
        for influence in vertex.influences:
            try:
                new_index = old_to_new[influence.bone_index]
            except KeyError as exc:
                raise VertexBoneOptimizationError(
                    f"Weighted stream references unmapped bone index "
                    f"{influence.bone_index}"
                ) from exc
            if new_index != influence.bone_index:
                remapped_count += 1
            influences.append(
                WeightedVertexInfluence(
                    bone_index=new_index,
                    x=influence.x,
                    y=influence.y,
                    weight=influence.weight,
                )
            )
        remapped_vertices.append(WeightedVertex(tuple(influences)))

    return (
        replace(attachment, vertices=encode_weighted_vertices(remapped_vertices)),
        remapped_count,
    )


def _rebuild_components(
    build: LegacyMeshDocumentBuildResult,
    compacted_bones: Tuple[Bone, ...],
    old_to_new: Mapping[int, int],
) -> tuple[Tuple[LegacyAttachmentComponent, ...], int]:
    """Rebuild component attachments and per-vertex canonical bone references."""

    rebuilt: list[LegacyAttachmentComponent] = []
    remapped_influence_count = 0

    for component_index, component in enumerate(build.components):
        attachment, remapped_count = _remap_weighted_stream(
            component.attachment,
            old_to_new,
        )
        remapped_influence_count += remapped_count
        decoded = decode_weighted_vertices(
            attachment.vertices,
            expected_vertex_count=len(component.request.vertices),
        )
        indices = tuple(
            weighted_vertex.influences[0].bone_index
            for weighted_vertex in decoded
        )
        if not indices:
            raise VertexBoneOptimizationError(
                f"Component {component_index} contains no weighted vertices"
            )
        try:
            vertex_bones = tuple(compacted_bones[index] for index in indices)
        except IndexError as exc:
            raise VertexBoneOptimizationError(
                f"Component {component_index} remapped outside compacted bone range"
            ) from exc

        rebuilt.append(
            replace(
                component,
                vertex_bone_start_index=indices[0],
                vertex_bones=vertex_bones,
                attachment=attachment,
            )
        )
    return tuple(rebuilt), remapped_influence_count


def _rebuild_skins(
    skins: Tuple[Skin, ...],
    components: Tuple[LegacyAttachmentComponent, ...],
) -> Tuple[Skin, ...]:
    """Replace component attachments in every skin while retaining all other data."""

    replacements = {
        (
            component.request.skin_name,
            component.request.slot_name,
            component.request.attachment_name,
        ): component.attachment
        for component in components
    }
    rebuilt: list[Skin] = []

    for skin in skins:
        skin_attachments: dict[str, dict[str, MeshAttachment | Mapping[str, Any]]] = {}
        for slot_name, attachments in skin.attachments.items():
            slot_attachments: dict[str, MeshAttachment | Mapping[str, Any]] = {}
            for attachment_name, attachment in attachments.items():
                slot_attachments[attachment_name] = replacements.get(
                    (skin.name, slot_name, attachment_name),
                    attachment,
                )
            skin_attachments[slot_name] = slot_attachments
        rebuilt.append(replace(skin, attachments=skin_attachments))
    return tuple(rebuilt)


def optimize_shared_vertex_bones(
    build: LegacyMeshDocumentBuildResult,
) -> LegacyMeshDocumentBuildResult:
    """Share equivalent segment vertex bones and remap all generated mesh weights.

    The operation is deterministic and idempotent.  Geometry arrays, UV arrays,
    triangle order, hull, edges, attachment paths, local influence coordinates, and
    influence weights are preserved exactly.
    """

    if not isinstance(build, LegacyMeshDocumentBuildResult):
        raise TypeError("build must be LegacyMeshDocumentBuildResult")

    try:
        component_indices = _decode_component_bindings(build)
        duplicate_to_master = _duplicate_to_master_map(
            build.document,
            component_indices,
        )
        if not duplicate_to_master:
            logger.debug("Vertex-bone optimization found no shareable duplicates")
            return build

        compacted_bones, old_to_new = _compact_bones(
            build.document.bones,
            duplicate_to_master,
        )
        components, remapped_influence_count = _rebuild_components(
            build,
            compacted_bones,
            old_to_new,
        )
        skins = _rebuild_skins(build.skins, components)
        document = replace(
            build.document,
            bones=compacted_bones,
            skins=skins,
        )
        optimized = replace(
            build,
            components=components,
            skins=skins,
            document=document,
        )
        SpineValidator().validate_or_raise(optimized.document)

        logger.info(
            "Shared vertex-bone optimization removed %d duplicate bone(s), "
            "remapped %d weighted influence(s), and reduced the document from %d "
            "to %d bones",
            len(duplicate_to_master),
            remapped_influence_count,
            len(build.document.bones),
            len(compacted_bones),
        )
        return optimized
    except VertexBoneOptimizationError:
        logger.exception("Shared vertex-bone optimization failed")
        raise
    except Exception as exc:
        logger.exception("Unexpected shared vertex-bone optimization failure")
        raise VertexBoneOptimizationError(
            f"Unable to optimize generated vertex bones: {exc}"
        ) from exc


__all__ = [
    "VertexBoneOptimizationError",
    "optimize_shared_vertex_bones",
]
