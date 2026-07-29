"""Validate exact material, attachment, and weighted-vertex correspondence for A1.

The object-bake pipeline owns several independent index streams: source loops,
UV-specific attachment vertices, triangle corners, vertex bones, and Spine's compact
weighted-vertex stream. A visually plausible bake is not sufficient evidence that
those streams still describe the same corners. This module validates the exact
relationships after projection and again after the final Spine component is built.

Attachment topology is defined in the projected local pixel plane stored in
``LegacyAttachmentVertex.bone_position_pixels``. Z-group parent transforms belong to
the animated rig and may legitimately collapse or unfold triangles in a particular
Spine pose; they must never redefine attachment hull membership or topology validity.
"""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING, Tuple

from ..domain.spine import LegacyAttachmentVertex, LegacyRigBuildResult
from ..domain.spine.legacy_attachment_builder import LegacyMeshDocumentBuildResult
from ..domain.spine.weighted_vertices import decode_weighted_vertices

if TYPE_CHECKING:
    from .a1_attachment_projection_service import A1AttachmentProjectionResult


Position2D = Tuple[float, float]


class A1MaterialCorrespondenceError(ValueError):
    """Raised when one projected corner no longer matches its Spine representation."""


def _require_projection(value: object) -> A1AttachmentProjectionResult:
    """Resolve the normalized public projection type without a module import cycle."""

    from .a1_attachment_projection_service import A1AttachmentProjectionResult

    if not isinstance(value, A1AttachmentProjectionResult):
        raise TypeError("projection must be A1AttachmentProjectionResult")
    return value


def _finite_position(value: object, *, label: str) -> Position2D:
    if not isinstance(value, tuple) or len(value) != 2:
        raise A1MaterialCorrespondenceError(f"{label} must contain two coordinates")
    try:
        resolved = tuple(float(component) for component in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise A1MaterialCorrespondenceError(
            f"{label} must contain numeric coordinates"
        ) from exc
    if not all(isfinite(component) for component in resolved):
        raise A1MaterialCorrespondenceError(f"{label} contains a non-finite coordinate")
    return resolved[0], resolved[1]


def _z_offset_by_index(rig: LegacyRigBuildResult) -> dict[int, float]:
    if not isinstance(rig, LegacyRigBuildResult):
        raise TypeError("rig must be LegacyRigBuildResult")

    offsets: dict[int, float] = {}
    for group in rig.info.z_groups:
        index = int(group.index)
        if index in offsets:
            raise A1MaterialCorrespondenceError(
                f"Rig contains duplicate Z-group index {index}"
            )
        offset = float(group.y_offset_pixels)
        if not isfinite(offset):
            raise A1MaterialCorrespondenceError(
                f"Rig Z-group {index} has a non-finite Y offset"
            )
        offsets[index] = offset
    if not offsets:
        raise A1MaterialCorrespondenceError("Rig contains no Z-group offsets")
    return offsets


def _attachment_projection_positions(
    vertices: Tuple[LegacyAttachmentVertex, ...],
    rig: LegacyRigBuildResult,
) -> Tuple[Position2D, ...]:
    """Return the local projected positions that own attachment topology.

    Every vertex must still reference a real Z-group, but the parent translation is
    deliberately not applied here. The rig can fold a valid attachment into a
    collinear setup pose and unfold it through controls; topology and hull validation
    therefore remain in the stable local projection plane used before rig animation.
    """

    if not isinstance(vertices, tuple) or not vertices:
        raise ValueError("vertices must be a non-empty tuple")
    if not all(isinstance(vertex, LegacyAttachmentVertex) for vertex in vertices):
        raise TypeError("vertices must contain LegacyAttachmentVertex values")

    valid_z_groups = _z_offset_by_index(rig)
    positions: list[Position2D] = []
    for vertex_index, vertex in enumerate(vertices):
        if vertex.z_group_index not in valid_z_groups:
            raise A1MaterialCorrespondenceError(
                f"Attachment vertex {vertex_index} references unknown Z-group index "
                f"{vertex.z_group_index}; available={tuple(sorted(valid_z_groups))}"
            )
        positions.append(
            _finite_position(
                vertex.bone_position_pixels,
                label=f"vertices[{vertex_index}].bone_position_pixels",
            )
        )
    return tuple(positions)


def attachment_setup_positions(
    vertices: Tuple[LegacyAttachmentVertex, ...],
    rig: LegacyRigBuildResult,
) -> Tuple[Position2D, ...]:
    """Return diagnostic positions after applying Z-group setup translations.

    This representation describes one rig pose only. It is useful for diagnostics,
    but it is intentionally not used to validate attachment triangle area or physical
    hull membership because valid animated rigs may collapse triangles in that pose.
    """

    if not isinstance(vertices, tuple) or not vertices:
        raise ValueError("vertices must be a non-empty tuple")
    if not all(isinstance(vertex, LegacyAttachmentVertex) for vertex in vertices):
        raise TypeError("vertices must contain LegacyAttachmentVertex values")

    offsets = _z_offset_by_index(rig)
    positions: list[Position2D] = []
    for vertex_index, vertex in enumerate(vertices):
        local_x, local_y = _finite_position(
            vertex.bone_position_pixels,
            label=f"vertices[{vertex_index}].bone_position_pixels",
        )
        try:
            z_offset = offsets[vertex.z_group_index]
        except KeyError as exc:
            raise A1MaterialCorrespondenceError(
                f"Attachment vertex {vertex_index} references unknown Z-group index "
                f"{vertex.z_group_index}; available={tuple(sorted(offsets))}"
            ) from exc
        setup_y = local_y + z_offset
        if not isfinite(setup_y):
            raise A1MaterialCorrespondenceError(
                f"Attachment vertex {vertex_index} setup Y is non-finite"
            )
        positions.append((local_x, setup_y))
    return tuple(positions)


def validate_projection_material_correspondence(
    projection: A1AttachmentProjectionResult,
    rig: LegacyRigBuildResult,
) -> Tuple[Position2D, ...]:
    """Validate projection identity and return stable local topology positions."""

    resolved_projection = _require_projection(projection)
    if not isinstance(rig, LegacyRigBuildResult):
        raise TypeError("rig must be LegacyRigBuildResult")

    request = resolved_projection.request
    if len(request.vertices) != len(resolved_projection.ordered_vertex_keys):
        raise A1MaterialCorrespondenceError(
            "Projection vertex count does not match ordered UV-specific keys"
        )

    for vertex_index, (vertex, key) in enumerate(
        zip(request.vertices, resolved_projection.ordered_vertex_keys, strict=True)
    ):
        if vertex.index != vertex_index:
            raise A1MaterialCorrespondenceError(
                f"Projection vertex {vertex_index} stores index {vertex.index}"
            )
        if vertex.uv != key.uv:
            raise A1MaterialCorrespondenceError(
                f"Projection vertex {vertex_index} UV {vertex.uv} does not match "
                f"its source-loop key UV {key.uv}"
            )

    mapped_indices = tuple(
        attachment_index
        for _loop_id, attachment_index in resolved_projection.loop_to_attachment_index
    )
    if mapped_indices != request.triangles:
        raise A1MaterialCorrespondenceError(
            "Projection loop-to-attachment mapping does not match triangle corners"
        )

    return _attachment_projection_positions(request.vertices, rig)


def validate_document_material_correspondence(
    projections: Tuple[A1AttachmentProjectionResult, ...],
    document_build: LegacyMeshDocumentBuildResult,
) -> None:
    """Validate that final Spine components preserve every projected index stream.

    Shared segment-boundary vertices may now reference a canonical bone introduced by
    another component. Therefore the weighted stream is the authoritative final index
    map; the old contiguous ``start + vertex`` assumption is intentionally rejected.
    """

    if not isinstance(projections, tuple) or not projections:
        raise ValueError("projections must be a non-empty tuple")
    resolved_projections = tuple(_require_projection(item) for item in projections)
    if not isinstance(document_build, LegacyMeshDocumentBuildResult):
        raise TypeError("document_build must be LegacyMeshDocumentBuildResult")
    if len(resolved_projections) != len(document_build.components):
        raise A1MaterialCorrespondenceError(
            f"Built {len(document_build.components)} Spine components for "
            f"{len(resolved_projections)} projections"
        )

    document_bones = document_build.document.bones
    for component_index, (projection, component) in enumerate(
        zip(resolved_projections, document_build.components, strict=True)
    ):
        request = projection.request
        if component.request != request:
            raise A1MaterialCorrespondenceError(
                f"Component {component_index} request differs from its projection"
            )

        expected_uvs = tuple(
            float(component_value)
            for vertex in request.vertices
            for component_value in vertex.uv
        )
        attachment = component.attachment
        if tuple(attachment.uvs) != expected_uvs:
            raise A1MaterialCorrespondenceError(
                f"Component {component_index} serialized UV order differs from "
                "projection vertex order"
            )
        if tuple(attachment.triangles) != request.triangles:
            raise A1MaterialCorrespondenceError(
                f"Component {component_index} serialized triangles differ from "
                "projection triangle corners"
            )
        if int(attachment.hull) != request.hull:
            raise A1MaterialCorrespondenceError(
                f"Component {component_index} serialized hull differs from projection"
            )
        if tuple(attachment.edges) != request.edges:
            raise A1MaterialCorrespondenceError(
                f"Component {component_index} serialized edges differ from projection"
            )

        weighted_vertices = decode_weighted_vertices(
            attachment.vertices,
            expected_vertex_count=len(request.vertices),
        )
        first_bone_index: int | None = None
        for vertex_index, weighted_vertex in enumerate(weighted_vertices):
            if len(weighted_vertex.influences) != 1:
                raise A1MaterialCorrespondenceError(
                    f"Component {component_index} vertex {vertex_index} has "
                    f"{len(weighted_vertex.influences)} influences instead of one"
                )
            influence = weighted_vertex.influences[0]
            actual_bone_index = influence.bone_index
            if first_bone_index is None:
                first_bone_index = actual_bone_index
            if (influence.x, influence.y, influence.weight) != (0.0, 0.0, 1.0):
                raise A1MaterialCorrespondenceError(
                    f"Component {component_index} vertex {vertex_index} has unexpected "
                    f"weighted local data {(influence.x, influence.y, influence.weight)}"
                )
            try:
                document_bone = document_bones[actual_bone_index]
                component_bone = component.vertex_bones[vertex_index]
            except IndexError as exc:
                raise A1MaterialCorrespondenceError(
                    f"Component {component_index} vertex {vertex_index} bone index "
                    f"{actual_bone_index} is outside the final document"
                ) from exc
            if document_bone != component_bone:
                raise A1MaterialCorrespondenceError(
                    f"Component {component_index} vertex {vertex_index} references "
                    f"bone {actual_bone_index}, which does not match its canonical "
                    "component bone"
                )

        if first_bone_index is None:
            raise A1MaterialCorrespondenceError(
                f"Component {component_index} contains no weighted vertices"
            )
        if component.vertex_bone_start_index != first_bone_index:
            raise A1MaterialCorrespondenceError(
                f"Component {component_index} start index "
                f"{component.vertex_bone_start_index} differs from its first weighted "
                f"bone index {first_bone_index}"
            )

        validate_projection_material_correspondence(projection, document_build.rig)


__all__ = [
    "A1MaterialCorrespondenceError",
    "Position2D",
    "attachment_setup_positions",
    "validate_document_material_correspondence",
    "validate_projection_material_correspondence",
]
