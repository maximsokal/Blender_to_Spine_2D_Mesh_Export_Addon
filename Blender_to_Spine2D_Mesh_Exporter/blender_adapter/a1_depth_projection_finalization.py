"""Finalize Depth Camera Projection texture crop without flattening its rig."""

from __future__ import annotations

from dataclasses import replace
import logging
from types import MappingProxyType
from typing import Mapping

from ..application import A1AttachmentVertexKey
from ..domain.baking import (
    A1TextureExportMode,
    CameraProjectionLayout,
)
from ..domain.spine import (
    LegacyAttachmentVertex,
    LegacyMeshAttachmentRequest,
    MeshAttachment,
    Skin,
    SpineDocument,
    SpineValidator,
)
from .a1_object_preparation import PreparedA1Object


logger = logging.getLogger(__name__)
_UV_EPSILON = 1.0e-6


class A1DepthProjectionFinalizationError(ValueError):
    """Raised when render crop cannot represent the prepared depth surface."""


def _clamped_unit(value: float, field_name: str) -> float:
    resolved = float(value)
    if resolved < -_UV_EPSILON or resolved > 1.0 + _UV_EPSILON:
        raise A1DepthProjectionFinalizationError(
            f"{field_name}={resolved} lies outside the camera render crop. The "
            "rendered alpha does not cover the generated depth surface; use opaque "
            "source materials or reduce the depth surface extent."
        )
    if resolved <= 0.0:
        return 0.0
    if resolved >= 1.0:
        return 1.0
    return resolved


def _crop_uv(
    uv: tuple[float, float],
    layout: CameraProjectionLayout,
    *,
    field_name: str,
) -> tuple[float, float]:
    if not isinstance(uv, tuple) or len(uv) != 2:
        raise TypeError(f"{field_name} must contain two values")
    full_x = float(uv[0]) * float(layout.full_width)
    full_y = (1.0 - float(uv[1])) * float(layout.full_height)
    cropped_u = (
        full_x - float(layout.crop.minimum_x)
    ) / float(layout.cropped_width)
    cropped_v = 1.0 - (
        full_y - float(layout.crop.minimum_y)
    ) / float(layout.cropped_height)
    return (
        _clamped_unit(cropped_u, f"{field_name}[0]"),
        _clamped_unit(cropped_v, f"{field_name}[1]"),
    )


def _remap_request(
    request: LegacyMeshAttachmentRequest,
    layout: CameraProjectionLayout,
) -> LegacyMeshAttachmentRequest:
    if not isinstance(request, LegacyMeshAttachmentRequest):
        raise TypeError("request must be LegacyMeshAttachmentRequest")
    vertices = tuple(
        replace(
            vertex,
            uv=_crop_uv(
                vertex.uv,
                layout,
                field_name=f"request.vertices[{index}].uv",
            ),
        )
        for index, vertex in enumerate(request.vertices)
    )
    if not all(isinstance(vertex, LegacyAttachmentVertex) for vertex in vertices):
        raise TypeError("remapped request vertices lost LegacyAttachmentVertex type")
    return replace(
        request,
        vertices=vertices,
        width=float(layout.cropped_width),
        height=float(layout.cropped_height),
    )


def _remap_attachment(
    attachment: MeshAttachment,
    layout: CameraProjectionLayout,
    *,
    path: str,
) -> MeshAttachment:
    if not isinstance(attachment, MeshAttachment):
        raise TypeError("attachment must be MeshAttachment")
    if len(attachment.uvs) % 2 != 0:
        raise A1DepthProjectionFinalizationError(
            f"Attachment {path} has an odd UV stream"
        )
    remapped: list[float] = []
    for vertex_index in range(len(attachment.uvs) // 2):
        uv = (
            float(attachment.uvs[vertex_index * 2]),
            float(attachment.uvs[vertex_index * 2 + 1]),
        )
        cropped = _crop_uv(
            uv,
            layout,
            field_name=f"{path}.uvs[{vertex_index}]",
        )
        remapped.extend(cropped)
    return replace(
        attachment,
        uvs=tuple(remapped),
        width=float(layout.cropped_width),
        height=float(layout.cropped_height),
    )


def _remap_skins(
    skins: tuple[Skin, ...],
    layout: CameraProjectionLayout,
) -> tuple[Skin, ...]:
    resolved: list[Skin] = []
    for skin in skins:
        groups: dict[str, dict[str, MeshAttachment | Mapping[str, object]]] = {}
        for slot_name, attachments in skin.attachments.items():
            group: dict[str, MeshAttachment | Mapping[str, object]] = {}
            for attachment_name, attachment in attachments.items():
                if isinstance(attachment, MeshAttachment):
                    group[attachment_name] = _remap_attachment(
                        attachment,
                        layout,
                        path=f"{skin.name}/{slot_name}/{attachment_name}",
                    )
                else:
                    group[attachment_name] = attachment
            groups[slot_name] = group
        resolved.append(replace(skin, attachments=groups))
    return tuple(resolved)


def _attachment_from_skins(
    skins: tuple[Skin, ...],
    request: LegacyMeshAttachmentRequest,
) -> MeshAttachment:
    for skin in skins:
        if skin.name != request.skin_name:
            continue
        group = skin.attachments.get(request.slot_name)
        if not isinstance(group, Mapping):
            break
        attachment = group.get(request.attachment_name)
        if isinstance(attachment, MeshAttachment):
            return attachment
        break
    raise A1DepthProjectionFinalizationError(
        "Final depth document lost component attachment "
        f"{request.skin_name}/{request.slot_name}/{request.attachment_name}"
    )


def _remap_projection_keys(
    projection: object,
    request: LegacyMeshAttachmentRequest,
    layout: CameraProjectionLayout,
) -> object:
    ordered = tuple(
        replace(
            key,
            uv=_crop_uv(
                key.uv,
                layout,
                field_name=(
                    f"projection[{request.slot_name}].ordered_vertex_keys[{index}].uv"
                ),
            ),
        )
        for index, key in enumerate(projection.ordered_vertex_keys)
    )
    if not all(isinstance(key, A1AttachmentVertexKey) for key in ordered):
        raise TypeError("projection keys lost A1AttachmentVertexKey type")
    hull_count = len(projection.hull_vertex_keys)
    return replace(
        projection,
        request=request,
        hull_vertex_keys=ordered[:hull_count],
        ordered_vertex_keys=ordered,
    )


def finalize_prepared_depth_camera_projection(
    prepared: PreparedA1Object,
    layout: CameraProjectionLayout | None,
) -> PreparedA1Object:
    """Apply one camera-render crop while preserving every depth bone and triangle."""

    if not isinstance(prepared, PreparedA1Object):
        raise TypeError("prepared must be PreparedA1Object")
    if (
        prepared.settings.bake_execution.texture_export_mode
        is not A1TextureExportMode.DEPTH_CAMERA_PROJECTION
    ):
        raise ValueError(
            "Depth projection finalization requires DEPTH_CAMERA_PROJECTION mode"
        )
    if not isinstance(layout, CameraProjectionLayout):
        raise A1DepthProjectionFinalizationError(
            "Depth Camera Projection did not produce a camera render layout"
        )

    assembly = prepared.document_assembly
    build = assembly.document_build
    remapped_skins = _remap_skins(build.skins, layout)
    remapped_components = []
    remapped_requests = []
    request_by_slot: dict[str, LegacyMeshAttachmentRequest] = {}
    for component_index, component in enumerate(build.components):
        request = _remap_request(component.request, layout)
        attachment = _attachment_from_skins(remapped_skins, request)
        remapped_components.append(
            replace(
                component,
                request=request,
                attachment=attachment,
            )
        )
        remapped_requests.append(request)
        request_by_slot[request.slot_name] = request

        if component.attachment.triangles != attachment.triangles:
            raise A1DepthProjectionFinalizationError(
                f"Component {component_index} triangles changed during crop remap"
            )
        if component.attachment.vertices != attachment.vertices:
            raise A1DepthProjectionFinalizationError(
                f"Component {component_index} weighted vertices changed during crop remap"
            )
        if component.attachment.hull != attachment.hull:
            raise A1DepthProjectionFinalizationError(
                f"Component {component_index} hull changed during crop remap"
            )

    document = replace(build.document, skins=remapped_skins)
    SpineValidator().validate_or_raise(document)
    remapped_build = replace(
        build,
        requests=tuple(remapped_requests),
        components=tuple(remapped_components),
        skins=remapped_skins,
        document=document,
    )
    remapped_projections = tuple(
        _remap_projection_keys(
            projection,
            request_by_slot[projection.request.slot_name],
            layout,
        )
        for projection in assembly.projections
    )
    remapped_assembly = replace(
        assembly,
        settings=replace(
            assembly.settings,
            attachment_width=float(layout.cropped_width),
            attachment_height=float(layout.cropped_height),
        ),
        projections=remapped_projections,
        document_build=remapped_build,
    )
    statistics = MappingProxyType(
        {
            **dict(prepared.statistics),
            "depth_projection_crop_minimum_x": layout.crop.minimum_x,
            "depth_projection_crop_minimum_y": layout.crop.minimum_y,
            "depth_projection_crop_maximum_x": layout.crop.maximum_x,
            "depth_projection_crop_maximum_y": layout.crop.maximum_y,
            "depth_projection_cropped_width": layout.cropped_width,
            "depth_projection_cropped_height": layout.cropped_height,
            "depth_projection_uv_remapped": 1,
            "depth_projection_weighted_topology_preserved": 1,
        }
    )
    logger.debug(
        "Finalized depth camera crop for %s: full=%dx%d crop=(%d,%d)-(%d,%d) "
        "attachments=%d",
        prepared.object_id,
        layout.full_width,
        layout.full_height,
        layout.crop.minimum_x,
        layout.crop.minimum_y,
        layout.crop.maximum_x,
        layout.crop.maximum_y,
        sum(
            len(attachments)
            for skin in remapped_skins
            for attachments in skin.attachments.values()
        ),
    )
    return replace(
        prepared,
        rig=remapped_assembly.rig,
        document_assembly=remapped_assembly,
        statistics=statistics,
    )


__all__ = [
    "A1DepthProjectionFinalizationError",
    "finalize_prepared_depth_camera_projection",
]
