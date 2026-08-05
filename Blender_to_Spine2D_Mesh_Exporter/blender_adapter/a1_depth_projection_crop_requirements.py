"""Collect reserve-view UV envelopes before camera images are physically cropped."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from ..domain.baking import ProjectionUvBounds
from ..domain.spine import MeshAttachment
from .a1_object_preparation import PreparedA1Object, PreparedDepthA1Object


_FRONT_VIEW_ID = "FRONT"


class A1DepthProjectionCropRequirementError(ValueError):
    """Raised when a prepared Depth document has inconsistent view-owned UV data."""


def depth_projection_view_id_for_slot(prefix: str, slot_name: str) -> str:
    """Resolve the established FRONT or ``<prefix>_Parallax_<VIEW>`` slot owner."""

    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string")
    if not isinstance(slot_name, str) or not slot_name.strip():
        raise ValueError("slot_name must be a non-empty string")
    reserve_prefix = f"{prefix}_Parallax_"
    if slot_name.startswith(reserve_prefix):
        view_id = slot_name[len(reserve_prefix) :].strip().upper()
        if not view_id:
            raise A1DepthProjectionCropRequirementError(
                f"Reserve slot {slot_name!r} has no view id"
            )
        return view_id
    return _FRONT_VIEW_ID


def _attachment_uvs(
    prepared: PreparedDepthA1Object,
    *,
    skin_name: str,
    slot_name: str,
    attachment_name: str,
) -> tuple[tuple[float, float], ...]:
    for skin in prepared.document_assembly.document_build.skins:
        if skin.name != skin_name:
            continue
        attachments = skin.attachments.get(slot_name)
        if not isinstance(attachments, Mapping):
            break
        attachment = attachments.get(attachment_name)
        if not isinstance(attachment, MeshAttachment):
            break
        if len(attachment.uvs) % 2 != 0:
            raise A1DepthProjectionCropRequirementError(
                "Prepared Depth attachment has an odd UV stream: "
                f"{skin_name}/{slot_name}/{attachment_name}"
            )
        return tuple(
            (
                float(attachment.uvs[index]),
                float(attachment.uvs[index + 1]),
            )
            for index in range(0, len(attachment.uvs), 2)
        )
    raise A1DepthProjectionCropRequirementError(
        "Prepared Depth document lost attachment while collecting crop requirements: "
        f"{skin_name}/{slot_name}/{attachment_name}"
    )


def depth_projection_required_uv_bounds(
    prepared: PreparedA1Object,
) -> Mapping[str, ProjectionUvBounds] | None:
    """Return exact UV bounds for reserve views without changing FRONT-only output.

    Non-Depth objects and Depth objects without reserve plans return ``None``. The active
    FRONT view therefore retains the established alpha-only crop, including the complete
    zero-horizon compatibility path. Positive parallax views receive geometry-required
    bounds because compact proxy UV envelopes can legitimately exceed rendered alpha.
    """

    if not isinstance(prepared, PreparedA1Object):
        raise TypeError("prepared must be PreparedA1Object")
    if not isinstance(prepared, PreparedDepthA1Object):
        return None

    expected_all = {_FRONT_VIEW_ID}
    expected_reserve = {
        plan.view_id.strip().upper()
        for plan in prepared.reserve_bake_plans
    }
    expected_all.update(expected_reserve)

    resolved_all: set[str] = set()
    resolved_reserve: dict[str, ProjectionUvBounds] = {}
    for projection_index, projection in enumerate(
        prepared.document_assembly.projections
    ):
        request = projection.request
        view_id = depth_projection_view_id_for_slot(
            prepared.prefix,
            request.slot_name,
        )
        if view_id in resolved_all:
            raise A1DepthProjectionCropRequirementError(
                f"Prepared Depth document contains duplicate projection view {view_id}"
            )
        resolved_all.add(view_id)

        if view_id == _FRONT_VIEW_ID:
            continue

        values: list[tuple[float, float]] = [
            (float(vertex.uv[0]), float(vertex.uv[1]))
            for vertex in request.vertices
        ]
        values.extend(
            (float(key.uv[0]), float(key.uv[1]))
            for key in projection.ordered_vertex_keys
        )
        values.extend(
            _attachment_uvs(
                prepared,
                skin_name=request.skin_name,
                slot_name=request.slot_name,
                attachment_name=request.attachment_name,
            )
        )
        resolved_reserve[view_id] = ProjectionUvBounds.from_uvs(
            values,
            field_name=(
                f"projections[{projection_index}].{view_id}.required_uvs"
            ),
        )

    missing = tuple(sorted(expected_all - resolved_all))
    unknown = tuple(sorted(resolved_all - expected_all))
    if missing or unknown:
        raise A1DepthProjectionCropRequirementError(
            "Prepared Depth projection views do not match texture plans; "
            f"missing={missing}, unknown={unknown}"
        )
    if set(resolved_reserve) != expected_reserve:
        raise A1DepthProjectionCropRequirementError(
            "Prepared Depth reserve UV bounds do not match reserve plans; "
            f"required={tuple(sorted(expected_reserve))}, "
            f"resolved={tuple(sorted(resolved_reserve))}"
        )
    if not resolved_reserve:
        return None
    return MappingProxyType(resolved_reserve)


__all__ = [
    "A1DepthProjectionCropRequirementError",
    "depth_projection_required_uv_bounds",
    "depth_projection_view_id_for_slot",
]
