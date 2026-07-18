"""Build one root-bound grouped B4 overlay while preserving connected rig structure."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Any, Mapping, Tuple

from ..domain.baking import CameraProjectionLayout, GroupedCameraProjectionPlan
from ..domain.spine import (
    LegacyAttachmentSequence,
    MeshAttachment,
    Skin,
    Slot,
    SpineDocument,
    SpineValidator,
)


class GroupedCameraOverlayError(ValueError):
    """Raised when a connected document cannot receive a grouped camera overlay."""


@dataclass(frozen=True, slots=True)
class GroupedCameraOverlayResult:
    document: SpineDocument
    slot_name: str
    attachment_name: str
    hidden_slot_names: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.document, SpineDocument):
            raise TypeError("document must be SpineDocument")
        for field_name in ("slot_name", "attachment_name"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if (
            not isinstance(self.hidden_slot_names, tuple)
            or not self.hidden_slot_names
            or not all(
                isinstance(value, str) and value.strip()
                for value in self.hidden_slot_names
            )
        ):
            raise ValueError("hidden_slot_names must contain non-empty strings")


def _normalized_pair(first: int, second: int) -> tuple[int, int]:
    if first == second:
        raise GroupedCameraOverlayError(
            f"grouped attachment edge repeats vertex {first}"
        )
    return (first, second) if first < second else (second, first)


def _attachment_edges(
    vertex_count: int,
    triangles: Tuple[tuple[int, int, int], ...],
) -> Tuple[int, ...]:
    boundary = tuple((index, (index + 1) % vertex_count) for index in range(vertex_count))
    result: list[int] = []
    seen: set[tuple[int, int]] = set()
    for first, second in boundary:
        seen.add(_normalized_pair(first, second))
        result.extend((first, second))
    for triangle in triangles:
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            pair = _normalized_pair(first, second)
            if pair in seen:
                continue
            seen.add(pair)
            result.extend((first, second))
    expected = vertex_count + max(0, vertex_count - 3)
    if len(result) // 2 != expected:
        raise GroupedCameraOverlayError(
            "grouped attachment edge count differs from disk triangulation; "
            f"expected={expected}, actual={len(result) // 2}"
        )
    return tuple(result)


def _attachment_path(
    plan: GroupedCameraProjectionPlan,
    image_relative_directory: str,
) -> str:
    relative = image_relative_directory.replace("\\", "/").strip("/")
    base_name = (
        f"{plan.settings.output_stem}_Baked_"
        if plan.sequence
        else plan.representative_task.image_name
    )
    return (
        PurePosixPath(relative, base_name).as_posix()
        if relative
        else base_name
    )


def _attachment_sequence(
    plan: GroupedCameraProjectionPlan,
) -> dict[str, int] | None:
    if not plan.sequence:
        return None
    return dict(
        LegacyAttachmentSequence(
            count=plan.settings.sequence_frame_count,
            start=plan.settings.sequence_start_frame,
            digits=plan.settings.sequence_frame_digits,
        ).to_spine_mapping()
    )


def _build_grouped_attachment(
    plan: GroupedCameraProjectionPlan,
    layout: CameraProjectionLayout,
    *,
    attachment_name: str,
    image_relative_directory: str,
) -> MeshAttachment:
    if not isinstance(plan, GroupedCameraProjectionPlan):
        raise TypeError("plan must be GroupedCameraProjectionPlan")
    if not isinstance(layout, CameraProjectionLayout):
        raise TypeError("layout must be CameraProjectionLayout")
    if layout.frame_count != len(plan.frame_tasks):
        raise GroupedCameraOverlayError(
            "grouped layout frame count does not match grouped plan"
        )
    if (
        layout.full_width != plan.settings.width
        or layout.full_height != plan.settings.height
    ):
        raise GroupedCameraOverlayError(
            "grouped layout dimensions do not match grouped plan"
        )

    triangles = layout.triangle_indices
    uvs = tuple(
        component
        for point in layout.hull
        for component in layout.spine_uv(point)
    )
    vertices = tuple(
        component
        for point in layout.hull
        for component in (
            layout.spine_position_pixels(point)[0],
            -layout.spine_position_pixels(point)[1],
        )
    )
    triangle_values = tuple(index for triangle in triangles for index in triangle)
    return MeshAttachment(
        name=attachment_name,
        path=_attachment_path(plan, image_relative_directory),
        uvs=uvs,
        triangles=triangle_values,
        vertices=vertices,
        hull=len(layout.hull),
        edges=_attachment_edges(len(layout.hull), triangles),
        width=float(layout.cropped_width),
        height=float(layout.cropped_height),
        sequence=_attachment_sequence(plan),
        extras={
            "spine2dGroupedCamera": True,
            "spine2dGroupedSourceCount": len(plan.source_object_ids),
        },
    )


def _strip_hidden_slot_timelines(
    animations: Mapping[str, Any],
    hidden_slot_names: set[str],
) -> dict[str, Any]:
    """Remove only source-slot timelines that could reveal hidden grouped layers.

    Bone, constraint, deform, draw-order and event timelines remain untouched. Grouped output
    flattens visual source slots into the rendered sequence, so color or attachment timelines on
    those slots must not be allowed to make the compatibility layers visible again.
    """

    result: dict[str, Any] = {}
    for animation_name, animation_payload in animations.items():
        if not isinstance(animation_payload, Mapping):
            result[str(animation_name)] = animation_payload
            continue
        copied_payload = dict(animation_payload)
        slot_timelines = copied_payload.get("slots")
        if isinstance(slot_timelines, Mapping):
            retained = {
                str(slot_name): timeline
                for slot_name, timeline in slot_timelines.items()
                if str(slot_name) not in hidden_slot_names
            }
            if retained:
                copied_payload["slots"] = retained
            else:
                copied_payload.pop("slots", None)
        result[str(animation_name)] = copied_payload
    return result


def apply_grouped_camera_overlay(
    document: SpineDocument,
    plan: GroupedCameraProjectionPlan,
    layout: CameraProjectionLayout,
    *,
    visual_slot_names: Tuple[str, ...],
    image_relative_directory: str,
    slot_name: str,
    attachment_name: str,
    skin_name: str = "default",
) -> GroupedCameraOverlayResult:
    """Hide individual B4 visuals and append one depth-correct root overlay.

    Source bones, constraints, attachments and non-slot animation data remain in the typed
    document. Source visual slots receive fully transparent setup color, and their slot color/
    attachment timelines are removed so they cannot reappear over the grouped render. The
    grouped slot is appended last inside the connected document and becomes the only visible B4
    layer.
    """

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    if not isinstance(plan, GroupedCameraProjectionPlan):
        raise TypeError("plan must be GroupedCameraProjectionPlan")
    if not isinstance(layout, CameraProjectionLayout):
        raise TypeError("layout must be CameraProjectionLayout")
    if (
        not isinstance(visual_slot_names, tuple)
        or len(visual_slot_names) != len(plan.source_object_ids)
        or not all(
            isinstance(value, str) and value.strip()
            for value in visual_slot_names
        )
    ):
        raise ValueError(
            "visual_slot_names must match grouped source objects"
        )
    if len(visual_slot_names) != len(set(visual_slot_names)):
        raise ValueError("visual_slot_names must be unique")
    for field_name, value in (
        ("slot_name", slot_name),
        ("attachment_name", attachment_name),
        ("skin_name", skin_name),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
    if not isinstance(image_relative_directory, str):
        raise TypeError("image_relative_directory must be str")

    bone_names = {bone.name for bone in document.bones}
    if "root" not in bone_names:
        raise GroupedCameraOverlayError(
            "connected document has no shared root bone for grouped B4 overlay"
        )
    existing_slot_names = {slot.name for slot in document.slots}
    missing = tuple(name for name in visual_slot_names if name not in existing_slot_names)
    if missing:
        raise GroupedCameraOverlayError(
            f"grouped B4 source slots are missing from connected document: {missing}"
        )
    if slot_name in existing_slot_names:
        raise GroupedCameraOverlayError(
            f"grouped B4 slot '{slot_name}' already exists"
        )
    if any(
        attachment_name in slot_attachments
        for skin in document.skins
        for slot_attachments in skin.attachments.values()
    ):
        raise GroupedCameraOverlayError(
            f"grouped B4 attachment '{attachment_name}' already exists"
        )

    hidden_set = set(visual_slot_names)
    slots = tuple(
        replace(slot, color="ffffff00") if slot.name in hidden_set else slot
        for slot in document.slots
    ) + (
        Slot(
            name=slot_name,
            bone="root",
            attachment=attachment_name,
            extras={"spine2dGroupedCamera": True},
        ),
    )
    attachment = _build_grouped_attachment(
        plan,
        layout,
        attachment_name=attachment_name,
        image_relative_directory=image_relative_directory,
    )

    skins: list[Skin] = []
    target_found = False
    for skin in document.skins:
        attachments = {
            current_slot: dict(current_attachments)
            for current_slot, current_attachments in skin.attachments.items()
        }
        if skin.name == skin_name:
            target_found = True
            attachments[slot_name] = {attachment_name: attachment}
        skins.append(replace(skin, attachments=attachments))
    if not target_found:
        skins.append(
            Skin(
                name=skin_name,
                attachments={slot_name: {attachment_name: attachment}},
            )
        )

    result_document = replace(
        document,
        slots=slots,
        skins=tuple(skins),
        animations=_strip_hidden_slot_timelines(
            document.animations,
            hidden_set,
        ),
        extras={
            **dict(document.extras),
            "spine2dGroupedCamera": {
                "group": plan.group_id,
                "sources": plan.source_object_ids,
                "slot": slot_name,
            },
        },
    )
    SpineValidator().validate_or_raise(result_document)
    return GroupedCameraOverlayResult(
        document=result_document,
        slot_name=slot_name,
        attachment_name=attachment_name,
        hidden_slot_names=visual_slot_names,
    )


__all__ = [
    "GroupedCameraOverlayError",
    "GroupedCameraOverlayResult",
    "apply_grouped_camera_overlay",
]
