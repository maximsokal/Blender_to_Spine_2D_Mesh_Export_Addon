"""Build A1 vertex bones and weighted mesh attachments without tolerance matching.

The legacy exporter creates one vertex bone for every exported mesh vertex and
binds that vertex to the new bone with one full-weight influence at local (0, 0).
This module preserves that external Spine contract while requiring an explicit
Z-group index and already-transformed pixel position for every vertex.

Single- and multi-attachment documents share the same component builder. Bone
indices are assigned against the final in-memory bone order, so no serialized JSON
merge or weighted-index remap is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Tuple

from .legacy_rig_builder import LegacyRigBuildResult
from .model import Bone, MeshAttachment, Skin, Slot, SpineDocument
from .validator import SpineValidator
from .weighted_vertices import (
    WeightedVertex,
    WeightedVertexInfluence,
    encode_weighted_vertices,
)


@dataclass(frozen=True, slots=True)
class LegacyAttachmentVertex:
    """One exported Spine mesh vertex and its exact legacy vertex-bone binding."""

    index: int
    uv: Tuple[float, float]
    bone_position_pixels: Tuple[float, float]
    z_group_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.index, int) or self.index < 0:
            raise ValueError("index must be a non-negative integer")
        for field_name, value in (
            ("uv", self.uv),
            ("bone_position_pixels", self.bone_position_pixels),
        ):
            if (
                not isinstance(value, tuple)
                or len(value) != 2
                or not all(
                    isinstance(component, (int, float))
                    and isfinite(float(component))
                    for component in value
                )
            ):
                raise ValueError(f"{field_name} must contain two finite values")
        if not isinstance(self.z_group_index, int) or self.z_group_index < 0:
            raise ValueError("z_group_index must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class LegacyAttachmentSequence:
    count: int
    start: int
    digits: int = 4
    setup: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.count, int) or self.count <= 0:
            raise ValueError("count must be a positive integer")
        if not isinstance(self.start, int) or self.start < 0:
            raise ValueError("start must be a non-negative integer")
        if not isinstance(self.digits, int) or not 1 <= self.digits <= 12:
            raise ValueError("digits must be in [1, 12]")
        if self.setup is not None and (
            not isinstance(self.setup, int) or not 0 <= self.setup < self.count
        ):
            raise ValueError("setup must be a valid sequence frame index or None")

    @property
    def resolved_setup(self) -> int:
        if self.setup is not None:
            return self.setup
        return 1 if self.count > 1 else 0

    def to_spine_mapping(self) -> Mapping[str, int]:
        return {
            "count": self.count,
            "start": self.start,
            "digits": self.digits,
            "setup": self.resolved_setup,
        }


@dataclass(frozen=True, slots=True)
class LegacyMeshAttachmentRequest:
    """Complete typed input for one legacy weighted mesh attachment."""

    slot_name: str
    attachment_name: str
    vertex_prefix: str
    image_path: str
    width: float
    height: float
    vertices: Tuple[LegacyAttachmentVertex, ...]
    triangles: Tuple[int, ...]
    hull: int
    edges: Tuple[int, ...] = ()
    sequence: LegacyAttachmentSequence | None = None
    skin_name: str = "default"

    def __post_init__(self) -> None:
        for field_name in (
            "slot_name",
            "attachment_name",
            "vertex_prefix",
            "image_path",
            "skin_name",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        for field_name in ("width", "height"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or not isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
        if not isinstance(self.vertices, tuple) or not self.vertices:
            raise ValueError("vertices must be a non-empty tuple")
        if not all(isinstance(vertex, LegacyAttachmentVertex) for vertex in self.vertices):
            raise TypeError("vertices must contain LegacyAttachmentVertex values")
        actual_indices = tuple(vertex.index for vertex in self.vertices)
        if actual_indices != tuple(range(len(self.vertices))):
            raise ValueError("vertex indices must be ordered and dense from zero")
        if not isinstance(self.triangles, tuple) or len(self.triangles) % 3 != 0:
            raise ValueError("triangles must be a tuple divisible into triples")
        if not isinstance(self.edges, tuple) or len(self.edges) % 2 != 0:
            raise ValueError("edges must be a tuple divisible into pairs")
        if not isinstance(self.hull, int) or not 0 <= self.hull <= len(self.vertices):
            raise ValueError("hull must be in [0, vertex_count]")
        if self.sequence is not None and not isinstance(
            self.sequence, LegacyAttachmentSequence
        ):
            raise TypeError("sequence must be LegacyAttachmentSequence or None")

        vertex_count = len(self.vertices)
        for field_name, values in (
            ("triangles", self.triangles),
            ("edges", self.edges),
        ):
            for value_index, value in enumerate(values):
                if not isinstance(value, int):
                    raise TypeError(f"{field_name}[{value_index}] must be int")
                if value < 0 or value >= vertex_count:
                    raise ValueError(
                        f"{field_name}[{value_index}]={value} is outside "
                        f"[0, {vertex_count})"
                    )


@dataclass(frozen=True, slots=True)
class LegacyAttachmentComponent:
    request: LegacyMeshAttachmentRequest
    vertex_bone_start_index: int
    vertex_bones: Tuple[Bone, ...]
    attachment: MeshAttachment
    slot: Slot

    def __post_init__(self) -> None:
        if not isinstance(self.vertex_bone_start_index, int) or self.vertex_bone_start_index < 0:
            raise ValueError("vertex_bone_start_index must be a non-negative integer")
        if len(self.vertex_bones) != len(self.request.vertices):
            raise ValueError("one vertex bone is required for every attachment vertex")


@dataclass(frozen=True, slots=True)
class LegacyMeshDocumentBuildResult:
    rig: LegacyRigBuildResult
    requests: Tuple[LegacyMeshAttachmentRequest, ...]
    components: Tuple[LegacyAttachmentComponent, ...]
    skins: Tuple[Skin, ...]
    document: SpineDocument

    @property
    def all_vertex_bones(self) -> Tuple[Bone, ...]:
        return tuple(
            bone for component in self.components for bone in component.vertex_bones
        )


@dataclass(frozen=True, slots=True)
class LegacyMeshAttachmentBuildResult:
    rig: LegacyRigBuildResult
    request: LegacyMeshAttachmentRequest
    vertex_bones: Tuple[Bone, ...]
    attachment: MeshAttachment
    slot: Slot
    skin: Skin
    document: SpineDocument
    vertex_bone_start_index: int

    @property
    def all_bones(self) -> Tuple[Bone, ...]:
        return self.document.bones


class LegacyMeshAttachmentBuildError(ValueError):
    """Raised when an explicit attachment binding is internally inconsistent."""


def _resolved_image_path(request: LegacyMeshAttachmentRequest) -> str:
    path = request.image_path.replace("\\", "/")
    if request.sequence is not None and not path.endswith("_"):
        path += "_"
    return path


def _z_parent_by_index(rig: LegacyRigBuildResult) -> dict[int, str]:
    return {group.index: group.bone_name for group in rig.info.z_groups}


def _build_vertex_bones(
    rig: LegacyRigBuildResult,
    request: LegacyMeshAttachmentRequest,
) -> Tuple[Bone, ...]:
    parent_by_index = _z_parent_by_index(rig)
    bones: list[Bone] = []
    for vertex in request.vertices:
        parent_name = parent_by_index.get(vertex.z_group_index)
        if parent_name is None:
            raise LegacyMeshAttachmentBuildError(
                f"Vertex {vertex.index} references unknown z_group_index "
                f"{vertex.z_group_index}; available indices are "
                f"{tuple(sorted(parent_by_index))}"
            )
        x_value, y_value = vertex.bone_position_pixels
        bones.append(
            Bone(
                name=rig.profile.vertex_bone(request.vertex_prefix, vertex.index),
                parent=parent_name,
                x=round(float(x_value), 2),
                y=round(float(y_value), 2),
            )
        )
    return tuple(bones)


def _build_weighted_stream(
    request: LegacyMeshAttachmentRequest,
    first_vertex_bone_index: int,
) -> Tuple[float | int, ...]:
    if not isinstance(first_vertex_bone_index, int) or first_vertex_bone_index < 0:
        raise ValueError("first_vertex_bone_index must be a non-negative integer")
    weighted = tuple(
        WeightedVertex(
            (
                WeightedVertexInfluence(
                    bone_index=first_vertex_bone_index + vertex.index,
                    x=0.0,
                    y=0.0,
                    weight=1.0,
                ),
            )
        )
        for vertex in request.vertices
    )
    return encode_weighted_vertices(weighted)


def _build_component(
    rig: LegacyRigBuildResult,
    request: LegacyMeshAttachmentRequest,
    *,
    first_vertex_bone_index: int,
) -> LegacyAttachmentComponent:
    vertex_bones = _build_vertex_bones(rig, request)
    weighted_vertices = _build_weighted_stream(
        request,
        first_vertex_bone_index,
    )
    uvs = tuple(
        float(component)
        for vertex in request.vertices
        for component in vertex.uv
    )
    sequence_mapping = (
        None if request.sequence is None else request.sequence.to_spine_mapping()
    )
    attachment = MeshAttachment(
        name=request.attachment_name,
        path=_resolved_image_path(request),
        uvs=uvs,
        triangles=request.triangles,
        vertices=weighted_vertices,
        hull=request.hull,
        edges=request.edges,
        width=float(request.width),
        height=float(request.height),
        sequence=sequence_mapping,
    )
    slot = Slot(
        name=request.slot_name,
        bone=rig.info.base_bone_name,
        attachment=request.attachment_name,
    )
    return LegacyAttachmentComponent(
        request=request,
        vertex_bone_start_index=first_vertex_bone_index,
        vertex_bones=vertex_bones,
        attachment=attachment,
        slot=slot,
    )


def _validate_request_set(
    requests: Tuple[LegacyMeshAttachmentRequest, ...],
) -> None:
    if not isinstance(requests, tuple) or not requests:
        raise ValueError("requests must be a non-empty tuple")
    if not all(isinstance(item, LegacyMeshAttachmentRequest) for item in requests):
        raise TypeError("requests must contain LegacyMeshAttachmentRequest values")

    checks = {
        "slot_name": tuple(request.slot_name for request in requests),
        "vertex_prefix": tuple(request.vertex_prefix for request in requests),
    }
    for field_name, values in checks.items():
        duplicates = tuple(sorted({value for value in values if values.count(value) > 1}))
        if duplicates:
            raise LegacyMeshAttachmentBuildError(
                f"Duplicate {field_name} values are not allowed: {duplicates}"
            )

    attachment_paths = tuple(
        (request.skin_name, request.slot_name, request.attachment_name)
        for request in requests
    )
    duplicate_paths = tuple(
        sorted({path for path in attachment_paths if attachment_paths.count(path) > 1})
    )
    if duplicate_paths:
        raise LegacyMeshAttachmentBuildError(
            f"Duplicate skin/slot/attachment paths are not allowed: {duplicate_paths}"
        )


def _build_skins(
    requests: Tuple[LegacyMeshAttachmentRequest, ...],
    components: Tuple[LegacyAttachmentComponent, ...],
) -> Tuple[Skin, ...]:
    skin_order: list[str] = []
    attachments_by_skin: dict[str, dict[str, dict[str, MeshAttachment]]] = {}
    for request, component in zip(requests, components):
        if request.skin_name not in attachments_by_skin:
            skin_order.append(request.skin_name)
            attachments_by_skin[request.skin_name] = {}
        slot_attachments = attachments_by_skin[request.skin_name].setdefault(
            request.slot_name,
            {},
        )
        slot_attachments[request.attachment_name] = component.attachment
    return tuple(
        Skin(name=skin_name, attachments=attachments_by_skin[skin_name])
        for skin_name in skin_order
    )


def _default_skeleton(
    rig: LegacyRigBuildResult,
    requests: Tuple[LegacyMeshAttachmentRequest, ...],
) -> dict[str, object]:
    return {
        "hash": "hash_value_placeholder",
        "spine": rig.profile.spine_version,
        "x": 0,
        "y": 0,
        "width": max(float(request.width) for request in requests),
        "height": max(float(request.height) for request in requests),
        "images": "",
        "audio": "./audio",
    }


def build_legacy_mesh_document(
    rig: LegacyRigBuildResult,
    requests: Tuple[LegacyMeshAttachmentRequest, ...],
    *,
    skeleton_metadata: Mapping[str, object] | None = None,
) -> LegacyMeshDocumentBuildResult:
    """Build one validated document containing several ordered mesh attachments."""

    if not isinstance(rig, LegacyRigBuildResult):
        raise TypeError("rig must be LegacyRigBuildResult")
    _validate_request_set(requests)
    rig.validate()

    components: list[LegacyAttachmentComponent] = []
    next_bone_index = len(rig.bones)
    for request in requests:
        component = _build_component(
            rig,
            request,
            first_vertex_bone_index=next_bone_index,
        )
        components.append(component)
        next_bone_index += len(component.vertex_bones)
    resolved_components = tuple(components)
    skins = _build_skins(requests, resolved_components)

    if skeleton_metadata is None:
        skeleton = _default_skeleton(rig, requests)
    else:
        if not isinstance(skeleton_metadata, Mapping):
            raise TypeError("skeleton_metadata must be a mapping or None")
        skeleton = dict(skeleton_metadata)
        skeleton.setdefault("spine", rig.profile.spine_version)
        skeleton.setdefault("width", max(float(request.width) for request in requests))
        skeleton.setdefault("height", max(float(request.height) for request in requests))

    vertex_bones = tuple(
        bone for component in resolved_components for bone in component.vertex_bones
    )
    document = SpineDocument(
        skeleton=skeleton,
        bones=rig.bones + vertex_bones,
        slots=tuple(component.slot for component in resolved_components),
        skins=skins,
        ik=rig.ik,
        transform=rig.transform,
        animations={"animation": {}},
    )
    try:
        SpineValidator().validate_or_raise(document)
    except Exception as exc:
        raise LegacyMeshAttachmentBuildError(
            f"Multi-attachment A1 document failed Spine validation: {exc}"
        ) from exc

    return LegacyMeshDocumentBuildResult(
        rig=rig,
        requests=requests,
        components=resolved_components,
        skins=skins,
        document=document,
    )


def build_legacy_mesh_attachment(
    rig: LegacyRigBuildResult,
    request: LegacyMeshAttachmentRequest,
    *,
    skeleton_metadata: Mapping[str, object] | None = None,
) -> LegacyMeshAttachmentBuildResult:
    """Build one attachment through the same cumulative in-memory composer."""

    if not isinstance(request, LegacyMeshAttachmentRequest):
        raise TypeError("request must be LegacyMeshAttachmentRequest")
    multi = build_legacy_mesh_document(
        rig,
        (request,),
        skeleton_metadata=skeleton_metadata,
    )
    component = multi.components[0]
    skin = multi.skins[0]
    return LegacyMeshAttachmentBuildResult(
        rig=rig,
        request=request,
        vertex_bones=component.vertex_bones,
        attachment=component.attachment,
        slot=component.slot,
        skin=skin,
        document=multi.document,
        vertex_bone_start_index=component.vertex_bone_start_index,
    )
