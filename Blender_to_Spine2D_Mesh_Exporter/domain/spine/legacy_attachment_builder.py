"""Build A1 vertex bones and weighted mesh attachments without tolerance matching.

The legacy exporter creates one vertex bone for every exported mesh vertex and
binds that vertex to the new bone with one full-weight influence at local (0, 0).
This module preserves that external Spine contract while requiring an explicit
Z-group index and already-transformed pixel position for every vertex.
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
class LegacyMeshAttachmentBuildResult:
    rig: LegacyRigBuildResult
    request: LegacyMeshAttachmentRequest
    vertex_bones: Tuple[Bone, ...]
    attachment: MeshAttachment
    slot: Slot
    skin: Skin
    document: SpineDocument

    @property
    def all_bones(self) -> Tuple[Bone, ...]:
        return self.document.bones

    @property
    def vertex_bone_start_index(self) -> int:
        return len(self.rig.bones)


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
    rig: LegacyRigBuildResult,
    request: LegacyMeshAttachmentRequest,
) -> Tuple[float | int, ...]:
    first_vertex_bone_index = len(rig.bones)
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


def build_legacy_mesh_attachment(
    rig: LegacyRigBuildResult,
    request: LegacyMeshAttachmentRequest,
    *,
    skeleton_metadata: Mapping[str, object] | None = None,
) -> LegacyMeshAttachmentBuildResult:
    """Append legacy vertex bones and build one fully validated Spine document."""

    if not isinstance(rig, LegacyRigBuildResult):
        raise TypeError("rig must be LegacyRigBuildResult")
    if not isinstance(request, LegacyMeshAttachmentRequest):
        raise TypeError("request must be LegacyMeshAttachmentRequest")
    rig.validate()

    vertex_bones = _build_vertex_bones(rig, request)
    weighted_vertices = _build_weighted_stream(rig, request)
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
    skin = Skin(
        name=request.skin_name,
        attachments={
            request.slot_name: {
                request.attachment_name: attachment,
            }
        },
    )

    if skeleton_metadata is None:
        skeleton = {
            "hash": "hash_value_placeholder",
            "spine": rig.profile.spine_version,
            "x": 0,
            "y": 0,
            "width": float(request.width),
            "height": float(request.height),
            "images": "",
            "audio": "./audio",
        }
    else:
        if not isinstance(skeleton_metadata, Mapping):
            raise TypeError("skeleton_metadata must be a mapping or None")
        skeleton = dict(skeleton_metadata)
        skeleton.setdefault("spine", rig.profile.spine_version)
        skeleton.setdefault("width", float(request.width))
        skeleton.setdefault("height", float(request.height))

    document = SpineDocument(
        skeleton=skeleton,
        bones=rig.bones + vertex_bones,
        slots=(slot,),
        skins=(skin,),
        ik=rig.ik,
        transform=rig.transform,
        animations={"animation": {}},
    )
    try:
        SpineValidator().validate_or_raise(document)
    except Exception as exc:
        raise LegacyMeshAttachmentBuildError(
            f"Attachment '{request.attachment_name}' failed Spine validation: {exc}"
        ) from exc

    return LegacyMeshAttachmentBuildResult(
        rig=rig,
        request=request,
        vertex_bones=vertex_bones,
        attachment=attachment,
        slot=slot,
        skin=skin,
        document=document,
    )
