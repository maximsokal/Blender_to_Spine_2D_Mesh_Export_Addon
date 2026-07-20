"""Local mesh identifiers and stable source-lineage identifiers.

Local identifiers are unique only inside one :class:`MeshSnapshot`. Source
identifiers point back to the original Blender mesh element and survive copying,
segment extraction and topology-preserving transformations.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import require_identity, require_integer


def _validate_non_negative_index(value: int, field_name: str) -> None:
    require_integer(value, field_name, minimum=0)


def _validate_object_id(value: str, field_name: str) -> None:
    require_identity(value, field_name)


@dataclass(frozen=True, order=True, slots=True)
class VertexId:
    index: int

    def __post_init__(self) -> None:
        _validate_non_negative_index(self.index, "VertexId.index")


@dataclass(frozen=True, order=True, slots=True)
class EdgeId:
    index: int

    def __post_init__(self) -> None:
        _validate_non_negative_index(self.index, "EdgeId.index")


@dataclass(frozen=True, order=True, slots=True)
class FaceId:
    index: int

    def __post_init__(self) -> None:
        _validate_non_negative_index(self.index, "FaceId.index")


@dataclass(frozen=True, order=True, slots=True)
class LoopId:
    index: int

    def __post_init__(self) -> None:
        _validate_non_negative_index(self.index, "LoopId.index")


@dataclass(frozen=True, order=True, slots=True)
class SourceVertexId:
    object_id: str
    vertex_index: int

    def __post_init__(self) -> None:
        _validate_object_id(self.object_id, "SourceVertexId.object_id")
        _validate_non_negative_index(self.vertex_index, "SourceVertexId.vertex_index")


@dataclass(frozen=True, order=True, slots=True)
class SourceEdgeId:
    object_id: str
    edge_index: int

    def __post_init__(self) -> None:
        _validate_object_id(self.object_id, "SourceEdgeId.object_id")
        _validate_non_negative_index(self.edge_index, "SourceEdgeId.edge_index")


@dataclass(frozen=True, order=True, slots=True)
class SourceFaceId:
    object_id: str
    face_index: int

    def __post_init__(self) -> None:
        _validate_object_id(self.object_id, "SourceFaceId.object_id")
        _validate_non_negative_index(self.face_index, "SourceFaceId.face_index")


@dataclass(frozen=True, order=True, slots=True)
class SourceLoopId:
    """Stable identity of one corner in one original source face.

    Blender's global mesh-loop index is not used as the public identity because
    face-local corner identity is easier to preserve when a face is copied into a
    segment or triangulated. A derived mesh may contain the same SourceLoopId more
    than once; local :class:`LoopId` values remain unique inside the snapshot.
    """

    object_id: str
    face_index: int
    corner_index: int

    def __post_init__(self) -> None:
        _validate_object_id(self.object_id, "SourceLoopId.object_id")
        _validate_non_negative_index(self.face_index, "SourceLoopId.face_index")
        _validate_non_negative_index(self.corner_index, "SourceLoopId.corner_index")
