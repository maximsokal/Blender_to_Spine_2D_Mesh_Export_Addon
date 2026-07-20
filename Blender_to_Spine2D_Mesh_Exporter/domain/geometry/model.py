"""Immutable, Blender-independent mesh snapshot model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from .contracts import (
    require_exact_type,
    require_finite_vector,
    require_identity,
    require_integer,
    require_non_empty_string,
    require_optional_exact_type,
    require_tuple_items,
)
from .ids import (
    EdgeId,
    FaceId,
    LoopId,
    SourceEdgeId,
    SourceFaceId,
    SourceLoopId,
    SourceVertexId,
    VertexId,
)

Vector2 = Tuple[float, float]
Vector3 = Tuple[float, float, float]
Matrix4x4 = Tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]

IDENTITY_MATRIX_4X4: Matrix4x4 = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


def _validate_name(value: str, field_name: str) -> None:
    require_non_empty_string(value, field_name)


def _validate_vector(value: tuple[float, ...], size: int, field_name: str) -> None:
    require_finite_vector(value, size, field_name)


def _validate_optional_layer_name(value: str | None, field_name: str) -> None:
    if value is None:
        return
    require_exact_type(value, str, field_name)
    _validate_name(value, field_name)


@dataclass(frozen=True, slots=True)
class LoopUV:
    layer_name: str
    coordinate: Vector2

    def __post_init__(self) -> None:
        _validate_name(self.layer_name, "layer_name")
        _validate_vector(self.coordinate, 2, "coordinate")


@dataclass(frozen=True, slots=True)
class MeshVertex:
    id: VertexId
    source_id: SourceVertexId
    position: Vector3
    normal: Vector3

    def __post_init__(self) -> None:
        require_exact_type(self.id, VertexId, "id")
        require_exact_type(self.source_id, SourceVertexId, "source_id")
        _validate_vector(self.position, 3, "position")
        _validate_vector(self.normal, 3, "normal")


@dataclass(frozen=True, slots=True)
class MeshEdge:
    id: EdgeId
    source_id: SourceEdgeId | None
    vertex_ids: Tuple[VertexId, VertexId]
    seam: bool = False
    sharp: bool = False

    def __post_init__(self) -> None:
        require_exact_type(self.id, EdgeId, "id")
        require_optional_exact_type(self.source_id, SourceEdgeId, "source_id")
        require_tuple_items(
            self.vertex_ids,
            VertexId,
            "vertex_ids",
            exact_length=2,
        )
        if self.vertex_ids[0] == self.vertex_ids[1]:
            raise ValueError("an edge cannot reference the same vertex twice")
        if not isinstance(self.seam, bool) or not isinstance(self.sharp, bool):
            raise TypeError("seam and sharp must be bool")


@dataclass(frozen=True, slots=True)
class MeshLoop:
    id: LoopId
    source_id: SourceLoopId
    vertex_id: VertexId
    edge_id: EdgeId
    uvs: Tuple[LoopUV, ...] = ()

    def __post_init__(self) -> None:
        require_exact_type(self.id, LoopId, "id")
        require_exact_type(self.source_id, SourceLoopId, "source_id")
        require_exact_type(self.vertex_id, VertexId, "vertex_id")
        require_exact_type(self.edge_id, EdgeId, "edge_id")
        require_tuple_items(self.uvs, LoopUV, "uvs")
        layer_names = tuple(uv.layer_name for uv in self.uvs)
        if len(layer_names) != len(set(layer_names)):
            raise ValueError(f"Loop {self.id.index} contains duplicate UV layer names")

    def uv(self, layer_name: str) -> Vector2 | None:
        _validate_name(layer_name, "layer_name")
        for entry in self.uvs:
            if entry.layer_name == layer_name:
                return entry.coordinate
        return None

    def with_uv(self, layer_name: str, coordinate: Vector2) -> "MeshLoop":
        replacement = LoopUV(layer_name=layer_name, coordinate=coordinate)
        updated = [entry for entry in self.uvs if entry.layer_name != layer_name]
        updated.append(replacement)
        updated.sort(key=lambda entry: entry.layer_name)
        return MeshLoop(
            id=self.id,
            source_id=self.source_id,
            vertex_id=self.vertex_id,
            edge_id=self.edge_id,
            uvs=tuple(updated),
        )


@dataclass(frozen=True, slots=True)
class MeshFace:
    id: FaceId
    source_id: SourceFaceId
    loop_ids: Tuple[LoopId, ...]
    material_index: int
    normal: Vector3
    smooth: bool = False

    def __post_init__(self) -> None:
        require_exact_type(self.id, FaceId, "id")
        require_exact_type(self.source_id, SourceFaceId, "source_id")
        require_tuple_items(
            self.loop_ids,
            LoopId,
            "loop_ids",
            minimum_length=3,
        )
        if len(self.loop_ids) != len(set(self.loop_ids)):
            raise ValueError(f"Face {self.id.index} contains duplicate LoopId values")
        require_integer(self.material_index, "material_index", minimum=0)
        _validate_vector(self.normal, 3, "normal")
        if not isinstance(self.smooth, bool):
            raise TypeError("smooth must be bool")


@dataclass(frozen=True, slots=True)
class MeshSnapshot:
    """One immutable mesh state with local IDs and stable source lineage."""

    snapshot_id: str
    source_object_id: str
    object_name: str
    vertices: Tuple[MeshVertex, ...]
    edges: Tuple[MeshEdge, ...]
    loops: Tuple[MeshLoop, ...]
    faces: Tuple[MeshFace, ...]
    uv_layer_names: Tuple[str, ...] = ()
    active_uv_layer: str | None = None
    world_matrix: Matrix4x4 = IDENTITY_MATRIX_4X4
    render_uv_layer: str | None = None

    def __post_init__(self) -> None:
        require_identity(self.snapshot_id, "snapshot_id")
        require_identity(self.source_object_id, "source_object_id")
        _validate_name(self.object_name, "object_name")
        require_tuple_items(self.vertices, MeshVertex, "vertices")
        require_tuple_items(self.edges, MeshEdge, "edges")
        require_tuple_items(self.loops, MeshLoop, "loops")
        require_tuple_items(self.faces, MeshFace, "faces")
        require_tuple_items(self.uv_layer_names, str, "uv_layer_names")
        for layer_name in self.uv_layer_names:
            _validate_name(layer_name, "uv_layer_name")
        if len(self.uv_layer_names) != len(set(self.uv_layer_names)):
            raise ValueError("uv_layer_names contains duplicates")

        _validate_optional_layer_name(self.active_uv_layer, "active_uv_layer")
        _validate_optional_layer_name(self.render_uv_layer, "render_uv_layer")
        if (
            self.active_uv_layer is not None
            and self.active_uv_layer not in self.uv_layer_names
        ):
            raise ValueError("active_uv_layer must be present in uv_layer_names")
        if self.render_uv_layer is None and self.active_uv_layer is not None:
            object.__setattr__(self, "render_uv_layer", self.active_uv_layer)
        if (
            self.render_uv_layer is not None
            and self.render_uv_layer not in self.uv_layer_names
        ):
            raise ValueError("render_uv_layer must be present in uv_layer_names")
        _validate_vector(self.world_matrix, 16, "world_matrix")

    def vertex_by_id(self) -> dict[VertexId, MeshVertex]:
        return {vertex.id: vertex for vertex in self.vertices}

    def edge_by_id(self) -> dict[EdgeId, MeshEdge]:
        return {edge.id: edge for edge in self.edges}

    def loop_by_id(self) -> dict[LoopId, MeshLoop]:
        return {loop.id: loop for loop in self.loops}

    def face_by_id(self) -> dict[FaceId, MeshFace]:
        return {face.id: face for face in self.faces}

    def iter_source_loop_ids(self) -> Iterable[SourceLoopId]:
        return (loop.source_id for loop in self.loops)
