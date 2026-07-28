"""Read a Blender 5.2 source mesh into an immutable domain snapshot.

The adapter uses the direct Mesh and generic Attribute APIs. It does not invoke
operators or allocate a BMesh, and it never mutates the source datablock.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from ..domain.geometry import (
    EdgeId,
    FaceId,
    LoopId,
    LoopUV,
    MeshEdge,
    MeshFace,
    MeshLoop,
    MeshSnapshot,
    MeshSnapshotValidator,
    MeshVertex,
    SourceEdgeId,
    SourceFaceId,
    SourceLoopId,
    SourceVertexId,
    VertexId,
)
from .mesh_edge_attributes import (
    MeshEdgeAttributeError,
    SHARP_EDGE_ATTRIBUTE,
    UV_SEAM_ATTRIBUTE,
    read_boolean_edge_attribute,
)
from .mesh_uv_attributes import MeshUvAttributeError, read_uv_coordinate


logger = logging.getLogger(__name__)


class MeshReadError(RuntimeError):
    """Raised when Blender mesh data cannot be converted safely."""


def _vector_tuple(value: Any, size: int, field_name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value[index]) for index in range(size))
    except Exception as exc:
        raise MeshReadError(f"Unable to read {field_name} as {size} floats") from exc
    return result


def _matrix_tuple(matrix: Any) -> tuple[float, ...]:
    try:
        return tuple(
            float(matrix[row][column]) for row in range(4) for column in range(4)
        )
    except Exception as exc:
        raise MeshReadError("Unable to read object.matrix_world as a 4x4 matrix") from exc


def _resolve_uv_layers(
    mesh: Any,
    requested_names: Iterable[str] | None,
) -> tuple[Any, ...]:
    try:
        available = tuple(mesh.uv_layers)
    except Exception as exc:
        raise MeshReadError("Unable to inspect Blender UV layers") from exc
    if requested_names is None:
        return available

    try:
        requested = tuple(requested_names)
    except Exception as exc:
        raise TypeError("requested_names must be iterable or None") from exc
    if any(not isinstance(name, str) or not name for name in requested):
        raise ValueError("requested_names must contain non-empty strings")
    if len(requested) != len(set(requested)):
        raise ValueError("requested_names cannot contain duplicates")

    available_by_name = {str(layer.name): layer for layer in available}
    missing = [name for name in requested if name not in available_by_name]
    if missing:
        raise MeshReadError("Requested UV layers are missing: " + ", ".join(missing))
    return tuple(available_by_name[name] for name in requested)


def _active_render_uv_name(
    resolved_uv_layers: tuple[Any, ...],
    active_layer: Any | None,
) -> str | None:
    """Resolve Blender 5.2's implicit shader-sampling UV layer.

    Real Blender 5.2 RNA can retain a stale ``active_render`` flag on the first
    UV layer after another layer becomes ``uv_layers.active``. Material nodes
    using ``Texture Coordinate: UV`` follow the active source UV in that state.
    The active layer is therefore authoritative when it belongs to the resolved
    layer set; ``active_render`` remains a compatibility fallback for snapshots
    or test doubles that do not expose an active collection item.
    """

    if active_layer is not None and active_layer in resolved_uv_layers:
        active_name = str(getattr(active_layer, "name", "") or "")
        if active_name:
            return active_name

    render_layers = tuple(
        layer for layer in resolved_uv_layers if bool(getattr(layer, "active_render", False))
    )
    if len(render_layers) > 1:
        raise MeshReadError(
            "Blender mesh reports more than one active_render UV layer without "
            "a resolvable active UV layer: "
            + str(tuple(layer.name for layer in render_layers))
        )
    if render_layers:
        return str(render_layers[0].name)
    return None


def _edge_boolean_attributes(mesh: Any) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    """Read Blender 5.2 `uv_seam` and `sharp_edge` generic attributes."""

    try:
        seams = read_boolean_edge_attribute(
            mesh,
            UV_SEAM_ATTRIBUTE,
            missing_value=False,
        )
        sharp_edges = read_boolean_edge_attribute(
            mesh,
            SHARP_EDGE_ATTRIBUTE,
            missing_value=False,
        )
        return seams, sharp_edges
    except MeshEdgeAttributeError as exc:
        raise MeshReadError(f"Unable to read mesh edge attributes: {exc}") from exc


def _loop_uvs(
    resolved_uv_layers: tuple[Any, ...],
    *,
    mesh_loop_index: int,
    mesh_loop_count: int,
) -> tuple[LoopUV, ...]:
    try:
        return tuple(
            LoopUV(
                layer_name=str(layer.name),
                coordinate=read_uv_coordinate(
                    layer,
                    mesh_loop_index,
                    expected_length=mesh_loop_count,
                ),
            )
            for layer in resolved_uv_layers
        )
    except MeshUvAttributeError as exc:
        raise MeshReadError(
            f"Unable to read UV coordinates for mesh loop {mesh_loop_index}: {exc}"
        ) from exc


def read_source_mesh_snapshot(
    obj: Any,
    *,
    snapshot_id: str | None = None,
    source_object_id: str | None = None,
    uv_layer_names: Iterable[str] | None = None,
) -> MeshSnapshot:
    """Convert one original Blender 5.2 Mesh without changing Blender state."""

    if obj is None:
        raise MeshReadError("obj cannot be None")
    if getattr(obj, "type", None) != "MESH":
        raise MeshReadError("obj must be a Blender MESH object")
    mesh = getattr(obj, "data", None)
    if mesh is None:
        raise MeshReadError("obj.data is missing")

    object_name = str(getattr(obj, "name_full", None) or getattr(obj, "name", ""))
    if not object_name:
        raise MeshReadError("object name is empty")
    resolved_source_object_id = source_object_id or object_name
    resolved_snapshot_id = snapshot_id or f"{resolved_source_object_id}:source"

    try:
        resolved_uv_layers = _resolve_uv_layers(mesh, uv_layer_names)
        resolved_uv_names = tuple(str(layer.name) for layer in resolved_uv_layers)
        active_layer = getattr(mesh.uv_layers, "active", None)
        active_uv_name = (
            str(active_layer.name)
            if active_layer is not None and active_layer.name in resolved_uv_names
            else None
        )
        render_uv_name = _active_render_uv_name(resolved_uv_layers, active_layer)
        seam_values, sharp_values = _edge_boolean_attributes(mesh)
        mesh_loop_count = len(mesh.loops)

        vertices = tuple(
            MeshVertex(
                id=VertexId(int(vertex.index)),
                source_id=SourceVertexId(
                    resolved_source_object_id,
                    int(vertex.index),
                ),
                position=_vector_tuple(
                    vertex.co,
                    3,
                    f"vertices[{vertex.index}].co",
                ),
                normal=_vector_tuple(
                    vertex.normal,
                    3,
                    f"vertices[{vertex.index}].normal",
                ),
            )
            for vertex in mesh.vertices
        )

        edges: list[MeshEdge] = []
        for edge in mesh.edges:
            edge_index = int(edge.index)
            if edge_index < 0 or edge_index >= len(seam_values):
                raise MeshReadError(
                    f"Mesh edge index {edge_index} is outside attribute data range"
                )
            edges.append(
                MeshEdge(
                    id=EdgeId(edge_index),
                    source_id=SourceEdgeId(
                        resolved_source_object_id,
                        edge_index,
                    ),
                    vertex_ids=(
                        VertexId(int(edge.vertices[0])),
                        VertexId(int(edge.vertices[1])),
                    ),
                    seam=seam_values[edge_index],
                    sharp=sharp_values[edge_index],
                )
            )

        domain_loops: list[MeshLoop] = []
        domain_faces: list[MeshFace] = []
        for polygon in mesh.polygons:
            polygon_loop_ids: list[LoopId] = []
            for corner_index in range(int(polygon.loop_total)):
                mesh_loop_index = int(polygon.loop_start) + corner_index
                mesh_loop = mesh.loops[mesh_loop_index]
                loop_id = LoopId(mesh_loop_index)
                polygon_loop_ids.append(loop_id)
                domain_loops.append(
                    MeshLoop(
                        id=loop_id,
                        source_id=SourceLoopId(
                            resolved_source_object_id,
                            int(polygon.index),
                            corner_index,
                        ),
                        vertex_id=VertexId(int(mesh_loop.vertex_index)),
                        edge_id=EdgeId(int(mesh_loop.edge_index)),
                        uvs=_loop_uvs(
                            resolved_uv_layers,
                            mesh_loop_index=mesh_loop_index,
                            mesh_loop_count=mesh_loop_count,
                        ),
                    )
                )

            domain_faces.append(
                MeshFace(
                    id=FaceId(int(polygon.index)),
                    source_id=SourceFaceId(
                        resolved_source_object_id,
                        int(polygon.index),
                    ),
                    loop_ids=tuple(polygon_loop_ids),
                    material_index=max(
                        0,
                        int(getattr(polygon, "material_index", 0)),
                    ),
                    normal=_vector_tuple(
                        polygon.normal,
                        3,
                        f"polygons[{polygon.index}].normal",
                    ),
                    smooth=bool(getattr(polygon, "use_smooth", False)),
                )
            )

        snapshot = MeshSnapshot(
            snapshot_id=resolved_snapshot_id,
            source_object_id=resolved_source_object_id,
            object_name=object_name,
            vertices=vertices,
            edges=tuple(edges),
            loops=tuple(domain_loops),
            faces=tuple(domain_faces),
            uv_layer_names=resolved_uv_names,
            active_uv_layer=active_uv_name,
            world_matrix=_matrix_tuple(obj.matrix_world),
            render_uv_layer=render_uv_name,
        )
        MeshSnapshotValidator().validate_or_raise(snapshot)
        logger.debug(
            "Read Blender 5.2 source mesh snapshot '%s': %d vertices, %d edges, "
            "%d loops, %d faces active_uv=%s render_uv=%s",
            snapshot.snapshot_id,
            len(snapshot.vertices),
            len(snapshot.edges),
            len(snapshot.loops),
            len(snapshot.faces),
            snapshot.active_uv_layer,
            snapshot.render_uv_layer,
        )
        return snapshot
    except MeshReadError:
        raise
    except Exception as exc:
        logger.exception("Failed to read source mesh '%s'", object_name)
        raise MeshReadError(f"Failed to read source mesh '{object_name}': {exc}") from exc
