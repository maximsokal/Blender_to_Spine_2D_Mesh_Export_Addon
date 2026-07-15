"""Read a Blender source mesh into an immutable domain snapshot.

This adapter deliberately reads the original Mesh datablock without operators or
BMesh allocation. Evaluated modifier topology will be handled by a separate adapter
because preserving source lineage through topology-changing modifiers requires an
explicit attribute propagation strategy.
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
    mesh: Any, requested_names: Iterable[str] | None
) -> tuple[Any, ...]:
    available = tuple(mesh.uv_layers)
    if requested_names is None:
        return available

    requested = tuple(requested_names)
    available_by_name = {layer.name: layer for layer in available}
    missing = [name for name in requested if name not in available_by_name]
    if missing:
        raise MeshReadError("Requested UV layers are missing: " + ", ".join(missing))
    return tuple(available_by_name[name] for name in requested)


def read_source_mesh_snapshot(
    obj: Any,
    *,
    snapshot_id: str | None = None,
    source_object_id: str | None = None,
    uv_layer_names: Iterable[str] | None = None,
) -> MeshSnapshot:
    """Convert one original ``bpy.types.Object`` mesh without changing Blender state."""

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
        resolved_uv_names = tuple(layer.name for layer in resolved_uv_layers)
        active_layer = getattr(mesh.uv_layers, "active", None)
        active_uv_name = (
            active_layer.name
            if active_layer is not None and active_layer.name in resolved_uv_names
            else None
        )

        vertices = tuple(
            MeshVertex(
                id=VertexId(int(vertex.index)),
                source_id=SourceVertexId(
                    resolved_source_object_id, int(vertex.index)
                ),
                position=_vector_tuple(
                    vertex.co, 3, f"vertices[{vertex.index}].co"
                ),
                normal=_vector_tuple(
                    vertex.normal, 3, f"vertices[{vertex.index}].normal"
                ),
            )
            for vertex in mesh.vertices
        )

        edges = tuple(
            MeshEdge(
                id=EdgeId(int(edge.index)),
                source_id=SourceEdgeId(resolved_source_object_id, int(edge.index)),
                vertex_ids=(
                    VertexId(int(edge.vertices[0])),
                    VertexId(int(edge.vertices[1])),
                ),
                seam=bool(getattr(edge, "use_seam", False)),
                sharp=bool(getattr(edge, "use_edge_sharp", False)),
            )
            for edge in mesh.edges
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
                loop_uvs = tuple(
                    LoopUV(
                        layer_name=layer.name,
                        coordinate=_vector_tuple(
                            layer.data[mesh_loop_index].uv,
                            2,
                            f"uv_layers[{layer.name}].data[{mesh_loop_index}].uv",
                        ),
                    )
                    for layer in resolved_uv_layers
                )
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
                        uvs=loop_uvs,
                    )
                )

            domain_faces.append(
                MeshFace(
                    id=FaceId(int(polygon.index)),
                    source_id=SourceFaceId(
                        resolved_source_object_id, int(polygon.index)
                    ),
                    loop_ids=tuple(polygon_loop_ids),
                    material_index=max(
                        0, int(getattr(polygon, "material_index", 0))
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
            edges=edges,
            loops=tuple(domain_loops),
            faces=tuple(domain_faces),
            uv_layer_names=resolved_uv_names,
            active_uv_layer=active_uv_name,
            world_matrix=_matrix_tuple(obj.matrix_world),
        )
        MeshSnapshotValidator().validate_or_raise(snapshot)
        logger.debug(
            "Read source mesh snapshot '%s': %d vertices, %d edges, %d loops, "
            "%d faces",
            snapshot.snapshot_id,
            len(snapshot.vertices),
            len(snapshot.edges),
            len(snapshot.loops),
            len(snapshot.faces),
        )
        return snapshot
    except MeshReadError:
        raise
    except Exception as exc:
        logger.exception("Failed to read source mesh '%s'", object_name)
        raise MeshReadError(f"Failed to read source mesh '{object_name}': {exc}") from exc
