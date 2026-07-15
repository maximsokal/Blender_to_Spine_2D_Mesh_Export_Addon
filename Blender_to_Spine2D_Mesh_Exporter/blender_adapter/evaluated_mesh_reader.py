"""Read an evaluated modifier stack through temporary lineage attributes.

No operator is used.  The original object and Mesh datablock are never modified:
all lineage attributes are stamped on a temporary object/data copy, evaluated by
the dependency graph, converted to a MeshSnapshot, and removed in ``finally``.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Iterable, Tuple
from uuid import uuid4

from ..domain.geometry import (
    EdgeId,
    EvaluatedLineageReport,
    FaceId,
    LoopId,
    LoopUV,
    MeshEdge,
    MeshFace,
    MeshLoop,
    MeshSnapshot,
    MeshSnapshotValidator,
    MeshVertex,
    ModifierLineagePolicy,
    SourceEdgeId,
    SourceFaceId,
    SourceLoopId,
    SourceVertexId,
    VertexId,
    analyse_evaluated_lineage,
    require_valid_evaluated_lineage,
)
from .mesh_reader import MeshReadError, _matrix_tuple, _resolve_uv_layers, _vector_tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LineageAttributeNames:
    vertex: str
    edge: str
    face: str
    corner_face: str
    corner_index: str


@dataclass(frozen=True, slots=True)
class EvaluatedMeshSnapshotResult:
    snapshot: MeshSnapshot
    lineage_report: EvaluatedLineageReport
    modifier_stack: Tuple[Tuple[str, str], ...]


class EvaluatedMeshReadError(MeshReadError):
    """Raised when evaluated geometry or lineage cannot be read safely."""


def _new_attribute_names() -> LineageAttributeNames:
    token = uuid4().hex
    prefix = f"__spine2d_lineage_{token}"
    return LineageAttributeNames(
        vertex=f"{prefix}_vertex",
        edge=f"{prefix}_edge",
        face=f"{prefix}_face",
        corner_face=f"{prefix}_corner_face",
        corner_index=f"{prefix}_corner_index",
    )


def _create_int_attribute(mesh: Any, name: str, domain: str) -> Any:
    attributes = getattr(mesh, "attributes", None)
    if attributes is None:
        raise EvaluatedMeshReadError("Mesh attributes API is unavailable")
    if attributes.get(name) is not None:
        raise EvaluatedMeshReadError(f"Temporary lineage attribute collision: {name}")
    try:
        return attributes.new(name=name, type="INT", domain=domain)
    except Exception as exc:
        raise EvaluatedMeshReadError(
            f"Unable to create INT/{domain} lineage attribute '{name}'"
        ) from exc


def _stamp_lineage_attributes(mesh: Any, names: LineageAttributeNames) -> None:
    """Stamp index+1 values; zero remains the generated/unknown sentinel."""

    vertex_attr = _create_int_attribute(mesh, names.vertex, "POINT")
    edge_attr = _create_int_attribute(mesh, names.edge, "EDGE")
    face_attr = _create_int_attribute(mesh, names.face, "FACE")
    corner_face_attr = _create_int_attribute(mesh, names.corner_face, "CORNER")
    corner_index_attr = _create_int_attribute(mesh, names.corner_index, "CORNER")

    try:
        for vertex in mesh.vertices:
            vertex_attr.data[int(vertex.index)].value = int(vertex.index) + 1
        for edge in mesh.edges:
            edge_attr.data[int(edge.index)].value = int(edge.index) + 1
        for polygon in mesh.polygons:
            face_attr.data[int(polygon.index)].value = int(polygon.index) + 1
            for corner_index in range(int(polygon.loop_total)):
                loop_index = int(polygon.loop_start) + corner_index
                corner_face_attr.data[loop_index].value = int(polygon.index) + 1
                corner_index_attr.data[loop_index].value = corner_index + 1
    except Exception as exc:
        raise EvaluatedMeshReadError("Unable to stamp source lineage attributes") from exc


def _decode_attribute_value(raw_value: Any) -> int | None:
    value = int(raw_value)
    if value == 0:
        return None
    return value - 1


def _read_int_attribute(
    mesh: Any,
    name: str,
    *,
    expected_domain: str,
    expected_length: int,
) -> Tuple[int | None, ...]:
    attributes = getattr(mesh, "attributes", None)
    attribute = attributes.get(name) if attributes is not None else None
    if attribute is None:
        return tuple(None for _ in range(expected_length))
    if str(getattr(attribute, "domain", "")) != expected_domain:
        raise EvaluatedMeshReadError(
            f"Lineage attribute '{name}' changed domain from {expected_domain} to "
            f"{getattr(attribute, 'domain', None)}"
        )
    if str(getattr(attribute, "data_type", "INT")) != "INT":
        raise EvaluatedMeshReadError(
            f"Lineage attribute '{name}' is no longer an INT attribute"
        )
    data = tuple(attribute.data)
    if len(data) != expected_length:
        raise EvaluatedMeshReadError(
            f"Lineage attribute '{name}' length {len(data)} does not match "
            f"evaluated domain length {expected_length}"
        )
    return tuple(_decode_attribute_value(item.value) for item in data)


def _remove_object_and_mesh(bpy_module: Any, obj: Any, mesh: Any) -> None:
    if obj is not None:
        try:
            bpy_module.data.objects.remove(obj, do_unlink=True)
        except Exception:
            logger.exception("Failed to remove temporary evaluated object")
    if mesh is not None:
        try:
            if getattr(mesh, "users", 0) == 0:
                bpy_module.data.meshes.remove(mesh)
        except Exception:
            logger.exception("Failed to remove temporary evaluated mesh datablock")


def read_evaluated_mesh_snapshot(
    obj: Any,
    *,
    depsgraph: Any | None = None,
    scene: Any | None = None,
    snapshot_id: str | None = None,
    source_object_id: str | None = None,
    uv_layer_names: Iterable[str] | None = None,
    lineage_policy: ModifierLineagePolicy = ModifierLineagePolicy.STRICT_PRESERVE,
) -> EvaluatedMeshSnapshotResult:
    """Evaluate modifiers on a temporary copy and return validated immutable data."""

    if obj is None or getattr(obj, "type", None) != "MESH":
        raise EvaluatedMeshReadError("obj must be a Blender MESH object")
    source_mesh = getattr(obj, "data", None)
    if source_mesh is None:
        raise EvaluatedMeshReadError("obj.data is missing")
    if not isinstance(lineage_policy, ModifierLineagePolicy):
        raise TypeError("lineage_policy must be ModifierLineagePolicy")

    try:
        import bpy
    except Exception as exc:
        raise EvaluatedMeshReadError("Blender bpy module is unavailable") from exc

    object_name = str(getattr(obj, "name_full", None) or getattr(obj, "name", ""))
    if not object_name:
        raise EvaluatedMeshReadError("object name is empty")
    resolved_source_object_id = source_object_id or object_name
    resolved_snapshot_id = snapshot_id or f"{resolved_source_object_id}:evaluated"
    resolved_scene = scene or getattr(bpy.context, "scene", None)
    if resolved_scene is None:
        raise EvaluatedMeshReadError("A Blender Scene is required for evaluation")

    names = _new_attribute_names()
    temporary_collection = None
    temporary_object = None
    temporary_mesh = None
    evaluated_object = None
    evaluated_mesh = None

    try:
        temporary_mesh = source_mesh.copy()
        temporary_object = obj.copy()
        temporary_object.data = temporary_mesh
        temporary_object.name = f"__Spine2D_Eval_{uuid4().hex}"
        temporary_mesh.name = f"{temporary_object.name}_Mesh"

        temporary_collection = bpy.data.collections.new(
            f"{temporary_object.name}_Collection"
        )
        resolved_scene.collection.children.link(temporary_collection)
        temporary_collection.objects.link(temporary_object)

        _stamp_lineage_attributes(temporary_mesh, names)
        modifier_stack = tuple(
            (str(modifier.name), str(modifier.type))
            for modifier in temporary_object.modifiers
        )

        resolved_depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
        update = getattr(resolved_depsgraph, "update", None)
        if callable(update):
            update()
        evaluated_object = temporary_object.evaluated_get(resolved_depsgraph)
        evaluated_mesh = evaluated_object.to_mesh(
            preserve_all_data_layers=True,
            depsgraph=resolved_depsgraph,
        )
        if evaluated_mesh is None:
            raise EvaluatedMeshReadError("evaluated_object.to_mesh() returned None")

        vertex_lineage = _read_int_attribute(
            evaluated_mesh,
            names.vertex,
            expected_domain="POINT",
            expected_length=len(evaluated_mesh.vertices),
        )
        edge_lineage = _read_int_attribute(
            evaluated_mesh,
            names.edge,
            expected_domain="EDGE",
            expected_length=len(evaluated_mesh.edges),
        )
        face_lineage = _read_int_attribute(
            evaluated_mesh,
            names.face,
            expected_domain="FACE",
            expected_length=len(evaluated_mesh.polygons),
        )
        corner_face_lineage = _read_int_attribute(
            evaluated_mesh,
            names.corner_face,
            expected_domain="CORNER",
            expected_length=len(evaluated_mesh.loops),
        )
        corner_index_lineage = _read_int_attribute(
            evaluated_mesh,
            names.corner_index,
            expected_domain="CORNER",
            expected_length=len(evaluated_mesh.loops),
        )

        source_face_corner_counts = tuple(
            int(polygon.loop_total) for polygon in source_mesh.polygons
        )
        lineage_report = analyse_evaluated_lineage(
            source_vertex_count=len(source_mesh.vertices),
            source_edge_count=len(source_mesh.edges),
            source_face_corner_counts=source_face_corner_counts,
            vertex_source_indices=vertex_lineage,
            edge_source_indices=edge_lineage,
            face_source_indices=face_lineage,
            corner_source_face_indices=corner_face_lineage,
            corner_source_corner_indices=corner_index_lineage,
            policy=lineage_policy,
        )
        require_valid_evaluated_lineage(lineage_report)

        resolved_uv_layers = _resolve_uv_layers(evaluated_mesh, uv_layer_names)
        resolved_uv_names = tuple(layer.name for layer in resolved_uv_layers)
        active_layer = getattr(evaluated_mesh.uv_layers, "active", None)
        active_uv_name = (
            active_layer.name
            if active_layer is not None and active_layer.name in resolved_uv_names
            else None
        )

        vertices = tuple(
            MeshVertex(
                id=VertexId(int(vertex.index)),
                source_id=SourceVertexId(
                    resolved_source_object_id,
                    int(vertex_lineage[int(vertex.index)]),
                ),
                position=_vector_tuple(
                    vertex.co, 3, f"evaluated.vertices[{vertex.index}].co"
                ),
                normal=_vector_tuple(
                    vertex.normal, 3, f"evaluated.vertices[{vertex.index}].normal"
                ),
            )
            for vertex in evaluated_mesh.vertices
        )
        edges = tuple(
            MeshEdge(
                id=EdgeId(int(edge.index)),
                source_id=(
                    None
                    if edge_lineage[int(edge.index)] is None
                    else SourceEdgeId(
                        resolved_source_object_id,
                        int(edge_lineage[int(edge.index)]),
                    )
                ),
                vertex_ids=(
                    VertexId(int(edge.vertices[0])),
                    VertexId(int(edge.vertices[1])),
                ),
                seam=bool(getattr(edge, "use_seam", False)),
                sharp=bool(getattr(edge, "use_edge_sharp", False)),
            )
            for edge in evaluated_mesh.edges
        )

        loops: list[MeshLoop] = []
        faces: list[MeshFace] = []
        for polygon in evaluated_mesh.polygons:
            polygon_index = int(polygon.index)
            source_face_index = face_lineage[polygon_index]
            if source_face_index is None:
                raise EvaluatedMeshReadError(
                    f"Evaluated face {polygon_index} has unknown lineage after validation"
                )
            polygon_loop_ids: list[LoopId] = []
            for local_corner_index in range(int(polygon.loop_total)):
                loop_index = int(polygon.loop_start) + local_corner_index
                mesh_loop = evaluated_mesh.loops[loop_index]
                source_loop_face = corner_face_lineage[loop_index]
                source_corner_index = corner_index_lineage[loop_index]
                if source_loop_face is None or source_corner_index is None:
                    raise EvaluatedMeshReadError(
                        f"Evaluated loop {loop_index} has unknown lineage after validation"
                    )
                loop_id = LoopId(loop_index)
                polygon_loop_ids.append(loop_id)
                loops.append(
                    MeshLoop(
                        id=loop_id,
                        source_id=SourceLoopId(
                            resolved_source_object_id,
                            int(source_loop_face),
                            int(source_corner_index),
                        ),
                        vertex_id=VertexId(int(mesh_loop.vertex_index)),
                        edge_id=EdgeId(int(mesh_loop.edge_index)),
                        uvs=tuple(
                            LoopUV(
                                layer_name=layer.name,
                                coordinate=_vector_tuple(
                                    layer.data[loop_index].uv,
                                    2,
                                    f"evaluated.uv_layers[{layer.name}]"
                                    f".data[{loop_index}].uv",
                                ),
                            )
                            for layer in resolved_uv_layers
                        ),
                    )
                )
            faces.append(
                MeshFace(
                    id=FaceId(polygon_index),
                    source_id=SourceFaceId(
                        resolved_source_object_id,
                        int(source_face_index),
                    ),
                    loop_ids=tuple(polygon_loop_ids),
                    material_index=max(0, int(getattr(polygon, "material_index", 0))),
                    normal=_vector_tuple(
                        polygon.normal,
                        3,
                        f"evaluated.polygons[{polygon_index}].normal",
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
            loops=tuple(loops),
            faces=tuple(faces),
            uv_layer_names=resolved_uv_names,
            active_uv_layer=active_uv_name,
            world_matrix=_matrix_tuple(obj.matrix_world),
        )
        MeshSnapshotValidator().validate_or_raise(snapshot)
        logger.info(
            "Evaluated '%s' with %d modifiers: %d vertices, %d edges, %d faces",
            object_name,
            len(modifier_stack),
            len(snapshot.vertices),
            len(snapshot.edges),
            len(snapshot.faces),
        )
        return EvaluatedMeshSnapshotResult(
            snapshot=snapshot,
            lineage_report=lineage_report,
            modifier_stack=modifier_stack,
        )
    except EvaluatedMeshReadError:
        raise
    except Exception as exc:
        logger.exception("Failed to read evaluated mesh '%s'", object_name)
        raise EvaluatedMeshReadError(
            f"Failed to read evaluated mesh '{object_name}': {exc}"
        ) from exc
    finally:
        if evaluated_object is not None and evaluated_mesh is not None:
            try:
                evaluated_object.to_mesh_clear()
            except Exception:
                logger.exception("Failed to clear evaluated to_mesh result")
        _remove_object_and_mesh(bpy, temporary_object, temporary_mesh)
        if temporary_collection is not None:
            try:
                if temporary_collection.name in resolved_scene.collection.children:
                    resolved_scene.collection.children.unlink(temporary_collection)
            except Exception:
                # Collection removal below uses do_unlink and is the authoritative
                # cleanup; explicit unlink is only an early release attempt.
                logger.debug("Temporary collection was already unlinked", exc_info=True)
            try:
                bpy.data.collections.remove(temporary_collection)
            except Exception:
                logger.exception("Failed to remove temporary evaluation collection")
