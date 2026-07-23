"""Read Blender 5.2 evaluated geometry through temporary lineage attributes.

The original Object and Mesh are never mutated. A private object/data copy is
linked to a temporary collection, stamped with lineage attributes, evaluated,
converted with ``Object.to_mesh()``, and released with ``to_mesh_clear()`` in
``finally``. UV seams and sharp edges are read from Blender's generic edge
attributes rather than retired ``MeshEdge`` flags.
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
from .mesh_edge_attributes import (
    MeshEdgeAttributeError,
    SHARP_EDGE_ATTRIBUTE,
    UV_SEAM_ATTRIBUTE,
    read_boolean_edge_attribute,
)
from .mesh_reader import (
    MeshReadError,
    _active_render_uv_name,
    _matrix_tuple,
    _resolve_uv_layers,
    _vector_tuple,
)


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

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, MeshSnapshot):
            raise TypeError("snapshot must be MeshSnapshot")
        if not isinstance(self.lineage_report, EvaluatedLineageReport):
            raise TypeError("lineage_report must be EvaluatedLineageReport")
        if not isinstance(self.modifier_stack, tuple):
            raise TypeError("modifier_stack must be tuple")


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


def _mesh_attributes(mesh: Any) -> Any:
    attributes = getattr(mesh, "attributes", None)
    if attributes is None:
        raise EvaluatedMeshReadError(
            "Blender 5.2 Mesh.attributes API is unavailable"
        )
    return attributes


def _create_int_attribute(mesh: Any, name: str, domain: str) -> Any:
    attributes = _mesh_attributes(mesh)
    try:
        existing = attributes.get(name)
    except Exception as exc:
        raise EvaluatedMeshReadError(
            f"Unable to check temporary lineage attribute '{name}'"
        ) from exc
    if existing is not None:
        raise EvaluatedMeshReadError(f"Temporary lineage attribute collision: {name}")
    try:
        return attributes.new(name=name, type="INT", domain=domain)
    except Exception as exc:
        raise EvaluatedMeshReadError(
            f"Unable to create INT/{domain} lineage attribute '{name}'"
        ) from exc


def _stamp_lineage_attributes(mesh: Any, names: LineageAttributeNames) -> None:
    """Stamp index+1 values; zero remains the generated/unknown sentinel."""

    vertex_attribute = _create_int_attribute(mesh, names.vertex, "POINT")
    edge_attribute = _create_int_attribute(mesh, names.edge, "EDGE")
    face_attribute = _create_int_attribute(mesh, names.face, "FACE")
    corner_face_attribute = _create_int_attribute(mesh, names.corner_face, "CORNER")
    corner_index_attribute = _create_int_attribute(mesh, names.corner_index, "CORNER")

    try:
        for vertex in mesh.vertices:
            index = int(vertex.index)
            vertex_attribute.data[index].value = index + 1
        for edge in mesh.edges:
            index = int(edge.index)
            edge_attribute.data[index].value = index + 1
        for polygon in mesh.polygons:
            polygon_index = int(polygon.index)
            face_attribute.data[polygon_index].value = polygon_index + 1
            for corner_index in range(int(polygon.loop_total)):
                loop_index = int(polygon.loop_start) + corner_index
                corner_face_attribute.data[loop_index].value = polygon_index + 1
                corner_index_attribute.data[loop_index].value = corner_index + 1
    except Exception as exc:
        raise EvaluatedMeshReadError("Unable to stamp source lineage attributes") from exc


def _decode_attribute_value(raw_value: Any) -> int | None:
    try:
        value = int(raw_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvaluatedMeshReadError(
            f"Lineage attribute contains non-integer value {raw_value!r}"
        ) from exc
    if value < 0:
        raise EvaluatedMeshReadError(
            f"Lineage attribute contains negative encoded value {value}"
        )
    return None if value == 0 else value - 1


def _read_int_attribute(
    mesh: Any,
    name: str,
    *,
    expected_domain: str,
    expected_length: int,
) -> Tuple[int | None, ...]:
    attributes = _mesh_attributes(mesh)
    try:
        attribute = attributes.get(name)
    except Exception as exc:
        raise EvaluatedMeshReadError(
            f"Unable to read lineage attribute '{name}'"
        ) from exc
    if attribute is None:
        return tuple(None for _ in range(expected_length))

    domain = str(getattr(attribute, "domain", "") or "")
    data_type = str(getattr(attribute, "data_type", "") or "")
    if domain != expected_domain:
        raise EvaluatedMeshReadError(
            f"Lineage attribute '{name}' changed domain from {expected_domain} "
            f"to {domain or None}"
        )
    if data_type != "INT":
        raise EvaluatedMeshReadError(
            f"Lineage attribute '{name}' changed data type from INT to "
            f"{data_type or None}"
        )
    try:
        data = tuple(attribute.data)
    except Exception as exc:
        raise EvaluatedMeshReadError(
            f"Unable to iterate lineage attribute '{name}'"
        ) from exc
    if len(data) != expected_length:
        raise EvaluatedMeshReadError(
            f"Lineage attribute '{name}' length {len(data)} does not match "
            f"evaluated domain length {expected_length}"
        )
    return tuple(_decode_attribute_value(item.value) for item in data)


def _read_edge_flags(mesh: Any) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    try:
        return (
            read_boolean_edge_attribute(
                mesh,
                UV_SEAM_ATTRIBUTE,
                missing_value=False,
            ),
            read_boolean_edge_attribute(
                mesh,
                SHARP_EDGE_ATTRIBUTE,
                missing_value=False,
            ),
        )
    except MeshEdgeAttributeError as exc:
        raise EvaluatedMeshReadError(
            f"Unable to read evaluated edge attributes: {exc}"
        ) from exc


def _require_lineage_value(
    values: Tuple[int | None, ...],
    index: int,
    *,
    label: str,
) -> int:
    if index < 0 or index >= len(values):
        raise EvaluatedMeshReadError(
            f"{label} index {index} is outside lineage data range"
        )
    value = values[index]
    if value is None:
        raise EvaluatedMeshReadError(
            f"{label} {index} has unknown lineage after validation"
        )
    return int(value)


def _remove_temporary_object_and_mesh(
    bpy_module: Any,
    obj: Any | None,
    mesh: Any | None,
) -> None:
    if obj is not None:
        try:
            bpy_module.data.objects.remove(obj, do_unlink=True)
        except Exception:
            logger.exception("Failed to remove temporary evaluated object")
    if mesh is not None:
        try:
            if int(getattr(mesh, "users", 0) or 0) == 0:
                bpy_module.data.meshes.remove(mesh)
        except Exception:
            logger.exception("Failed to remove temporary evaluated Mesh datablock")


def _remove_temporary_collection(
    bpy_module: Any,
    collection: Any | None,
) -> None:
    if collection is None:
        return
    try:
        bpy_module.data.collections.remove(collection, do_unlink=True)
    except Exception:
        logger.exception("Failed to remove temporary evaluation collection")


def _build_snapshot_from_evaluated_mesh(
    *,
    evaluated_mesh: Any,
    source_mesh: Any,
    source_object: Any,
    object_name: str,
    source_object_id: str,
    snapshot_id: str,
    names: LineageAttributeNames,
    modifier_stack: Tuple[Tuple[str, str], ...],
    uv_layer_names: Iterable[str] | None,
    lineage_policy: ModifierLineagePolicy,
) -> EvaluatedMeshSnapshotResult:
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
    resolved_uv_names = tuple(str(layer.name) for layer in resolved_uv_layers)
    active_layer = getattr(evaluated_mesh.uv_layers, "active", None)
    active_uv_name = (
        str(active_layer.name)
        if active_layer is not None and active_layer.name in resolved_uv_names
        else None
    )
    render_uv_name = _active_render_uv_name(resolved_uv_layers, active_layer)
    seam_values, sharp_values = _read_edge_flags(evaluated_mesh)

    vertices: list[MeshVertex] = []
    for vertex in evaluated_mesh.vertices:
        vertex_index = int(vertex.index)
        vertices.append(
            MeshVertex(
                id=VertexId(vertex_index),
                source_id=SourceVertexId(
                    source_object_id,
                    _require_lineage_value(
                        vertex_lineage,
                        vertex_index,
                        label="Evaluated vertex",
                    ),
                ),
                position=_vector_tuple(
                    vertex.co,
                    3,
                    f"evaluated.vertices[{vertex_index}].co",
                ),
                normal=_vector_tuple(
                    vertex.normal,
                    3,
                    f"evaluated.vertices[{vertex_index}].normal",
                ),
            )
        )

    edges: list[MeshEdge] = []
    for edge in evaluated_mesh.edges:
        edge_index = int(edge.index)
        if edge_index < 0 or edge_index >= len(seam_values):
            raise EvaluatedMeshReadError(
                f"Evaluated edge index {edge_index} is outside attribute data range"
            )
        source_edge_index = edge_lineage[edge_index]
        edges.append(
            MeshEdge(
                id=EdgeId(edge_index),
                source_id=(
                    None
                    if source_edge_index is None
                    else SourceEdgeId(source_object_id, int(source_edge_index))
                ),
                vertex_ids=(
                    VertexId(int(edge.vertices[0])),
                    VertexId(int(edge.vertices[1])),
                ),
                seam=seam_values[edge_index],
                sharp=sharp_values[edge_index],
            )
        )

    loops: list[MeshLoop] = []
    faces: list[MeshFace] = []
    for polygon in evaluated_mesh.polygons:
        polygon_index = int(polygon.index)
        source_face_index = _require_lineage_value(
            face_lineage,
            polygon_index,
            label="Evaluated face",
        )
        polygon_loop_ids: list[LoopId] = []
        for local_corner_index in range(int(polygon.loop_total)):
            loop_index = int(polygon.loop_start) + local_corner_index
            mesh_loop = evaluated_mesh.loops[loop_index]
            source_loop_face = _require_lineage_value(
                corner_face_lineage,
                loop_index,
                label="Evaluated loop face",
            )
            source_corner_index = _require_lineage_value(
                corner_index_lineage,
                loop_index,
                label="Evaluated loop corner",
            )
            loop_id = LoopId(loop_index)
            polygon_loop_ids.append(loop_id)
            loops.append(
                MeshLoop(
                    id=loop_id,
                    source_id=SourceLoopId(
                        source_object_id,
                        source_loop_face,
                        source_corner_index,
                    ),
                    vertex_id=VertexId(int(mesh_loop.vertex_index)),
                    edge_id=EdgeId(int(mesh_loop.edge_index)),
                    uvs=tuple(
                        LoopUV(
                            layer_name=str(layer.name),
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
                source_id=SourceFaceId(source_object_id, source_face_index),
                loop_ids=tuple(polygon_loop_ids),
                material_index=max(
                    0,
                    int(getattr(polygon, "material_index", 0)),
                ),
                normal=_vector_tuple(
                    polygon.normal,
                    3,
                    f"evaluated.polygons[{polygon_index}].normal",
                ),
                smooth=bool(getattr(polygon, "use_smooth", False)),
            )
        )

    snapshot = MeshSnapshot(
        snapshot_id=snapshot_id,
        source_object_id=source_object_id,
        object_name=object_name,
        vertices=tuple(vertices),
        edges=tuple(edges),
        loops=tuple(loops),
        faces=tuple(faces),
        uv_layer_names=resolved_uv_names,
        active_uv_layer=active_uv_name,
        world_matrix=_matrix_tuple(source_object.matrix_world),
        render_uv_layer=render_uv_name,
    )
    MeshSnapshotValidator().validate_or_raise(snapshot)
    return EvaluatedMeshSnapshotResult(
        snapshot=snapshot,
        lineage_report=lineage_report,
        modifier_stack=modifier_stack,
    )


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
    """Evaluate a Blender 5.2 modifier stack and return immutable geometry."""

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

        result = _build_snapshot_from_evaluated_mesh(
            evaluated_mesh=evaluated_mesh,
            source_mesh=source_mesh,
            source_object=obj,
            object_name=object_name,
            source_object_id=resolved_source_object_id,
            snapshot_id=resolved_snapshot_id,
            names=names,
            modifier_stack=modifier_stack,
            uv_layer_names=uv_layer_names,
            lineage_policy=lineage_policy,
        )
        logger.info(
            "Evaluated Blender 5.2 Mesh '%s' with %d modifiers: %d vertices, "
            "%d edges, %d faces active_uv=%s render_uv=%s",
            object_name,
            len(modifier_stack),
            len(result.snapshot.vertices),
            len(result.snapshot.edges),
            len(result.snapshot.faces),
            result.snapshot.active_uv_layer,
            result.snapshot.render_uv_layer,
        )
        return result
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
                logger.exception("Failed to clear evaluated Object.to_mesh result")
        _remove_temporary_object_and_mesh(
            bpy,
            temporary_object,
            temporary_mesh,
        )
        _remove_temporary_collection(bpy, temporary_collection)


__all__ = [
    "EvaluatedMeshReadError",
    "EvaluatedMeshSnapshotResult",
    "LineageAttributeNames",
    "read_evaluated_mesh_snapshot",
]
