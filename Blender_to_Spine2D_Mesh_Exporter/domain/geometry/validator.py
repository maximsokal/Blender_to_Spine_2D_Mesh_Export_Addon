"""Cross-reference validation for immutable mesh snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Iterable, Tuple

from .ids import EdgeId, FaceId, LoopId, VertexId
from .model import MeshSnapshot


class MeshValidationSeverity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass(frozen=True, slots=True)
class MeshValidationIssue:
    severity: MeshValidationSeverity
    code: str
    path: str
    message: str


class MeshValidationError(ValueError):
    def __init__(self, issues: Iterable[MeshValidationIssue]):
        self.issues = tuple(issues)
        message = "Mesh snapshot validation failed:\n" + "\n".join(
            f"- [{issue.code}] {issue.path}: {issue.message}" for issue in self.issues
        )
        super().__init__(message)


def _duplicates(values: Iterable[object]) -> set[object]:
    seen: set[object] = set()
    duplicates: set[object] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _expected_dense_ids(count: int, id_type):
    return tuple(id_type(index) for index in range(count))


class MeshSnapshotValidator:
    """Validate local topology and source-lineage invariants."""

    def validate(self, snapshot: MeshSnapshot) -> Tuple[MeshValidationIssue, ...]:
        if not isinstance(snapshot, MeshSnapshot):
            raise TypeError("snapshot must be MeshSnapshot")

        issues: list[MeshValidationIssue] = []
        vertex_ids = tuple(vertex.id for vertex in snapshot.vertices)
        edge_ids = tuple(edge.id for edge in snapshot.edges)
        loop_ids = tuple(loop.id for loop in snapshot.loops)
        face_ids = tuple(face.id for face in snapshot.faces)

        for code, path, values in (
            ("DUPLICATE_VERTEX_ID", "vertices", vertex_ids),
            ("DUPLICATE_EDGE_ID", "edges", edge_ids),
            ("DUPLICATE_LOOP_ID", "loops", loop_ids),
            ("DUPLICATE_FACE_ID", "faces", face_ids),
        ):
            for duplicate in sorted(_duplicates(values), key=lambda value: value.index):
                issues.append(
                    MeshValidationIssue(
                        MeshValidationSeverity.ERROR,
                        code,
                        path,
                        f"Local id {duplicate.index} is duplicated",
                    )
                )

        for path, actual, expected, code in (
            (
                "vertices",
                vertex_ids,
                _expected_dense_ids(len(vertex_ids), VertexId),
                "NON_DENSE_VERTEX_IDS",
            ),
            (
                "edges",
                edge_ids,
                _expected_dense_ids(len(edge_ids), EdgeId),
                "NON_DENSE_EDGE_IDS",
            ),
            (
                "loops",
                loop_ids,
                _expected_dense_ids(len(loop_ids), LoopId),
                "NON_DENSE_LOOP_IDS",
            ),
            (
                "faces",
                face_ids,
                _expected_dense_ids(len(face_ids), FaceId),
                "NON_DENSE_FACE_IDS",
            ),
        ):
            if actual != expected:
                issues.append(
                    MeshValidationIssue(
                        MeshValidationSeverity.ERROR,
                        code,
                        path,
                        "Local IDs must be unique, ordered and dense from zero",
                    )
                )

        vertex_map = snapshot.vertex_by_id()
        edge_map = snapshot.edge_by_id()
        loop_map = snapshot.loop_by_id()

        for vertex in snapshot.vertices:
            if vertex.source_id.object_id != snapshot.source_object_id:
                issues.append(
                    MeshValidationIssue(
                        MeshValidationSeverity.ERROR,
                        "FOREIGN_SOURCE_VERTEX",
                        f"vertices[{vertex.id.index}].source_id",
                        "SourceVertexId belongs to another source object",
                    )
                )
            if not all(
                isfinite(float(value)) for value in (*vertex.position, *vertex.normal)
            ):
                issues.append(
                    MeshValidationIssue(
                        MeshValidationSeverity.ERROR,
                        "NON_FINITE_VERTEX",
                        f"vertices[{vertex.id.index}]",
                        "Vertex coordinates and normal must be finite",
                    )
                )

        for edge in snapshot.edges:
            if (
                edge.source_id is not None
                and edge.source_id.object_id != snapshot.source_object_id
            ):
                issues.append(
                    MeshValidationIssue(
                        MeshValidationSeverity.ERROR,
                        "FOREIGN_SOURCE_EDGE",
                        f"edges[{edge.id.index}].source_id",
                        "SourceEdgeId belongs to another source object",
                    )
                )
            for vertex_id in edge.vertex_ids:
                if vertex_id not in vertex_map:
                    issues.append(
                        MeshValidationIssue(
                            MeshValidationSeverity.ERROR,
                            "MISSING_EDGE_VERTEX",
                            f"edges[{edge.id.index}].vertex_ids",
                            f"Vertex {vertex_id.index} does not exist",
                        )
                    )

        referenced_loops: set[LoopId] = set()
        for face in snapshot.faces:
            if face.source_id.object_id != snapshot.source_object_id:
                issues.append(
                    MeshValidationIssue(
                        MeshValidationSeverity.ERROR,
                        "FOREIGN_SOURCE_FACE",
                        f"faces[{face.id.index}].source_id",
                        "SourceFaceId belongs to another source object",
                    )
                )
            face_loops = []
            for loop_id in face.loop_ids:
                loop = loop_map.get(loop_id)
                if loop is None:
                    issues.append(
                        MeshValidationIssue(
                            MeshValidationSeverity.ERROR,
                            "MISSING_FACE_LOOP",
                            f"faces[{face.id.index}].loop_ids",
                            f"Loop {loop_id.index} does not exist",
                        )
                    )
                    continue
                face_loops.append(loop)
                referenced_loops.add(loop_id)
                if loop.source_id.object_id != snapshot.source_object_id:
                    issues.append(
                        MeshValidationIssue(
                            MeshValidationSeverity.ERROR,
                            "FOREIGN_SOURCE_LOOP",
                            f"loops[{loop.id.index}].source_id",
                            "SourceLoopId belongs to another source object",
                        )
                    )
                if loop.source_id.face_index != face.source_id.face_index:
                    issues.append(
                        MeshValidationIssue(
                            MeshValidationSeverity.ERROR,
                            "LOOP_FACE_LINEAGE_MISMATCH",
                            f"faces[{face.id.index}].loop_ids",
                            f"Loop {loop.id.index} does not originate from source face "
                            f"{face.source_id.face_index}",
                        )
                    )

            if len(face_loops) == len(face.loop_ids):
                for index, current_loop in enumerate(face_loops):
                    next_loop = face_loops[(index + 1) % len(face_loops)]
                    edge = edge_map.get(current_loop.edge_id)
                    if edge is None:
                        issues.append(
                            MeshValidationIssue(
                                MeshValidationSeverity.ERROR,
                                "MISSING_LOOP_EDGE",
                                f"loops[{current_loop.id.index}].edge_id",
                                f"Edge {current_loop.edge_id.index} does not exist",
                            )
                        )
                        continue
                    expected_vertices = {current_loop.vertex_id, next_loop.vertex_id}
                    if set(edge.vertex_ids) != expected_vertices:
                        issues.append(
                            MeshValidationIssue(
                                MeshValidationSeverity.ERROR,
                                "FACE_EDGE_CONNECTIVITY",
                                f"faces[{face.id.index}].loop_ids[{index}]",
                                f"Edge {edge.id.index} does not connect loop vertices "
                                f"{current_loop.vertex_id.index} and "
                                f"{next_loop.vertex_id.index}",
                            )
                        )

        for loop in snapshot.loops:
            if loop.vertex_id not in vertex_map:
                issues.append(
                    MeshValidationIssue(
                        MeshValidationSeverity.ERROR,
                        "MISSING_LOOP_VERTEX",
                        f"loops[{loop.id.index}].vertex_id",
                        f"Vertex {loop.vertex_id.index} does not exist",
                    )
                )
            if loop.edge_id not in edge_map:
                issues.append(
                    MeshValidationIssue(
                        MeshValidationSeverity.ERROR,
                        "MISSING_LOOP_EDGE",
                        f"loops[{loop.id.index}].edge_id",
                        f"Edge {loop.edge_id.index} does not exist",
                    )
                )
            actual_layers = tuple(entry.layer_name for entry in loop.uvs)
            if set(actual_layers) != set(snapshot.uv_layer_names):
                issues.append(
                    MeshValidationIssue(
                        MeshValidationSeverity.ERROR,
                        "UV_LAYER_SET_MISMATCH",
                        f"loops[{loop.id.index}].uvs",
                        "Every loop must contain exactly the snapshot UV layer set",
                    )
                )

        unreferenced_loops = set(loop_ids) - referenced_loops
        for loop_id in sorted(unreferenced_loops, key=lambda value: value.index):
            issues.append(
                MeshValidationIssue(
                    MeshValidationSeverity.ERROR,
                    "UNREFERENCED_LOOP",
                    f"loops[{loop_id.index}]",
                    "Loop is not referenced by any face",
                )
            )

        return tuple(issues)

    def validate_or_raise(self, snapshot: MeshSnapshot) -> None:
        errors = tuple(
            issue
            for issue in self.validate(snapshot)
            if issue.severity == MeshValidationSeverity.ERROR
        )
        if errors:
            raise MeshValidationError(errors)
