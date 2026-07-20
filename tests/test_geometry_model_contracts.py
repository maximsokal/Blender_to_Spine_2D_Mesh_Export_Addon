from __future__ import annotations

from math import inf, nan

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    EdgeId,
    FaceId,
    IDENTITY_MATRIX_4X4,
    LoopId,
    LoopUV,
    MeshEdge,
    MeshFace,
    MeshLoop,
    MeshSnapshot,
    MeshSnapshotValidator,
    MeshValidationSeverity,
    MeshVertex,
    SourceEdgeId,
    SourceFaceId,
    SourceLoopId,
    SourceVertexId,
    VertexId,
)


_SOURCE = "Object"


def _triangle_snapshot(
    *,
    snapshot_id: str = "triangle",
    source_object_id: str = _SOURCE,
    vertex_zero_normal: bool = False,
    face_zero_normal: bool = False,
) -> MeshSnapshot:
    vertex_normals = (
        (0.0, 0.0, 0.0) if vertex_zero_normal else (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
    )
    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(source_object_id, index),
            position=position,
            normal=vertex_normals[index],
        )
        for index, position in enumerate(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            )
        )
    )
    edges = (
        MeshEdge(
            EdgeId(0),
            SourceEdgeId(source_object_id, 0),
            (VertexId(0), VertexId(1)),
        ),
        MeshEdge(
            EdgeId(1),
            SourceEdgeId(source_object_id, 1),
            (VertexId(1), VertexId(2)),
        ),
        MeshEdge(
            EdgeId(2),
            SourceEdgeId(source_object_id, 2),
            (VertexId(2), VertexId(0)),
        ),
    )
    loops = (
        MeshLoop(
            LoopId(0),
            SourceLoopId(source_object_id, 0, 0),
            VertexId(0),
            EdgeId(0),
        ),
        MeshLoop(
            LoopId(1),
            SourceLoopId(source_object_id, 0, 1),
            VertexId(1),
            EdgeId(1),
        ),
        MeshLoop(
            LoopId(2),
            SourceLoopId(source_object_id, 0, 2),
            VertexId(2),
            EdgeId(2),
        ),
    )
    faces = (
        MeshFace(
            FaceId(0),
            SourceFaceId(source_object_id, 0),
            (LoopId(0), LoopId(1), LoopId(2)),
            0,
            (0.0, 0.0, 0.0) if face_zero_normal else (0.0, 0.0, 1.0),
        ),
    )
    return MeshSnapshot(
        snapshot_id=snapshot_id,
        source_object_id=source_object_id,
        object_name="Display Object",
        vertices=vertices,
        edges=edges,
        loops=loops,
        faces=faces,
    )


@pytest.mark.parametrize(
    "factory",
    (
        lambda: VertexId(True),
        lambda: EdgeId(False),
        lambda: FaceId(True),
        lambda: LoopId(False),
        lambda: SourceVertexId(_SOURCE, True),
        lambda: SourceEdgeId(_SOURCE, False),
        lambda: SourceFaceId(_SOURCE, True),
        lambda: SourceLoopId(_SOURCE, False, 0),
        lambda: SourceLoopId(_SOURCE, 0, True),
    ),
)
def test_geometry_indices_reject_bool(factory):
    with pytest.raises(TypeError):
        factory()


@pytest.mark.parametrize(
    "factory",
    (
        lambda: SourceVertexId(" Object", 0),
        lambda: SourceEdgeId("Object ", 0),
        lambda: SourceFaceId("\tObject", 0),
        lambda: SourceLoopId("Object\n", 0, 0),
    ),
)
def test_source_object_identifiers_are_canonical(factory):
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        factory()


@pytest.mark.parametrize(
    "factory",
    (
        lambda: LoopUV("UVMap", (True, 0.0)),
        lambda: LoopUV("UVMap", (0.0, False)),
        lambda: LoopUV("UVMap", (nan, 0.0)),
        lambda: MeshVertex(
            VertexId(0),
            SourceVertexId(_SOURCE, 0),
            (True, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        lambda: MeshVertex(
            VertexId(0),
            SourceVertexId(_SOURCE, 0),
            (0.0, 0.0, 0.0),
            (0.0, inf, 1.0),
        ),
        lambda: MeshSnapshot(
            snapshot_id="snapshot",
            source_object_id=_SOURCE,
            object_name="Object",
            vertices=(),
            edges=(),
            loops=(),
            faces=(),
            world_matrix=(True,) + IDENTITY_MATRIX_4X4[1:],
        ),
    ),
)
def test_vectors_reject_bool_and_non_finite_components(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


@pytest.mark.parametrize(
    "factory",
    (
        lambda: MeshVertex(
            EdgeId(0),
            SourceVertexId(_SOURCE, 0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        lambda: MeshVertex(
            VertexId(0),
            SourceEdgeId(_SOURCE, 0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        lambda: MeshEdge(
            VertexId(0),
            SourceEdgeId(_SOURCE, 0),
            (VertexId(0), VertexId(1)),
        ),
        lambda: MeshEdge(
            EdgeId(0),
            SourceVertexId(_SOURCE, 0),
            (VertexId(0), VertexId(1)),
        ),
        lambda: MeshLoop(
            FaceId(0),
            SourceLoopId(_SOURCE, 0, 0),
            VertexId(0),
            EdgeId(0),
        ),
        lambda: MeshLoop(
            LoopId(0),
            SourceFaceId(_SOURCE, 0),
            VertexId(0),
            EdgeId(0),
        ),
        lambda: MeshFace(
            LoopId(0),
            SourceFaceId(_SOURCE, 0),
            (LoopId(0), LoopId(1), LoopId(2)),
            0,
            (0.0, 0.0, 1.0),
        ),
        lambda: MeshFace(
            FaceId(0),
            SourceLoopId(_SOURCE, 0, 0),
            (LoopId(0), LoopId(1), LoopId(2)),
            0,
            (0.0, 0.0, 1.0),
        ),
    ),
)
def test_mesh_elements_require_exact_identifier_classes(factory):
    with pytest.raises(TypeError):
        factory()


@pytest.mark.parametrize(
    "factory",
    (
        lambda: MeshEdge(
            EdgeId(0),
            SourceEdgeId(_SOURCE, 0),
            (VertexId(0), EdgeId(1)),
        ),
        lambda: MeshLoop(
            LoopId(0),
            SourceLoopId(_SOURCE, 0, 0),
            VertexId(0),
            EdgeId(0),
            (object(),),
        ),
        lambda: MeshFace(
            FaceId(0),
            SourceFaceId(_SOURCE, 0),
            (LoopId(0), LoopId(1), object()),
            0,
            (0.0, 0.0, 1.0),
        ),
        lambda: MeshSnapshot(
            snapshot_id="snapshot",
            source_object_id=_SOURCE,
            object_name="Object",
            vertices=(object(),),
            edges=(),
            loops=(),
            faces=(),
        ),
        lambda: MeshSnapshot(
            snapshot_id="snapshot",
            source_object_id=_SOURCE,
            object_name="Object",
            vertices=(),
            edges=(),
            loops=(),
            faces=(),
            uv_layer_names=(object(),),
        ),
    ),
)
def test_tuple_payloads_validate_item_types_before_using_items(factory):
    with pytest.raises(TypeError):
        factory()


def test_material_index_rejects_bool():
    with pytest.raises(TypeError):
        MeshFace(
            FaceId(0),
            SourceFaceId(_SOURCE, 0),
            (LoopId(0), LoopId(1), LoopId(2)),
            True,
            (0.0, 0.0, 1.0),
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("snapshot_id", " snapshot"),
        ("snapshot_id", "snapshot "),
        ("source_object_id", " Object"),
        ("source_object_id", "Object\t"),
    ),
)
def test_snapshot_and_source_object_ids_are_canonical(field_name, value):
    kwargs = {
        "snapshot_id": "snapshot",
        "source_object_id": _SOURCE,
        "object_name": " Display name may retain boundary spaces ",
        "vertices": (),
        "edges": (),
        "loops": (),
        "faces": (),
    }
    kwargs[field_name] = value
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        MeshSnapshot(**kwargs)


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("active_uv_layer", 1),
        ("render_uv_layer", object()),
    ),
)
def test_optional_uv_layer_names_require_string_items(field_name, value):
    kwargs = {
        "snapshot_id": "snapshot",
        "source_object_id": _SOURCE,
        "object_name": "Object",
        "vertices": (),
        "edges": (),
        "loops": (),
        "faces": (),
        "uv_layer_names": ("UVMap",),
    }
    kwargs[field_name] = value
    with pytest.raises(TypeError):
        MeshSnapshot(**kwargs)


def test_zero_normals_are_non_fatal_explicit_diagnostics():
    snapshot = _triangle_snapshot(vertex_zero_normal=True, face_zero_normal=True)

    issues = MeshSnapshotValidator().validate(snapshot)

    assert tuple((issue.code, issue.severity) for issue in issues) == (
        ("ZERO_VERTEX_NORMAL", MeshValidationSeverity.WARNING),
        ("ZERO_FACE_NORMAL", MeshValidationSeverity.WARNING),
    )
    MeshSnapshotValidator().validate_or_raise(snapshot)


def test_non_zero_normal_snapshot_keeps_clean_validator_result():
    snapshot = _triangle_snapshot()
    assert MeshSnapshotValidator().validate(snapshot) == ()
