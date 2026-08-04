"""Regressions for modifier-duplicated Depth evaluated topology."""

from __future__ import annotations

from pathlib import Path

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_depth_source_geometry_preparation import (
    _canonicalize_depth_evaluated_identity,
    _normal_camera_request_settings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    A1TextureExportMode,
    BakeExecutionSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    EdgeId,
    FaceId,
    LoopId,
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
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.evaluated_identity import (
    rebase_mesh_snapshot_to_evaluated_identity,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import A1ProjectionDirection
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    A1RigProfile,
    SpineJsonTarget,
)


_OBJECT_ID = "DuplicatedEvaluatedObject"


def _duplicated_snapshot() -> MeshSnapshot:
    """Build two disconnected triangles sharing stamped source lineage."""

    positions = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (2.0, 1.0, 0.0),
    )
    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(_OBJECT_ID, index % 3),
            position=position,
            normal=(0.0, 0.0, 1.0),
        )
        for index, position in enumerate(positions)
    )
    edges = tuple(
        MeshEdge(
            id=EdgeId(index),
            source_id=SourceEdgeId(_OBJECT_ID, index % 3),
            vertex_ids=(
                VertexId((index // 3) * 3 + (index % 3)),
                VertexId((index // 3) * 3 + ((index + 1) % 3)),
            ),
        )
        for index in range(6)
    )
    loops = tuple(
        MeshLoop(
            id=LoopId(index),
            source_id=SourceLoopId(_OBJECT_ID, 0, index % 3),
            vertex_id=VertexId(index),
            edge_id=EdgeId(index),
        )
        for index in range(6)
    )
    faces = (
        MeshFace(
            id=FaceId(0),
            source_id=SourceFaceId(_OBJECT_ID, 0),
            loop_ids=(LoopId(0), LoopId(1), LoopId(2)),
            normal=(0.0, 0.0, 1.0),
        ),
        MeshFace(
            id=FaceId(1),
            source_id=SourceFaceId(_OBJECT_ID, 0),
            loop_ids=(LoopId(3), LoopId(4), LoopId(5)),
            normal=(0.0, 0.0, 1.0),
        ),
    )
    snapshot = MeshSnapshot(
        snapshot_id="duplicated-evaluated",
        source_object_id=_OBJECT_ID,
        object_name="Duplicated Evaluated",
        vertices=vertices,
        edges=edges,
        loops=loops,
        faces=faces,
    )
    MeshSnapshotValidator().validate_or_raise(snapshot)
    return snapshot


def _depth_settings() -> A1SingleObjectExportSettings:
    target = SpineJsonTarget.SPINE_4_2
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=64,
            texture_height=64,
            output_directory=Path("output"),
            spine_version=target.exact_version,
            rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE.value,
        ),
        bake_execution=BakeExecutionSettings(
            texture_export_mode=A1TextureExportMode.DEPTH_CAMERA_PROJECTION,
        ),
    )


def test_rebase_assigns_unique_local_identity_without_geometry_mutation() -> None:
    source = _duplicated_snapshot()

    result = rebase_mesh_snapshot_to_evaluated_identity(source)
    rebased = result.snapshot

    assert result.changed is True
    assert result.duplicate_vertex_source_id_count == 3
    assert result.duplicate_edge_source_id_count == 3
    assert result.duplicate_face_source_id_count == 1
    assert result.duplicate_loop_source_id_count == 3
    assert result.missing_edge_source_id_count == 0
    MeshSnapshotValidator().validate_or_raise(rebased)

    assert tuple(vertex.id for vertex in rebased.vertices) == tuple(
        vertex.id for vertex in source.vertices
    )
    assert tuple(vertex.position for vertex in rebased.vertices) == tuple(
        vertex.position for vertex in source.vertices
    )
    assert tuple(edge.vertex_ids for edge in rebased.edges) == tuple(
        edge.vertex_ids for edge in source.edges
    )
    assert tuple(face.loop_ids for face in rebased.faces) == tuple(
        face.loop_ids for face in source.faces
    )

    assert len({vertex.source_id for vertex in rebased.vertices}) == 6
    assert len({edge.source_id for edge in rebased.edges}) == 6
    assert len({face.source_id for face in rebased.faces}) == 2
    assert len({loop.source_id for loop in rebased.loops}) == 6
    assert tuple(
        vertex.source_id.vertex_index for vertex in rebased.vertices
    ) == tuple(range(6))
    assert tuple(
        edge.source_id.edge_index for edge in rebased.edges
    ) == tuple(range(6))
    assert tuple(face.source_id.face_index for face in rebased.faces) == (0, 1)

    loops = rebased.loop_by_id()
    for face in rebased.faces:
        for corner_index, loop_id in enumerate(face.loop_ids):
            source_id = loops[loop_id].source_id
            assert source_id.face_index == face.id.index
            assert source_id.corner_index == corner_index

    # The immutable incoming snapshot keeps its original duplicate provenance.
    assert len({vertex.source_id for vertex in source.vertices}) == 3
    assert len({face.source_id for face in source.faces}) == 1


def test_rebase_is_idempotent_after_local_identity_is_canonical() -> None:
    first = rebase_mesh_snapshot_to_evaluated_identity(_duplicated_snapshot())
    second = rebase_mesh_snapshot_to_evaluated_identity(first.snapshot)

    assert second.changed is False
    assert second.snapshot is first.snapshot
    assert second.duplicate_vertex_source_id_count == 0
    assert second.duplicate_edge_source_id_count == 0
    assert second.duplicate_face_source_id_count == 0
    assert second.duplicate_loop_source_id_count == 0


def test_depth_route_validates_duplication_then_uses_evaluated_identity() -> None:
    settings = _depth_settings()

    resolved = _normal_camera_request_settings(settings)

    assert settings.modifier_lineage_policy is ModifierLineagePolicy.STRICT_PRESERVE
    assert resolved.modifier_lineage_policy is (
        ModifierLineagePolicy.ALLOW_SOURCE_DUPLICATION
    )
    assert resolved.source_geometry_mode is A1SourceGeometryMode.EVALUATED
    assert resolved.projection_direction is A1ProjectionDirection.ACTIVE_CAMERA
    assert resolved.bake_execution.texture_export_mode is (
        A1TextureExportMode.NORMAL_UV_SEGMENTS
    )


def test_depth_canonicalization_reports_collisions_and_warning() -> None:
    snapshot = _duplicated_snapshot()

    rebased, warnings, statistics, result = _canonicalize_depth_evaluated_identity(
        snapshot,
        (),
        {},
        object_id=_OBJECT_ID,
    )

    assert rebased == result.snapshot
    assert result.changed is True
    assert statistics["evaluated_identity_rebased"] == 1
    assert statistics["evaluated_identity_duplicate_vertex_source_ids"] == 3
    assert statistics["evaluated_identity_duplicate_edge_source_ids"] == 3
    assert statistics["evaluated_identity_duplicate_face_source_ids"] == 1
    assert statistics["evaluated_identity_duplicate_loop_source_ids"] == 3
    assert len(warnings) == 1
    assert warnings[0].code == "EVALUATED_IDENTITY_REBASED"
    assert warnings[0].object_id == _OBJECT_ID
