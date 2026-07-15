from dataclasses import replace

from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    EdgeId,
    FaceId,
    LoopUV,
    SegmentationSettings,
    materialize_segment_snapshots,
    segment_mesh,
)

from test_geometry_domain import build_square_snapshot


def _boundary_reason_names(plan):
    return {
        boundary.edge_id.index: {reason.value for reason in boundary.reasons}
        for boundary in plan.boundary_edges
    }


def test_coplanar_triangles_form_one_deterministic_segment():
    snapshot = build_square_snapshot()
    plan = segment_mesh(snapshot)

    assert tuple(segment.face_ids for segment in plan.segments) == (
        (FaceId(0), FaceId(1)),
    )
    assert plan.segments[0].topology.euler_characteristic == 1
    assert plan.segments[0].topology.boundary_component_count == 1
    assert plan.segments[0].topology.manifold


def test_seam_splits_faces_and_marks_shared_edge():
    snapshot = build_square_snapshot()
    edges = tuple(
        replace(edge, seam=True) if edge.id == EdgeId(2) else edge
        for edge in snapshot.edges
    )
    plan = segment_mesh(replace(snapshot, edges=edges))

    assert tuple(segment.face_ids for segment in plan.segments) == (
        (FaceId(0),),
        (FaceId(1),),
    )
    assert "SEAM" in _boundary_reason_names(plan)[2]


def test_material_and_angle_policies_are_independent():
    snapshot = build_square_snapshot()
    faces = (
        snapshot.faces[0],
        replace(snapshot.faces[1], material_index=3, normal=(0.0, 1.0, 0.0)),
    )
    plan = segment_mesh(replace(snapshot, faces=faces))
    reasons = _boundary_reason_names(plan)[2]
    assert reasons == {"ANGLE", "MATERIAL"}

    relaxed = segment_mesh(
        replace(snapshot, faces=faces),
        SegmentationSettings(
            split_by_angle=False,
            split_materials=False,
            split_uv_boundaries=False,
        ),
    )
    assert len(relaxed.segments) == 1


def test_uv_discontinuity_splits_without_coordinate_matching():
    snapshot = build_square_snapshot()
    changed_loop = replace(
        snapshot.loops[3],
        uvs=(LoopUV("UVMap", (0.25, 0.25)),),
    )
    changed = replace(
        snapshot,
        loops=snapshot.loops[:3] + (changed_loop,) + snapshot.loops[4:],
    )
    plan = segment_mesh(
        changed,
        SegmentationSettings(split_by_angle=False, split_materials=False),
    )

    assert len(plan.segments) == 2
    assert "UV_DISCONTINUITY" in _boundary_reason_names(plan)[2]


def test_segment_snapshots_preserve_source_lineage():
    snapshot = build_square_snapshot()
    edges = tuple(
        replace(edge, seam=True) if edge.id == EdgeId(2) else edge
        for edge in snapshot.edges
    )
    changed = replace(snapshot, edges=edges)
    plan = segment_mesh(changed)
    segments = materialize_segment_snapshots(changed, plan)

    assert tuple(segment.object_name for segment in segments) == (
        "Cube_Segment_000",
        "Cube_Segment_001",
    )
    assert segments[0].faces[0].source_id == snapshot.faces[0].source_id
    assert segments[1].faces[0].source_id == snapshot.faces[1].source_id


def test_segmentation_is_repeatable():
    snapshot = build_square_snapshot()
    first = segment_mesh(snapshot)
    second = segment_mesh(snapshot)
    assert first == second
