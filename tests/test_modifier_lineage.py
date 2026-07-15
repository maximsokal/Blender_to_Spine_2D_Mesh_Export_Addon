import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    EvaluatedLineageError,
    LineageSeverity,
    ModifierLineagePolicy,
    analyse_evaluated_lineage,
    require_valid_evaluated_lineage,
)


SOURCE_FACE_CORNERS = (3, 4)


def analyse(
    *,
    vertices=(0, 1, 2, 3),
    edges=(0, 1, 2, 3, 4),
    faces=(0, 1),
    corner_faces=(0, 0, 0, 1, 1, 1, 1),
    corner_indices=(0, 1, 2, 0, 1, 2, 3),
    policy=ModifierLineagePolicy.STRICT_PRESERVE,
):
    return analyse_evaluated_lineage(
        source_vertex_count=4,
        source_edge_count=5,
        source_face_corner_counts=SOURCE_FACE_CORNERS,
        vertex_source_indices=vertices,
        edge_source_indices=edges,
        face_source_indices=faces,
        corner_source_face_indices=corner_faces,
        corner_source_corner_indices=corner_indices,
        policy=policy,
    )


def test_strict_preserve_accepts_exact_topology_lineage():
    report = analyse()
    assert report.valid
    assert report.issues == ()
    require_valid_evaluated_lineage(report)


def test_strict_preserve_rejects_duplicated_source_elements():
    report = analyse(
        vertices=(0, 1, 2, 3, 3),
        faces=(0, 1, 1),
        corner_faces=(0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1),
        corner_indices=(0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3),
    )
    assert not report.valid
    assert any(issue.code == "TOPOLOGY_COUNT_CHANGED" for issue in report.issues)
    assert any(issue.code == "SOURCE_ELEMENTS_DUPLICATED" for issue in report.issues)
    with pytest.raises(EvaluatedLineageError):
        require_valid_evaluated_lineage(report)


def test_duplicate_policy_allows_mirror_or_triangulate_style_copying():
    report = analyse(
        vertices=(0, 1, 2, 3, 0, 1, 2, 3),
        edges=(0, 1, 2, 3, 4, None),
        faces=(0, 1, 1),
        corner_faces=(0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1),
        corner_indices=(0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3),
        policy=ModifierLineagePolicy.ALLOW_SOURCE_DUPLICATION,
    )
    assert report.valid
    assert report.vertices.duplicated_source_indices == (0, 1, 2, 3)
    assert report.faces.duplicated_source_indices == (1,)
    assert any(
        issue.code == "GENERATED_EDGES" and issue.severity is LineageSeverity.WARNING
        for issue in report.issues
    )


def test_generated_vertex_face_or_corner_is_rejected():
    report = analyse(
        vertices=(0, 1, 2, None),
        faces=(0, None),
        corner_faces=(0, 0, 0, 1, 1, 1, None),
        corner_indices=(0, 1, 2, 0, 1, 2, None),
        policy=ModifierLineagePolicy.ALLOW_SOURCE_DUPLICATION,
    )
    assert not report.valid
    unknown_channels = {
        issue.channel
        for issue in report.issues
        if issue.code == "UNKNOWN_SOURCE_LINEAGE"
    }
    assert unknown_channels == {"vertices", "faces", "corners"}


def test_out_of_range_corner_identity_is_rejected_with_context():
    report = analyse(
        corner_faces=(0, 0, 0, 1, 1, 1, 1),
        corner_indices=(0, 1, 7, 0, 1, 2, 3),
        policy=ModifierLineagePolicy.ALLOW_SOURCE_DUPLICATION,
    )
    assert not report.valid
    assert any(issue.code == "CORNER_INDEX_OUT_OF_RANGE" for issue in report.issues)
