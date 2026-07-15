from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1GeometryPreparationSettings,
    prepare_a1_geometry_regions,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    DecompositionSettings,
    SegmentationSettings,
    TriangulationSettings,
    analyse_face_region,
    is_simple_disk,
)

from test_complex_decomposition import build_quad_ring
from test_geometry_domain import build_square_snapshot
from test_geometry_triangulation import build_polygon_snapshot


def test_square_prepares_one_triangulated_disk_region():
    source = build_square_snapshot()
    result = prepare_a1_geometry_regions(source)

    assert result.source_snapshot_id == source.snapshot_id
    assert len(result.segmentation.segments) == 1
    assert len(result.decomposition.regions) == 1
    assert len(result.regions) == 1
    region = result.regions[0]
    assert region.region_index == 0
    assert tuple(face.id.index for face in region.snapshot.faces) == (0, 1)
    assert all(len(face.loop_ids) == 3 for face in region.snapshot.faces)
    assert is_simple_disk(
        analyse_face_region(
            region.snapshot,
            tuple(face.id for face in region.snapshot.faces),
        )
    )


def test_single_quad_ngon_is_triangulated_after_disk_materialization():
    source = build_polygon_snapshot(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 2.0, 0.0),
            (0.0, 2.0, 0.0),
        ),
        name="QuadNgon",
    )
    result = prepare_a1_geometry_regions(source)

    assert len(result.regions) == 1
    region = result.regions[0]
    assert region.source_face_ids == (source.faces[0].source_id,)
    assert len(region.snapshot.faces) == 2
    assert len(region.triangulation.generated_edge_ids) == 1


def test_ring_is_decomposed_into_complete_disjoint_disk_regions():
    source = build_quad_ring()
    result = prepare_a1_geometry_regions(source)

    assert len(result.segmentation.segments) == 1
    assert len(result.regions) > 1
    covered_source_faces = [
        source_face_id
        for region in result.regions
        for source_face_id in region.source_face_ids
    ]
    assert len(covered_source_faces) == len(source.faces)
    assert len(covered_source_faces) == len(set(covered_source_faces))
    assert set(covered_source_faces) == {face.source_id for face in source.faces}
    for region in result.regions:
        assert all(len(face.loop_ids) == 3 for face in region.snapshot.faces)
        topology = analyse_face_region(
            region.snapshot,
            tuple(face.id for face in region.snapshot.faces),
        )
        assert is_simple_disk(topology)


def test_geometry_preparation_is_repeatable():
    source = build_quad_ring()
    settings = A1GeometryPreparationSettings(
        segmentation=SegmentationSettings(
            split_by_angle=False,
            split_materials=False,
            split_uv_boundaries=False,
        ),
        decomposition=DecompositionSettings(),
        triangulation=TriangulationSettings(),
    )

    first = prepare_a1_geometry_regions(source, settings)
    second = prepare_a1_geometry_regions(source, settings)

    assert first == second


def test_prepared_region_indices_are_dense_and_source_segment_ids_are_stable():
    result = prepare_a1_geometry_regions(build_quad_ring())

    assert tuple(region.region_index for region in result.regions) == tuple(
        range(len(result.regions))
    )
    assert all(region.source_segment_id == 0 for region in result.regions)
