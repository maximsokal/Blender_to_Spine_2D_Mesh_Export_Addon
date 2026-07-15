from dataclasses import replace

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1GeometryPreparationSettings,
    build_a1_texturing_topology,
    prepare_a1_geometry_regions,
    propagate_texturing_uv_to_regions,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    LoopUV,
    SegmentationSettings,
)

from test_a1_segmentation_decomposition import (
    build_quad_ring,
    build_three_quad_strip,
)
from test_geometry_domain import build_square_snapshot


def test_seed_normal_internal_cut_becomes_shared_texturing_seam():
    source = build_three_quad_strip()
    geometry = prepare_a1_geometry_regions(
        source,
        A1GeometryPreparationSettings(
            segmentation=SegmentationSettings(
                angle_limit_degrees=30.0,
                split_uv_boundaries=False,
            )
        ),
    )

    topology = build_a1_texturing_topology(source, geometry)

    assert len(geometry.segmentation.segments) == 2
    assert len(topology.segmentation_seam_edge_ids) == 1
    seam_id = topology.segmentation_seam_edge_ids[0]
    assert topology.snapshot.edge_by_id()[seam_id].seam
    assert seam_id in topology.all_seam_edge_ids


def test_decomposition_cuts_are_added_to_full_object_texturing_snapshot():
    source = build_quad_ring()
    geometry = prepare_a1_geometry_regions(
        source,
        A1GeometryPreparationSettings(
            segmentation=SegmentationSettings(
                split_by_angle=False,
                split_materials=False,
                split_uv_boundaries=False,
            )
        ),
    )

    topology = build_a1_texturing_topology(source, geometry)

    assert len(geometry.regions) > 1
    assert topology.decomposition_seam_edge_ids
    assert set(topology.decomposition_seam_edge_ids).issubset(
        set(topology.all_seam_edge_ids)
    )
    assert all(
        topology.snapshot.edge_by_id()[edge_id].seam
        for edge_id in topology.decomposition_seam_edge_ids
    )


def test_existing_source_seams_are_preserved():
    source = build_square_snapshot()
    marked = replace(
        source,
        edges=(replace(source.edges[0], seam=True),) + source.edges[1:],
    )
    geometry = prepare_a1_geometry_regions(marked)

    topology = build_a1_texturing_topology(marked, geometry)

    assert topology.existing_seam_edge_ids == (source.edges[0].id,)
    assert topology.snapshot.edges[0].seam


def test_global_uvs_transfer_to_every_duplicated_triangulation_loop():
    source = build_square_snapshot()
    geometry = prepare_a1_geometry_regions(source)
    topology = build_a1_texturing_topology(source, geometry)

    propagation = propagate_texturing_uv_to_regions(
        topology.snapshot,
        geometry,
        source_layer_name="UVMap",
        target_layer_name="SpineBakeUV",
    )

    assert len(propagation.regions) == 1
    region = propagation.regions[0]
    assert region.transfer_report.complete
    assert region.transfer_report.updated_loop_count == 6
    assert len(region.snapshot.loops) == 6
    assert region.snapshot.active_uv_layer == "SpineBakeUV"
    assert tuple(loop.uv("SpineBakeUV") for loop in region.snapshot.loops) == (
        (1.0, 1.0),
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    )


def test_global_uv_propagation_uses_changed_unwrap_layer_not_old_region_uv():
    source = build_square_snapshot()
    geometry = prepare_a1_geometry_regions(source)
    topology = build_a1_texturing_topology(source, geometry)
    changed_loops = tuple(
        replace(
            loop,
            uvs=tuple(
                uv
                for uv in loop.uvs
                if uv.layer_name != "GlobalBakeUV"
            )
            + (
                LoopUV(
                    "GlobalBakeUV",
                    (
                        loop.uv("UVMap")[0] * 0.25 + 0.1,
                        loop.uv("UVMap")[1] * 0.25 + 0.2,
                    ),
                ),
            ),
        )
        for loop in topology.snapshot.loops
    )
    textured = replace(
        topology.snapshot,
        loops=changed_loops,
        uv_layer_names=tuple(sorted(set(topology.snapshot.uv_layer_names) | {"GlobalBakeUV"})),
        active_uv_layer="GlobalBakeUV",
    )

    propagation = propagate_texturing_uv_to_regions(
        textured,
        geometry,
        source_layer_name="GlobalBakeUV",
        target_layer_name="SpineBakeUV",
    )

    region_uvs = tuple(loop.uv("SpineBakeUV") for loop in propagation.regions[0].snapshot.loops)
    assert region_uvs == (
        (0.35, 0.45),
        (0.1, 0.2),
        (0.35, 0.2),
        (0.1, 0.2),
        (0.35, 0.45),
        (0.1, 0.45),
    )


def test_texturing_layout_pipeline_is_repeatable():
    source = build_square_snapshot()
    geometry = prepare_a1_geometry_regions(source)
    first_topology = build_a1_texturing_topology(source, geometry)
    second_topology = build_a1_texturing_topology(source, geometry)
    assert first_topology == second_topology

    first_uv = propagate_texturing_uv_to_regions(
        first_topology.snapshot,
        geometry,
        source_layer_name="UVMap",
    )
    second_uv = propagate_texturing_uv_to_regions(
        second_topology.snapshot,
        geometry,
        source_layer_name="UVMap",
    )
    assert first_uv == second_uv
