from dataclasses import replace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_uv_preparation import (
    transfer_normal_uv_to_material_bake_snapshot,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    MissingSourceLoopError,
    SourceLoopId,
)

from test_geometry_domain import build_square_snapshot


_GENERATED_LAYER = "SpineBakeUV"


def _generated_uv_snapshot():
    source = build_square_snapshot(snapshot_id="projected", layer="UVMap")
    generated = tuple(
        (
            0.1 + 0.1 * loop.id.index,
            0.2 + 0.05 * loop.id.index,
        )
        for loop in source.loops
    )
    return replace(
        source,
        vertices=tuple(
            replace(
                vertex,
                position=(
                    vertex.position[1] * 7.0 + 10.0,
                    -vertex.position[0] * 3.0 - 5.0,
                    vertex.position[0] - vertex.position[1],
                ),
                normal=(1.0, 0.0, 0.0),
            )
            for vertex in source.vertices
        ),
        loops=tuple(
            loop.with_uv(_GENERATED_LAYER, coordinate)
            for loop, coordinate in zip(source.loops, generated, strict=True)
        ),
        faces=tuple(
            replace(face, normal=(1.0, 0.0, 0.0))
            for face in source.faces
        ),
        uv_layer_names=("UVMap", _GENERATED_LAYER),
        active_uv_layer=_GENERATED_LAYER,
        render_uv_layer="UVMap",
        world_matrix=(
            1.0,
            0.0,
            0.0,
            500.0,
            0.0,
            1.0,
            0.0,
            -250.0,
            0.0,
            0.0,
            1.0,
            90.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    )


def _material_snapshot():
    source = build_square_snapshot(snapshot_id="material", layer="UVMap")
    return replace(
        source,
        render_uv_layer="UVMap",
        world_matrix=(
            0.0,
            -1.0,
            0.0,
            4.0,
            1.0,
            0.0,
            0.0,
            8.0,
            0.0,
            0.0,
            1.0,
            12.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    )


def test_generated_uv_transfer_preserves_material_geometry_exactly():
    projected = _generated_uv_snapshot()
    material = _material_snapshot()

    result, report = transfer_normal_uv_to_material_bake_snapshot(
        projected,
        material,
        layer_name=_GENERATED_LAYER,
    )

    assert report.complete
    assert report.updated_loop_count == len(material.loops)
    assert report.missing_source_loop_ids == ()
    assert report.unused_source_loop_ids == ()

    assert result.vertices == material.vertices
    assert result.edges == material.edges
    assert result.faces == material.faces
    assert result.world_matrix == material.world_matrix
    assert result.render_uv_layer == material.render_uv_layer == "UVMap"
    assert result.active_uv_layer == _GENERATED_LAYER

    expected_by_source_loop = {
        loop.source_id: loop.uv(_GENERATED_LAYER)
        for loop in projected.loops
    }
    assert {
        loop.source_id: loop.uv(_GENERATED_LAYER)
        for loop in result.loops
    } == expected_by_source_loop


def test_generated_uv_transfer_rejects_invalid_contracts():
    projected = _generated_uv_snapshot()
    material = _material_snapshot()

    with pytest.raises(TypeError, match="projected_uv_snapshot"):
        transfer_normal_uv_to_material_bake_snapshot(
            object(),
            material,
            layer_name=_GENERATED_LAYER,
        )
    with pytest.raises(TypeError, match="material_snapshot"):
        transfer_normal_uv_to_material_bake_snapshot(
            projected,
            object(),
            layer_name=_GENERATED_LAYER,
        )
    with pytest.raises(ValueError, match="layer_name"):
        transfer_normal_uv_to_material_bake_snapshot(
            projected,
            material,
            layer_name="",
        )


def test_generated_uv_transfer_fails_on_lineage_loss():
    projected = _generated_uv_snapshot()
    material = _material_snapshot()
    material = replace(
        material,
        loops=(
            replace(
                material.loops[0],
                source_id=SourceLoopId("Cube", 99, 0),
            ),
            *material.loops[1:],
        ),
    )

    with pytest.raises(MissingSourceLoopError):
        transfer_normal_uv_to_material_bake_snapshot(
            projected,
            material,
            layer_name=_GENERATED_LAYER,
        )
